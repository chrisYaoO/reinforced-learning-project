import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 5e-6
    kl_coef: float = 0.1
    clip_range: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 8
    max_new_tokens: int = 40
    entropy_coef: float = 0.0
    kl_threshold: float = 3.0


class PPOTrainer:
    """
    使用 PPO 对 DistilGPT2（或其它 CausalLM）进行 RLHF 微调：

    - policy: 可训练策略模型
    - ref: 冻结参考模型（通常是 SFT 模型复制）
    - reward_fn: 接受 (prompt, response) → (final_reward, ...)
    """

    def __init__(self, model_path: str, reward_fn):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.ref = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        for p in self.ref.parameters():
            p.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_fn = reward_fn
        self.cfg = PPOConfig()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    # ========= 采样 =========

    def generate(self, prompt: str) -> str:
        """
        从 policy 采样一个 response，用于 Rollout。
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.policy.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if full.startswith(prompt):
            return full[len(prompt):].strip()
        return full.strip()

    # ========= logprobs =========

    def compute_logprobs(self, model, input_ids, attention_mask, labels):
        """
        计算给定 labels 下，响应 token 的平均 log-prob:
            - 使用 CrossEntropyLoss 得到 per-token NLL
            - 对非 -100 的位置取平均 → sequence_logprob
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())  # [B, T-1]

        mask = (shift_labels != -100).float()
        token_count = mask.sum(dim=1)
        token_count[token_count == 0] = 1.0

        seq_nll = (loss * mask).sum(dim=1)
        seq_logprobs = -seq_nll / token_count  # 平均 logprob

        return seq_logprobs  # [B]

    # ========= PPO 更新 =========

    def ppo_step(self, prompts, responses, rewards, old_logprobs):
        batch_size = len(prompts)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        old_logprobs = torch.cat(old_logprobs).detach().to(self.device)
        if old_logprobs.dim() == 0:
            old_logprobs = old_logprobs.unsqueeze(0)

        for epoch in range(self.cfg.ppo_epochs):
            new_lps = []
            ref_lps = []

            for i in range(batch_size):
                p = prompts[i]
                r = responses[i]
                text = p + r

                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                labels = input_ids.clone()
                prompt_len = len(self.tokenizer(p).input_ids)
                labels[:, :prompt_len] = -100
                labels[labels == self.tokenizer.pad_token_id] = -100

                new_lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)
                with torch.no_grad():
                    ref_lp = self.compute_logprobs(self.ref, input_ids, attention_mask, labels)

                new_lps.append(new_lp)
                ref_lps.append(ref_lp)

            new_lps = torch.cat(new_lps).to(self.device)
            ref_lps = torch.cat(ref_lps).to(self.device)

            # KL approx (average per sequence)
            kl = new_lps - ref_lps
            mean_kl = kl.mean()

            if mean_kl.item() > self.cfg.kl_threshold:
                print(f"[PPO] KL too high ({mean_kl.item():.4f}), early stop this epoch.")
                break

            # non-score reward: KL penalty
            non_score_reward = -self.cfg.kl_coef * kl
            advantages = rewards + non_score_reward

            if batch_size > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_lps - old_logprobs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range) * advantages

            ppo_loss = -torch.min(surr1, surr2).mean()
            loss = ppo_loss  # 目前不加 entropy，可后续 ablation

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            print(f"[PPO] epoch {epoch+1}/{self.cfg.ppo_epochs}, loss={loss.item():.4f}, mean KL={mean_kl.item():.4f}")

    # ========= 主训练循环 =========

    def train(self, prompts):
        print("Starting PPO RLHF training...\n")

        for start in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[start: start + self.cfg.batch_size]

            responses = []
            rewards = []
            old_logprobs = []

            for p in batch_prompts:
                r = self.generate(p)
                responses.append(r)

                reward_tuple = self.reward_fn(prompt=p, response=r)
                rew = float(reward_tuple[0])  # final_reward
                rewards.append(rew)

                text = p + r
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                labels = input_ids.clone()
                prompt_len = len(self.tokenizer(p).input_ids)
                labels[:, :prompt_len] = -100
                labels[labels == self.tokenizer.pad_token_id] = -100

                with torch.no_grad():
                    lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)

                old_logprobs.append(lp)

            self.ppo_step(batch_prompts, responses, rewards, old_logprobs)

        print("\nFinished PPO RLHF training!")

    def save_policy(self, save_path: str):
        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[PPOTrainer] policy model saved to {save_path}")
