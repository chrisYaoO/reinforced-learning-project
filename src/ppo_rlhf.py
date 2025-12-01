import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 2e-6
    kl_coef: float = 0.3
    clip_range: float = 0.1
    ppo_epochs: int = 8
    batch_size: int = 8
    max_new_tokens: int = 40
    entropy_coef: float = 0.0  # Kept for reference, but set to 0.0
    kl_threshold: float = 1.0  # early stop if KL too large
       



class PPOTrainer:
    def __init__(self, model_path: str, reward_fn):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Policy & reference models
        self.policy = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.ref = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        for p in self.ref.parameters():
            p.requires_grad = False

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 确保 pad_token 设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_fn = reward_fn
        self.cfg = PPOConfig()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    # Generation (Rollout)
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.policy.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.2, # ADDED: Apply penalty during PPO sampling to fight repetition
                pad_token_id=self.tokenizer.eos_token_id
            )
        # 解码并去除 prompt 部分
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # 简单处理：找到 prompt 结束后的内容作为 response
        if full.startswith(prompt):
            return full[len(prompt):].strip()
        else:
            # 如果模型在生成过程中修改了 prompt，返回全部生成内容（非最佳做法，但应对复杂场景）
            return full.strip()

    
    # 1. 优化 logprobs 计算：改为返回 token-level 的 mean (平均值)
    def compute_logprobs(self, model, input_ids, attention_mask, labels):
        """
        计算模型在给定 labels 下的平均 log-probability。
        返回的是 token-level 的平均值，而不是 per-sequence 的总和。
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift logits and labels for language modeling
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        
        # CrossEntropyLoss 默认计算 Negative Log Likelihood (NLL)
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        # 拉平张量进行计算
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        # 还原形状 [Batch_Size, Sequence_Length - 1]
        loss = loss.view(shift_labels.size())
        
        # 掩码：只计算非 -100 的 token (即 response token)
        mask = (shift_labels != -100).float()
        
        # 响应长度 (非 -100 token 的数量)
        response_lengths = mask.sum(dim=1)
        # 防止除以零
        response_lengths[response_lengths == 0] = 1 
        
        # Sequence NLL: (loss * mask).sum(dim=1)
        sequence_nll = (loss * mask).sum(dim=1)
        
        # Sequence Logprob (Mean): -NLL_sum / N_tokens
        sequence_logprobs = -sequence_nll / response_lengths
        
        # 如果输入是 [1, L]，则返回 size [1] 的 tensor
        return sequence_logprobs
    
    # 2. PPO Step (重构为批处理，并引入 Advantage Normalization)
    def ppo_step(self, prompts, responses, rewards, old_logprobs):
        batch_size = len(prompts)

        # 将 rewards 和 old_logprobs 转换为 tensor 方便批处理
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        # old_logprobs 是一个列表，需要合并成一个 tensor
        old_logprobs_tensor = torch.cat(old_logprobs).to(self.device)
        
        # 确保 old_logprobs_tensor 形状正确
        if old_logprobs_tensor.dim() == 0:
             old_logprobs_tensor = old_logprobs_tensor.unsqueeze(0)

        for epoch in range(self.cfg.ppo_epochs):
            
            # --- 1. 重新计算当前 Logprobs, Ref Logprobs ---
            
            new_lps = []
            ref_lps = []
            
            # 简化起见，这里可以再次遍历，但更高效的做法是使用 DataLoader 或 padding/truncation
            # 鉴于当前代码结构，我们先保持逐个编码
            for i in range(batch_size):
                p = prompts[i]
                r = responses[i]
                
                # 注意：这里我们只处理单个序列，所以无需 padding
                text = p + r
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                # Build labels (masking prompt and padding)
                labels = input_ids.clone()
                prompt_ids = self.tokenizer(p).input_ids
                prompt_len = len(prompt_ids)

                labels[:, :prompt_len] = -100
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Compute logprobs (token-level mean)
                new_lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)
                with torch.no_grad():
                    ref_lp = self.compute_logprobs(self.ref, input_ids, attention_mask, labels)
                    
                new_lps.append(new_lp)
                ref_lps.append(ref_lp)
            
            # 将收集到的 logprobs 合并成批次 tensor
            new_lps = torch.cat(new_lps).to(self.device)
            ref_lps = torch.cat(ref_lps).to(self.device)
            
            # --- 2. PPO 核心计算 (批处理) ---
            
            # KL divergence (token-level mean)
            kl = new_lps - ref_lps 
            mean_kl = kl.mean()

            # KL early stopping
            if mean_kl.item() > self.cfg.kl_threshold:
                print(f"⚠ KL too high ({mean_kl.item():.4f}), early stopping PPO epoch")
                break
                
            # Compute Advantage (Total Reward: R(x,y) - kl_coef * KL)
            non_score_reward = -self.cfg.kl_coef * kl
            advantages = rewards + non_score_reward
            
            # === [关键修改] Advantage Normalization ===
            if batch_size > 1:
                # 标准化: (Adv - Mean) / (Std + epsilon)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO ratio
            ratio = torch.exp(new_lps - old_logprobs_tensor)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 
                                 1.0 - self.cfg.clip_range, 
                                 1.0 + self.cfg.clip_range) * advantages

            ppo_loss = -torch.min(surr1, surr2).mean() # Mean across the batch

            # Final Loss: 只使用 PPO Loss (移除有误的 Entropy 项)
            loss = ppo_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            print(f"[PPO] epoch {epoch+1}/{self.cfg.ppo_epochs}, avg loss={loss.item():.4f}, mean KL={mean_kl.item():.4f}")

    
    # Main Trainer
    
    def train(self, prompts):
        print(" Starting PPO RLHF training...\n")

        for start in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[start:start + self.cfg.batch_size]

            responses = []
            rewards = []
            old_logprobs = []

            # Rollout
            for p in batch_prompts:
                r = self.generate(p)
                responses.append(r)

                # Call the reward function with both prompt and response
                reward_tuple = self.reward_fn(prompt=p, response=r)
                # The first element is the final_reward
                rew = float(reward_tuple[0])
                rewards.append(rew)

                # Calculate old logprobs for PPO update
                text = p + r
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                labels = input_ids.clone()
                # 再次获取 prompt token id 长度，确保一致性
                prompt_len = len(self.tokenizer(p).input_ids)
                labels[:, :prompt_len] = -100

                with torch.no_grad():
                    # 现在 compute_logprobs 返回的是 token-level mean
                    lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)

                old_logprobs.append(lp)

            # PPO step (现在支持批处理)
            self.ppo_step(batch_prompts, responses, rewards, old_logprobs)

        print("\n🎉 Finished PPO RLHF training!")
