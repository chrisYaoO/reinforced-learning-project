import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 2e-6
    kl_coef: float = 0.02
    clip_range: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 4
    max_new_tokens: int = 40
    entropy_coef: float = 0.01
    kl_threshold: float = 4.0     # early stop if KL too large


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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.reward_fn = reward_fn
        self.cfg = PPOConfig()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    # Generation
    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.policy.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    
    # Safe logprob extraction (supports -100 masking)
    
    def compute_logprobs(self, model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100)

        # replace -100 for gather (but mask removes contribution)
        safe_labels = shift_labels.clone()
        safe_labels[~mask] = 0

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        selected = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        selected = selected * mask  # ignore masked positions

        return selected.sum(dim=1)  # per-sequence logprob

    
    # PPO Step
    
    def ppo_step(self, prompts, responses, rewards, old_logprobs):
        batch_size = len(prompts)

        for epoch in range(self.cfg.ppo_epochs):
            losses = []

            for i in range(batch_size):

                p = prompts[i]
                r = responses[i]

                text = p + r
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                # Build labels
                labels = input_ids.clone()
                prompt_ids = self.tokenizer(p, return_tensors="pt").input_ids
                prompt_len = prompt_ids.shape[1]

                labels[:, :prompt_len] = -100
                labels[labels == self.tokenizer.pad_token_id] = -100

                # logprob (policy & ref)
                new_lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)
                with torch.no_grad():
                    ref_lp = self.compute_logprobs(self.ref, input_ids, attention_mask, labels)

                # KL divergence
                kl = (new_lp - ref_lp).mean()

                # KL early stopping
                if kl.item() > self.cfg.kl_threshold:
                    print(f"⚠ KL too high ({kl.item():.2f}), early stopping PPO epoch")
                    break

                # Advantage
                adv = rewards[i] - self.cfg.kl_coef * kl.item()
                adv = torch.tensor([adv], device=self.device)

                # PPO ratio
                ratio = torch.exp(new_lp - old_logprobs[i])

                # Clipped objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio,
                                    1 - self.cfg.clip_range,
                                    1 + self.cfg.clip_range) * adv

                ppo_loss = -torch.min(surr1, surr2)

                # Entropy bonus (stabilizes learning)
                entropy = -(new_lp * torch.exp(new_lp)).mean()
                loss = ppo_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                losses.append(loss.item())

            print(f"[PPO] epoch {epoch+1}/{self.cfg.ppo_epochs}, avg loss={np.mean(losses):.4f}")

    
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

                rew = float(self.reward_fn(r))
                rewards.append(rew)

                text = p + r
                enc = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = enc.input_ids
                attention_mask = enc.attention_mask

                labels = input_ids.clone()
                prompt_len = len(self.tokenizer(p).input_ids)
                labels[:, :prompt_len] = -100

                with torch.no_grad():
                    lp = self.compute_logprobs(self.policy, input_ids, attention_mask, labels)

                old_logprobs.append(lp)

            # PPO step
            rewards_tensor = torch.tensor(rewards, device=self.device)
            self.ppo_step(batch_prompts, responses, rewards_tensor, old_logprobs)

        print("\n🎉 Finished PPO RLHF training!")
