import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    lr: float = 2e-6
    kl_coef: float = 0.2
    clip_range: float = 0.2
    ppo_epochs: int = 1
    batch_size: int = 8
    max_new_tokens: int = 40
    entropy_coef: float = 0.01
    kl_threshold: float = 2.0     # early stop if KL too large


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

        self.train_logs = {
            "iteration": [],
            "reward": [],
            "kl": [],
            "length": [],
            "distinct_1": []
        }

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
        selected = torch.clamp(selected, -30, 30)


        return selected.sum(dim=1)  # per-sequence logprob

    
    # PPO Step
    
    def ppo_step(self, prompts, responses, rewards, old_logprobs):
        batch_size = len(prompts)
        current_kl = 0.0  

        

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
                # --- NaN CHECK (must be BEFORE KL computation) ---
                if torch.isnan(new_lp).any() or torch.isnan(ref_lp).any():
                    print("⚠ NaN detected in logprob — skipping this sample")
                    continue

                # KL divergence
                kl = (new_lp - ref_lp).mean()
                kl = torch.clamp(kl, -2.0, 2.0)
                current_kl = kl.item()


                # KL early stopping
                if kl.item() > self.cfg.kl_threshold:
                    print(f"⚠ KL too high ({kl.item():.2f}), early stopping PPO epoch")
                    break

                # Advantage
                adv = rewards[i] - self.cfg.kl_coef * kl.item()
                adv = torch.tensor([adv], device=self.device)

                # PPO ratio
                ratio = torch.exp(torch.clamp(new_lp - old_logprobs[i], -10, 10))

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
        return current_kl 


    
    # Main Trainer
    
    def train(self, prompts):
        print(" Starting PPO RLHF training...\n")

        for start in range(0, len(prompts), self.cfg.batch_size):
            batch_prompts = prompts[start:start + self.cfg.batch_size]
            if len(batch_prompts) < self.cfg.batch_size:
                print("⚠ skipping last incomplete batch")
                continue

            # print("batch_prompts =", batch_prompts)

            valid_prompts = []
            responses = []
            rewards = []
            old_logprobs = []

            # Rollout
            for p in batch_prompts:
                r = self.generate(p)
                # --- EMPTY RESPONSE CHECK ---
                if len(r.strip()) == 0:
                    print(f"⚠ Empty response for prompt: {p} — skipping")
                    continue
                # ----------------------------
                valid_prompts.append(p)
                responses.append(r)

                # --- FIX ---
                # The original line was:
                # rew = float(self.reward_fn(r))
                # This was wrong for two reasons:
                # 1. self.reward_fn (which is RewardModel.compute_reward)
                #    needs BOTH the prompt and the response.
                # 2. It returns a TUPLE of 4 values, not a single float.
                
                # Call the reward function with both prompt and response
                # (Your PPO trainer now correctly uses the prompt in reward calculation)
                reward_tuple = self.reward_fn(prompt=p, response=r)
                
                # The first element is the final_reward
                rew = float(reward_tuple[0])
                # --- END FIX ---
                
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

                

            if len(rewards) == 0:
                print("⚠ No valid samples in this batch, skipping PPO step.")
                continue
            # --- Logging ---
            avg_reward = float(np.mean(rewards))
            avg_length = float(np.mean([len(r.split()) for r in responses]))

            # distinct-1
            all_tokens = []
            for r in responses:
                toks = r.lower().split()
                all_tokens += toks
            distinct_1 = len(set(all_tokens)) / (len(all_tokens) + 1e-8)

            self.train_logs["iteration"].append(start)
            self.train_logs["reward"].append(avg_reward)
            self.train_logs["length"].append(avg_length)
            self.train_logs["distinct_1"].append(distinct_1)

            # Normalize
            rewards = np.array(rewards, dtype=np.float32)
            if rewards.std() > 1e-8:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                rewards = rewards - rewards.mean()


            # PPO step
            rewards_tensor = torch.tensor(rewards, device=self.device)
            current_kl = self.ppo_step(valid_prompts, responses, rewards_tensor, old_logprobs)
            self.train_logs["kl"].append(current_kl)
        import json
        with open("ppo_training_logs.json", "w") as f:
            json.dump(self.train_logs, f, indent=2)
        print("Saved PPO logs to ppo_training_logs.json")


        print("\n🎉 Finished PPO RLHF training!")
