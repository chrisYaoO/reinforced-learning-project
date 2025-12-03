
# run_ppo.py
from reward_model import RewardModel
from ppo_rlhf import PPOTrainer
import json

# load prompts
with open("src/prompts.txt", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)

print(type(PROMPTS))



def main():
    # 1. initializae reward model
    rm = RewardModel()

    # 2. initialize PPO
    sft_model_path = "../models/sft_model"
    trainer = PPOTrainer(model_path=sft_model_path, reward_fn=rm.compute_reward)

    # 3. train
    trainer.train(PROMPTS)

    # 4. save
    rlhf_save_path = "../models/rlhf_model"
    trainer.save_policy(rlhf_save_path)


if __name__ == "__main__":
    main()