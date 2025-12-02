
# run_ppo.py
from reward_model import RewardModel
from ppo_rlhf import PPOTrainer

# 你自己的 prompt 列表，可以直接用 compare_model.py 里那套 10 个 prompt
PROMPTS = [
    "Write a positive review of a newly opened Italian restaurant, using an enthusiastic tone.",
    "The terrible service at this restaurant made me very unhappy. Write a strong negative complaint.",
    "Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone.",
    "Recommend the most delicious dish you have ever eaten, using an extremely excited tone.",
    "How would you review a coffee shop where the coffee is bad but the desserts are excellent?",
    "Briefly state your opinion on a restaurant that offers takeout service, focusing on convenience.",
    "Write a cautionary review about a restaurant's hygiene issues.",
    "Describe a heartwarming experience during a family dinner.",
    "Rate the restaurant's parking and accessibility.",
    "You dined at an upscale restaurant. Write a complaint about the price.",
]


def main():
    # 1. 初始化 RewardModel（judge）
    rm = RewardModel(
        alpha=0.7,
        beta=0.4,
        gamma=0.3,
        delta=0.3,
    )

    # 2. 从 SFT 模型初始化 PPO
    sft_model_path = "../models/sft_model"
    trainer = PPOTrainer(model_path=sft_model_path, reward_fn=rm.compute_reward)

    # 3. PPO 训练
    trainer.train(PROMPTS)

    # 4. 保存 RLHF 模型
    rlhf_save_path = "../models/rlhf_model"
    trainer.save_policy(rlhf_save_path)


if __name__ == "__main__":
    main()
