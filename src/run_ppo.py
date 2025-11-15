from ppo_rlhf import PPOTrainer
from sft import toy_reward_fn

prompts = [
    "Write a positive review:",
    "Write a negative review:",
    "Describe your day:",
    "Tell me something funny:"
]

ppo = PPOTrainer(model_path="../models/sft_model", reward_fn=toy_reward_fn)
ppo.train(prompts)
ppo.policy.save_pretrained("../models/rlhf_model")
