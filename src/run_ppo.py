from ppo_rlhf import PPOTrainer
# from sft import toy_reward_fn # No longer needed
from reward_model import RewardModel # Import the class

prompts = [
    "Write a positive review:",
    "Write a negative review:",
    "Describe your day:",
    "Tell me something funny:"
]

# --- FIX (Part 1) ---
# 1. Create an INSTANCE of the RewardModel first.
# This will load your fine-tuned sentiment classifier *once*.
print("Initializing Reward Model...")
reward_model_instance = RewardModel()
print("Reward Model Initialized.")
# --- END FIX ---

# --- FIX (Part 2) ---
# 2. Pass the specific INSTANCE METHOD (.compute_reward) to the trainer.
# Now, self.reward_fn inside the trainer will be a callable function.
ppo = PPOTrainer(
    model_path="../models/sft_model", 
    reward_fn=reward_model_instance.compute_reward # Pass the method
)
# --- END FIX ---

ppo.train(prompts)

print("Saving final RLHF model...")
ppo.policy.save_pretrained("../models/rlhf_model")
print("Done.")