from ppo_rlhf import PPOTrainer
# from sft import toy_reward_fn # No longer needed
from reward_model import RewardModel # Import the class

# prompts = [
#     "Write a positive review:",
#     "Write a negative review:",
#     "Describe your day:",
#     "Tell me something funny:"
# ]
prompts = [
    "Write a positive review of a newly opened Italian restaurant, using an enthusiastic tone.",
    "The terrible service at this restaurant made me very unhappy. Write a strong negative complaint.",
    "Objectively describe the dishes and environment of a Chinese restaurant, maintaining a neutral tone.",
    "Recommend the most delicious dish you have ever eaten, using an extremely excited tone.",
    "How would you review a coffee shop where the coffee is bad but the desserts are excellent?",
    "Briefly state your opinion on a restaurant that offers takeout service, focusing on convenience.",
    "Write a cautionary review about a restaurant's hygiene issues.",
    "Describe a heartwarming experience during a family dinner.",
    "Rate the restaurant's parking and accessibility.",
    "You dined at an upscale restaurant. Write a complaint about the price."
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
