from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sft import toy_reward_fn
from reward_model_PAST import RewardModel
import re
import numpy as np # Added numpy for mean calculation

# ============================================================
# Diversity helpers
# ============================================================
def distinct_n_ratio(texts, n=2):
    def tok(s):
        return re.findall(r"\w+|\S", s)

    ngrams = []
    for t in texts:
        toks = tok(t)
        ngrams += [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]

    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def trigram_repeat_rate(texts):
    def tok(s):
        return re.findall(r"\w+|\S", s)

    total, dup = 0, 0
    for t in texts:
        toks = tok(t)
        tri = [tuple(toks[i:i+3]) for i in range(len(toks)-2)]
        total += len(tri)
        dup += (len(tri) - len(set(tri)))
    return dup / total if total > 0 else 0.0


def evaluate_diversity(texts):
    if len(texts) == 0:
        return {
            "avg_len": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "tri_repeat": 0.0
        }

    avg_len = sum(len(t.split()) for t in texts) / len(texts)
    return {
        "avg_len": avg_len,
        "distinct_1": distinct_n_ratio(texts, n=1),
        "distinct_2": distinct_n_ratio(texts, n=2),
        "tri_repeat": trigram_repeat_rate(texts)
    }


# ============================================================
# RLHF model evaluator
# ============================================================
class RLHFEvaluator:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        # Load tokenizer from SFT model path assumption
        # Note: We assume the SFT model path is derived by replacing "rlhf" with "sft"
        sft_path = model_path.replace("rlhf", "sft")
        self.tokenizer = AutoTokenizer.from_pretrained(sft_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.2, # ADDED: Introduce a penalty for repeating tokens to combat mode collapse
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ============================================================
# Main evaluation function
# ============================================================
def evaluate_rlhf_model(model_path, prompts, reward_model: RewardModel, n_samples=10):
    evaluator = RLHFEvaluator(model_path)
    generations = {}

    # Generate many samples per prompt
    print(f"Generating {n_samples} samples for {len(prompts)} prompts...")
    for p in prompts:
        gens = []
        for _ in range(n_samples):
            full_text = evaluator.generate(p)
            response_only = full_text[len(p):].strip()
            gens.append(response_only)
        generations[p] = gens

    # --- REWARD CALCULATION FIX ---
    # We must iterate over prompts (p) and responses (o) together
    all_rewards = []
    for p, outs in generations.items(): # Iterate over items to get prompt (p)
        for o in outs: # o is the response
            # 1. Call the correct method: .compute_reward()
            # 2. Pass both prompt (p) and response (o)
            # 3. Extract the final reward (index [0])
            reward_tuple = reward_model.compute_reward(prompt=p, response=o)
            all_rewards.append(reward_tuple[0])

    mean_reward = np.mean(all_rewards) if all_rewards else 0.0

    # diversity (all generated text)
    all_texts = [o for outs in generations.values() for o in outs]
    diversity = evaluate_diversity(all_texts)

    return {
        "mean_reward": mean_reward,
        "diversity": diversity,
        "samples": {p: outs[:2] for p, outs in generations.items()}
    }


# Run evaluation
if __name__ == "__main__":
    # --- REWARD MODEL INSTANTIATION FIX ---
    # Removed the trailing comma to correctly instantiate the RewardModel object
    # If the RewardModel is slow to load, it's better to instantiate it once here.
    print("Loading Reward Model...")
    reward_model_instance = RewardModel()
    print("Reward Model loaded.")
    
    # Use a small set of Yelp-appropriate prompts for testing
    test_prompts = [
        "Write a positive review of a newly opened Italian restaurant, using an enthusiastic tone.",
        "The terrible service at this restaurant made me very unhappy. Write a strong negative complaint."
    ]

    rlhf_report = evaluate_rlhf_model(
        model_path="../models/rlhf_model",
        prompts=test_prompts,
        reward_model=reward_model_instance, # Pass the instantiated object
        n_samples=10
    )

    print("\n================= RLHF EVALUATION REPORT =================")
    print(f"Mean Reward: {rlhf_report['mean_reward']:.3f}")
    print(f"Avg Len:     {rlhf_report['diversity']['avg_len']:.2f}")
    print(f"Distinct-1:  {rlhf_report['diversity']['distinct_1']:.3f}")
    print(f"Distinct-2:  {rlhf_report['diversity']['distinct_2']:.3f}")
    print(f"Tri-Repeat:  {rlhf_report['diversity']['tri_repeat']:.3f}")
    print("\nSample Outputs:")
    for p, samples in rlhf_report['samples'].items():
        print(f"\nPrompt: {p}")
        print(f" - Sample 1: {samples[0]}")
        print(f" - Sample 2: {samples[1]}")
    print("==========================================================")
