import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import Counter
from reward_model import RewardModel 

# ===============================
#      CONFIGURATION
# ===============================
SFT_MODEL_PATH = "../models/sft_model"
RLHF_MODEL_PATH = "../models/rlhf_model"

N_SAMPLES = 200   # 🔥 每个 prompt 生成 200 条样本
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 全面 Prompt 列表：指令型 + 续写型 + query 型
PROMPTS = [
    # === 指令型：明确正负 ===
    "Write a positive one-sentence review:",
    "Write a negative one-sentence review:",
    "Say something positive about a restaurant:",
    "Say something negative about a product:",

    # === 续写型（情绪可从 prompt 直接读出）
    "The restaurant was",
    "The meal was absolutely",
    "I will never go back because",
    "This experience made me feel",
]

# =======================================================
#                  Diversity Calculations
# =======================================================
def evaluate_diversity(texts):
    tokens = [t.lower().split() for t in texts]

    # Distinct-1
    d1_total = sum(len(tk) for tk in tokens)
    d1_unique = sum(len(set(tk)) for tk in tokens)
    distinct_1 = d1_unique / d1_total if d1_total else 0

    # Distinct-2
    d2_total = 0
    d2_unique = 0
    for tk in tokens:
        bigrams = set()
        for i in range(len(tk) - 1):
            bigrams.add(tuple(tk[i:i+2]))
        d2_total += max(len(tk) - 1, 0)
        d2_unique += len(bigrams)
    distinct_2 = d2_unique / d2_total if d2_total else 0

    # Tri-gram repetition
    tri_rep_rates = []
    for tk in tokens:
        if len(tk) < 3:
            continue
        tri = Counter()
        for i in range(len(tk) - 2):
            tri[tuple(tk[i:i+3])] += 1
        total = sum(tri.values())
        uniq = len(tri)
        tri_rep_rates.append((total - uniq) / total)
    tri_repeat = np.mean(tri_rep_rates) if tri_rep_rates else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "tri_repeat": tri_repeat
    }

# =======================================================
#                MODEL EVALUATION
# =======================================================
def evaluate_model(model_path: str, tokenizer, reward_model, prompts):
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    generations = {}
    all_rewards = []
    all_texts = []

    print(f"Generating {N_SAMPLES} samples × {len(prompts)} prompts = {N_SAMPLES * len(prompts)} responses...")
    for p in prompts:
        prompt_outputs = []
        for _ in range(N_SAMPLES):

            # Generate
            inputs = tokenizer(p, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )

            full = tokenizer.decode(out[0], skip_special_tokens=True)
            response = full[len(p):].strip()

            prompt_outputs.append(response)
            all_texts.append(response)

            # True reward
            reward_tuple = reward_model.compute_reward(prompt=p, response=response)
            all_rewards.append(reward_tuple[0])   # final reward

        generations[p] = prompt_outputs

    # Evaluate
    mean_reward = np.mean(all_rewards)
    diversity = evaluate_diversity(all_texts)

    # Only save first 3 outputs for inspection
    sample_out = {p: generations[p][:3] for p in prompts}

    return {
        "mean_reward": mean_reward,
        "diversity": diversity,
        "samples": sample_out
    }

# =======================================================
#                MODEL COMPARISON
# =======================================================
def compare_models():
    print("Loading RewardModel judge...")
    reward_model = RewardModel()  # true judge
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # ---- Evaluate SFT ----
    print("\n===== Evaluating SFT (Baseline) =====")
    sft = evaluate_model(SFT_MODEL_PATH, tokenizer, reward_model, PROMPTS)

    # ---- Evaluate RLHF ----
    print("\n===== Evaluating RLHF (PPO Model) =====")
    rlhf = evaluate_model(RLHF_MODEL_PATH, tokenizer, reward_model, PROMPTS)

    # ---- Table ----
    print("\n=================== MODEL COMPARISON ===================")
    print(f"{'Metric':<20}{'SFT':<20}{'RLHF':<20}")
    print("--------------------------------------------------------")
    print(f"{'Mean Reward':<20}{sft['mean_reward']:<20.4f}{rlhf['mean_reward']:<20.4f}")
    print(f"{'Distinct-1':<20}{sft['diversity']['distinct_1']:<20.4f}{rlhf['diversity']['distinct_1']:<20.4f}")
    print(f"{'Distinct-2':<20}{sft['diversity']['distinct_2']:<20.4f}{rlhf['diversity']['distinct_2']:<20.4f}")
    print(f"{'Tri-Repeat':<20}{sft['diversity']['tri_repeat']:<20.4f}{rlhf['diversity']['tri_repeat']:<20.4f}")

    # ---- Qualitative output ----
    print("\n=================== SAMPLE OUTPUTS ===================")
    for p in PROMPTS:
        print(f"\nPrompt: {p}")
        print("-" * 60)
        print(f"SFT  #1: {sft['samples'][p][0]}")
        print(f"SFT  #2: {sft['samples'][p][1]}")
        print(f"SFT  #3: {sft['samples'][p][2]}")
        print(f"RLHF #1: {rlhf['samples'][p][0]}")
        print(f"RLHF #2: {rlhf['samples'][p][1]}")
        print(f"RLHF #3: {rlhf['samples'][p][2]}")

    print("\n=======================================================")


if __name__ == "__main__":
    compare_models()
