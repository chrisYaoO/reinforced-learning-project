import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from collections import Counter
from reward_model import RewardModel 

# --- 配置 ---
SFT_MODEL_PATH = "../models/sft_model"
RLHF_MODEL_PATH = "../models/rlhf_model"
N_SAMPLES = 20 # 运行更多样本以获得更稳定的统计数据
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

# PROMPTS = [
#     "Write a positive review:",
#     "Write a negative review:",
#     "The restaurant was",
#     "I will never go back because"
# ]

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
    "You dined at an upscale restaurant. Write a complaint about the price."
]


# --- 辅助函数：多样性计算 ---
# (从你的 'evaluate_sft_model' 中提取)
def evaluate_diversity(texts):
    """Calculates distinct-1, distinct-2, and tri-gram repetition."""
    tokens = [t.lower().split() for t in texts]
    
    # Distinct-1
    d1_total = 0
    d1_unique = 0
    for tk in tokens:
        d1_total += len(tk)
        d1_unique += len(set(tk))
    distinct_1 = d1_unique / d1_total if d1_total > 0 else 0

    # Distinct-2
    d2_total = 0
    d2_unique = 0
    for tk in tokens:
        bigrams = set()
        for i in range(len(tk) - 1):
            bigrams.add(tuple(tk[i:i+2]))
        d2_total += len(tk) - 1
        d2_unique += len(bigrams)
    distinct_2 = d2_unique / d2_total if d2_total > 0 else 0

    # Tri-gram repetition
    tri_rep_rates = []
    for tk in tokens:
        if len(tk) < 3:
            continue
        trigrams = Counter()
        for i in range(len(tk) - 2):
            trigrams[tuple(tk[i:i+3])] += 1
        
        if not trigrams:
            continue
            
        total_trigrams = sum(trigrams.values())
        unique_trigrams = len(trigrams)
        tri_rep_rates.append((total_trigrams - unique_trigrams) / total_trigrams)
        
    tri_repeat = np.mean(tri_rep_rates) if tri_rep_rates else 0

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "tri_repeat": tri_repeat
    }

# --- 核心评估函数 ---
def evaluate_model(model_path: str, tokenizer: AutoTokenizer, reward_model: RewardModel, prompts: list):
    """
    Loads a model, generates responses for all prompts, and evaluates them
    using the REAL reward model.
    """
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)
    model.eval()

    generations = {}
    all_rewards = []
    all_texts = []
    
    print(f"Generating {N_SAMPLES} samples for {len(prompts)} prompts...")
    for p in prompts:
        prompt_generations = []
        for _ in range(N_SAMPLES):
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
            
            prompt_generations.append(response)
            all_texts.append(response)
            
            # --- 关键修复 ---
            # 调用真实的奖励函数，它需要 prompt 和 response
            # 并且它返回一个元组，我们取第一个元素
            reward_tuple = reward_model.compute_reward(prompt=p, response=response)
            all_rewards.append(reward_tuple[0]) # [0] is the final_reward
        
        generations[p] = prompt_generations

    # --- 计算指标 ---
    mean_reward = np.mean(all_rewards)
    diversity_metrics = evaluate_diversity(all_texts)
    
    return {
        "mean_reward": mean_reward,
        "diversity": diversity_metrics,
        "samples": {p: generations[p][:2] for p in prompts} # Save 2 samples for qualitative review
    }

# --- 主对比逻辑 ---
def compare_models():
    
    # --- 关键修复：实例化你真正的 RewardModel ---
    print("Initializing Real Reward Model (Judge)...")
    # 这会加载你（Wei Wang）的分类器
    reward_model_instance = RewardModel() 
    print("Reward Model loaded.")

    # 我们需要一个 SFT 模型的 Tokenizer 来加载
    # 两个模型都使用相同的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Evaluating SFT Model (Baseline) ===")
    sft_report = evaluate_model(
        model_path=SFT_MODEL_PATH,
        tokenizer=tokenizer,
        reward_model=reward_model_instance, # 使用真正的裁判
        prompts=PROMPTS
    )

    print("\n=== Evaluating RLHF Model (PPO-Tuned) ===")
    rlhf_report = evaluate_model(
        model_path=RLHF_MODEL_PATH,
        tokenizer=tokenizer,
        reward_model=reward_model_instance, # 使用同一个真正的裁判
        prompts=PROMPTS
    )

    # --- 打印对比表 ---
    print("\n================= MODEL COMPARISON =================")
    print(f"{'Metric':<20}{'SFT (Baseline)':<20}{'RLHF (PPO)':<20}")
    print("-" * 60)
    print(f"{'Mean Reward':<20}{sft_report['mean_reward']:<20.3f}{rlhf_report['mean_reward']:<20.3f}")
    print(f"{'Distinct-1':<20}{sft_report['diversity']['distinct_1']:<20.3f}{rlhf_report['diversity']['distinct_1']:<20.3f}")
    print(f"{'Distinct-2':<20}{sft_report['diversity']['distinct_2']:<20.3f}{rlhf_report['diversity']['distinct_2']:<20.3f}")
    print(f"{'Tri-Repeat':<20}{sft_report['diversity']['tri_repeat']:<20.3f}{rlhf_report['diversity']['tri_repeat']:<20.3f}")

    # --- 打印定性样本 ---
    print("\n================= SAMPLE OUTPUTS =================")
    for p in PROMPTS:
        print(f"\nPrompt: {p}")
        print("--------------------------------------------------")
        print(f"SFT Sample 1:  {sft_report['samples'][p][0]}")
        print(f"SFT Sample 2:  {sft_report['samples'][p][1]}")
        print(f"RLHF Sample 1: {rlhf_report['samples'][p][0]}")
        print(f"RLHF Sample 2: {rlhf_report['samples'][p][1]}")

    print("\n====================================================")


if __name__ == "__main__":
    compare_models()
