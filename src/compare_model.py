
# compare_model.py
import torch
import numpy as np
import re
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from reward_model import RewardModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SFT_MODEL_PATH = "../models/sft_model"
RLHF_MODEL_PATH = "../models/rlhf_model"

N_SAMPLES_PER_PROMPT = 5
MAX_NEW_TOKENS = 40
TEMPERATURE = 0.7
TOP_P = 0.9

RESULTS_DIR = "../results/stable"

PROMPTS_0 = [
    "Write a cheerful and enthusiastic review about how delicious the pasta dishes were.",
    "Give a positive one-sentence comment praising the friendly service at the café.",
    "Write a short energetic review about the fresh sushi you enjoyed today.",
    "Write a warm and heartwarming review about your cozy family dinner.",
    "Write a positive review celebrating the flavorful ramen and amazing broth.",
    "Write a happy and uplifting review about the refreshing smoothie you ordered.",
    "Write a brief positive praise for the polite waiter and excellent customer service.",
    "Write an encouraging review about your great experience at a newly opened brunch spot.",
    "Write a harsh complaint about the slow service and rude staff at the restaurant.",
    "Give a strong negative review about the burnt pizza and awful seasoning.",
    "Write a one-sentence complaint about the cold food and long waiting time.",
    "Write a critical review describing the dirty restroom and poor hygiene conditions.",
    "Write a disappointed review about the bland curry and flavorless soup.",
    "Write a negative comment about the overpriced dishes and terrible coffee.",
    "Write a harsh review criticizing the soggy fries and oily dishes.",
    "Write a complaint about the chaotic management and unprofessional staff."
]

PROMPTS = [
    "Write a neutral description of the restaurant’s seating capacity.",
    "Objectively describe the variety of soups available on the menu.",
    "Write a factual summary of the restaurant’s lunch specials.",
    "Provide a neutral comment about the overall portion sizes.",
    "Write a brief objective description of the bar area.",
    "Describe the drink refill process in a neutral tone.",
    "Write a neutral comment about the restaurant’s location.",
    "Objectively describe the style of plates and bowls used.",
    "Give a neutral statement about the restaurant’s service speed.",
    "Write a factual remark about how customers place orders.",
    
    "Write a positive review praising the warm hospitality.",
    "Give an enthusiastic comment about the flavorful curry.",
    "Write a cheerful review about the restaurant’s breakfast dishes.",
    "Describe how amazing the handmade dumplings were in an excited tone.",
    "Write a positive remark about the fresh seafood and delicious broth.",
    "Give a happy and uplifting comment about the cozy atmosphere.",
    "Write a positive review celebrating the creative fusion menu.",
    "Give an enthusiastic recommendation of the dessert platter.",
    "Write a joyful compliment about the friendly cashier.",
    "Write a vibrant and excited review of the grilled dishes.",
    
    "Write a strong negative review about the burnt toast.",
    "Complain about the oily and soggy fries in a harsh tone.",
    "Write a negative comment about careless staff behavior.",
    "Describe the unpleasant smell inside the restaurant negatively.",
    "Write a harsh complaint about the extremely slow checkout process.",
    "Give a negative review about the stale pastries.",
    "Write a complaint about the restaurant’s misleading menu photos.",
    "Write a strong critique of the poorly prepared noodles.",
    "Complain about the excessive noise during dinner hours.",
    "Write a negative comment about the bland and watery soup.",
    
    "Objectively describe the restaurant’s exterior appearance.",
    "Write a neutral remark about the drink container sizes.",
    "Provide a factual comment about the temperature of the dining area.",
    "Write a neutral description of the restaurant’s seasonal menu changes.",
    "Give an unbiased observation about weekday crowd levels.",
    "Write a factual note on whether reservations are required.",
    "Objectively describe the staff workflow at the counter.",
    "Write a neutral comparison between dine-in and delivery packaging.",
    "Provide an unbiased comment on menu readability.",
    "Write a factual remark about the available condiment choices."
]

def generate_many(
    model,
    tokenizer,
    prompts,
    n_samples_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    """
    generate n samples for each prompt
    return:
        all_prompts: [p0, p0, ..., p1, p1, ...]
        all_texts
    
    """
    model.eval()
    all_prompts = []
    all_texts = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            ).to(DEVICE)

            for _ in range(n_samples_per_prompt):
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

                gen = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                ).strip()

                all_prompts.append(prompt)
                all_texts.append(gen)

    return all_prompts, all_texts


# ========= diversity metrics =========
def _tok(s: str):
    return re.findall(r"\w+|\S", s)


def distinct_n_ratio(texts: list[str], n: int = 1) -> float:
    ngrams = []
    for t in texts:
        toks = _tok(t)
        ngrams += [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def trigram_repeat_rate(texts: list[str]) -> float:
    total, dup = 0, 0
    for t in texts:
        toks = _tok(t)
        trigrams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        total += len(trigrams)
        dup += (len(trigrams) - len(set(trigrams)))
    return (dup / total) if total > 0 else 0.0


def evaluate_model(model_path: str, reward_model: RewardModel):

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(DEVICE)

    all_prompts, all_texts = generate_many(
        model=model,
        tokenizer=tokenizer,
        prompts=PROMPTS,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    # compute reward
    rewards = []
    for p, r in zip(all_prompts, all_texts):
        final_r, r_sent, r_rep, r_flu, r_task = reward_model.compute_reward(p, r)
        rewards.append(final_r)
    rewards = np.array(rewards, dtype=np.float32)

    mean_reward = float(rewards.mean()) if len(rewards) > 0 else 0.0
    std_reward = float(rewards.std()) if len(rewards) > 1 else 0.0

    # diversity
    d1 = distinct_n_ratio(all_texts, n=1)
    d2 = distinct_n_ratio(all_texts, n=2)
    tri_rep = trigram_repeat_rate(all_texts)

    stats = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "distinct_1": d1,
        "distinct_2": d2,
        "tri_repeat": tri_rep,
    }

    return stats, all_prompts, all_texts, rewards


def print_comparison_table(sft_stats, rlhf_stats):
    print("\n================= MODEL COMPARISON =================")
    print(f"{'Metric':20s}{'SFT (Baseline)':>18s}{'RLHF (PPO)':>18s}")
    print("------------------------------------------------------------")
    print(
        f"{'Mean Reward':20s}"
        f"{sft_stats['mean_reward']:>18.3f}"
        f"{rlhf_stats['mean_reward']:>18.3f}"
    )
    print(
        f"{'Std Reward':20s}"
        f"{sft_stats['std_reward']:>18.3f}"
        f"{rlhf_stats['std_reward']:>18.3f}"
    )
    print(
        f"{'Distinct-1':20s}"
        f"{sft_stats['distinct_1']:>18.3f}"
        f"{rlhf_stats['distinct_1']:>18.3f}"
    )
    print(
        f"{'Distinct-2':20s}"
        f"{sft_stats['distinct_2']:>18.3f}"
        f"{rlhf_stats['distinct_2']:>18.3f}"
    )
    print(
        f"{'Tri-Repeat':20s}"
        f"{sft_stats['tri_repeat']:>18.3f}"
        f"{rlhf_stats['tri_repeat']:>18.3f}"
    )


def print_samples_side_by_side(
    sft_texts,
    sft_rewards,
    rlhf_texts,
    rlhf_rewards,
    n_prompts_to_show: int = 5,
):
    """
    show first sample（sample_idx = 0）
    """
    print("\n================= SAMPLE OUTPUTS (subset) =================")
    n_prompts_to_show = min(n_prompts_to_show, len(PROMPTS))

    for i in range(n_prompts_to_show):
        prompt = PROMPTS[i]
        base_idx = i * N_SAMPLES_PER_PROMPT

        sft_text = sft_texts[base_idx]
        sft_reward = sft_rewards[base_idx]

        rlhf_text = rlhf_texts[base_idx]
        rlhf_reward = rlhf_rewards[base_idx]

        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        print(f"SFT Sample:  {sft_text}")
        print(f"SFT Reward:  {sft_reward:.3f}")
        print()
        print(f"RLHF Sample: {rlhf_text}")
        print(f"RLHF Reward: {rlhf_reward:.3f}")


# ========= 把结果写入 JSON / TXT =========
def build_json_report(
    sft_stats,
    rlhf_stats,
    sft_texts,
    sft_rewards,
    rlhf_texts,
    rlhf_rewards,
):
    """
    json report: 
    - overall metrics
    - per-prompt, per-model, per-sample 文本 + reward
    """
    report = {
        "sft_stats": sft_stats,
        "rlhf_stats": rlhf_stats,
        "prompts": PROMPTS,
        "per_prompt": []
    }

    sft_rewards_list = sft_rewards.tolist()
    rlhf_rewards_list = rlhf_rewards.tolist()

    for i, prompt in enumerate(PROMPTS):
        start = i * N_SAMPLES_PER_PROMPT
        end = start + N_SAMPLES_PER_PROMPT

        sft_samples = [
            {
                "text": sft_texts[j],
                "reward": float(sft_rewards_list[j]),
            }
            for j in range(start, end)
        ]
        rlhf_samples = [
            {
                "text": rlhf_texts[j],
                "reward": float(rlhf_rewards_list[j]),
            }
            for j in range(start, end)
        ]

        report["per_prompt"].append(
            {
                "prompt": prompt,
                "sft_samples": sft_samples,
                "rlhf_samples": rlhf_samples,
            }
        )

    return report


def build_txt_report_string(
    sft_stats,
    rlhf_stats,
    sft_texts,
    sft_rewards,
    rlhf_texts,
    rlhf_rewards,
    n_prompts_to_show: int = 5,
) -> str:
    """
    terminal output version
    """
    lines = []
    lines.append("================= MODEL COMPARISON =================")
    lines.append(f"{'Metric':20s}{'SFT (Baseline)':>18s}{'RLHF (PPO)':>18s}")
    lines.append("------------------------------------------------------------")
    lines.append(
        f"{'Mean Reward':20s}"
        f"{sft_stats['mean_reward']:>18.3f}"
        f"{rlhf_stats['mean_reward']:>18.3f}"
    )
    lines.append(
        f"{'Std Reward':20s}"
        f"{sft_stats['std_reward']:>18.3f}"
        f"{rlhf_stats['std_reward']:>18.3f}"
    )
    lines.append(
        f"{'Distinct-1':20s}"
        f"{sft_stats['distinct_1']:>18.3f}"
        f"{rlhf_stats['distinct_1']:>18.3f}"
    )
    lines.append(
        f"{'Distinct-2':20s}"
        f"{sft_stats['distinct_2']:>18.3f}"
        f"{rlhf_stats['distinct_2']:>18.3f}"
    )
    lines.append(
        f"{'Tri-Repeat':20s}"
        f"{sft_stats['tri_repeat']:>18.3f}"
        f"{rlhf_stats['tri_repeat']:>18.3f}"
    )

    lines.append("")
    lines.append("================= SAMPLE OUTPUTS (subset) =================")

    n_prompts_to_show = min(n_prompts_to_show, len(PROMPTS))

    for i in range(n_prompts_to_show):
        prompt = PROMPTS[i]
        base_idx = i * N_SAMPLES_PER_PROMPT

        sft_text = sft_texts[base_idx]
        sft_reward = float(sft_rewards[base_idx])

        rlhf_text = rlhf_texts[base_idx]
        rlhf_reward = float(rlhf_rewards[base_idx])

        lines.append("")
        lines.append(f"Prompt: {prompt}")
        lines.append("--------------------------------------------------")
        lines.append(f"SFT Sample:  {sft_text}")
        lines.append(f"SFT Reward:  {sft_reward:.3f}")
        lines.append("")
        lines.append(f"RLHF Sample: {rlhf_text}")
        lines.append(f"RLHF Reward: {rlhf_reward:.3f}")

    return "\n".join(lines)


def save_results(
    sft_stats,
    rlhf_stats,
    sft_texts,
    sft_rewards,
    rlhf_texts,
    rlhf_rewards,
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_report = build_json_report(
        sft_stats,
        rlhf_stats,
        sft_texts,
        sft_rewards,
        rlhf_texts,
        rlhf_rewards,
    )
    txt_report = build_txt_report_string(
        sft_stats,
        rlhf_stats,
        sft_texts,
        sft_rewards,
        rlhf_texts,
        rlhf_rewards,
        n_prompts_to_show=5,
    )

    json_path = os.path.join(RESULTS_DIR, f"compare_results_{timestamp}.json")
    txt_path = os.path.join(RESULTS_DIR, f"compare_results_{timestamp}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  TXT : {txt_path}")


def main():
    print("Initializing Real Reward Model (Judge)...")
    reward_model = RewardModel()
    print("Reward Model loaded.\n")

    print("=== Evaluating SFT (Baseline) ===")
    sft_stats, sft_prompts, sft_texts, sft_rewards = evaluate_model(
        SFT_MODEL_PATH, reward_model
    )

    print("\n=== Evaluating RLHF (PPO) ===")
    rlhf_stats, rlhf_prompts, rlhf_texts, rlhf_rewards = evaluate_model(
        RLHF_MODEL_PATH, reward_model
    )

    print_comparison_table(sft_stats, rlhf_stats)
    print_samples_side_by_side(
        sft_texts,
        sft_rewards,
        rlhf_texts,
        rlhf_rewards,
        n_prompts_to_show=5,
    )

    save_results(
        sft_stats,
        rlhf_stats,
        sft_texts,
        sft_rewards,
        rlhf_texts,
        rlhf_rewards,
    )


if __name__ == "__main__":
    main()


