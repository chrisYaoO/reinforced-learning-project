from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sft import SFT, toy_reward_fn
from rlhf_eval import evaluate_rlhf_model, evaluate_diversity
import numpy as np



# Generate samples from a model

def generate_from_model(model, tokenizer, prompt, device="cpu", n=3):
    tokenizer.pad_token = tokenizer.eos_token
    results = []

    for _ in range(n):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
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
        results.append(full[len(prompt):].strip())

    return results



# Evaluate SFT model

def evaluate_sft_model(sft_model_dir, token_dir, prompts, reward_fn, n_samples=10):
    sft = SFT(model_name=sft_model_dir, token_dir=token_dir)

    generations = {}
    all_rewards = []

    for p in prompts:
        gens = sft.generate_batch([p], n_per_prompt=n_samples)[p]
        generations[p] = gens
        for g in gens:
            all_rewards.append(reward_fn(g))

    mean_reward = np.mean(all_rewards)
    all_texts = [g for outs in generations.values() for g in outs]
    diversity = SFT.evaluate_diversity(all_texts)

    return {
        "mean_reward": mean_reward,
        "diversity": diversity,
        "samples": {p: generations[p][:2] for p in prompts}
    }



# Compare SFT vs RLHF

def compare_models():
    prompts = [
        "Write a positive review:",
        "Write a negative review:"
    ]

    print("=== Evaluating SFT Model ===")
    sft_report = evaluate_sft_model(
        sft_model_dir="../models/sft_model",
        token_dir="../data/tokenized_data",
        prompts=prompts,
        reward_fn=toy_reward_fn,
        n_samples=10
    )

    print("\n=== Evaluating RLHF Model ===")
    rlhf_report = evaluate_rlhf_model(
        model_path="../models/rlhf_model",
        prompts=prompts,
        reward_fn=toy_reward_fn,
        n_samples=10
    )

    # Print comparison table
    print("\n================= MODEL COMPARISON =================")
    print(f"{'Metric':<20}{'SFT':<20}{'RLHF':<20}")
    print("-" * 60)
    print(f"{'Mean Reward':<20}{sft_report['mean_reward']:<20.3f}{rlhf_report['mean_reward']:<20.3f}")
    print(f"{'Distinct-1':<20}{sft_report['diversity']['distinct_1']:<20.3f}{rlhf_report['diversity']['distinct_1']:<20.3f}")
    print(f"{'Distinct-2':<20}{sft_report['diversity']['distinct_2']:<20.3f}{rlhf_report['diversity']['distinct_2']:<20.3f}")
    print(f"{'Tri-Repeat':<20}{sft_report['diversity']['tri_repeat']:<20.3f}{rlhf_report['diversity']['tri_repeat']:<20.3f}")

    # Show qualitative samples
    print("\n================= SAMPLE OUTPUTS =================")
    for p in prompts:
        print(f"\nPrompt: {p}")
        print(f"SFT Sample 1: {sft_report['samples'][p][0]}")
        print(f"SFT Sample 2: {sft_report['samples'][p][1]}")

        print(f"RLHF Sample 1: {rlhf_report['samples'][p][0]}")
        print(f"RLHF Sample 2: {rlhf_report['samples'][p][1]}")

    print("\n====================================================")


if __name__ == "__main__":
    compare_models()
