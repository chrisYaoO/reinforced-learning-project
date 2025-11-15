from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sft import toy_reward_fn
import re


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

        # Load tokenizer from SFT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.replace("rlhf", "sft"))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=40):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# ============================================================
# Main evaluation function
# ============================================================
def evaluate_rlhf_model(model_path, prompts, reward_fn, n_samples=10):
    evaluator = RLHFEvaluator(model_path)
    generations = {}

    # Generate many samples per prompt
    for p in prompts:
        gens = []
        for _ in range(n_samples):
            full_text = evaluator.generate(p)
            response_only = full_text[len(p):].strip()
            gens.append(response_only)
        generations[p] = gens

    # reward
    all_rewards = []
    for outs in generations.values():
        for o in outs:
            all_rewards.append(reward_fn(o))

    mean_reward = sum(all_rewards) / len(all_rewards)

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
    rlhf_report = evaluate_rlhf_model(
        model_path="../models/rlhf_model",
        prompts=[
            "Write a positive review:",
            "Write a negative review:"
        ],
        reward_fn=toy_reward_fn,
        n_samples=10
    )

    print(rlhf_report)
