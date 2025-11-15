from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch, random
import numpy as np
from collections.abc import Callable
from typing import Any


class SFT:
    def __init__(self, model_name, token_dir, seed=42):
        # fix seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.token_dir = token_dir
        tokenized = load_from_disk(token_dir)
        self.train_dataset = tokenized["train"]
        self.eval_dataset = tokenized["eval"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def train(self):
        use_fp16 = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir="../results/sft_results",
            overwrite_output_dir=True,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=500,
            logging_steps=50,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=use_fp16,
            report_to="tensorboard",
            disable_tqdm=True,
            run_name="sft_distilgpt2_yelp",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        trainer.train()
        print("Finished Training")
        trainer.save_model("../models/sft_model")
        self.tokenizer.save_pretrained("../models/sft_model")
        print(" model saved")

    #        Generation
    def _generate_one(self, prompt: str,
                      max_new_tokens: int = 40,
                      temperature: float = 0.7,
                      top_p: float = 0.9) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # return without prompt
        return full[len(prompt):].strip()

    def generate_batch(self, prompts: list[str],
                       n_per_prompt: int = 5,
                       max_new_tokens: int = 40,
                       temperature: float = 0.7,
                       top_p: float = 0.9) -> dict[str, list[str]]:
        results = {}
        for p in prompts:
            results[p] = [
                self._generate_one(p, max_new_tokens, temperature, top_p)
                for _ in range(n_per_prompt)
            ]
        return results

    # Evaluation: perplexity
    def evaluate_ppl(self, per_device_eval_batch_size: int = 4) -> dict[str, float]:
        args = TrainingArguments(
            output_dir="../results/tmp_eval",
            per_device_eval_batch_size=per_device_eval_batch_size,
            report_to="none"
        )
        trainer = Trainer(model=self.model, args=args,
                          eval_dataset=self.eval_dataset, processing_class=self.tokenizer)
        metrics = trainer.evaluate()
        eval_loss = float(metrics["eval_loss"])
        ppl = np.exp(eval_loss) if eval_loss < 20 else float("inf")  # set safe threshold
        return {"eval_loss": eval_loss, "perplexity": ppl}

    # Evaluation: Diversity
    @staticmethod
    def _distinct_n_ratio(texts: list[str], n: int = 2) -> float:
        import re
        def tok(s):
            return re.findall(r"\w+|\S", s)

        ngrams = []
        for t in texts:
            toks = tok(t)
            ngrams += [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def _trigram_repeat_rate(texts: list[str]) -> float:
        import re
        def tok(s): return re.findall(r"\w+|\S", s)

        total, dup = 0, 0
        for t in texts:
            toks = tok(t)
            trigrams = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
            total += len(trigrams)
            dup += (len(trigrams) - len(set(trigrams)))
        return (dup / total) if total > 0 else 0.0
    @staticmethod
    def evaluate_diversity(generations: list[str]) -> dict[str, float]:
        if len(generations) == 0:
            return {"avg_len": 0.0, "distinct_1": 0.0, "distinct_2": 0.0, "tri_repeat": 0.0}
        avg_len = sum(len(g.split()) for g in generations) / len(generations)
        d1 = SFT._distinct_n_ratio(generations, n=1)
        d2 = SFT._distinct_n_ratio(generations, n=2)
        tri_rep = SFT._trigram_repeat_rate(generations)
        return {"avg_len": avg_len, "distinct_1": d1, "distinct_2": d2, "tri_repeat": tri_rep}

    #   Evaluation: Reward
    #  reward function not decided
    def evaluate_reward(self,
                        prompts: list[str],
                        reward_fn: Callable[[str], float],
                        n_per_prompt: int = 5,
                        max_new_tokens: int = 40,
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        gen_map=None,
                        ) -> dict[str, Any]:

        assert reward_fn is not None, "reward_fn must be provided."

        if gen_map is None:
            assert prompts is not None, "Need prompts when gen_map is None."
            gen_map = self.generate_batch(
                prompts, n_per_prompt, max_new_tokens, temperature, top_p
            )

        all_scores, by_prompt = [], {}
        for p, outs in gen_map.items():
            scores = [float(reward_fn(o)) for o in outs]
            by_prompt[p] = {"samples": outs, "scores": scores}
            all_scores.extend(scores)

        if len(all_scores) == 0:
            return {"mean_reward": 0.0, "std_reward": 0.0, "detail": by_prompt}

        return {
            "mean_reward": float(np.mean(all_scores)),
            "std_reward": float(np.std(all_scores)) if len(all_scores) > 1 else 0.0,
            "detail": by_prompt
        }

    #  together eval
    def evaluate_all(self,
                     prompts: list[str],
                     reward_fn: Callable[[str], float],
                     n_per_prompt: int = 5,
                     max_new_tokens: int = 40,
                     temperature: float = 0.7,
                     top_p: float = 0.9) -> dict[str, Any]:
        """
        PPL
        reward
        diversity

        """
        # PPL
        ppl_metrics = self.evaluate_ppl()

        # generate + reward
        gen_map = self.generate_batch(prompts, n_per_prompt, max_new_tokens, temperature, top_p)
        all_generations = [g for outs in gen_map.values() for g in outs]

        reward_report = self.evaluate_reward(
            prompts=prompts,
            reward_fn=reward_fn,
            n_per_prompt=n_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # diversity
        diversity_report = self.evaluate_diversity(all_generations)

        qualitative = {p: outs[:2] for p, outs in gen_map.items()}

        report = {
            "ppl_metrics": ppl_metrics,
            "reward_metrics": {
                "mean_reward": reward_report["mean_reward"],
                "std_reward": reward_report["std_reward"],
            },
            "diversity_metrics": diversity_report,
            "samples": qualitative,
            "reward_details": reward_report["detail"]  # reward per sample
        }

        print("\n===== SFT Baseline Evaluation =====")
        print(f"PPL -> eval_loss: {ppl_metrics['eval_loss']:.4f}, perplexity: {ppl_metrics['perplexity']:.2f}")
        print(
            f"Reward -> mean: {report['reward_metrics']['mean_reward']:.4f}, std: {report['reward_metrics']['std_reward']:.4f}")
        print(f"Diversity -> AvgLen: {diversity_report['avg_len']:.1f}, "
              f"Distinct-1: {diversity_report['distinct_1']:.3f}, Distinct-2: {diversity_report['distinct_2']:.3f}, "
              f"Tri-Repeat: {diversity_report['tri_repeat']:.3f}")
        for p, outs in qualitative.items():
            print(f"\nPrompt: {p}")
            for i, s in enumerate(outs, 1):
                print(f"  Sample {i}: {s}")

        return report


def toy_reward_fn(text: str) -> float:
    text_low = text.lower()
    pos = sum(w in text_low for w in ["good", "great", "excellent", "love", "amazing"])
    neg = sum(w in text_low for w in ["bad", "terrible", "awful", "hate", "poor"])
    return float(pos - neg)


if __name__ == "__main__":
    sft = SFT(model_name="distilgpt2", token_dir="../data/tokenized_data")
    sft.train()
    prompts = [
        "Write a positive one-sentence review:",
        "Write a negative one-sentence review:"
    ]

    report = sft.evaluate_all(
        prompts=prompts,
        reward_fn=toy_reward_fn,
        n_per_prompt=10,
        max_new_tokens=40,
        temperature=0.7,
        top_p=0.9
    )
