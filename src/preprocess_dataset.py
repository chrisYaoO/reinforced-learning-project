import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


class data_loader:
    def __init__(self, dataset_name: str, tokenizer_name: str, save_dir: str = "../data/tokenized_data"):
        """
        init loader
        """
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.save_dir = save_dir

        # 初始化 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def build_sft_example(self, ex):
        sentiment = "positive" if ex["label"] == 1 else "negative"
        prompt = f"Write a {sentiment} one-sentence review:"
        target = ex["text"].splitlines()[0].strip()
        return {"prompt": prompt, "target": target, "sentiment": sentiment}

    def tokenize_example(self, example, max_length=128):
        """build token"""
        prompt = example["prompt"]
        target = example["target"]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or self.tokenizer.unk_token

        # 3) get prompt token length
        enc_prompt = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False
        )
        prompt_len = len(enc_prompt["input_ids"])

        # concat
        enc_all = self.tokenizer(
            prompt + " " + target,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            add_special_tokens=False,
            return_attention_mask=True
        )

        input_ids = enc_all["input_ids"]
        attn_mask = enc_all["attention_mask"]

        # 5) block prompt and padding
        labels = input_ids.copy()

        cut = min(prompt_len, max_length)
        for i in range(cut):
            labels[i] = -100

        labels = [tok if m == 1 else -100 for tok, m in zip(labels, attn_mask)]

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

    def prepare(self, train_size=5000, eval_size=1000, max_length=128):
        print(f"Loading dataset: {self.dataset_name}")
        ds = load_dataset(self.dataset_name)
        cols = ds["train"].column_names

        # build sample
        ds_sft = ds.map(self.build_sft_example, remove_columns=cols)

        # build train and eval dataset
        train_ds = ds_sft["train"].shuffle(seed=42).select(range(train_size))
        eval_ds = ds_sft["test"].shuffle(seed=42).select(range(eval_size))
        data_dict = DatasetDict({"train": train_ds, "eval": eval_ds})

        # tokenize
        tokenized = data_dict.map(
            lambda x: self.tokenize_example(x, max_length=max_length),
            remove_columns=data_dict["train"].column_names,
            batched=False
        )

        # save tokens
        print(f"Saving tokenized dataset to: {self.save_dir}")

        tokenized.save_to_disk(self.save_dir)

        return tokenized


if __name__ == "__main__":
    print(torch.__version__)
    preparer = data_loader(dataset_name="yelp_polarity", tokenizer_name="distilgpt2")
    tokenized = preparer.prepare()
    print(tokenized["train"][0])
