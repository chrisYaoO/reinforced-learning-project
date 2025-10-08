import torch

from datasets import load_dataset


def build_sft_example(ex):
    sentiment = "positive" if ex["label"] == 1 else "negative"
    prompt = f"Write a {sentiment} one-sentence review:"
    target = ex["text"].splitlines()[0].strip()
    return {"prompt": prompt, "target": target, "sentiment": sentiment}


def load_data(dataset_name):
    ds = load_dataset(dataset_name)
    cols = ds["train"].column_names
    ds_sft = ds.map(build_sft_example, remove_columns=cols)
    small_train = ds_sft["train"].shuffle(seed=42).select(range(5000))
    small_eval = ds_sft["test"].shuffle(seed=42).select(range(1000))

    return small_train, small_eval


if __name__ == '__main__':
    print(torch.__version__)
    small_train, small_eval = load_data("yelp_polarity")
    print(small_train[0])
