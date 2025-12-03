import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
import evaluate

# --- Configuration ---
MODEL_PATH = "../models/sentiment_classifier_yelp"
DATASET_NAME = "yelp_polarity"
MAX_LENGTH = 256
N_TEST_SAMPLES = 2000  # number of test examples for automatic evaluation


def predict_sentiment(texts, model, tokenizer, device):
    """
    Run sentiment prediction on a list of texts.
    """
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()

    return preds, probs.cpu().numpy()


def main():
    # 1. Load model & tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # Label mapping for yelp_polarity: 0 = negative, 1 = positive
    label_names = {0: "negative", 1: "positive"}

    # 2. Manual tests on a few example sentences
    print("\n=== Manual examples ===")
    example_texts = [
        "The food was absolutely delicious and the service was great!",  # 明显正面
        "I waited for an hour and the pasta was cold. Terrible.",       # 明显负面
        "It was okay, not the best but not the worst.",                 # 中性/模糊
        "The ambiance is nice, but the food is overpriced.",            # 混合评价
        "I will definitely come back again.",                         # 正面意图
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

    preds, probs = predict_sentiment(example_texts, model, tokenizer, device)
    for text, pred, prob in zip(example_texts, preds, probs):
        print("=" * 80)
        print(f"Text: {text}")
        print(f"Predicted label: {label_names[int(pred)]}")
        print(f"Probabilities [neg, pos]: {prob}")

    # 3. Evaluate on a subset of the Yelp test split
    print("\n=== Evaluating on Yelp test split subset ===")
    raw_datasets = load_dataset(DATASET_NAME)
    test_dataset = raw_datasets["test"].shuffle(seed=42).select(range(N_TEST_SAMPLES))

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    accuracy_metric = evaluate.load("accuracy")

    all_preds = []
    all_labels = []

    batch_size = 32
    for i in range(0, len(tokenized_test), batch_size):
        batch = tokenized_test[i : i + batch_size]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].numpy()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    print(f"\nTest accuracy on {N_TEST_SAMPLES} samples: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
