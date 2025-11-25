import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate 

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased" # Use a base model suitable for classification

DATASET_NAME = "yelp_polarity"         
OUTPUT_DIR = "../models/sentiment_classifier_yelp" # Directory to save the final model
MAX_LENGTH = 256 # Max token length for reviews
TRAIN_SAMPLES = 10000 # Use a subset for faster training
EVAL_SAMPLES = 2000

# 1. Load Tokenizer
# We load the tokenizer for the CLASSIFICATION model, not the SFT model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 2. Tokenization Function be applied to every example in the dataset
def tokenize_function(examples):
    # Truncation=True ensures we cut off reviews longer than MAX_LENGTH
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)

# 3. Load and Prepare Dataset
print(f"Loading dataset: {DATASET_NAME}")
raw_datasets = load_dataset(DATASET_NAME)

# Apply the tokenization
# batched=True makes it much faster
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# We don't need the original "text" column anymore
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
# The Trainer expects the label column to be named "labels"
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# Set the format to PyTorch tensors
tokenized_datasets.set_format("torch")

print("Subsetting datasets for training...")
# The Yelp dataset is huge.select a subset for a quicker training run.
# .shuffle() ensures we get a random sample.
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(EVAL_SAMPLES))

# 4. Data Collator
# The DataCollator batches examples together and applies padding
# This is more efficient than padding all examples to MAX_LENGTH
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 5. Evaluation Metric
# We need a function to compute metrics during evaluation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Get the most likely class (0 or 1) from the logits
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# 6. Load Model
# We use AutoModelForSequenceClassification
# This adds a classification head (a linear layer) on top of DistilBERT
# num_labels=2 tells it we have 2 classes: (0: negative, 1: positive)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)

# 7. Training Arguments
# This class holds all the configuration for the training run
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2, # 2-3 epochs is usually good for fine-tuning
    weight_decay=0.01,
    eval_strategy="epoch", # Run evaluation at the end of each epoch
    save_strategy="epoch",       # Save a checkpoint at the end of each epoch
    load_best_model_at_end=True, # Load the best model (by val_loss) at the end
    push_to_hub=False,
)

# 8. Initialize Trainer
# The Trainer class handles the entire training and evaluation loop
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 9. Train
print("Starting fine-tuning...")
trainer.train()

# 10. Save the final model
print(f"Training complete. Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
# Also save the tokenizer so they are bundled together
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n--- Example of how to use the saved model ---")
print(f"model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")
print(f"tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
print("inputs = tokenizer('This food is amazing!', return_tensors='pt')")
print("with torch.no_grad(): logits = model(**inputs).logits")
print("print(logits)")