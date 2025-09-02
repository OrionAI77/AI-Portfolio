from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
import torch
import sys
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log to a file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(os.getcwd(), f"training_log_{timestamp}.txt")
print(f"Script started... Log file: {log_file}", file=sys.stderr)
try:
    with open(log_file, "w") as f:
        print(f"Script started at {sys.version} on {datetime.now()}", file=f)
        print(f"Current directory: {os.getcwd()}", file=f)
except Exception as e:
    print(f"Log file error: {e}", file=sys.stderr)
    print(f"Log file error: {e}")

# Expanded dummy dataset with more examples
reviews = [
    "amazing product", "excellent value", "fantastic item", "superb quality", "outstanding performance",
    "terrible service", "awful quality", "poor support", "bad experience", "horrible customer service",
    "okay experience", "neutral feedback", "decent buy", "mediocre result",
    "highly recommend this item", "worst purchase ever", "it's average", "love it so much", "hate the design", "neither good nor bad",
    "top-notch quality", "disappointing service", "fair price", "brilliant design", "subpar performance", "average product",
    "great service", "poor quality", "okay product", "fantastic deal", "awful experience", "neutral review",
    "wonderful item", "terrible value", "fair deal", "amazing service", "bad design", "mixed feelings",
    "excellent support", "poor performance", "great value", "awful product", "decent experience", "positive feedback",
    "negative review", "superb item", "bad quality", "lovely design", "neutral opinion", "impressive result"
]
sentiments = [
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative", "negative",
    "neutral", "neutral", "neutral", "neutral",
    "positive", "negative", "neutral", "positive", "negative", "neutral",
    "positive", "negative", "neutral", "positive", "negative", "neutral",
    "positive", "negative", "neutral", "positive", "negative", "neutral",
    "positive", "negative", "neutral", "positive", "negative", "neutral",
    "positive", "negative", "positive", "negative", "neutral", "positive",
    "negative", "positive", "negative", "positive", "neutral", "positive"
]

# Convert to dataset and split
data = {"text": reviews, "label": [0 if s == "negative" else 1 if s == "positive" else 2 for s in sentiments]}
dataset = Dataset.from_dict(data)
train_test_split_data = dataset.train_test_split(test_size=0.2)
dataset_dict = DatasetDict({
    "train": train_test_split_data["train"],
    "eval": train_test_split_data["test"]
})

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Define model and move to device
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    lr_scheduler_type="cosine",
    learning_rate=2e-05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print("Training model...", file=sys.stderr)
try:
    with open(log_file, "a") as f:
        print("Training model...", file=f)
    trainer.train()
except Exception as e:
    with open(log_file, "a") as f:
        print(f"Training error: {e}", file=f)
    print(f"Training error: {e}")

print("Evaluating model...", file=sys.stderr)
try:
    with open(log_file, "a") as f:
        print("Evaluating model...", file=f)
    eval_result = trainer.evaluate()
    with open(log_file, "a") as f:
        print(f"Evaluation result: {eval_result}", file=f)
    print(f"Evaluation result: {eval_result}")
except Exception as e:
    with open(log_file, "a") as f:
        print(f"Evaluation error: {e}", file=f)
    print(f"Evaluation error: {e}")

print("Enter reviews to test (type 'exit' to quit) or 'save' to export model (type within the script prompt):", file=sys.stderr)
try:
    with open(log_file, "a") as f:
        print("Enter reviews to test (type 'exit' to quit) or 'save' to export model (type within the script prompt):", file=f)
    model.eval()
    while True:
        try:
            new_review = input("> ").lower()
            if new_review == "exit":
                with open(log_file, "a") as f:
                    print("Thanks for testing—Orion’s analyzer signing off!", file=f)
                print("Thanks for testing—Orion’s analyzer signing off!")
                break
            elif new_review == "save":
                model.save_pretrained("./saved_model")
                tokenizer.save_pretrained("./saved_model")
                with open(log_file, "a") as f:
                    print("Model and tokenizer saved to ./saved_model/", file=f)
                print("Model and tokenizer saved to ./saved_model/")
            else:
                inputs = tokenizer(new_review, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted = outputs.logits.argmax().item()
                    predicted_sentiment = ["negative", "positive", "neutral"][predicted]
                with open(log_file, "a") as f:
                    print(f"Predicted sentiment: {predicted_sentiment}", file=f)
                print(f"Predicted sentiment: {predicted_sentiment}")
        except ValueError as e:
            with open(log_file, "a") as f:
                print(f"Error processing input: {e}", file=f)
            print(f"Error processing input: {e}")
except Exception as e:
    print(f"Input loop error: {e}", file=sys.stderr)
    print(f"Input loop error: {e}")

print("Orion’s AI sentiment analyzer—impress your clients!", file=sys.stderr)
try:
    with open(log_file, "a") as f:
        print("Orion’s AI sentiment analyzer—impress your clients!", file=f)
except Exception as e:
    print(f"Final log error: {e}", file=sys.stderr)
    print(f"Final log error: {e}")
except Exception as e:
    print(f"Global error: {e}", file=sys.stderr)
    print(f"Global error: {e}")