import pandas as pd
import torch
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict, ClassLabel, Value, Features
from transformers import pipeline
import math
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from sklearn.metrics import classification_report
import torch
from sklearn.metrics import accuracy_score


# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
PYTORCH_CUDA_ALLOC_CONF= True
print(f"Using device: {device}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Calculate each metric
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    prec = precision_metric.compute(predictions=preds, references=labels, average="binary")
    rec = recall_metric.compute(predictions=preds, references=labels, average="binary")
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")

    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1["f1"]
    }





def convert_data():
    # Define column names
    columns = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation",
        "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts",
        "context"
    ]

    # Load data
    df_train = pd.read_csv("train.tsv", sep="\t")
    df_valid = pd.read_csv("valid.tsv", sep="\t")
    df_test = pd.read_csv("test.tsv", sep="\t")

    df_train.columns = columns
    df_valid.columns = columns
    df_test.columns = columns

    # Select relevant columns
    df_train = df_train[["statement", "label", "speaker", "context","barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts"]]

    df_valid = df_valid[["statement", "label", "speaker", "context","barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts"]]

    df_test = df_test[["statement", "label", "speaker", "context","barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts"]]


    df_train = df_train.fillna(0)
    df_valid = df_train.fillna(0)
    df_test = df_test.fillna(0)
    # Map labels to numerical values


    df_train["label"] = df_train["label"].str.lower().map(label_map).astype(int)
    df_valid["label"] = df_valid["label"].str.lower().map(label_map).astype(int)
    df_test["label"] = df_test["label"].str.lower().map(label_map).astype(int)


    # Define valid class labels
    label_classes = ["true", "mostly-true", "half-true", "barely-true",
        "false", "pants-fire"]

    # Define dataset features
    features = Features({
        'statement': Value('string'),
        'label': ClassLabel(num_classes=len(label_classes), names=label_classes),
        'speaker': Value('string'),
        'context': Value('string'),
        'false_counts' : Value('int16'),
        'true_counts' : Value('int16')
    })


    # Create train and test datasets
    train_dataset = Dataset.from_dict({
        'statement': df_train["statement"],
        'label': df_train["label"],
        'speaker': df_train["speaker"],
        'context': df_train["context"],
        'false_counts': df_train['false_counts'] + df_train['pants_on_fire_counts'] + (df_train['barely_true_counts'] / 2),
        'true_counts': df_train['mostly_true_counts'] + df_train['half_true_counts'] + (df_train['barely_true_counts'] / 2)
    }, features=features)


    valid_dataset = Dataset.from_dict({
        'statement': df_valid["statement"],
        'label': df_valid["label"],
        'speaker': df_valid["speaker"],
        'context': df_valid["context"],
        'false_counts': df_valid['false_counts'] + df_valid['pants_on_fire_counts'] + (df_valid['barely_true_counts'] / 2),
        'true_counts': df_valid['mostly_true_counts'] + df_valid['half_true_counts'] + (df_valid['barely_true_counts'] / 2)
    }, features=features)




    test_dataset = Dataset.from_dict({
        'statement': df_test["statement"],
        'label': df_test["label"],
        'speaker': df_test["speaker"],
        'context': df_test["context"],
        'false_counts': df_test['false_counts'] + df_test['pants_on_fire_counts'] + (df_test['barely_true_counts'] / 2),
        'true_counts': df_test['mostly_true_counts'] + df_test['half_true_counts'] + (df_test['barely_true_counts'] / 2)

    }, features=features)


    # Define DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset,
        'test': test_dataset
    })

    return dataset_dict


def preprocess_function(samples):
    return tokenizer(
        [" ".join([
            f'statement: {s}',
            f'speaker: {sp}',
            f'context: {c}',
            f'true_counts: {tc}',
            f'false_counts: {fc}'
        ]) for s, sp, c, tc, fc in zip(
            samples["statement"],
            samples["speaker"],
            samples["context"],
            samples["true_counts"],
            samples["false_counts"]
        )],
        truncation=True,
        padding=True,
        max_length=512,
    )

# Convert raw data into HF dataset format
label_map = {
        "true": 0,
        "mostly-true": 0,
        "half-true": 0,
        "barely-true": 1,
        "false": 1,
        "pants-fire": 1
    }
dataset = convert_data(label_map)

tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base')

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

label2id = {
    "true": 0,
    "false": 1,
}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping

training_args = TrainingArguments(
    output_dir="liar-dataset-model",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print(test_results)


# 1. Save the model
trainer.save_model("my_saved_model")  # The directory name is up to you.

# 2. Save the tokenizer (optional, but highly recommended)
tokenizer.save_pretrained("my_saved_model")

# Folder to be zipped
folder_to_zip = "my_saved_model"

# Output ZIP file name (without .zip extension)
output_zip_name = "my_saved_model_backup"

# Create the ZIP file
shutil.make_archive(output_zip_name, 'zip', folder_to_zip)

print(f"Zipped folder saved as {output_zip_name}.zip")

shutil.unpack_archive("my_saved_model_backup.zip", "my_saved_model")

print("Unzipped successfully!")

#True Label Map:
label_map = {
        "true": 0,
        "mostly-true": 1,
        "half-true": 2,
        "barely-true": 3,
        "false": 4,
        "pants-fire": 5
    }

dataset = convert_data(label_map)
tokenizer = AutoTokenizer.from_pretrained(
'roberta-base')

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
PYTORCH_CUDA_ALLOC_CONF= True
print(f"Using device: {device}")

# Load the saved model and tokenizer
model_path = "my_saved_model"
loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)

def evaluate_saved_model(model, tokenizer, dataset):
    """
    Evaluate a saved model on a dataset.
    """
    model.eval()  # Set model to evaluation mode
    output_list = []
    correct_list = []
    correct_list.extend(dataset["label"])
    with torch.no_grad():  # Disable gradient calculations
        for sample in dataset:
            inputs = tokenizer(
                sample["statement"],
                padding=True, truncation=True, return_tensors="pt"
            ).to(device)  # Move to GPU if available

            logits = model(**inputs).logits  # Get logits from model
            probs = softmax(logits.cpu().numpy(), axis=1)  # Convert to probabilities
            true_value = probs[0][0]
            false_value = probs[0][1]


            if true_value > false_value:
                if true_value >= 0.80:
                    output_list.append(0)
                elif true_value >= 0.40:
                    output_list.append(1)
                else:
                    output_list.append(2)
            else:
                if false_value >= 0.80:
                    output_list.append(3)
                elif false_value >= 0.40:
                    output_list.append(4)
                else:
                    output_list.append(5)


    print("output_list", output_list)
    print("correct_lis", correct_list)
    # Compute metrics
    report = classification_report(
        output_list,
        correct_list,
        labels=[0, 1, 2, 3, 4, 5],
        target_names=["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"],
        digits=3
    )
    print(report)

    acc = accuracy_score(correct_list, output_list)
    print(f"Accuracy: {acc:.3f}")


# Example usage: Evaluate the saved model on the test dataset
evaluate_saved_model(loaded_model, loaded_tokenizer, tokenized_dataset["test"])

