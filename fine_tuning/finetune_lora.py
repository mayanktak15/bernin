import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from evaluate import load

# Load dataset
dataset = load_from_disk("medical_dataset")

# Split into train (80%) and test (20%)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Preprocess dataset
def preprocess_function(examples):
    inputs = [f"Question: {q} Answer:" for q in examples["question"]]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    # Replace padding token ID with -100 for loss calculation
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i][model_inputs["labels"][i] == tokenizer.pad_token_id] = -100
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_flan_t5_small",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
)

# Evaluation metrics
bleu = load("bleu")
rouge = load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Handle predictions: convert logits to token IDs if necessary
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract logits if eval_pred is a tuple
    if predictions.ndim == 3:  # Shape: (batch_size, sequence_length, vocab_size)
        predictions = np.argmax(predictions, axis=-1)  # Convert logits to token IDs

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Handle labels: replace -100 with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # BLEU
    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # ROUGE
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rougeL": rouge_result["rougeL"]
    }


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Fine-tune
trainer.train()

# Save model
model.save_pretrained("./lora_flan_t5_small/finetuned")
tokenizer.save_pretrained("./lora_flan_t5_small/finetuned")

# Evaluate on test set
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

print("Fine-tuning complete. Model saved to ./lora_flan_t5_small/finetuned")