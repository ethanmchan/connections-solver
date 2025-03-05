# STEP 5: Train and fine-tune LLM
# Saves model into 'llama_finetuned'

import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# Load tokenized dataset
dataset = load_from_disk("llama_nyt_connections_tokenized")

# Load Llama-2 model & tokenizer
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Modify Dataset to Include Labels
def preprocess_function(examples):
    inputs = examples["prompt"]
    targets = examples["response"]  # Ensure "response" is formatted properly

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
dataset = dataset.map(preprocess_function, batched=True)

# Define training parameters
training_args = TrainingArguments(
    output_dir="llama_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=1,  # Reduce if running out of memory
    per_device_eval_batch_size=1,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    push_to_hub=False,  # Set to True if you want to upload to Hugging Face
    fp16=torch.cuda.is_available(),  # Use mixed precision training if on GPU
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("llama_finetuned")
print("🎉 Fine-tuning complete! Model saved to 'llama_finetuned'.")
