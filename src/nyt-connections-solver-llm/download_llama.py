# STEP 4: Downloads tokenized dataset and Llama-2 model from Hugging Face 

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

# Load dataset from disk
dataset = load_from_disk("llama_nyt_connections_dataset")

# Choose Llama-2 model (7B)
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set a padding token (Llama-2 doesn't have one by default)
tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token for padding

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset
tokenized_datasets.save_to_disk("llama_nyt_connections_tokenized")

print("Llama-2 model and dataset loaded successfully!")