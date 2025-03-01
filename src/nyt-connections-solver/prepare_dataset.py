# STEP 3: Converts the formatted JSON into a Hugging Face dataset format so it can be used for LLM fine-tuning.
# Creates training and validation datasets

from datasets import Dataset
import json

# Load formatted dataset
with open("formatted_nyt_connections.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert dataset into a Hugging Face dataset
dataset = Dataset.from_dict({
    "prompt": [entry["prompt"] for entry in data],
    "response": [json.dumps(entry["response"]) for entry in data]  # Convert response dict to string
})

# Split into training and validation sets (90% train, 10% test)
dataset = dataset.train_test_split(test_size=0.1)

# Save dataset for quick reloading
dataset.save_to_disk("llama_nyt_connections_dataset")

print("Dataset loaded and saved successfully!")