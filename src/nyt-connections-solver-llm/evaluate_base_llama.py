import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
from tqdm import tqdm

# Load dataset
with open("nyt_connections_dataset.json", "r") as file:
    dataset = json.load(file)

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

def generate_groupings(words):
    """Uses Llama-2 to generate 4 groups of 4 words from a given list."""
    prompt = f"""
    Organize the following 16 words into 4 groups of 4 words each based on common themes:
    {', '.join(words)}
    
    Provide the output in JSON format as follows:
    {{
        "Group 1": ["word1", "word2", "word3", "word4"],
        "Group 2": ["word5", "word6", "word7", "word8"],
        "Group 3": ["word9", "word10", "word11", "word12"],
        "Group 4": ["word13", "word14", "word15", "word16"]
    }}
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_data = json.loads(response[json_start:json_end])
        return json_data
    except:
        return None

def evaluate_results(predictions, actual):
    """Evaluates accuracy by counting correct groupings."""
    correct_count = sum(set(pred) in actual.values() for pred in predictions.values())
    return correct_count

# Evaluation metrics
correct_distribution = Counter()
total_puzzles = len(dataset)
total_correct = 0

evaluation_results = {}

# Add progress bar
for puzzle_id, data in tqdm(dataset.items(), desc="Processing Puzzles"):
    words = data["words"]
    actual_groups = data["categories"]
    
    predicted_groups = generate_groupings(words)
    if not predicted_groups:
        continue
    
    correct_count = evaluate_results(predicted_groups, actual_groups)
    correct_distribution[correct_count] += 1
    
    if correct_count == 4:
        total_correct += 1
    
    evaluation_results[puzzle_id] = {
        "predicted_groups": predicted_groups,
        "correct_count": correct_count
    }

# Print summary
print("Total puzzles evaluated:", total_puzzles)
print("Total fully correct puzzles:", total_correct)
print("Correct grouping distribution:")
for k, v in sorted(correct_distribution.items()):
    print(f"{k}/4 correct: {v} puzzles")
