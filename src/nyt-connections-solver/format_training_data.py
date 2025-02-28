import json

# Load raw puzzle data
with open("nyt_connections_dataset.json", "r", encoding="utf-8") as f:
    puzzles = json.load(f)

formatted_data = []

for puzzle in puzzles.values():
    words = ", ".join(puzzle["words"])
    
    for category, words_list in puzzle["categories"].items():
        reasoning = puzzle["reasoning"].get(category, "No reasoning available.")

        # Construct structured prompts (removing extra spaces/newlines)
        prompt = f"Words: {words}\n\nStep 1: Identify the connection.\nStep 2: Assign words to a category.\nStep 3: Explain why they belong together."

        response = {
            "category": category,
            "words": words_list,
            "explanation": reasoning
        }

        formatted_data.append({"prompt": prompt, "response": response})

# Save formatted training data
with open("formatted_nyt_connections.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, indent=4)

print("Training data formatted successfully!")