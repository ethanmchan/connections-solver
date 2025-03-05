# STEP 1: Scrapes the NYT Connections Archive Website for past Connections answers for building dataset.
# Scrapes into nyt_connections_dataset.json

import requests
from bs4 import BeautifulSoup
import json
import os

BASE_URL = "https://connections.swellgarfo.com/nyt/"
OUTPUT_FILE = "nyt_connections_dataset.json"

# Load existing answers
def load_existing_answers():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Save scraped puzzles
def save_answers(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# Scrape puzzle words
def extract_puzzle(puzzle_id):
    url = f"{BASE_URL}{puzzle_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "lxml")
        script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if script_tag:
            json_data = json.loads(script_tag.string)
            answers = json_data.get("props", {}).get("pageProps", {}).get("answers", [])
            
            # Extract words and category reasoning
            puzzle_data = {
                "puzzle_id": puzzle_id,
                "words": [],
                "categories": {},
                "reasoning": {}
            }
            
            for group in answers:
                category = group["description"]
                words = group["words"]
                puzzle_data["words"].extend(words)
                puzzle_data["categories"][category] = words
                puzzle_data["reasoning"][category] = f"These words all share the concept of {category.lower()}."
            
            return puzzle_data
    return None

# Scrape multiple puzzles
def scrape_puzzles(start_id, end_id):
    puzzles = load_existing_answers()
    
    for puzzle_id in range(start_id, end_id + 1):
        if str(puzzle_id) in puzzles:
            print(f"Skipping puzzle {puzzle_id} (already scraped).")
            continue
        
        print(f"Scraping puzzle {puzzle_id}...")
        puzzle_data = extract_puzzle(puzzle_id)
        if puzzle_data:
            puzzles[str(puzzle_id)] = puzzle_data
            save_answers(puzzles)
    
    print("Scraping complete.")

if __name__ == "__main__":
    scrape_puzzles(1, 540)
