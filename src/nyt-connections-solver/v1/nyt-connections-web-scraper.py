import os
import requests
from bs4 import BeautifulSoup
import json

base_url = 'https://connections.swellgarfo.com/nyt/'

# Function to extract answers from a single page
def extract_answers(page_id):
    url = f"{base_url}{page_id}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if script_tag:
            json_data = json.loads(script_tag.string)
            answers = json_data.get('props', {}).get('pageProps', {}).get('answers', [])
            return answers
    else:
        print(f"Failed to retrieve page {page_id}. Status code: {response.status_code}")
    return None

# Load existing answers from file
def load_existing_answers(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Save answers to file
def save_answers(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

# Loop through all page IDs and collect answers
def scrape_connections_answers(start_id, end_id, output_file):
    # Load existing answers
    all_answers = load_existing_answers(output_file)
    
    for page_id in range(start_id, end_id + 1):
        if str(page_id) in all_answers:  # Check if page is already scraped
            print(f"Page {page_id} already scraped. Skipping.")
            continue
        
        print(f"Scraping page {page_id}...")
        answers = extract_answers(page_id)
        if answers:
            all_answers[str(page_id)] = answers  # Save answers with page_id as string
            save_answers(output_file, all_answers)  # Incrementally save after each scrape
        else:
            print(f"Failed to scrape page {page_id}.")
    
    print(f"Scraping completed. Answers saved to '{output_file}'.")

# Main execution
if __name__ == "__main__":
    scrape_connections_answers(1, 545, "nyt_connections_answers.json")