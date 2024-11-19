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

# Loop through all page IDs and collect answers
def scrape_connections_answers(start_id, end_id):
    all_answers = {}
    for page_id in range(start_id, end_id + 1):
        print(f"Scraping page {page_id}...")
        answers = extract_answers(page_id)
        if answers:
            all_answers[page_id] = answers

    with open('nyt_connections_answers.json', 'w') as f:
        json.dump(all_answers, f, indent=4)

scrape_connections_answers(1, 527)
print("Scraping completed. Answers saved to 'nyt_connections_answers.json'.")
