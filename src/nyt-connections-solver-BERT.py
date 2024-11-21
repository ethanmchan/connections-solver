import requests
from bs4 import BeautifulSoup
import json
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

# Initialize BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Function to scrape puzzle words by puzzle ID
def get_puzzle_words(puzzle_id):
    url = f"https://connections.swellgarfo.com/nyt/{puzzle_id}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'lxml')
        script_tag = soup.find('script', {'id': '__NEXT_DATA__'})
        if script_tag:
            # Parse JSON from the script tag
            json_data = json.loads(script_tag.string)
            # Extract the words
            answers = json_data.get('props', {}).get('pageProps', {}).get('answers', [])
            words = []
            for group in answers:
                words.extend(group['words'])
            return words
        else:
            print("Error: Puzzle data not found!")
            return []
    else:
        print(f"Failed to fetch puzzle {puzzle_id}. Status code: {response.status_code}")
        return []

# Function to get BERT embeddings for words
def get_bert_embeddings(words):
    embeddings = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=5)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the CLS token embedding as the representation of the word
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

# Perform K-means clustering
def cluster_words(vectors, n_clusters=4):
    print("Clustering words...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    with tqdm(total=100, desc="K-means clustering") as pbar:
        kmeans.fit(vectors)
        pbar.update(100)
    return kmeans.labels_

# Solve the puzzle using BERT embeddings
def solve_puzzle_with_bert(words):
    # Step 1: Get word embeddings using BERT
    vectors = get_bert_embeddings(words)
    
    # Step 2: Cluster words
    labels = cluster_words(vectors)
    
    # Step 3: Group words based on clustering
    groups = {i: [] for i in range(4)}
    for word, label in zip(words, labels):
        groups[label].append(word)
    
    return groups

# Display the puzzle as a 4x4 table
def display_puzzle_table(words):
    max_word_length = max(len(word) for word in words)
    column_width = max_word_length + 2
    separator = "-" * ((column_width + 3) * 4 + 1)

    print("\nPuzzle Grid:")
    print(separator)
    for i in range(0, 16, 4):
        row = "|".join(f" {words[j]:<{column_width}}" for j in range(i, i + 4))
        print(f"|{row} |")
        print(separator)

# Main function
def main():
    # Input: Puzzle ID
    puzzle_id = input("Enter the puzzle ID to solve: ")
    
    # Fetch puzzle words
    words = get_puzzle_words(puzzle_id)
    if not words:
        print("No words found for the given puzzle!")
        return
    
    # Display the puzzle table
    display_puzzle_table(words)
    
    # Solve the puzzle using BERT embeddings
    groups = solve_puzzle_with_bert(words)
    
    # Print grouped words
    print("\nSolution:")
    for group_id, group_words in groups.items():
        print(f"Group {group_id}: {group_words}")

# Run the program
if __name__ == "__main__":
    main()
