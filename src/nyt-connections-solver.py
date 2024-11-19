import os
import pickle
import json
import requests
import numpy as np
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from tqdm import tqdm

# Import GloVe-related functions from glove_utils
from glove_utils import load_embeddings

# Scrape puzzle words by puzzle ID
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

# Normalize and get embeddings for words
def get_word_embeddings(words, embeddings):
    word_vectors = []
    for word in words:
        word = word.lower()
        if word in embeddings:
            word_vectors.append(embeddings[word])
        else:
            print(f"Word '{word}' not in embeddings!")  # Handle unknown words
            word_vectors.append(np.zeros(100))  # Use zero vector as fallback
    return np.array(word_vectors)

# Calculate pairwise cosine similarities
def calculate_cosine_similarities(vectors):
    num_words = len(vectors)
    similarity_matrix = np.zeros((num_words, num_words))
    for i in tqdm(range(num_words), desc="Calculating cosine similarities"):
        for j in range(num_words):
            if i != j:
                similarity_matrix[i][j] = 1 - cosine(vectors[i], vectors[j])
    return similarity_matrix

# Perform K-means clustering
def cluster_words(vectors, n_clusters=4):
    print("Clustering words...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    with tqdm(total=100, desc="K-means clustering") as pbar:
        kmeans.fit(vectors)
        pbar.update(100)
    return kmeans.labels_

def solve_puzzle(words, embeddings):
    # Step 1: Get word embeddings
    vectors = get_word_embeddings(words, embeddings)
    
    # Step 2: Calculate similarities (for debugging)
    similarity_matrix = calculate_cosine_similarities(vectors)
    
    # Step 3: Cluster words
    labels = cluster_words(vectors)
    
    # Step 4: Group words based on clustering
    groups = {i: [] for i in range(4)}
    for word, label in zip(words, labels):
        groups[label].append(word)
    
    return groups

# Display the puzzle as a 4x4 table
def display_puzzle_table(words):
    print("\nPuzzle Grid:")
    print("-" * 33)
    for i in range(0, 16, 4):
        print(f"| {words[i]:<8} | {words[i+1]:<8} | {words[i+2]:<8} | {words[i+3]:<8} |")
        print("-" * 33)

# Main Function
def main():
    # Load GloVe embeddings with reduced overhead
    glove_embeddings = load_embeddings(
        glove_path="glove.6B.100d.txt",
        pickle_path="glove.6B.100d.pkl"
    )
    
    # Input: Puzzle ID
    puzzle_id = input("Enter the puzzle ID to solve: ")
    
    # Fetch puzzle words
    words = get_puzzle_words(puzzle_id)
    if not words:
        print("No words found for the given puzzle!")
        return
    
    # Display the puzzle table
    display_puzzle_table(words)
    
    # Solve the puzzle
    groups = solve_puzzle(words, glove_embeddings)
    
    # Print grouped words
    print("\nSolution:")
    for group_id, group_words in groups.items():
        print(f"Group {group_id}: {group_words}")

# Run the program
if __name__ == "__main__":
    main()