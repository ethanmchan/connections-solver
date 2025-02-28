import os
import pickle
import numpy as np
from tqdm import tqdm

# Functions for loading/saving GloVe embeddings and pickle files

# Save embeddings to a pickle file (one-time setup)
def save_glove_embeddings_to_pickle(glove_path, pickle_path):
    embeddings = load_glove_embeddings(glove_path)
    with open(pickle_path, 'wb') as f:
        pickle.dump(embeddings, f)

# Load embeddings from pickle
def load_glove_embeddings_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

# General function to load embeddings
def load_embeddings(glove_path="glove.6B.100d.txt", pickle_path="glove.6B.100d.pkl"):
    if os.path.exists(pickle_path):
        print("Loading embeddings from pickle...")
        return load_glove_embeddings_from_pickle(pickle_path)
    else:
        print("Loading embeddings from GloVe file and saving to pickle...")
        save_glove_embeddings_to_pickle(glove_path, pickle_path)
        return load_glove_embeddings_from_pickle(pickle_path)

# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading GloVe embeddings"):
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
