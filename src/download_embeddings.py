"""
Download and cache GloVe embeddings (300d).
"""
import os
import sys
import gzip
import urllib.request
import numpy as np
from tqdm import tqdm

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
CACHE_DIR = "data"


def download_glove(target_dir=CACHE_DIR):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "glove.6B.zip")
    txt_path = os.path.join(target_dir, "glove.6B.300d.txt")

    if os.path.exists(txt_path):
        print(f"GloVe already at {txt_path}")
        return txt_path

    if not os.path.exists(zip_path):
        print(f"Downloading GloVe 6B from {GLOVE_URL} ...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
        print("Downloaded.")

    print("Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract("glove.6B.300d.txt", target_dir)
    print(f"Extracted to {txt_path}")
    return txt_path


def load_glove_embeddings(filepath="data/glove.6B.300d.txt", max_vocab=None):
    if not os.path.exists(filepath):
        filepath = download_glove()
    embeddings = {}
    print(f"Loading GloVe from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading vectors")):
            if max_vocab and i >= max_vocab:
                break
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vector
    emb_dim = len(next(iter(embeddings.values())))
    print(f"Loaded {len(embeddings)} words, dim={emb_dim}")
    return embeddings, emb_dim


if __name__ == "__main__":
    download_glove()
    embeddings, dim = load_glove_embeddings()
    print(f"GloVe ready: {len(embeddings)} vectors x {dim}d")
