# api/similarity.py
import numpy as np

def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2))  # embeddings already normalized
