import numpy as np

def consensus(candidates, scores):
    best_idx = np.argmax(scores)
    return candidates[best_idx]
