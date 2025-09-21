"""Consensus module for selecting the best candidate based on scores."""

import numpy as np

def consensus(candidates, scores):
    """
    Select the candidate with the highest score.

    Args:
        candidates (list[str]): List of generated candidate outputs.
        scores (list[float] or np.ndarray): Corresponding evaluation scores.

    Returns:
        str: The candidate with the highest score.
    """
    best_idx = np.argmax(scores)
    return candidates[best_idx]
