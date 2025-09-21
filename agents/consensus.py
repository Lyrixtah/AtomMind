"""Consensus module for selecting the best candidate based on scores."""

from typing import List, Union
import numpy as np

def consensus(candidates: List[str], scores: Union[List[float], np.ndarray]) -> str:
    """
    Select the candidate with the highest score.

    Args:
        candidates (List[str]): List of generated candidate outputs.
        scores (List[float] | np.ndarray): Corresponding evaluation scores.

    Returns:
        str: The candidate with the highest score.
    """
    best_idx = np.argmax(scores)
    return candidates[best_idx]
