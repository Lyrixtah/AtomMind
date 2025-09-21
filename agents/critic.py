"""CriticAgent module for evaluating candidate outputs."""

import random
from typing import List

class CriticAgent:
    """
    The CriticAgent assigns evaluation scores to generated candidates.
    Scores are randomized in this simple baseline version.
    """

    def evaluate(self, candidates: List[str]) -> List[float]:
        """
        Assign a random score between 0 and 1 to each candidate.

        Args:
            candidates (List[str]): List of generated candidate outputs.

        Returns:
            List[float]: Random scores corresponding to each candidate.
        """
        return [random.uniform(0, 1) for _ in candidates]

    def evaluate_metrics(self, candidates: List[str]) -> List[float]:
        """
        Backward-compatible alias for `evaluate`.

        Args:
            candidates (List[str]): List of generated candidate outputs.

        Returns:
            List[float]: Random scores corresponding to each candidate.
        """
        return self.evaluate(candidates)
