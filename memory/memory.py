"""
This module implements a simple memory system for storing episodic
traces and querying the top scored traces.
"""

from typing import Any, List, Dict, Optional

class MemorySystem:
    """
    MemorySystem handles storage, retrieval, and pruning of episodic
    memories with optional scoring.

    Attributes:
        episodic (List[Dict]):
            A list of episodic memory traces.
        long_term (List[Dict]):
            Placeholder for long-term memory traces.
    """

    def __init__(self) -> None:
        """
        Initialize the MemorySystem with empty episodic and long-term memory lists.
        """
        self.episodic: List[Dict[str, Any]] = []
        self.long_term: List[Dict[str, Any]] = []

    def store(self, trace: Any, score: Optional[float] = None) -> None:
        """
        Store a new trace in episodic memory, optionally with a score.
        Maintains a maximum of 1000 episodic traces.

        Args:
            trace (Any):
                The memory trace to store.
            score (Optional[float]):
                The associated score for ranking (default: None).
        """
        self.episodic.append({"trace": trace, "score": score})
        if len(self.episodic) > 1000:
            self.episodic.pop(0)

    def query(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k traces sorted by score in descending order.

        Args:
            top_k (int): The number of top traces to return (default: 5).
        Returns:
            List[Dict[str, Any]]: The list of top_k traces with highest scores.
        """
        sorted_mem = sorted(
            self.episodic, key=lambda x: x.get("score", 0), reverse=True
        )
        return sorted_mem[:top_k]

    def prune(self) -> None:
        """
        Remove traces with scores below 0.1 from episodic memory.
        """
        self.episodic = [t for t in self.episodic if t.get("score", 0) > 0.1]
