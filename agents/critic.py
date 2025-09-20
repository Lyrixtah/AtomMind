import random
class CriticAgent:
    def evaluate(self, candidates):
        scores = [random.uniform(0, 1) for _ in candidates]
        return scores

    def evaluate_metrics(self, candidates):
        """Alias for backward compatibility with main.py"""
        return self.evaluate(candidates)
