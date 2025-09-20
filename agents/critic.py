import random

class CriticAgent:
    def evaluate(self, candidates):
        scores = [random.uniform(0, 1) for _ in candidates]
        return scores
