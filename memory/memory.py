class MemorySystem:
    def __init__(self):
        self.episodic = []
        self.long_term = []

    def store(self, trace, score=None):
        self.episodic.append({"trace": trace, "score": score})
        if len(self.episodic) > 1000:
            self.episodic.pop(0)

    def query(self, top_k=5):
        sorted_mem = sorted(self.episodic, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_mem[:top_k]

    def prune(self):
        self.episodic = [t for t in self.episodic if t.get("score", 0) > 0.1]
