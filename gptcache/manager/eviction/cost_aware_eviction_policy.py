class CostAwareCache:
    def __init__(self, maxsize, alpha=1.0, beta=1.0):
        self.maxsize = maxsize
        self.alpha = alpha  # weight for generation cost
        self.beta = beta    # weight for frequency
        self.cache = {}
        self.metadata = {}

    def __setitem__(self, key, value):
        # Add or update
        self.cache[key] = value
        self.metadata.setdefault(key, {
            "generation_time": self._estimate_cost(value),
            "access_count": 0,
        })
        # Evict if needed
        if len(self.cache) > self.maxsize:
            self._evict()

    def __getitem__(self, key):
        self.metadata[key]["access_count"] += 1
        return self.cache[key]

    def _score(self, key):
        data = self.metadata[key]
        return self.alpha * data["generation_time"] + self.beta * data["access_count"]

    def _evict(self):
        # Find item with lowest score
        lowest = min(self.cache, key=self._score)
        del self.cache[lowest]
        del self.metadata[lowest]

    def _estimate_cost(self, value):
        # Could be response length, tokens, etc.
        return len(str(value)) / 100.0
