import time
from cachetools import Cache

class CostAwarePolicy(Cache):
    def __init__(self, maxsize, alpha=1.0, beta=1.0):
        """
        maxsize: max number of entries
        alpha: weight for generation cost
        beta: weight for access frequency
        """
        self.maxsize = maxsize
        self.alpha = alpha
        self.beta = beta
        self.cache = {}         # key -> value
        self.metadata = {}      # key -> {'cost': float, 'access': int, 'timestamp': float}

    def __setitem__(self, key, value):
        # On insert, estimate cost and initialize metadata
        now = time.time()
        cost = self.estimate_cost(value)
        self.cache[key] = value
        self.metadata[key] = {
            "cost": cost,
            "access": 0,
            "timestamp": now,
        }
        # Evict if needed
        if len(self.cache) > self.maxsize:
            self.evict()

    def __getitem__(self, key):
        if key not in self.cache:
            raise KeyError(key)
        # Update access count and timestamp
        self.metadata[key]["access"] += 1
        self.metadata[key]["timestamp"] = time.time()
        return self.cache[key]

    def estimate_cost(self, value):
        # Simple cost estimate: length of response string (can customize)
        return len(str(value))

    def eviction_score(self, key):
        data = self.metadata[key]
        # Score = weighted cost + weighted access count (lower score = candidate for eviction)
        return self.alpha * data["cost"] + self.beta * data["access"]

    def evict(self):
        # Evict the key with the lowest eviction score
        key_to_evict = min(self.cache.keys(), key=self.eviction_score)
        del self.cache[key_to_evict]
        del self.metadata[key_to_evict]

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)
