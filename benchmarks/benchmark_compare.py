import time
import csv
from typing import Callable

try:
    from gptcache.manager.eviction.cost_aware import make_cost_aware_cache
except Exception:
    raise RuntimeError("No cost-aware cache implementation found; install it before running benchmarks")

from cachetools import LRUCache


def make_cost(maxsize=100):
        return make_cost_aware_cache(maxsize=maxsize)


def synth_workload(num_unique=100, ops=10000, skew=0.8):
    # Generate keys where low ids are hotter
    import random
    keys = [f"k{i}" for i in range(num_unique)]
    ops_list = []
    for _ in range(ops):
        # Zipf-like skew by sampling small indices more often
        r = random.random()
        if r < skew:
            idx = random.randint(0, max(0, int(num_unique * 0.2) - 1))
        else:
            idx = random.randint(0, num_unique - 1)
        ops_list.append(keys[idx])
    return ops_list


def run_benchmark(cache_factory: Callable, ops_list):
    cache = cache_factory()
    start = time.perf_counter()
    hits = 0
    misses = 0
    for k in ops_list:
        try:
            _ = cache[k]
            hits += 1
        except Exception:
            # miss -> generate and insert
            cache[k] = "v" * (len(k) % 50 + 1)
            misses += 1
    duration = time.perf_counter() - start
    # attempt to read get_stats
    stats = getattr(cache, "get_stats", lambda: None)()
    return {"hits": hits, "misses": misses, "duration": duration, "stats": stats}


def csv_header():
    return ["policy", "num_ops", "hits", "misses", "duration_sec", "extra_stats"]


def main():
    ops_list = synth_workload(num_unique=200, ops=20000, skew=0.85)
    rows = []
    print(','.join(csv_header()))
    # LRU
    r = run_benchmark(lambda: LRUCache(maxsize=200), ops_list)
    print(f"LRU,{len(ops_list)},{r['hits']},{r['misses']},{r['duration']},{r['stats']}")
    # Cost-aware
    r2 = run_benchmark(lambda: make_cost(maxsize=200), ops_list)
    print(f"COST_AWARE,{len(ops_list)},{r2['hits']},{r2['misses']},{r2['duration']},{r2['stats']}")

if __name__ == '__main__':
    main()