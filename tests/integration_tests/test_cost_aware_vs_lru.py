# File: tests/integration/test_cost_aware_vs_lru.py
"""
Integration tests comparing cost-aware policy against LRU and other policies.
"""

import pytest
import random
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from gptcache.manager.eviction.cost_aware_policy import CostAwareCache

try:
    from cachetools import LRUCache, LFUCache

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False


@dataclass
class WorkloadResult:
    """Results from running a workload against a cache."""
    policy_name: str
    hit_rate: float
    total_cost_saved: float
    avg_latency_ms: float
    evictions: int
    final_cache_size: int


class WorkloadGenerator:
    """Generates synthetic workloads for cache testing."""

    def __init__(self, seed: int = 42):
        """Initialize workload generator with random seed."""
        random.seed(seed)
        self.models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
        self.model_costs = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03, "claude-3": 0.015}

    def generate_query(self, length_category: str = "medium") -> Tuple[str, str]:
        """Generate a query-response pair."""
        if length_category == "short":
            query = f"short query {random.randint(1, 1000)}"
            response = f"brief answer {random.randint(1, 100)}"
        elif length_category == "long":
            query = "very detailed question " * random.randint(10, 50)
            response = "comprehensive detailed answer " * random.randint(20, 100)
        else:  # medium
            query = "medium complexity query " * random.randint(2, 8)
            response = "reasonable length response " * random.randint(5, 20)

        return query, response

    def generate_read_heavy_workload(self,
                                     num_operations: int = 1000,
                                     read_ratio: float = 0.9,
                                     skew_factor: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate read-heavy workload with expensive tail.

        Args:
            num_operations: Total number of operations
            read_ratio: Fraction of operations that are reads
            skew_factor: Proportion of expensive queries (Zipf-like distribution)
        """
        operations = []
        unique_queries = []

        # Generate pool of unique queries
        expensive_count = int(100 * skew_factor)
        cheap_count = 100 - expensive_count

        # Expensive queries (longer, GPT-4)
        for i in range(expensive_count):
            query, response = self.generate_query("long")
            unique_queries.append({
                'key': f"expensive_{i}",
                'query': query,
                'response': response,
                'model': 'gpt-4',
                'cost_category': 'high'
            })

        # Cheap queries (shorter, GPT-3.5)
        for i in range(cheap_count):
            query, response = self.generate_query("short")
            unique_queries.append({
                'key': f"cheap_{i}",
                'query': query,
                'response': response,
                'model': 'gpt-3.5-turbo',
                'cost_category': 'low'
            })

        # Generate operations with Zipf-like access pattern
        num_reads = int(num_operations * read_ratio)
        num_writes = num_operations - num_reads

        # Writes (populate cache)
        for i in range(min(num_writes, len(unique_queries))):
            operations.append({
                'type': 'write',
                'data': unique_queries[i]
            })

        # Reads (with skewed access - expensive items accessed more frequently)
        for _ in range(num_reads):
            if random.random() < 0.7:  # 70% access to expensive items
                idx = random.randint(0, expensive_count - 1)
            else:  # 30% access to cheap items
                idx = expensive_count + random.randint(0, cheap_count - 1)

            if idx < len(unique_queries):
                operations.append({
                    'type': 'read',
                    'data': unique_queries[idx]
                })

        random.shuffle(operations)
        return operations

    def generate_mixed_workload(self,
                                num_operations: int = 1000,
                                read_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """Generate mixed read/write workload with uniform costs."""
        operations = []
        query_pool = []

        # Generate diverse query pool
        for i in range(200):
            model = random.choice(self.models)
            length = random.choice(["short", "medium", "long"])
            query, response = self.generate_query(length)

            query_pool.append({
                'key': f"query_{i}",
                'query': query,
                'response': response,
                'model': model,
                'cost_category': 'mixed'
            })

        num_reads = int(num_operations * read_ratio)
        num_writes = num_operations - num_reads

        # Generate writes
        for _ in range(num_writes):
            query_data = random.choice(query_pool)
            operations.append({
                'type': 'write',
                'data': query_data
            })

        # Generate reads
        for _ in range(num_reads):
            query_data = random.choice(query_pool)
            operations.append({
                'type': 'read',
                'data': query_data
            })

        random.shuffle(operations)
        return operations

    def generate_write_heavy_workload(self,
                                      num_operations: int = 1000,
                                      read_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """Generate write-heavy workload."""
        operations = []

        num_reads = int(num_operations * read_ratio)
        num_writes = num_operations - num_reads

        # Generate unique writes (no repeats to stress eviction)
        for i in range(num_writes):
            model = random.choice(self.models)
            length = random.choice(["short", "medium", "long"])
            query, response = self.generate_query(length)

            operations.append({
                'type': 'write',
                'data': {
                    'key': f"unique_{i}",
                    'query': query,
                    'response': response,
                    'model': model,
                    'cost_category': 'varied'
                }
            })

        # Generate some reads to recently written data
        for _ in range(num_reads):
            idx = max(0, num_writes - random.randint(1, 50))  # Read recent writes
            operations.append({
                'type': 'read',
                'data': {
                    'key': f"unique_{idx}",
                    'query': f"query for unique_{idx}",
                    'response': f"response for unique_{idx}",
                    'model': random.choice(self.models),
                    'cost_category': 'varied'
                }
            })

        random.shuffle(operations)
        return operations


class CacheSimulator:
    """Simulates cache behavior for benchmarking."""

    def __init__(self, cache_policy: str, maxsize: int = 100, **policy_params):
        """Initialize cache simulator."""
        self.policy_name = cache_policy
        self.maxsize = maxsize

        if cache_policy == "cost_aware":
            self.cache = CostAwareCache(maxsize=maxsize, **policy_params)
        elif cache_policy == "lru" and CACHETOOLS_AVAILABLE:
            self.cache = LRUCache(maxsize=maxsize)
        elif cache_policy == "lfu" and CACHETOOLS_AVAILABLE:
            self.cache = LFUCache(maxsize=maxsize)
        else:
            # Fallback to cost-aware cache
            self.cache = CostAwareCache(maxsize=maxsize, **policy_params)

        # Track metrics for all cache types
        self.hits = 0
        self.misses = 0
        self.total_cost_saved = 0.0
        self.evictions = 0
        self.latencies = []

    def _calculate_cost(self, query: str, response: str, model: str) -> float:
        """Calculate cost for non-cost-aware caches."""
        model_prices = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03, "claude-3": 0.015}
        price_per_1k = model_prices.get(model, 0.002)

        input_tokens = max(1, len(query.split()) + len(query) // 4)
        output_tokens = max(1, len(response.split()) + len(response) // 4)

        return (input_tokens + output_tokens) * price_per_1k / 1000

    def run_workload(self, operations: List[Dict[str, Any]]) -> WorkloadResult:
        """Run workload against cache and collect metrics."""
        for operation in operations:
            start_time = time.time()

            data = operation['data']
            key = data['key']

            if operation['type'] == 'read':
                # Simulate cache get
                if hasattr(self.cache, 'get'):
                    result = self.cache.get(key)
                else:
                    # Standard dict interface
                    result = self.cache.get(key) if key in self.cache else None

                if result is not None:
                    self.hits += 1
                    # Calculate cost saved
                    cost = self._calculate_cost(
                        data.get('query', ''),
                        data.get('response', ''),
                        data.get('model', 'gpt-3.5-turbo')
                    )
                    self.total_cost_saved += cost
                else:
                    self.misses += 1

            else:  # write
                # Simulate cache put
                if hasattr(self.cache, 'put'):
                    self.cache.put(
                        key, data['response'],
                        query=data['query'],
                        model=data['model']
                    )
                else:
                    # Standard dict interface
                    if len(self.cache) >= self.maxsize and key not in self.cache:
                        self.evictions += 1
                    self.cache[key] = data['response']

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)

        # Collect final stats
        if hasattr(self.cache, 'get_stats'):
            cache_stats = self.cache.get_stats()
            self.evictions = cache_stats.get('evictions', self.evictions)

        total_accesses = self.hits + self.misses
        hit_rate = self.hits / max(1, total_accesses)
        avg_latency = sum(self.latencies) / max(1, len(self.latencies))

        return WorkloadResult(
            policy_name=self.policy_name,
            hit_rate=hit_rate,
            total_cost_saved=self.total_cost_saved,
            avg_latency_ms=avg_latency,
            evictions=self.evictions,
            final_cache_size=len(self.cache)
        )


class TestCostAwareVsLRU:
    """Test cost-aware policy against LRU baseline."""

    def setup_method(self):
        """Set up test environment."""
        self.workload_gen = WorkloadGenerator(seed=42)
        self.cache_size = 50

    @pytest.mark.skipif(not CACHETOOLS_AVAILABLE, reason="cachetools not available")
    def test_read_heavy_expensive_tail(self):
        """Test read-heavy workload with expensive tail distribution."""
        workload = self.workload_gen.generate_read_heavy_workload(
            num_operations=500, read_ratio=0.9, skew_factor=0.8
        )

        # Test cost-aware policy
        cost_aware_sim = CacheSimulator("cost_aware", maxsize=self.cache_size)
        cost_aware_result = cost_aware_sim.run_workload(workload)

        # Test LRU policy
        lru_sim = CacheSimulator("lru", maxsize=self.cache_size)
        lru_result = lru_sim.run_workload(workload)

        # Cost-aware should save more money on expensive queries
        assert cost_aware_result.total_cost_saved >= lru_result.total_cost_saved

        # Both should have reasonable hit rates
        assert cost_aware_result.hit_rate > 0.1
        assert lru_result.hit_rate > 0.1

        print(f"\nRead-heavy with expensive tail:")
        print(f"Cost-aware: hit_rate={cost_aware_result.hit_rate:.3f}, "
              f"cost_saved=${cost_aware_result.total_cost_saved:.4f}")
        print(f"LRU: hit_rate={lru_result.hit_rate:.3f}, "
              f"cost_saved=${lru_result.total_cost_saved:.4f}")

    @pytest.mark.skipif(not CACHETOOLS_AVAILABLE, reason="cachetools not available")
    def test_mixed_workload_uniform_costs(self):
        """Test mixed workload with uniform cost distribution."""
        workload = self.workload_gen.generate_mixed_workload(
            num_operations=400, read_ratio=0.6
        )

        # Test cost-aware policy
        cost_aware_sim = CacheSimulator("cost_aware", maxsize=self.cache_size)
        cost_aware_result = cost_aware_sim.run_workload(workload)

        # Test LRU policy
        lru_sim = CacheSimulator("lru", maxsize=self.cache_size)
        lru_result = lru_sim.run_workload(workload)

        # With uniform costs, performance should be comparable
        hit_rate_diff = abs(cost_aware_result.hit_rate - lru_result.hit_rate)
        assert hit_rate_diff < 0.2  # Within 20%

        print(f"\nMixed workload with uniform costs:")
        print(f"Cost-aware: hit_rate={cost_aware_result.hit_rate:.3f}")
        print(f"LRU: hit_rate={lru_result.hit_rate:.3f}")

    @pytest.mark.skipif(not CACHETOOLS_AVAILABLE, reason="cachetools not available")
    def test_write_heavy_workload(self):
        """Test write-heavy workload with high eviction pressure."""
        workload = self.workload_gen.generate_write_heavy_workload(
            num_operations=300, read_ratio=0.1
        )

        # Test cost-aware policy
        cost_aware_sim = CacheSimulator("cost_aware", maxsize=self.cache_size)
        cost_aware_result = cost_aware_sim.run_workload(workload)

        # Test LRU policy
        lru_sim = CacheSimulator("lru", maxsize=self.cache_size)
        lru_result = lru_sim.run_workload(workload)

        # Both should handle write pressure
        assert cost_aware_result.evictions > 0
        assert lru_result.evictions > 0

        # Cache sizes should be at limit
        assert cost_aware_result.final_cache_size <= self.cache_size
        assert lru_result.final_cache_size <= self.cache_size

        print(f"\nWrite-heavy workload:")
        print(f"Cost-aware: evictions={cost_aware_result.evictions}, "
              f"final_size={cost_aware_result.final_cache_size}")
        print(f"LRU: evictions={lru_result.evictions}, "
              f"final_size={lru_result.final_cache_size}")

    def test_cost_aware_configuration_impact(self):
        """Test different cost-aware configurations."""
        workload = self.workload_gen.generate_read_heavy_workload(
            num_operations=300, read_ratio=0.8, skew_factor=0.9
        )

        # Test different cost weights
        configs = [
            {"cost_weight": 0.5, "name": "low_cost_weight"},
            {"cost_weight": 1.0, "name": "medium_cost_weight"},
            {"cost_weight": 2.0, "name": "high_cost_weight"}
        ]

        results = []
        for config in configs:
            config_copy = config.copy()
            name = config_copy.pop("name")

            sim = CacheSimulator("cost_aware", maxsize=self.cache_size, **config_copy)
            result = sim.run_workload(workload)
            result.policy_name = name
            results.append(result)

        # Higher cost weight should generally save more money
        cost_savings = [r.total_cost_saved for r in results]
        print(f"\nCost weight impact:")
        for i, result in enumerate(results):
            print(f"{result.policy_name}: cost_saved=${result.total_cost_saved:.4f}")

        # Generally expect increasing trend (though not guaranteed due to access patterns)
        assert max(cost_savings) > min(cost_savings)

    def test_eviction_strategy_comparison(self):
        """Test heuristic vs greedy eviction strategies."""
        workload = self.workload_gen.generate_read_heavy_workload(
            num_operations=200, read_ratio=0.9, skew_factor=0.7
        )

        # Test heuristic strategy
        heuristic_sim = CacheSimulator("cost_aware", maxsize=20, strategy="heuristic")
        heuristic_result = heuristic_sim.run_workload(workload)

        # Test greedy strategy
        greedy_sim = CacheSimulator("cost_aware", maxsize=20, strategy="greedy")
        greedy_result = greedy_sim.run_workload(workload)

        # Both should work reasonably well
        assert heuristic_result.hit_rate > 0.0
        assert greedy_result.hit_rate > 0.0

        print(f"\nEviction strategy comparison:")
        print(f"Heuristic: hit_rate={heuristic_result.hit_rate:.3f}, "
              f"cost_saved=${heuristic_result.total_cost_saved:.4f}")
        print(f"Greedy: hit_rate={greedy_result.hit_rate:.3f}, "
              f"cost_saved=${greedy_result.total_cost_saved:.4f}")

    def test_small_cache_behavior(self):
        """Test behavior with very small cache sizes."""
        workload = self.workload_gen.generate_mixed_workload(
            num_operations=100, read_ratio=0.7
        )

        # Test with tiny cache
        small_cache_sim = CacheSimulator("cost_aware", maxsize=5)
        result = small_cache_sim.run_workload(workload)

        # Should handle small cache gracefully
        assert result.final_cache_size <= 5
        assert result.evictions > 0  # Should definitely evict with tiny cache

        print(f"\nSmall cache (size=5):")
        print(f"Final size: {result.final_cache_size}, evictions: {result.evictions}")

    def test_deterministic_workload(self):
        """Test with deterministic workload for reproducible results."""
        # Create deterministic workload
        operations = []

        # Add expensive items
        for i in range(10):
            operations.append({
                'type': 'write',
                'data': {
                    'key': f"expensive_{i}",
                    'query': "very long expensive query " * 20,
                    'response': "detailed expensive response " * 30,
                    'model': 'gpt-4'
                }
            })

        # Add cheap items
        for i in range(10):
            operations.append({
                'type': 'write',
                'data': {
                    'key': f"cheap_{i}",
                    'query': "short query",
                    'response': "brief response",
                    'model': 'gpt-3.5-turbo'
                }
            })

        # Access expensive items more frequently
        for i in range(5):
            for j in range(5):  # 5 accesses each
                operations.append({
                    'type': 'read',
                    'data': {'key': f"expensive_{i}"}
                })

        # Access cheap items less frequently
        for i in range(5, 10):
            operations.append({
                'type': 'read',
                'data': {'key': f"cheap_{i}"}
            })

        # Test with small cache to force evictions
        cache_sim = CacheSimulator("cost_aware", maxsize=15, strategy="greedy")
        result = cache_sim.run_workload(operations)

        # Should preserve expensive frequently-accessed items
        assert result.total_cost_saved > 0
        print(f"\nDeterministic workload:")
        print(f"Hit rate: {result.hit_rate:.3f}, cost saved: ${result.total_cost_saved:.4f}")


class TestCostAwareMultiPolicy:
    """Test cost-aware policy against multiple baselines."""

    @pytest.mark.skipif(not CACHETOOLS_AVAILABLE, reason="cachetools not available")
    def test_policy_comparison_comprehensive(self):
        """Compare cost-aware against multiple policies comprehensively."""
        workload = WorkloadGenerator(seed=123).generate_read_heavy_workload(
            num_operations=400, read_ratio=0.85, skew_factor=0.8
        )

        policies = ["cost_aware", "lru", "lfu"]
        results = []

        for policy in policies:
            try:
                sim = CacheSimulator(policy, maxsize=40)
                result = sim.run_workload(workload)
                results.append(result)
            except Exception as e:
                print(f"Policy {policy} failed: {e}")

        # Print comparison
        print(f"\nPolicy comparison (cache_size=40):")
        print(f"{'Policy':<12} {'Hit Rate':<10} {'Cost Saved':<12} {'Evictions':<10}")
        print("-" * 50)
        for result in results:
            print(f"{result.policy_name:<12} {result.hit_rate:<10.3f} "
                  f"${result.total_cost_saved:<11.4f} {result.evictions:<10}")

        # Cost-aware should be competitive
        cost_aware_result = next(r for r in results if r.policy_name == "cost_aware")
        assert cost_aware_result.hit_rate > 0.1
        assert cost_aware_result.total_cost_saved > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])