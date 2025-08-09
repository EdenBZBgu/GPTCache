# File: tests/unit/test_cost_aware_policy.py
"""
Unit tests for cost-aware cache policy.
"""

import pytest
import time
from unittest.mock import Mock, patch

from gptcache.manager.eviction.cost_aware_policy import (
    CostAwareCache, CostFunction, BenefitCalculator, CacheItem
)


class TestCostFunction:
    """Test cost function implementation."""

    def test_default_pricing(self):
        """Test default model pricing."""
        cost_fn = CostFunction()

        # Test with default model
        cost = cost_fn("hello world", "response", {"model": "gpt-3.5-turbo"})
        assert cost > 0

        # Test with expensive model
        cost_expensive = cost_fn("hello world", "response", {"model": "gpt-4"})
        assert cost_expensive > cost

    def test_custom_pricing(self):
        """Test custom model pricing."""
        custom_prices = {"custom-model": 0.1}
        cost_fn = CostFunction(model_prices=custom_prices)

        cost = cost_fn("test", "response", {"model": "custom-model"})
        assert cost > 0

    def test_token_estimation(self):
        """Test token estimation logic."""
        cost_fn = CostFunction()

        short_tokens = cost_fn.estimate_tokens("hello")
        long_tokens = cost_fn.estimate_tokens("hello world this is a longer sentence")

        assert long_tokens > short_tokens
        assert cost_fn.estimate_tokens("") == 0

    def test_minimum_cost(self):
        """Test minimum cost enforcement."""
        cost_fn = CostFunction({"zero-model": 0.0})
        cost = cost_fn("", "", {"model": "zero-model"})
        assert cost >= 0.0001  # Minimum cost


class TestBenefitCalculator:
    """Test benefit calculation logic."""

    def test_hit_probability_calculation(self):
        """Test hit probability calculation."""
        calc = BenefitCalculator()
        current_time = time.time()

        # New item with no hits
        item = CacheItem("key", "value", 1.0, hits=0, access_count=1)
        prob = calc.calculate_hit_probability(item, current_time)
        assert prob == 0.0

        # Item with high hit rate
        item.hits = 8
        item.access_count = 10
        item.last_access = current_time
        prob = calc.calculate_hit_probability(item, current_time)
        assert prob > 0.4  # Should be reasonably high

    def test_benefit_calculation(self):
        """Test overall benefit calculation."""
        calc = BenefitCalculator(cost_weight=1.0)
        current_time = time.time()

        # High cost, high hit rate item
        expensive_item = CacheItem(
            "exp", "value", 10.0, hits=9, access_count=10,
            last_access=current_time
        )

        # Low cost, high hit rate item
        cheap_item = CacheItem(
            "cheap", "value", 0.1, hits=9, access_count=10,
            last_access=current_time
        )

        expensive_benefit = calc.calculate_benefit(expensive_item, current_time)
        cheap_benefit = calc.calculate_benefit(cheap_item, current_time)

        assert expensive_benefit > cheap_benefit


class TestCostAwareCache:
    """Test cost-aware cache implementation."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = CostAwareCache(maxsize=3)

        # Test miss
        assert cache.get("missing") is None
        assert "missing" not in cache

        # Test put and get
        cache.put("key1", "value1", query="test query", model="gpt-3.5-turbo")
        assert cache.get("key1") == "value1"
        assert "key1" in cache
        assert len(cache) == 1

    def test_dict_interface(self):
        """Test dictionary-like interface."""
        cache = CostAwareCache(maxsize=5)

        # Test setitem/getitem
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"

        # Test KeyError on missing key
        with pytest.raises(KeyError):
            _ = cache["missing"]

        # Test deletion
        del cache["key1"]
        assert "key1" not in cache

    def test_eviction_by_cost(self):
        """Test that expensive items are kept longer."""
        cache = CostAwareCache(maxsize=2, strategy="greedy")  # Use greedy for deterministic results

        # Add cheap item
        cache.put("cheap", "val1", query="hi", model="gpt-3.5-turbo")

        # Add expensive item
        cache.put("expensive", "val2",
                  query="very long query " * 50, model="gpt-4")

        # Cache should be full
        assert len(cache) == 2

        # Add third item to trigger eviction
        cache.put("medium", "val3", query="medium length query here", model="gpt-3.5-turbo")

        # Should still have 2 items
        assert len(cache) == 2

        # Expensive item should still be there
        assert cache.get("expensive") == "val2"

        # Cheap item likely evicted (though not guaranteed due to access patterns)
        stats = cache.get_stats()
        assert stats['evictions'] > 0

    def test_hit_tracking_updates(self):
        """Test that hit tracking works correctly."""
        cache = CostAwareCache(maxsize=5)

        cache.put("key1", "value1", query="test", model="gpt-3.5-turbo")

        # Multiple accesses should update hit stats
        for _ in range(5):
            cache.get("key1")

        item = cache.data["key1"]
        assert item.hits == 5
        assert item.access_count == 6  # 1 put + 5 gets
        assert item.ewma_hit_rate > 0

    def test_stats_collection(self):
        """Test statistics collection."""
        cache = CostAwareCache(maxsize=3)

        # Initial stats
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['size'] == 0

        # Add item and access
        cache.put("key1", "value1", query="test", model="gpt-3.5-turbo")
        cache.get("key1")  # hit
        cache.get("missing")  # miss

        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['total_cost_saved'] > 0

    def test_empty_cache_handling(self):
        """Test behavior with empty cache."""
        cache = CostAwareCache(maxsize=1)

        # Operations on empty cache should not crash
        assert len(cache) == 0
        assert cache.get("key") is None
        assert list(cache.keys()) == []
        assert list(cache.values()) == []
        assert list(cache.items()) == []

        cache.clear()
        assert len(cache) == 0

    def test_single_item_cache(self):
        """Test cache with single item capacity."""
        cache = CostAwareCache(maxsize=1)

        cache.put("key1", "val1", query="test1", model="gpt-3.5-turbo")
        assert len(cache) == 1
        assert cache.get("key1") == "val1"

        cache.put("key2", "val2", query="test2", model="gpt-3.5-turbo")
        assert len(cache) == 1

        # First item should be evicted
        stats = cache.get_stats()
        assert stats['evictions'] >= 1

    def test_cost_calculation_integration(self):
        """Test integration with cost calculation."""
        cache = CostAwareCache(maxsize=10)

        # Add items with different costs
        cache.put("cheap", "short", query="hi", model="gpt-3.5-turbo")
        cache.put("expensive", "long response " * 100,
                  query="complex question " * 50, model="gpt-4")

        stats = cache.get_stats()
        assert stats['avg_cost_per_item'] > 0

        # Access expensive item multiple times
        for _ in range(5):
            cache.get("expensive")

        # Should have accumulated significant cost savings
        stats = cache.get_stats()
        assert stats['total_cost_saved'] > 0

    def test_concurrent_access(self):
        """Test thread safety of cache operations."""
        import threading

        cache = CostAwareCache(maxsize=100)
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(10):
                    key = f"key_{thread_id}_{i}"
                    cache.put(key, f"value_{thread_id}_{i}",
                              query=f"query {thread_id} {i}", model="gpt-3.5-turbo")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not have any thread safety errors
        assert len(errors) == 0
        assert len(cache) <= 100

    def test_reset_stats(self):
        """Test statistics reset functionality."""
        cache = CostAwareCache(maxsize=5)

        cache.put("key1", "value1", query="test", model="gpt-3.5-turbo")
        cache.get("key1")
        cache.get("missing")

        stats_before = cache.get_stats()
        assert stats_before['hits'] > 0
        assert stats_before['misses'] > 0

        cache.reset_stats()

        stats_after = cache.get_stats()
        assert stats_after['hits'] == 0
        assert stats_after['misses'] == 0
        assert stats_after['total_cost_saved'] == 0.0
        assert stats_after['evictions'] == 0

        # Cache contents should remain
        assert len(cache) == 1
        assert cache.get("key1") == "value1"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_huge_cost_item(self):
        """Test handling of items with very high cost."""
        cache = CostAwareCache(maxsize=2, strategy="greedy")

        # Create custom cost function that returns huge cost
        def huge_cost_fn(query, response, metadata):
            if "huge" in query:
                return 999999.0
            return 0.001

        cache.cost_function = huge_cost_fn

        cache.put("huge", "value", query="huge cost query", model="gpt-4")
        cache.put("normal", "value", query="normal query", model="gpt-3.5-turbo")

        # Add third item
        cache.put("another", "value", query="another normal query", model="gpt-3.5-turbo")

        # Huge cost item should be preserved
        assert cache.get("huge") == "value"

    def test_zero_maxsize(self):
        """Test cache with zero maximum size."""
        cache = CostAwareCache(maxsize=0)

        cache.put("key1", "value1", query="test", model="gpt-3.5-turbo")

        # Should not store anything
        assert len(cache) == 0
        assert cache.get("key1") is None

    def test_invalid_cost_function(self):
        """Test handling of invalid cost function."""

        def broken_cost_fn(query, response, metadata):
            raise ValueError("Broken cost function")

        cache = CostAwareCache(maxsize=5, cost_function=broken_cost_fn)

        # Should handle gracefully without crashing
        with pytest.raises(Exception):
            cache.put("key1", "value1", query="test", model="gpt-3.5-turbo")


if __name__ == "__main__":
    pytest.main([__file__])