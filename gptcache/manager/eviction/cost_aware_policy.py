# # File: gptcache/manager/eviction/cost_aware.py
# """
# Cost-Aware Cache Eviction Policy for GPTCache
#
# This module implements a cost-aware eviction policy that considers the inference
# cost of responses when making eviction decisions.
# """
#
# import heapq
# import threading
# import time
# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, Optional
# import logging
#
# try:
#     from cachetools import Cache
#
#     CACHETOOLS_AVAILABLE = True
# except ImportError:
#     CACHETOOLS_AVAILABLE = False
#
#
#     # Fallback minimal cache interface
#     class Cache:
#         def __init__(self, maxsize):
#             self.maxsize = maxsize
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class CacheItem:
#     """Represents an item in the cache with cost and access statistics."""
#     key: str
#     value: Any
#     cost: float
#     hits: int = 0
#     last_access: float = 0.0
#     creation_time: float = 0.0
#     access_count: int = 0
#     ewma_hit_rate: float = 0.0
#     size: int = 1
#
#
# class CostFunction:
#     """Default cost function implementation with token estimation."""
#
#     DEFAULT_MODEL_PRICES = {
#         "gpt-4": 0.03,
#         "gpt-4o": 0.02,
#         "gpt-3.5-turbo": 0.002,
#         "gpt-3.5": 0.002,
#         "claude-3": 0.015,
#         "claude-2": 0.008,
#         "text-davinci-003": 0.02,
#         "text-curie-001": 0.002,
#     }
#
#     def __init__(self, model_prices: Optional[Dict[str, float]] = None):
#         """
#         Initialize cost function with model pricing.
#
#         Args:
#             model_prices: Dict mapping model names to price per 1k tokens
#         """
#         self.model_prices = model_prices or self.DEFAULT_MODEL_PRICES.copy()
#
#     def estimate_tokens(self, text: str) -> int:
#         """Simple whitespace-based token estimation."""
#         if not text:
#             return 0
#         # Rough approximation: ~4 characters per token for English
#         return max(1, len(text.split()) + len(text) // 4)
#
#     def __call__(self, query: str, response: Any, metadata: Dict[str, Any]) -> float:
#         """
#         Calculate cost for a query-response pair.
#
#         Args:
#             query: Input query text
#             response: Response object/text
#             metadata: Additional metadata (should contain 'model' key)
#
#         Returns:
#             Estimated cost in dollars
#         """
#         model = metadata.get('model', 'gpt-3.5-turbo')
#         price_per_1k = self.model_prices.get(model, 0.002)
#
#         input_tokens = self.estimate_tokens(str(query))
#         output_tokens = self.estimate_tokens(str(response))
#
#         total_cost = (input_tokens + output_tokens) * price_per_1k / 1000
#         return max(0.0001, total_cost)  # Minimum cost to avoid division by zero
#
#
# class BenefitCalculator:
#     """Calculates eviction benefit based on cost and hit probability."""
#
#     def __init__(self,
#                  cost_weight: float = 1.0,
#                  frequency_weight: float = 0.5,
#                  recency_weight: float = 0.3,
#                  ewma_alpha: float = 0.1):
#         """
#         Initialize benefit calculator.
#
#         Args:
#             cost_weight: Weight for cost component
#             frequency_weight: Weight for frequency component
#             recency_weight: Weight for recency component
#             ewma_alpha: Exponential weighted moving average decay factor
#         """
#         self.cost_weight = cost_weight
#         self.frequency_weight = frequency_weight
#         self.recency_weight = recency_weight
#         self.ewma_alpha = ewma_alpha
#
#     def calculate_hit_probability(self, item: CacheItem, current_time: float) -> float:
#         """Estimate probability of cache hit for an item."""
#         if item.access_count == 0:
#             return 0.0
#
#         # Frequency component
#         frequency_score = item.hits / max(1, item.access_count)
#
#         # Recency component (decays over time)
#         time_since_access = current_time - item.last_access
#         recency_score = max(0, 1.0 - time_since_access / 3600)  # 1 hour decay
#
#         # Combine with EWMA hit rate
#         hit_prob = (
#                 self.frequency_weight * frequency_score +
#                 self.recency_weight * recency_score +
#                 (1 - self.frequency_weight - self.recency_weight) * item.ewma_hit_rate
#         )
#
#         return min(1.0, max(0.0, hit_prob))
#
#     def calculate_benefit(self, item: CacheItem, current_time: float) -> float:
#         """
#         Calculate eviction benefit (higher = keep longer).
#
#         benefit = hit_probability * cost_weight * cost
#         """
#         hit_prob = self.calculate_hit_probability(item, current_time)
#         benefit = hit_prob * self.cost_weight * item.cost
#         return benefit
#
#
# class HeapEntry:
#     """Entry in the eviction priority heap."""
#
#     def __init__(self, priority: float, item: CacheItem, entry_id: int):
#         self.priority = priority
#         self.item = item
#         self.entry_id = entry_id
#         self.removed = False
#
#     def __lt__(self, other):
#         return self.priority < other.priority
#
#
# class CostAwareCache:
#     """
#     Cost-aware cache implementation that integrates with GPTCache's architecture.
#
#     This cache prioritizes keeping expensive responses and evicts based on
#     estimated cost-saving benefit.
#     """
#
#     def __init__(self,
#                  maxsize: int = 1000,
#                  cost_function: Optional[Callable] = None,
#                  strategy: str = "heuristic",
#                  cost_weight: float = 1.0,
#                  frequency_weight: float = 0.5,
#                  recency_weight: float = 0.3,
#                  ewma_alpha: float = 0.1,
#                  on_evict: Optional[Callable[[str, CacheItem], None]] = None):
#         """
#         Initialize cost-aware cache.
#
#         Args:
#             maxsize: Maximum number of items in cache
#             cost_function: Function to calculate item costs
#             strategy: "heuristic" or "greedy" eviction strategy
#             cost_weight: Weight for cost in benefit calculation
#             frequency_weight: Weight for access frequency
#             recency_weight: Weight for access recency
#             ewma_alpha: EWMA decay factor for hit rate tracking
#             on_evict: Optional callback called as on_evict(key, CacheItem) when an item is removed
#         """
#         self.maxsize = maxsize
#         self.cost_function = cost_function or CostFunction()
#         self.strategy = strategy
#
#         # Core data structures
#         self.data: Dict[str, CacheItem] = {}
#         self.heap: List[HeapEntry] = []
#         self.heap_map: Dict[str, HeapEntry] = {}
#         self.entry_counter = 0
#
#         self.benefit_calc = BenefitCalculator(
#             cost_weight, frequency_weight, recency_weight, ewma_alpha
#         )
#
#         self._lock = threading.RLock()
#
#         # Metrics tracking
#         self.hits = 0
#         self.misses = 0
#         self.total_cost_saved = 0.0
#         self.evictions = 0
#
#         # optional eviction callback
#         self.on_evict = on_evict
#
#     def _current_time(self) -> float:
#         """Get current timestamp."""
#         return time.time()
#
#     def _update_heap_entry(self, key: str, item: CacheItem):
#         """Update heap entry for an item with new priority."""
#         current_time = self._current_time()
#         new_priority = -self.benefit_calc.calculate_benefit(item, current_time)
#
#         # Mark old entry as removed
#         if key in self.heap_map:
#             self.heap_map[key].removed = True
#
#         # Add new entry
#         entry = HeapEntry(new_priority, item, self.entry_counter)
#         self.entry_counter += 1
#         heapq.heappush(self.heap, entry)
#         self.heap_map[key] = entry
#
#     def _clean_heap(self):
#         """Remove marked entries from heap top."""
#         while self.heap and self.heap[0].removed:
#             heapq.heappop(self.heap)
#
#     def __contains__(self, key: str) -> bool:
#         """Check if key is in cache."""
#         with self._lock:
#             return key in self.data
#
#     def __getitem__(self, key: str) -> Any:
#         """Get item from cache (dict-like interface)."""
#         with self._lock:
#             if key not in self.data:
#                 self.misses += 1
#                 raise KeyError(key)
#
#             item = self.data[key]
#             current_time = self._current_time()
#
#             # Update access statistics
#             item.hits += 1
#             item.access_count += 1
#             item.last_access = current_time
#
#             # Update EWMA hit rate
#             item.ewma_hit_rate = (
#                     self.benefit_calc.ewma_alpha +
#                     (1 - self.benefit_calc.ewma_alpha) * item.ewma_hit_rate
#             )
#
#             # Update heap priority
#             self._update_heap_entry(key, item)
#
#             self.hits += 1
#             self.total_cost_saved += item.cost
#
#             return item.value
#
#     def __setitem__(self, key: str, value: Any):
#         """Set item in cache (dict-like interface)."""
#         # Default metadata for dict interface
#         metadata = {'model': 'gpt-3.5-turbo', 'query': str(key)}
#         self.put(key, value, **metadata)
#
#     def __delitem__(self, key: str):
#         """Delete item from cache."""
#         with self._lock:
#             if key not in self.data:
#                 raise KeyError(key)
#             self._remove_item(key)
#
#     def __len__(self) -> int:
#         """Get cache size."""
#         return len(self.data)
#
#     def get(self, key: str, default: Any = None) -> Any:
#         """Get item with default value."""
#         try:
#             return self[key]
#         except KeyError:
#             return default
#
#     def put(self, key: str, value: Any, query: str = "", model: str = "gpt-3.5-turbo", **kwargs):
#         """
#         Put item in cache with cost calculation.
#
#         Args:
#             key: Cache key
#             value: Value to cache
#             query: Original query for cost calculation
#             model: Model name for cost calculation
#             **kwargs: Additional metadata
#         """
#         with self._lock:
#             metadata = {'model': model, 'query': query, **kwargs}
#             cost = self.cost_function(query, value, metadata)
#             current_time = self._current_time()
#
#             # If key exists, update it
#             if key in self.data:
#                 item = self.data[key]
#                 item.value = value
#                 item.cost = cost
#                 item.access_count += 1
#                 item.last_access = current_time
#                 self._update_heap_entry(key, item)
#                 return
#
#             # Create new item
#             item = CacheItem(
#                 key=key,
#                 value=value,
#                 cost=cost,
#                 last_access=current_time,
#                 creation_time=current_time,
#                 access_count=1,
#                 size=self._estimate_size(value)
#             )
#
#             # Check if eviction needed
#             while len(self.data) >= self.maxsize:
#                 evicted = self._evict()
#                 if not evicted:
#                     break  # Avoid infinite loop
#
#             # Add new item
#             self.data[key] = item
#             self._update_heap_entry(key, item)
#
#     def _estimate_size(self, value: Any) -> int:
#         """Estimate memory size of a value."""
#         try:
#             import sys
#             return sys.getsizeof(value)
#         except:
#             return 1  # Fallback to unit size
#
#     def _evict(self) -> bool:
#         """Evict item(s) to make space. Returns True if eviction occurred."""
#         if self.strategy == "greedy":
#             return self._evict_greedy()
#         else:
#             return self._evict_heuristic()
#
#     def _evict_heuristic(self) -> bool:
#         """Fast heuristic eviction using heap."""
#         self._clean_heap()
#
#         if not self.heap:
#             # Fallback: evict oldest item
#             if self.data:
#                 oldest_key = min(self.data.keys(),
#                                  key=lambda k: self.data[k].creation_time)
#                 self._remove_item(oldest_key)
#                 return True
#             return False
#
#         # Evict item with lowest benefit
#         entry = heapq.heappop(self.heap)
#         if not entry.removed and entry.item.key in self.data:
#             self._remove_item(entry.item.key)
#             return True
#         return False
#
#     def _evict_greedy(self) -> bool:
#         """Exact greedy eviction (for testing/small caches)."""
#         if not self.data:
#             return False
#
#         current_time = self._current_time()
#         min_benefit = float('inf')
#         evict_key = None
#
#         for key, item in self.data.items():
#             benefit = self.benefit_calc.calculate_benefit(item, current_time)
#             if benefit < min_benefit:
#                 min_benefit = benefit
#                 evict_key = key
#
#         if evict_key:
#             self._remove_item(evict_key)
#             return True
#         return False
#
#     def _remove_item(self, key: str):
#         """Remove item from cache and heap and call on_evict if provided."""
#         item = None
#         if key in self.data:
#             item = self.data[key]
#             del self.data[key]
#             self.evictions += 1
#
#         if key in self.heap_map:
#             self.heap_map[key].removed = True
#             del self.heap_map[key]
#
#         # call optional callback (use try/except so eviction can't break runtime)
#         if self.on_evict is not None and item is not None:
#             try:
#                 # Call without lock to avoid deadlocks if callback interacts with other systems
#                 self.on_evict(key, item)
#             except Exception:
#                 logger.exception("on_evict callback raised for key=%s", key)
#
#     def clear(self):
#         """Clear entire cache."""
#         with self._lock:
#             self.data.clear()
#             self.heap.clear()
#             self.heap_map.clear()
#             self.entry_counter = 0
#
#     def keys(self):
#         """Get cache keys."""
#         return self.data.keys()
#
#     def values(self):
#         """Get cache values."""
#         return [item.value for item in self.data.values()]
#
#     def items(self):
#         """Get cache items."""
#         return [(key, item.value) for key, item in self.data.items()]
#
#     def get_stats(self) -> Dict[str, Any]:
#         """Get cache performance statistics."""
#         with self._lock:
#             total_accesses = self.hits + self.misses
#             hit_rate = self.hits / max(1, total_accesses)
#
#             return {
#                 'hits': self.hits,
#                 'misses': self.misses,
#                 'hit_rate': hit_rate,
#                 'total_cost_saved': self.total_cost_saved,
#                 'evictions': self.evictions,
#                 'size': len(self.data),
#                 'maxsize': self.maxsize,
#                 'total_accesses': total_accesses,
#                 'avg_cost_per_item': sum(item.cost for item in self.data.values()) / max(1, len(self.data))
#             }
#
#     def reset_stats(self):
#         """Reset performance statistics."""
#         with self._lock:
#             self.hits = 0
#             self.misses = 0
#             self.total_cost_saved = 0.0
#             self.evictions = 0
#
#
# def cost_aware_cache_factory(maxsize: int = 1000, **kwargs) -> CostAwareCache:
#     """Factory function for creating cost-aware cache instances."""
#     return CostAwareCache(maxsize=maxsize, **kwargs)