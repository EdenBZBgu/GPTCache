# File: gptcache/manager/eviction/cost_aware.py
"""
Cost-aware cache adapter for GPTCache memory eviction.

This module implements:
 - CostFunction: estimate request/response cost
 - BenefitCalculator: compute benefit (keep score) per item
 - CostAwareCacheAdapter: a mapping-like cache that exposes a popitem()
   method (returns (key,value)) so it can be used by MemoryCacheEviction.

MemoryCacheEviction expects objects that behave like cachetools caches:
 - __setitem__(key, value)
 - __getitem__(key)
 - __contains__(key)
 - get(key)
 - popitem() -> (key, value)  # evict one item
 - pop(key, default=None)
 - __len__()
"""
from dataclasses import dataclass
import heapq
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheItem:
    key: Any
    value: Any
    cost: float = 1.0
    hits: int = 0
    last_access: float = 0.0
    creation_time: float = 0.0
    access_count: int = 0
    ewma_hit_rate: float = 0.0
    size: int = 1


class CostFunction:
    """Default cost function implementation with model pricing."""

    DEFAULT_MODEL_PRICES = {
        "gpt-4": 0.03,
        "gpt-4o": 0.02,
        "gpt-3.5-turbo": 0.002,
        "gpt-3.5": 0.002,
        "claude-3": 0.015,
        "claude-2": 0.008,
        "text-davinci-003": 0.02,
        "text-curie-001": 0.002,
    }

    def __init__(self, model_prices: Optional[Dict[str, float]] = None):
        self.model_prices = (model_prices or self.DEFAULT_MODEL_PRICES.copy())

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        # Keep simple: whitespace-based + length-based
        words = len(text.split())
        approx_from_chars = max(0, len(text) // 4)
        return max(1, words + approx_from_chars)

    def __call__(self, query: str, response: Any, metadata: Dict[str, Any]) -> float:
        model = metadata.get("model", "gpt-3.5-turbo")
        price_per_1k = self.model_prices.get(model, 0.002)
        input_tokens = self.estimate_tokens(str(query))
        output_tokens = self.estimate_tokens(str(response))
        total_cost = (input_tokens + output_tokens) * price_per_1k / 1000.0
        return max(0.0001, total_cost)


class BenefitCalculator:
    """Calculates a benefit (keep-score) for a cache item."""

    def __init__(
        self,
        cost_weight: float = 1.0,
        frequency_weight: float = 0.5,
        recency_weight: float = 0.3,
        ewma_alpha: float = 0.1,
    ):
        self.cost_weight = cost_weight
        self.frequency_weight = frequency_weight
        self.recency_weight = recency_weight
        self.ewma_alpha = ewma_alpha

    def calculate_hit_probability(self, item: CacheItem, current_time: float) -> float:
        if item.access_count == 0:
            return 0.0
        frequency_score = item.hits / max(1, item.access_count)
        time_since_access = max(0.0, current_time - item.last_access)
        # 1 hour decay baseline — tune as needed
        recency_score = max(0.0, 1.0 - time_since_access / 3600.0)
        # Combine: frequency + recency + ewma (remaining weight)
        remaining = max(0.0, 1.0 - self.frequency_weight - self.recency_weight)
        hit_prob = (
            self.frequency_weight * frequency_score
            + self.recency_weight * recency_score
            + remaining * item.ewma_hit_rate
        )
        return min(1.0, max(0.0, hit_prob))

    def calculate_benefit(self, item: CacheItem, current_time: float) -> float:
        hit_prob = self.calculate_hit_probability(item, current_time)
        # higher benefit means "worth keeping"
        benefit = hit_prob * self.cost_weight * (item.cost or 1.0)
        return benefit


class HeapEntry:
    def __init__(self, priority: float, key: Any, entry_id: int):
        self.priority = priority
        self.key = key
        self.entry_id = entry_id
        self.removed = False

    def __lt__(self, other: "HeapEntry"):
        # Python heapq is min-heap. We want smallest priority popped first for eviction.
        return self.priority < other.priority


class CostAwareCacheAdapter:
    """
    Adapter that exposes the minimal mapping API MemoryCacheEviction expects:
      - __setitem__(key, value)
      - __getitem__(key)
      - __contains__(key)
      - get(key)
      - popitem() -> (key, value)  # evict one item
      - pop(key, default=None)
      - __len__()
    """

    def __init__(
        self,
        maxsize: int = 1000,
        on_evict: Optional[Callable[[List[Any]], None]] = None,
        cost_function: Optional[Callable] = None,
        fetch_metadata: Optional[Callable[[Any], Dict[str, Any]]] = None,
        strategy: str = "heuristic",
        cost_weight: float = 1.0,
        frequency_weight: float = 0.5,
        recency_weight: float = 0.3,
        ewma_alpha: float = 0.1,
    ):
        self.maxsize = int(maxsize)
        self._data: Dict[Any, CacheItem] = {}
        self.on_evict = on_evict
        self._lock = threading.RLock()

        self.cost_function = cost_function or CostFunction()
        self.fetch_metadata = fetch_metadata  # callable(key) -> dict with 'query','response','model'

        self.strategy = strategy
        self.benefit_calc = BenefitCalculator(
            cost_weight, frequency_weight, recency_weight, ewma_alpha
        )

        self.heap: List[HeapEntry] = []
        self.heap_map: Dict[Any, HeapEntry] = {}
        self._counter = 0

        # metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ---------- internal helpers ----------
    def _now(self) -> float:
        return time.time()

    def _estimate_size(self, value: Any) -> int:
        try:
            import sys

            return sys.getsizeof(value)
        except Exception:
            return 1

    def _compute_cost_for_key(self, key: Any, value: Any) -> float:
        # Try to fetch metadata if provided
        metadata = {}
        if self.fetch_metadata:
            try:
                metadata = self.fetch_metadata(key) or {}
            except Exception:
                logger.exception("fetch_metadata failed for key=%s", key)
                metadata = {}

        # allow metadata to include model/query/response
        query = metadata.get("query", str(key))
        response = metadata.get("response", value)
        try:
            return float(self.cost_function(query, response, metadata))
        except Exception:
            logger.exception("cost_function failed for key=%s", key)
            return 1.0

    def _update_heap_entry(self, key: Any):
        """Recompute priority and push a new HeapEntry; mark old one removed."""
        now = self._now()
        item = self._data.get(key)
        if item is None:
            return
        # priority we want to evict first = lowest benefit
        priority = self.benefit_calc.calculate_benefit(item, now)
        # push priority as-is (min-heap) -> lowest benefit is popped first
        if key in self.heap_map:
            self.heap_map[key].removed = True
        entry = HeapEntry(priority, key, self._counter)
        self._counter += 1
        heapq.heappush(self.heap, entry)
        self.heap_map[key] = entry

    def _clean_heap(self):
        while self.heap and self.heap[0].removed:
            heapq.heappop(self.heap)

    def _evict_one(self) -> Any:
        """Evict a single item using chosen strategy and return the evicted key."""
        with self._lock:
            if not self._data:
                raise KeyError("cache is empty")

            if self.strategy == "greedy":
                # compute benefit for all items and evict min
                now = self._now()
                min_benefit = float("inf")
                evict_key = None
                for k, it in self._data.items():
                    b = self.benefit_calc.calculate_benefit(it, now)
                    if b < min_benefit:
                        min_benefit = b
                        evict_key = k
                if evict_key is None:
                    raise KeyError("no candidate")
                value = self._remove_key(evict_key)
                return evict_key, value
            else:
                # heuristic via heap
                self._clean_heap()
                if not self.heap:
                    # fallback to oldest
                    oldest = min(self._data.items(), key=lambda kv: kv[1].creation_time)[0]
                    value = self._remove_key(oldest)
                    return oldest, value

                entry = heapq.heappop(self.heap)
                # Ensure entry not already removed
                if entry.removed:
                    return self._evict_one()
                k = entry.key
                if k not in self._data:
                    return self._evict_one()
                value = self._remove_key(k)
                return k, value

    def _remove_key(self, key: Any):
        """Remove and return value."""
        item = self._data.pop(key, None)
        if key in self.heap_map:
            self.heap_map[key].removed = True
            del self.heap_map[key]
        if item:
            self.evictions += 1
            return item.value
        return None

    # ---------- public mapping API ----------
    def __setitem__(self, key: Any, value: Any):
        """Insert or update an entry. No return value (like cachetools behavior)."""
        with self._lock:
            now = self._now()
            if key in self._data:
                item = self._data[key]
                item.value = value
                item.last_access = now
                item.access_count += 1
                # recalc cost if we can
                try:
                    item.cost = self._compute_cost_for_key(key, value)
                except Exception:
                    pass
                self._update_heap_entry(key)
                return

            cost = self._compute_cost_for_key(key, value)
            item = CacheItem(
                key=key,
                value=value,
                cost=cost,
                last_access=now,
                creation_time=now,
                access_count=1,
                size=self._estimate_size(value),
            )
            self._data[key] = item
            self._update_heap_entry(key)

            # Evict until below maxsize if needed (evict one-by-one)
            while len(self._data) > self.maxsize:
                try:
                    evicted_key, evicted_value = self._evict_one()
                    # call on_evict for each eviction immediately (MemoryCacheEviction
                    # wraps popitem and calls on_evict in batch; both ways are fine)
                    if self.on_evict:
                        try:
                            self.on_evict([evicted_key])
                        except Exception:
                            logger.exception("on_evict raised for key=%s", evicted_key)
                except KeyError:
                    break

    def __getitem__(self, key: Any):
        with self._lock:
            if key not in self._data:
                self.misses += 1
                raise KeyError(key)
            item = self._data[key]
            now = self._now()
            item.hits += 1
            item.access_count += 1
            item.last_access = now
            # update EWMA correctly: reward = 1.0 for a hit
            a = self.benefit_calc.ewma_alpha
            item.ewma_hit_rate = a * 1.0 + (1.0 - a) * item.ewma_hit_rate
            self._update_heap_entry(key)
            self.hits += 1
            return item.value

    def get(self, key: Any, default: Any = None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def popitem(self):
        """
        Pop a single item using the adapter's eviction logic.
        Returns a (key, value) tuple (matches cachetools API).
        If no item to pop, raise KeyError.
        """
        with self._lock:
            if not self._data:
                raise KeyError("popitem(): cache is empty")
            key, value = self._evict_one()
            return key, value

    def pop(self, key: Any, default: Any = None):
        with self._lock:
            if key in self._data:
                val = self._remove_key(key)
                return val
            return default


def make_cost_aware_cache(
    maxsize: int = 1000,
    on_evict: Optional[Callable[[List[Any]], None]] = None,
    fetch_metadata: Optional[Callable[[Any], Dict[str, Any]]] = None,
    cost_function: Optional[Callable[[str, Any, Dict[str, Any]], float]] = None,
    **kwargs,
) -> CostAwareCacheAdapter:
    """
    Factory that returns a CostAwareCacheAdapter instance.

    Parameters:
      - maxsize: maximum number of items (count-based)
      - on_evict: optional callback invoked with list-of-keys when eviction happens
      - fetch_metadata: optional callable(key) -> metadata dict used by cost fn
      - cost_function: optional cost function callable (query, response, metadata) -> float
      - kwargs: passed to BenefitCalculator / adapter (e.g., strategy, cost_weight)
    """
    return CostAwareCacheAdapter(
        maxsize=int(maxsize),
        on_evict=on_evict,
        fetch_metadata=fetch_metadata,
        cost_function=cost_function,
        **kwargs,
    )
