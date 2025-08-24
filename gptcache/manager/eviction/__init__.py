__all__ = ["EvictionBase"]

from gptcache.utils.lazy_import import LazyImport

eviction_manager = LazyImport(
    "eviction_manager", globals(), "gptcache.manager.eviction.manager"
)


def EvictionBase(name: str, **kwargs):
    """Generate specific CacheStorage with the configuration.

    :param name: the name of the eviction, like: memory
    :type name: str

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]

    Example:
        .. code-block:: python

            from gptcache.manager import EvictionBase

            cache_base = EvictionBase('memory', policy='lru', maxsize=10, clean_size=2, on_evict=lambda x: print(x))
    """
    return eviction_manager.EvictionBase.get(name, **kwargs)


"""
GPTCache eviction policy implementations.
"""

from typing import Any, Dict, Callable
from .cost_aware import CostAwareCacheAdapter, make_cost_aware_cache

# Import existing eviction policies
try:
    from cachetools import LRUCache, LFUCache, TTLCache

    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    LRUCache = None
    LFUCache = None
    TTLCache = None


# Factory functions with consistent signatures
def lru_cache_factory(maxsize: int = 1000, **kwargs) -> Any:
    """Factory function for LRU cache."""
    if not CACHETOOLS_AVAILABLE:
        raise ImportError("cachetools not available for LRU cache")
    return LRUCache(maxsize=maxsize)


def lfu_cache_factory(maxsize: int = 1000, **kwargs) -> Any:
    """Factory function for LFU cache."""
    if not CACHETOOLS_AVAILABLE:
        raise ImportError("cachetools not available for LFU cache")
    return LFUCache(maxsize=maxsize)


def ttl_cache_factory(maxsize: int = 1000, ttl: int = 600, **kwargs) -> Any:
    """Factory function for TTL cache."""
    if not CACHETOOLS_AVAILABLE:
        raise ImportError("cachetools not available for TTL cache")
    return TTLCache(maxsize=maxsize, ttl=ttl)


# Eviction policy registry with consistent factory function types
EVICTION_POLICIES: Dict[str, Callable[[int], Any]] = {
    'cost_aware': make_cost_aware_cache,
}

if CACHETOOLS_AVAILABLE:
    EVICTION_POLICIES.update({
        'lru': lru_cache_factory,
        'lfu': lfu_cache_factory,
        'ttl': ttl_cache_factory,
    })


def get_eviction_policy(policy_name: str, maxsize: int = 1000, **kwargs) -> Any:
    """
    Get eviction policy by name.

    Args:
        policy_name: Name of the eviction policy
        maxsize: Maximum cache size
        **kwargs: Additional policy parameters

    Returns:
        Cache instance with specified eviction policy
    """
    if policy_name not in EVICTION_POLICIES:
        available = list(EVICTION_POLICIES.keys())
        raise ValueError(f"Unknown eviction policy: {policy_name}. Available: {available}")

    policy_factory = EVICTION_POLICIES[policy_name]
    return policy_factory(maxsize=maxsize, **kwargs)


__all__ = ['CostAwareCache', 'make_cost_aware_cache', 'get_eviction_policy', 'EVICTION_POLICIES']
