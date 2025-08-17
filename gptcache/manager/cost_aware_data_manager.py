# File: gptcache/manager/data_manager/cost_aware_data_manager.py
"""
Cost-aware Map-style DataManager for GPTCache

This DataManager adapts a cost-aware in-memory container (either the
legacy `CostAwareCache` implementation or the adapter-based
`make_cost_aware_cache` factory) to MapDataManager persistence and
vector-store eviction semantics.

Key features:
- Uses a cost-aware in-memory cache for eviction (keeps expensive items).
- Persists a plain dict snapshot on `flush()` and restores by reinserting
  into a fresh cost-aware container on `init()`.
- Optional `vector_store` deletion on eviction (robust to several
  callback shapes emitted by different cache implementations).

Usage: drop this file into your repo at
`gptcache/manager/data_manager/cost_aware_data_manager.py` and ensure the
cost-aware cache module you implemented is importable as either:
 - `gptcache.manager.eviction.cost_aware.make_cost_aware_cache` (adapter)
 - OR `gptcache.manager.eviction.cost_aware_policy.CostAwareCache` (legacy)

The manager detects which one is available and adapts automatically.
"""

from typing import Callable, Optional, Dict, Any, cast, Iterable
import pickle
import os
import logging

# Try importing MapDataManager from common locations in the project.
try:
    # Preferred: MapDataManager exported under manager.data_manager
    from gptcache.manager.data_manager import MapDataManager
except Exception:
    # Fallback import path — adjust if your repo structure differs
    try:
        from gptcache.manager import MapDataManager  # type: ignore
    except Exception:
        raise ImportError("Could not import MapDataManager from gptcache.manager.data_manager or gptcache.manager")

logger = logging.getLogger(__name__)

# --- Try to import cost-aware cache implementations ---
COST_AWARE_FACTORY = False
COST_AWARE_LEGACY = False

# We'll attempt to import the adapter factory first (preferred), then the
# legacy CostAwareCache class if present.
try:
    # Adapter variant (from the refactor): make_cost_aware_cache(...)
    from gptcache.manager.eviction.cost_aware import make_cost_aware_cache as CostAwareCacheFactory
    # Optional: adapter might expose a CacheItem dataclass; not required here
    COST_AWARE_FACTORY = True
except Exception:
    try:
        # Legacy name you previously used in your repo
        from gptcache.manager.eviction.cost_aware_policy import CostAwareCache as CostAwareCacheLegacy
        from gptcache.manager.eviction.cost_aware_policy import CostFunction, CacheItem
        COST_AWARE_LEGACY = True
    except Exception:
        # If neither import succeeded, raise a helpful ImportError
        raise ImportError(
            "No cost-aware cache implementation found. Please add either: \n"
            " - gptcache.manager.eviction.cost_aware.make_cost_aware_cache(...) \n"
            "or\n - gptcache.manager.eviction.cost_aware_policy.CostAwareCache(...)"
        )


class CostAwareDataManager(MapDataManager):
    """
    Map-style DataManager that uses a cost-aware in-memory container.

    If your project's MapDataManager constructor accepts `get_data_container`
    as a keyword, this class will pass a factory so MapDataManager will create
    the in-memory container. If not, it falls back to creating the container
    itself after calling super().__init__.
    """

    def __init__(
        self,
        data_path: str,
        max_size: int,
        *,
        cost_function: Optional[Callable] = None,
        strategy: str = "heuristic",
        cost_weight: float = 1.0,
        frequency_weight: float = 0.5,
        recency_weight: float = 0.3,
        ewma_alpha: float = 0.1,
        vector_store: Optional[Any] = None,
        get_data_container: Optional[Callable[[int], Any]] = None,
    ):
        """
        Args:
            data_path: path to persist snapshot (pickle)
            max_size: maximum number of items (passed to cost-aware container)
            cost_function: optional CostFunction or callable
            strategy: "heuristic" | "greedy"
            cost_weight/frequency_weight/recency_weight/ewma_alpha: benefit params
            vector_store: optional vector store client with `delete(id)` method
            get_data_container: optional custom container factory (for advanced use)
        """
        self.data_path = data_path
        self.max_size = int(max_size)
        self._cost_function = cost_function
        self._strategy = strategy
        self._cost_weight = cost_weight
        self._frequency_weight = frequency_weight
        self._recency_weight = recency_weight
        self._ewma_alpha = ewma_alpha
        self.vector_store = vector_store

        # If user passed a custom factory use it; otherwise create one depending
        # on which cost-aware implementation is available.
        if get_data_container is None:
            def _get_data_container(sz: int):
                # Create container depending on which implementation we have
                if COST_AWARE_FACTORY:
                    # Adapter factory exposes make_cost_aware_cache(...)
                    # It expects keyword args like maxsize, on_evict, cost_function etc.
                    return CostAwareCacheFactory(
                        maxsize=sz,
                        on_evict=self._on_evict_callback_adapter,
                        cost_function=self._cost_function,
                        strategy=self._strategy,
                        cost_weight=self._cost_weight,
                        frequency_weight=self._frequency_weight,
                        recency_weight=self._recency_weight,
                        ewma_alpha=self._ewma_alpha,
                    )
                else:
                    # Legacy constructor expecting (maxsize, cost_function=..., on_evict=...)
                    return CostAwareCacheLegacy(
                        maxsize=sz,
                        cost_function=self._cost_function,
                        strategy=self._strategy,
                        cost_weight=self._cost_weight,
                        frequency_weight=self._frequency_weight,
                        recency_weight=self._recency_weight,
                        ewma_alpha=self._ewma_alpha,
                        on_evict=self._on_evict_callback,
                    )

            get_data_container = _get_data_container

        # Attempt to pass get_data_container into parent if parent supports it.
        # Some MapDataManager implementations accept this factory and will call
        # it to create the underlying container; others do not. We handle both.
        try:
            super().__init__(data_path=self.data_path, max_size=self.max_size, get_data_container=get_data_container)
        except TypeError:
            # Parent does not accept get_data_container; call with standard args,
            # then create the container ourselves and attach it to self.data.
            super().__init__(data_path=self.data_path, max_size=self.max_size)
            try:
                self.data = get_data_container(self.max_size)
            except Exception:
                logger.exception("Failed to create cost-aware container via factory; creating empty dict instead.")
                self.data = {}

    # ---------- persistence: snapshot plain dict ----------
    def init(self):
        """
        Load persisted entries (if present) into a fresh container.

        Persists format expected by this manager is a plain mapping: key -> value
        (i.e. we do not serialize locks/heap internals). On restore we reinsert
        keys which will cause the in-memory container to recompute costs.
        """
        # Ensure the in-memory container exists (parent MapDataManager may have created it)
        if not hasattr(self, "data") or self.data is None:
            try:
                self.data = self._create_container(self.max_size)
            except Exception:
                logger.exception("Failed to create cost-aware container during init(); falling back to dict.")
                self.data = {}

        if not os.path.exists(self.data_path):
            return

        try:
            with open(self.data_path, "rb") as f:
                raw = pickle.load(f)
        except Exception as e:
            logger.exception("Failed to load persisted map from %s: %s", self.data_path, e)
            return

        if not isinstance(raw, dict):
            logger.warning("Persisted file at %s does not contain a dict; skipping restore.", self.data_path)
            return

        # Ensure fresh container (so locks/heaps are clean)
        try:
            self.data = self._create_container(self.max_size)
        except Exception:
            logger.exception("Failed to create cost-aware container for restore; aborting restore.")
            return

        # Reinsert items using the container API so cost & stats are computed.
        for k, v in raw.items():
            try:
                # Prefer __setitem__ semantics; if container is plain dict use assignment
                if hasattr(self.data, "__setitem__"):
                    self.data[k] = v
                else:
                    self.data[k] = v
            except Exception:
                logger.exception("Failed to restore key %s into cost-aware container", k)

    def flush(self):
        """
        Persist a plain dictionary snapshot of current items.
        Avoid serializing internal cache structures (locks, heaps etc).
        """
        try:
            snapshot: Dict[Any, Any] = {}

            # Try to use an items() view if present and behaving
            items_attr = getattr(self.data, "items", None)
            if callable(items_attr):
                try:
                    # Explicitly cast the returned value to an Iterable for the type checker
                    raw_items = list(cast(Iterable[Any], items_attr()))
                    for pair in raw_items:
                        if isinstance(pair, tuple) and len(pair) == 2:
                            k, val = pair
                            # If adapter returns CacheItem-like objects, extract .value
                            if hasattr(val, "value"):
                                snapshot[k] = val.value
                            else:
                                snapshot[k] = val
                except Exception:
                    # items() existed but raised or returned something unexpected; fallback
                    items_attr = None

            # Fallback: iterate keys and read values (safe copy of keys)
            if items_attr is None:
                # get a keys() callable or default to list(self.data.keys())
                keys_callable = getattr(self.data, "keys", None)
                if callable(keys_callable):
                    try:
                        # Cast the result to an Iterable so the type checker accepts list(...)
                        keys_iter = list(cast(Iterable[Any], keys_callable()))
                    except Exception:
                        # If keys() raised, fallback to an explicit list(self.data.keys())
                        keys_iter = list(self.data.keys())
                else:
                    keys_iter = list(self.data.keys())

                for k in keys_iter:
                    try:
                        snapshot[k] = self.data[k]
                    except Exception:
                        logger.exception("Failed to read key %s during flush; skipping", k)

            # Ensure directory exists and write snapshot using binary pickle
            os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
            with open(self.data_path, "wb") as f:
                # cast to Any silences type-checker complaints about file-like types
                pickle.dump(snapshot, cast(Any, f), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.exception("Failed to flush CostAwareDataManager to %s: %s", self.data_path, e)

    def _create_container(self, size: int):
        """Create a new cost-aware container with current settings (used by init/ctor)."""
        if COST_AWARE_FACTORY:
            return CostAwareCacheFactory(
                maxsize=size if size > 0 else self.max_size,
                on_evict=self._on_evict_callback_adapter,
                cost_function=self._cost_function,
                strategy=self._strategy,
                cost_weight=self._cost_weight,
                frequency_weight=self._frequency_weight,
                recency_weight=self._recency_weight,
                ewma_alpha=self._ewma_alpha,
            )
        else:
            return CostAwareCacheLegacy(
                maxsize=size if size > 0 else self.max_size,
                cost_function=self._cost_function,
                strategy=self._strategy,
                cost_weight=self._cost_weight,
                frequency_weight=self._frequency_weight,
                recency_weight=self._recency_weight,
                ewma_alpha=self._ewma_alpha,
                on_evict=self._on_evict_callback,
            )

    # ---------- vector deletion helpers ----------
    def _delete_vector_for_key(self, key: Any):
        """
        Delete the vector associated with `key` using the provided vector_store.

        The vector_store API may accept a single id or a list of ids, so we try
        both. This helper swallows exceptions and logs them.
        """
        if self.vector_store is None or key is None:
            return
        try:
            try:
                # Try single-id deletion
                self.vector_store.delete(key)
            except TypeError:
                # Maybe expects a list
                self.vector_store.delete([key])
        except Exception:
            logger.exception("Failed to delete vector for key %s", key)

    def _on_evict_callback_adapter(self, ev: Any):
        """
        Adapter callback to handle multiple shapes of eviction callbacks:
          - (key, item)
          - single key
          - list/tuple of keys

        Normalizes and deletes vectors accordingly.
        """
        # tuple (key, item)
        if isinstance(ev, tuple) and len(ev) == 2:
            key, _ = ev
            self._delete_vector_for_key(key)
            return

        # list/tuple of keys
        if isinstance(ev, (list, tuple)):
            for k in ev:
                self._delete_vector_for_key(k)
            return

        # single key
        self._delete_vector_for_key(ev)

    def _on_evict_callback(self, key: Any, item: Any = None):
        """
        Backwards-compatible on_evict signature for legacy CostAwareCache which
        calls on_evict(key, item). Normalize to vector deletion.
        """
        self._delete_vector_for_key(key)


# End of file
