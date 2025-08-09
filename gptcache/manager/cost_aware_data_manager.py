# File: gptcache/manager/data_manager/cost_aware_data_manager.py
from typing import Callable, Optional, Dict, Any
import pickle
import os
import logging

# Try importing MapDataManager from common locations in the project.
try:
    from gptcache.manager.data_manager import MapDataManager
except Exception:
    # Fallback import path — adjust if your repo structure differs
    from gptcache.manager import MapDataManager  # type: ignore

from gptcache.manager.eviction.cost_aware_policy import CostAwareCache, CostFunction, CacheItem

logger = logging.getLogger(__name__)


class CostAwareDataManager(MapDataManager):
    """
    Map-style DataManager that uses CostAwareCache as the in-memory container.

    - Persists a plain dict snapshot (avoids pickling locks/heap).
    - Restores snapshot into a fresh CostAwareCache on init.
    - Optionally accepts a vector_store and will delete vector entries on eviction.
    """

    def __init__(self,
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
                 get_data_container: Optional[Callable[[int], Any]] = None):
        """
        Args:
            data_path: path to persist snapshot (pickle)
            max_size: maximum number of items (passed to CostAwareCache)
            cost_function: optional CostFunction or callable
            strategy: "heuristic" | "greedy"
            cost_weight/frequency_weight/recency_weight/ewma_alpha: benefit params
            vector_store: optional vector store client with `delete(id)` method — used on eviction
            get_data_container: optional custom container factory (for advanced use)
        """
        self.data_path = data_path
        self.max_size = max_size
        self._cost_function = cost_function if cost_function is not None else CostFunction()
        self._strategy = strategy
        self._cost_weight = cost_weight
        self._frequency_weight = frequency_weight
        self._recency_weight = recency_weight
        self._ewma_alpha = ewma_alpha
        self.vector_store = vector_store

        # Prepare get_data_container if not provided
        if get_data_container is None:
            def _get_data_container(sz: int):
                # attach an on_evict callback so manager can remove vectors if needed
                return CostAwareCache(
                    maxsize=sz,
                    cost_function=self._cost_function,
                    strategy=self._strategy,
                    cost_weight=self._cost_weight,
                    frequency_weight=self._frequency_weight,
                    recency_weight=self._recency_weight,
                    ewma_alpha=self._ewma_alpha,
                    on_evict=self._on_evict_callback
                )
            get_data_container = _get_data_container

        # Initialize MapDataManager (the parent will call get_data_container to set up self.data)
        super().__init__(data_path=self.data_path, max_size=self.max_size, get_data_container=get_data_container)

    # ---------- persistence: snapshot plain dict ----------
    def init(self):
        """
        Load persisted entries (if present) into a fresh CostAwareCache container.
        The persisted format must be a mapping: key -> value (whatever you stored).
        """
        # Ensure the in-memory container exists (parent MapDataManager likely already created it)
        if not hasattr(self, "data") or self.data is None:
            self.data = self._create_container(self.max_size)

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
        self.data = self._create_container(self.max_size)

        for k, v in raw.items():
            try:
                # Insert via __setitem__ so CostAwareCache.put() runs and cost is (re-)calculated.
                # If you need to preserve original cost, extend snapshot format and call put() with metadata.
                self.data[k] = v
            except Exception:
                logger.exception("Failed to restore key %s into CostAwareCache", k)

    def flush(self):
        """
        Persist a plain dictionary snapshot of current items.
        Avoid serializing internal cache structures.
        """
        try:
            snapshot = {}
            # iterate through keys and retrieve stored "value"
            # use list(...) to avoid runtime changes during iteration
            for k in list(self.data.keys()):
                try:
                    snapshot[k] = self.data[k]
                except Exception:
                    logger.exception("Failed to read key %s during flush; skipping", k)
            # ensure directory exists
            os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
            with open(self.data_path, "wb") as f:
                pickle.dump(snapshot, f)
        except Exception as e:
            logger.exception("Failed to flush CostAwareDataManager to %s: %s", self.data_path, e)

    def _create_container(self, size: int):
        """Create a new CostAwareCache with current settings (used by init/ctor)."""
        return CostAwareCache(
            maxsize=size if size > 0 else self.max_size,
            cost_function=self._cost_function,
            strategy=self._strategy,
            cost_weight=self._cost_weight,
            frequency_weight=self._frequency_weight,
            recency_weight=self._recency_weight,
            ewma_alpha=self._ewma_alpha,
            on_evict=self._on_evict_callback
        )

    # ---------- vector deletion callback ----------
    def _on_evict_callback(self, key: str, item: CacheItem):
        """
        Called by CostAwareCache when an item is evicted.
        If a vector_store is provided, attempt to delete the vector for this key.
        Adjust this method if your vector store API differs (id extraction etc).
        """
        if self.vector_store is None:
            return
        try:
            # By default assume key == vector_id. If your vector id is inside item.value,
            # adapt accordingly, e.g. vector_id = item.value.get("vector_id")
            vector_id = key
            # Common vector store delete signature: delete(id) or delete([id1, id2]) — try both
            try:
                # try single-id deletion first
                self.vector_store.delete(vector_id)
            except TypeError:
                # maybe expects a list
                self.vector_store.delete([vector_id])
        except Exception:
            logger.exception("Failed to delete vector for evicted key %s", key)
