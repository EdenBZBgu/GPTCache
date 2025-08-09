import pickle
from abc import abstractmethod, ABCMeta
from typing import List, Any, Optional, Union, Dict

import cachetools
import numpy as np
import requests

from gptcache.manager.eviction import EvictionBase
from gptcache.manager.eviction.distributed_cache import NoOpEviction
from gptcache.manager.eviction_manager import EvictionManager
from gptcache.manager.object_data.base import ObjectBase
from gptcache.manager.scalar_data.base import (
    CacheStorage,
    CacheData,
    DataType,
    Answer,
    Question,
)
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils.error import CacheError, ParamError
from gptcache.utils.log import gptcache_log
from ..manager.eviction import get_eviction_policy


class DataManager(metaclass=ABCMeta):
    """DataManager manage the cache data, including save and search"""

    @abstractmethod
    def save(self, question, answer, embedding_data, **kwargs):
        pass

    @abstractmethod
    def import_data(
            self,
            questions: List[Any],
            answers: List[Any],
            embedding_datas: List[Any],
            session_ids: List[Optional[str]],
            **kwargs,
    ):
        pass

    @abstractmethod
    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        pass

    def hit_cache_callback(self, res_data, **kwargs):
        pass

    @abstractmethod
    def search(self, embedding_data, **kwargs):
        """search the data in the cache store accrodding to the embedding data

        :return: a list of search result, [[score, id], [score, id], ...]
        """
        pass

    def flush(self):
        pass

    @abstractmethod
    def add_session(self, res_data, session_id, pre_embedding_data):
        pass

    @abstractmethod
    def list_sessions(self, session_id, key):
        pass

    @abstractmethod
    def delete_session(self, session_id):
        pass

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        pass

    @abstractmethod
    def close(self):
        pass


class MapDataManager(DataManager):
    """MapDataManager, store all data in a map data structure.

    :param data_path: the path to save the map data, defaults to 'data_map.txt'.
    :type data_path:  str
    :param max_size: the max size for the cache, defaults to 1000.
    :type max_size: int
    :param get_data_container: a Callable to get the data container, defaults to None.
    :type get_data_container:  Callable


    Example:
        .. code-block:: python

            from gptcache.manager import get_data_manager

            data_manager = get_data_manager("data_map.txt", 1000)
    """

    def __init__(self, data_path, max_size, get_data_container=None):
        if get_data_container is None:
            self.data = cachetools.LRUCache(max_size)
        else:
            self.data = get_data_container(max_size)
        self.data_path = data_path
        self.init()

    def init(self):
        try:
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            return
        except PermissionError:
            raise CacheError(  # pylint: disable=W0707
                f"You don't have permission to access this file <{self.data_path}>."
            )

    def save(self, question, answer, embedding_data, **kwargs):
        if isinstance(question, Question):
            question = question.content
        session = kwargs.get("session", None)
        session_id = {session.name} if session else set()
        self.data[embedding_data] = (question, answer, embedding_data, session_id)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Any],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
        **_,
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        for i, embedding_data in enumerate(embedding_datas):
            self.data[embedding_data] = (
                questions[i],
                answers[i],
                embedding_datas[i],
                {session_ids[i]} if session_ids[i] else set(),
            )

    def get_scalar_data(self, res_data, **kwargs) -> CacheData:
        session = kwargs.get("session", None)
        if session:
            answer = (
                res_data[1].answer if isinstance(res_data[1], Answer) else res_data[1]
            )
            if not session.check_hit_func(
                session.name, list(res_data[3]), [res_data[0]], answer
            ):
                return None
        return CacheData(question=res_data[0], answers=res_data[1])

    def search(self, embedding_data, **kwargs):
        try:
            return [self.data[embedding_data]]
        except KeyError:
            return []

    def flush(self):
        try:
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
        except PermissionError:
            gptcache_log.error(
                "You don't have permission to access this file %s.", self.data_path
            )

    def add_session(self, res_data, session_id, pre_embedding_data):
        res_data[3].add(session_id)

    def list_sessions(self, session_id=None, key=None):
        session_ids = set()
        for k in self.data:
            if session_id and session_id in self.data[k][3]:
                session_ids.add(k)
            elif len(self.data[k][3]) > 0:
                session_ids.update(self.data[k][3])
        return list(session_ids)

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        for k in keys:
            self.data[k][3].remove(session_id)
            if len(self.data[k][3]) == 0:
                del self.data[k]

    def close(self):
        self.flush()


def normalize(vec):
    magnitude = np.linalg.norm(vec)
    normalized_v = vec / magnitude
    return normalized_v


class SSDataManager(DataManager):
    """Generate SSDataManage to manager the data.

    :param s: CacheStorage to manager the scalar data, it can be generated with :meth:`gptcache.manager.CacheBase`.
    :type s: CacheStorage
    :param v: VectorBase to manager the vector data, it can be generated with :meth:`gptcache.manager.VectorBase`.
    :type v:  VectorBase
    :param o: ObjectBase to manager the object data, it can be generated with :meth:`gptcache.manager.ObjectBase`.
    :type o:  ObjectBase
    :param e: EvictionBase to manager the eviction data, it can be generated with :meth:`gptcache.manager.EvictionBase`.
    :type e:  EvictionBase
    """

    def __init__(
        self,
        s: CacheStorage,
        v: VectorBase,
        o: Optional[ObjectBase],
        e: Optional[EvictionBase],
        max_size,
        clean_size,
        policy="LRU"
    ):
        self.s = s
        self.v = v
        self.o = o
        self.eviction_manager = EvictionManager(self.s, self.v)
        if e is None:
            e = EvictionBase(name="memory",
                             maxsize=max_size,
                             clean_size=clean_size,
                             policy=policy,
                             on_evict=self._clear)
        self.eviction_base = e

        if not isinstance(self.eviction_base, NoOpEviction):
            # if eviction manager is no op redis, we don't need to put data into eviction base
            self.eviction_base.put(self.s.get_ids(deleted=False))

    def _clear(self, marked_keys):
        self.eviction_manager.soft_evict(marked_keys)
        if self.eviction_manager.check_evict():
            self.eviction_manager.delete()

    def save(self, question, answer, embedding_data, **kwargs):
        """Save the data and vectors to cache and vector storage.

        :param question: question data.
        :type question: str
        :param answer: answer data.
        :type answer: str, Answer or (Any, DataType)
        :param embedding_data: vector data.
        :type embedding_data: np.ndarray

        Example:
            .. code-block:: python

                import numpy as np
                from gptcache.manager import get_data_manager, CacheBase, VectorBase

                data_manager = get_data_manager(CacheBase('sqlite'), VectorBase('faiss', dimension=128))
                data_manager.save('hello', 'hi', np.random.random((128, )).astype('float32'))
        """
        session = kwargs.get("session", None)
        session_id = session.name if session else None
        self.import_data([question], [answer], [embedding_data], [session_id], **kwargs)

    def _process_answer_data(self, answers: Union[Answer, List[Answer]]):
        if isinstance(answers, Answer):
            answers = [answers]
        new_ans = []
        for ans in answers:
            if ans.answer_type != DataType.STR:
                new_ans.append(Answer(self.o.put(ans.answer), ans.answer_type))
            else:
                new_ans.append(ans)
        return new_ans

    def _process_question_data(self, question: Union[str, Question]):
        if isinstance(question, Question):
            if question.deps is None:
                return question

            for dep in question.deps:
                if dep.dep_type == DataType.IMAGE_URL:
                    dep.dep_type.data = self.o.put(requests.get(dep.data).content)
            return question

        return Question(question)

    def import_data(
        self,
        questions: List[Any],
        answers: List[Answer],
        embedding_datas: List[Any],
        session_ids: List[Optional[str]],
        **kwargs,
    ):
        if (
            len(questions) != len(answers)
            or len(questions) != len(embedding_datas)
            or len(questions) != len(session_ids)
        ):
            raise ParamError("Make sure that all parameters have the same length")
        cache_datas = []
        embedding_datas = [
            normalize(embedding_data) for embedding_data in embedding_datas
        ]
        for i, embedding_data in enumerate(embedding_datas):
            if self.o is not None and not isinstance(answers[i], str):
                ans = self._process_answer_data(answers[i])
            else:
                ans = answers[i]

            cache_datas.append(
                CacheData(
                    question=self._process_question_data(questions[i]),
                    answers=ans,
                    embedding_data=embedding_data.astype("float32"),
                    session_id=session_ids[i],
                )
            )
        ids = self.s.batch_insert(cache_datas)
        self.v.mul_add(
            [
                VectorData(id=ids[i], data=embedding_data)
                for i, embedding_data in enumerate(embedding_datas)
            ],
            **kwargs,
        )
        self.eviction_base.put(ids)

    def get_scalar_data(self, res_data, **kwargs) -> Optional[CacheData]:
        session = kwargs.get("session", None)
        cache_data = self.s.get_data_by_id(res_data[1])
        if cache_data is None:
            return None

        if session:
            cache_answer = (
                cache_data.answers[0].answer
                if isinstance(cache_data.answers[0], Answer)
                else cache_data.answers[0]
            )
            res_list = self.list_sessions(key=res_data[1])
            cache_session_ids, cache_questions = [r.session_id for r in res_list], [
                r.session_question for r in res_list
            ]
            if not session.check_hit_func(
                session.name, cache_session_ids, cache_questions, cache_answer
            ):
                return None

        for ans in cache_data.answers:
            if ans.answer_type != DataType.STR:
                ans.answer = self.o.get(ans.answer)
        return cache_data

    def hit_cache_callback(self, res_data, **kwargs):
        self.eviction_base.get(res_data[1])

    def search(self, embedding_data, **kwargs):
        embedding_data = normalize(embedding_data)
        top_k = kwargs.pop("top_k", -1)
        return self.v.search(data=embedding_data, top_k=top_k, **kwargs)

    def flush(self):
        self.s.flush()
        self.v.flush()

    def add_session(self, res_data, session_id, pre_embedding_data):
        self.s.add_session(res_data[1], session_id, pre_embedding_data)

    def list_sessions(self, session_id=None, key=None):
        res = self.s.list_sessions(session_id, key)
        if key:
            return res
        if session_id:
            return list(r.id for r in res)
        return list(set(r.session_id for r in res))

    def delete_session(self, session_id):
        keys = self.list_sessions(session_id=session_id)
        self.s.delete_session(keys)

    def report_cache(
        self,
        user_question,
        cache_question,
        cache_question_id,
        cache_answer,
        similarity_value,
        cache_delta_time,
    ):
        self.s.report_cache(
            user_question,
            cache_question,
            cache_question_id,
            cache_answer,
            similarity_value,
            cache_delta_time,
        )

    def close(self):
        self.s.close()
        self.v.close()


class CostAwareDataManager:
    """
    Data manager that integrates cost-aware caching with GPTCache's storage layers.

    This manager coordinates between cache storage, vector storage, and the
    cost-aware eviction policy.
    """

    def __init__(self,
                 cache_base,
                 vector_base,
                 eviction_policy: str = "cost_aware",
                 maxsize: int = 1000,
                 eviction_params: Optional[Dict[str, Any]] = None):
        """
        Initialize cost-aware data manager.

        Args:
            cache_base: Cache storage backend (e.g., SQLite, Redis)
            vector_base: Vector storage backend (e.g., Faiss, Milvus)
            eviction_policy: Eviction policy name
            maxsize: Maximum cache size
            eviction_params: Parameters for eviction policy
        """
        self.cache_base = cache_base
        self.vector_base = vector_base

        # Initialize eviction policy
        eviction_params = eviction_params or {}
        self.cache = get_eviction_policy(
            eviction_policy,
            maxsize=maxsize,
            **eviction_params
        )

        self.eviction_policy = eviction_policy
        self.maxsize = maxsize

        # Track metadata for cost calculation
        self.query_metadata: Dict[str, Dict[str, Any]] = {}

    def save(self, question: str, answer: str, embedding: List[float], **kwargs):
        """
        Save question-answer pair with cost-aware caching.

        Args:
            question: Input question/query
            answer: Response/answer
            embedding: Question embedding vector
            **kwargs: Additional metadata (model, etc.)
        """
        # Generate cache key
        cache_key = self._generate_cache_key(question, **kwargs)

        # Extract metadata for cost calculation
        metadata = {
            'model': kwargs.get('model', 'gpt-3.5-turbo'),
            'query': question,
            **kwargs
        }

        # Store in underlying storage
        if hasattr(self.cache_base, 'save'):
            self.cache_base.save(question, answer, **kwargs)

        if hasattr(self.vector_base, 'save'):
            self.vector_base.save(embedding, cache_key)

        # Store in cost-aware cache
        self.cache.put(cache_key, answer, query=question, **metadata)

        # Track metadata
        self.query_metadata[cache_key] = metadata

    def get_scalar_data(self, res: Any, **kwargs) -> Optional[str]:
        """
        Get cached response if available.

        Args:
            res: Search result from similarity evaluation
            **kwargs: Additional parameters

        Returns:
            Cached response or None if not found
        """
        if not res:
            return None

        # Extract cache key from search result
        cache_key = self._extract_cache_key(res)
        if not cache_key:
            return None

        # Try cost-aware cache first
        cached_answer = self.cache.get(cache_key)
        if cached_answer is not None:
            return cached_answer

        # Fallback to underlying storage
        if hasattr(self.cache_base, 'get_scalar_data'):
            return self.cache_base.get_scalar_data(res, **kwargs)

        return None

    def search_data(self, embedding: List[float], **kwargs) -> List[Any]:
        """
        Search for similar embeddings.

        Args:
            embedding: Query embedding vector
            **kwargs: Search parameters

        Returns:
            List of similar items
        """
        if hasattr(self.vector_base, 'search'):
            return self.vector_base.search(embedding, **kwargs)
        return []

    def _generate_cache_key(self, question: str, **kwargs) -> str:
        """Generate cache key from question and metadata."""
        # Simple key generation - can be enhanced with hashing
        model = kwargs.get('model', 'default')
        return f"{model}:{hash(question)}"

    def _extract_cache_key(self, search_result: Any) -> Optional[str]:
        """Extract cache key from search result."""
        if hasattr(search_result, 'id'):
            return search_result.id
        elif hasattr(search_result, 'cache_key'):
            return search_result.cache_key
        elif isinstance(search_result, dict):
            return search_result.get('id') or search_result.get('cache_key')
        return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cost-aware cache statistics."""
        if hasattr(self.cache, 'get_stats'):
            return self.cache.get_stats()
        return {}

    def clear_cache(self):
        """Clear the cost-aware cache."""
        self.cache.clear()
        self.query_metadata.clear()


def get_data_manager(cache_base, vector_base=None, **kwargs):
    """
    Factory function for creating data managers.

    Args:
        cache_base: Cache storage backend
        vector_base: Vector storage backend
        **kwargs: Additional parameters including eviction policy config

    Returns:
        Data manager instance
    """
    eviction_policy = kwargs.get('eviction_policy', 'cost_aware')

    if eviction_policy == 'cost_aware' or 'cost_' in eviction_policy:
        return CostAwareDataManager(
            cache_base=cache_base,
            vector_base=vector_base,
            **kwargs
        )
    else:
        # Fallback to standard data manager if available
        try:
            from ..manager import DataManager
            return DataManager(cache_base, vector_base, **kwargs)
        except ImportError:
            return CostAwareDataManager(
                cache_base=cache_base,
                vector_base=vector_base,
                eviction_policy=eviction_policy,
                **kwargs
            )
