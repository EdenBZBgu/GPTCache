#!/usr/bin/env python3
"""
Refactored GPTCache benchmark runner.

- Flushes cache (POST /flush) before each policy run.
- Uses requests.Session with retries.
- Encapsulates HTTP in CacheClient.
"""

import argparse
import json
import random
import string
import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------------------
# Config / Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("gptcache-bench")


# ----------------------------
# Utility Functions
# ----------------------------
def random_string(n: int = 16) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


# ----------------------------
# Cache HTTP Client
# ----------------------------
class CacheClient:
    """
    Simple HTTP client wrapper for cache server endpoints.
    Uses requests.Session with retries for robustness.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, timeout: float = 5.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get(self, key: str) -> Optional[str]:
        try:
            r = self.session.get(self._url("/get"), params={"prompt": key}, timeout=self.timeout)
            if r.status_code == 200 and r.text.strip():
                return r.text
        except Exception as e:
            logger.debug("Cache GET error for key %s: %s", key, e)
        return None

    def put(self, key: str, value: str) -> None:
        try:
            self.session.post(self._url("/put"), json={"prompt": key, "answer": value}, timeout=self.timeout)
        except Exception as e:
            logger.debug("Cache PUT error for key %s: %s", key, e)

    def flush(self, namespace: Optional[str] = None) -> bool:
        """
        Attempt to flush the cache for a given namespace.
        Tries POST /flush with {"namespace": namespace} first.
        Falls back to POST /flush with no body if the first fails.
        Returns True if any POST returned successful (2xx).
        """
        try:
            payload = {"namespace": namespace} if namespace else {}
            r = self.session.post(self._url("/flush"), json=payload, timeout=self.timeout)
            if 200 <= r.status_code < 300:
                logger.info("Flush succeeded with payload %s (status %d)", payload, r.status_code)
                return True
            # fallback: no body
            r2 = self.session.post(self._url("/flush"), timeout=self.timeout)
            if 200 <= r2.status_code < 300:
                logger.info("Flush succeeded (fallback, no payload) (status %d)", r2.status_code)
                return True
            logger.warning("Flush attempts failed: %d / %d", r.status_code, r2.status_code)
        except Exception as e:
            logger.debug("Flush exception (will be treated as failure): %s", e)
        return False

    def stats(self) -> Optional[Dict]:
        try:
            r = self.session.get(self._url("/stats"), timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug("Stats fetch error: %s", e)
        return None


# ----------------------------
# Workloads
# ----------------------------
def workload_sequential(dataset: List[str], client: CacheClient, namespace: str) -> Tuple[int, int, float]:
    """
    Sequentially get each prompt (key) from cache and put on miss.
    Returns (hits, misses, duration_seconds).
    """
    hits = 0
    misses = 0
    start = time.perf_counter()

    # Prepend namespace to keys (keeps them isolated per test)
    for line in dataset:
        key = f"{namespace}:{line.strip()}"
        val = client.get(key)
        if val is None:
            misses += 1
            client.put(key, line.strip())
        else:
            hits += 1

    duration = time.perf_counter() - start
    return hits, misses, duration


# ----------------------------
# Benchmark Runner
# ----------------------------
def run_benchmarks(
    policies: List[str],
    dataset: List[str],
    client: CacheClient,
    attempt_stats: bool = False,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    for policy in policies:
        namespace = f"test-{policy}-{random_string(6)}"
        logger.info("Running policy '%s' using namespace '%s'...", policy, namespace)

        # Flush cache before each test (user indicated this is desired)
        flushed = client.flush(namespace)
        if not flushed:
            logger.warning("Flush for namespace '%s' did not report success. Continuing anyway.", namespace)

        hits, misses, duration = workload_sequential(dataset, client, namespace)
        ratio = hits / (hits + misses) if (hits + misses) > 0 else 0.0
        results[policy] = {
            "hits": hits,
            "misses": misses,
            "hit_ratio": ratio,
            "duration": duration,
            "namespace": namespace,
        }

        if attempt_stats:
            stats = client.stats()
            if stats is not None:
                results[policy]["server_stats"] = stats

        logger.info(
            "Policy '%s' finished: hits=%d misses=%d hit_ratio=%.3f duration=%.3fs",
            policy,
            hits,
            misses,
            ratio,
            duration,
        )

    return results


# ----------------------------
# Visualization
# ----------------------------
def visualize(results: Dict[str, Dict], attempt_stats: bool = False) -> None:
    policies = list(results.keys())
    hit_ratios = [results[p]["hit_ratio"] for p in policies]
    durations = [results[p]["duration"] for p in policies]
    hits = [results[p]["hits"] for p in policies]
    misses = [results[p]["misses"] for p in policies]

    ncols = 2
    nrows = 3 if attempt_stats else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    axes = axes.flatten()

    axes[0].bar(policies, hit_ratios)
    axes[0].set_title("Hit Ratios")
    axes[0].set_ylabel("Fraction")

    axes[1].bar(policies, durations)
    axes[1].set_title("Durations (s)")
    axes[1].set_ylabel("Seconds")

    axes[2].bar(policies, hits)
    axes[2].set_title("Hits")

    axes[3].bar(policies, misses)
    axes[3].set_title("Misses")

    if attempt_stats:
        server_keys = []
        server_values = defaultdict(list)
        for p in policies:
            if "server_stats" in results[p]:
                for k, v in results[p]["server_stats"].items():
                    if k not in server_keys:
                        server_keys.append(k)
                    server_values[k].append(v)

        # Plot each server stat on its own axis if space allows
        for idx, k in enumerate(server_keys):
            plot_idx = idx + 4
            if plot_idx < len(axes):
                axes[plot_idx].bar(policies, server_values[k])
                axes[plot_idx].set_title(f"Server: {k}")

    plt.tight_layout()
    plt.show()


# ----------------------------
# Dataset loader
# ----------------------------
def load_dataset(path: str) -> List[str]:
    dataset: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
                dataset.append(obj.get("prompt", line))
            except Exception:
                dataset.append(line)
    return dataset


# ----------------------------
# CLI / Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GPTCache server with flush per-policy.")
    parser.add_argument("--dataset", required=True, help="Path to dataset file (jsonl)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8000, type=int, help="Server port")
    parser.add_argument(
        "--policies", nargs="+", default=["cost_aware", "lru", "fifo"], help="Policies to test"
    )
    parser.add_argument("--attempt-stats", action="store_true", help="Attempt fetching server stats if supported")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if not dataset:
        logger.error("Loaded dataset is empty. Check the file: %s", args.dataset)
        return

    client = CacheClient(host=args.host, port=args.port)
    results = run_benchmarks(args.policies, dataset, client, attempt_stats=args.attempt_stats)
    visualize(results, attempt_stats=args.attempt_stats)


if __name__ == "__main__":
    main()