#!/usr/bin/env python3
"""
benchmark_costaware_only.py

Benchmark harness for GPTCache cost_aware policy only.
Runs all workloads (unique, repeated_hot, zipfian, burst) for cost_aware policy.
Visualizes results in a 2x3 matplotlib grid.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import psutil
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# Utilities

def now() -> float:
    return time.time()

def try_json(resp: requests.Response) -> Optional[Dict]:
    try:
        return resp.json()
    except Exception:
        return None

# Workload generators

def load_prompts(path: str, max_prompts: Optional[int] = None) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                p = obj.get("prompt") or obj.get("text")
                if p:
                    prompts.append(p)
            except Exception:
                continue
            if max_prompts and len(prompts) >= max_prompts:
                break
    return prompts

def workload_unique(prompts: List[str]) -> List[str]:
    unique_prompts = load_prompts("synthetic_datasets/all_unique_prompts.jsonl")
    return list(unique_prompts)

def workload_repeated(prompts: List[str], hot_ratio: float = 0.1, repeats: int = 5) -> List[str]:
    k = max(1, int(len(prompts) * hot_ratio))
    hot = prompts[:k]
    cold = prompts[k:]
    out: List[str] = []
    for _ in range(repeats):
        out.extend(hot)
    out.extend(cold)
    return out

def workload_zipf(prompts: List[str], n_requests: int, a: float = 1.2) -> List[str]:
    N = len(prompts)
    if N == 0:
        return []
    ranks = list(range(1, N + 1))
    weights = [1.0 / (r ** a) for r in ranks]
    s = sum(weights)
    probs = [w / s for w in weights]
    choices = random.choices(range(N), weights=probs, k=n_requests)
    return [prompts[i] for i in choices]

def workload_burst(prompts: List[str], burst_size: int = 100, idle: float = 0.5) -> List[str]:
    hot = prompts[:max(1, int(len(prompts) * 0.02))]
    out: List[str] = []
    out.extend(random.choices(hot, k=burst_size))
    time.sleep(idle)
    out.extend(random.sample(prompts, k=min(len(prompts), burst_size)))
    return out

# Single-request wrapper

def single_get(
    base_url: str,
    prompt: str,
    get_path: str = "/get",
    put_path: str = "/put",
    policy_param: str = "policy",
    policy_name: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict:
    url = base_url.rstrip("/") + get_path
    payload = {"prompt": prompt}
    if policy_name and policy_param:
        payload[policy_param] = policy_name
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except Exception as e:
        return {"ok": False, "error": str(e), "latency": None, "status": None}
    t = r.elapsed.total_seconds() if hasattr(r, "elapsed") else None
    js = try_json(r)
    answer = None
    if js and isinstance(js, dict):
        answer = js.get("answer")
    if answer is None:
        put_url = base_url.rstrip("/") + put_path
        try:
            requests.post(
                put_url, json={"prompt": prompt, "answer": "default answer", policy_param: policy_name}, timeout=timeout
            )
        except Exception:
            pass
        return {"ok": True, "hit": False, "latency": t, "status": r.status_code}
    return {"ok": True, "hit": True, "latency": t, "status": r.status_code}

# Server helpers

def clear_server(base_url: str, policy_name: Optional[str] = None, policy_param: str = "policy", timeout: float = 5.0) -> bool:
    url = base_url.rstrip("/") + "/clear"
    try:
        r = requests.post(url, timeout=timeout)
        if 200 <= getattr(r, "status_code", 0) < 300:
            print(f"Clear succeeded (status {r.status_code})")
            return True
        print(f"Clear attempt failed: {getattr(r, 'status_code', 'err')} / {getattr(r, 'status_code', 'err')}")
    except Exception as e:
        print(f"Clear exception: {e}")
    return False

# Benchmark runner

def run_workload(
    base_url: str,
    prompts: List[str],
    policy_name: str,
    concurrency: int = 8,
    policy_param: str = "policy",
    get_path: str = "/get",
    put_path: str = "/put",
) -> Dict:
    client_proc = psutil.Process()
    metrics = {
        "hits": 0,
        "misses": 0,
        "latencies": [],
        "timestamps": [],
        "server_memory": [],
        "client_memory": [],
    }
    def worker(prompt: str) -> Dict:
        s = now()
        res = single_get(base_url, prompt, get_path=get_path, put_path=put_path, policy_param=policy_param, policy_name=policy_name)
        elapsed = now() - s
        res["measured_latency"] = elapsed
        return res
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, p) for p in prompts]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"policy={policy_name}"):
            try:
                r = fut.result()
            except Exception:
                continue
            if r.get("hit"):
                metrics["hits"] += 1
            else:
                metrics["misses"] += 1
            lat = r.get("measured_latency") or r.get("latency") or 0.0
            metrics["latencies"].append(lat)
            metrics["timestamps"].append(now())
            try:
                metrics["client_memory"].append(client_proc.memory_info().rss / (1024 * 1024))
            except Exception:
                metrics["client_memory"].append(None)
    return metrics

def run_benchmarks(base_url: str, workloads: List[Tuple[str, List[str]]], concurrency: int = 8, policy_param: str = "policy"):
    results = defaultdict(dict)
    policy = "lru"
    for wname, wprompts in workloads:
        print(f"== LRU | Workload={wname} | Requests={len(wprompts)} | Concurrency={concurrency}")
        cleared = clear_server(base_url, policy_name=policy, policy_param=policy_param)
        if not cleared:
            print(f"Warning: Clear did not return success for lru, workload={wname}. Continuing anyway.")
        metrics = run_workload(base_url, wprompts, policy, concurrency=concurrency, policy_param=policy_param)
        results[policy][wname] = metrics
    return results

def visualize_grid(results: Dict[str, Dict], workloads_order: List[str]) -> None:
    agg: Dict[str, Dict] = {}
    p = "lru"
    agg[p] = {
        "hit_ratio": [],
        "avg_latency": [],
        "latencies_all": [],
        "timestamps_all": [],
        "client_mem_all": [],
    }
    for w in workloads_order:
        m = results[p].get(w, {})
        hits = m.get("hits", 0)
        misses = m.get("misses", 0)
        total = hits + misses
        hr = hits / max(1, total)
        agg[p]["hit_ratio"].append(hr)
        lats = m.get("latencies", [])
        if lats:
            agg[p]["avg_latency"].append(sum(lats) / len(lats))
            agg[p]["latencies_all"].extend(lats)
        agg[p]["timestamps_all"].extend(m.get("timestamps", []))
        agg[p]["client_mem_all"].extend([x for x in m.get("client_memory", []) if x is not None])
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs[0, 0]
    vals = agg[p]["hit_ratio"]
    ax.plot(workloads_order, vals, marker="o", label=p)
    ax.set_title("Hit ratio per workload")
    ax.set_ylabel("Hit ratio")
    ax.set_ylim(0, 1)
    ax.legend()
    ax = axs[0, 1]
    vals = agg[p]["avg_latency"]
    ax.plot(workloads_order, vals, marker="o", label=p)
    ax.set_title("Avg latency per workload")
    ax.set_ylabel("seconds")
    ax.legend()
    ax = axs[0, 2]
    lats = sorted(agg[p]["latencies_all"]) or [0]
    if len(lats) > 1:
        xs = sorted(lats)
        ys = [i / len(xs) for i in range(1, len(xs) + 1)]
        ax.plot(xs, ys, label=p)
    ax.set_title("Latency CDF (combined)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("CDF")
    ax.legend()
    ax = axs[1, 0]
    lats = agg[p]["latencies_all"][:1000]
    ax.plot(range(len(lats)), lats, label=p)
    ax.set_title("Latency over time (first 1000 samples)")
    ax.set_ylabel("seconds")
    ax.legend()
    ax = axs[1, 1]
    mem = agg[p]["client_mem_all"]
    if mem:
        window = max(1, int(len(mem) / 50))
        smoothed = [sum(mem[max(0, i - window): i + 1]) / max(1, min(i + 1, window)) for i in range(len(mem))]
        ax.plot(smoothed, label=p)
    ax.set_title("Client memory (MB) over time")
    ax.legend()
    ax = axs[1, 2]
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def build_workloads(prompts: List[str], max_requests: int) -> List[Tuple[str, List[str]]]:
    workloads: List[Tuple[str, List[str]]] = []
    workloads.append(("unique", workload_unique(prompts[:max_requests])))
    workloads.append(("repeated_hot", workload_repeated(prompts[:max_requests], hot_ratio=0.05, repeats=10)))
    workloads.append(("zipfian", workload_zipf(prompts, n_requests=max_requests, a=1.2)))
    workloads.append(("burst", workload_burst(prompts[:max(500, len(prompts))], burst_size=min(500, max_requests))))
    return workloads

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="jsonl dataset with 'prompt' fields")
    parser.add_argument("--host", default="127.0.0.1", help="server host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="server port (default: 8000)")
    parser.add_argument("--base-url", default=None, help="base url to the running server (overrides host/port)")
    parser.add_argument("--max-prompts", type=int, default=2000, help="max number of prompts to use")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrency for executor")
    parser.add_argument("--policy-param", default="policy", help="JSON key used to tell server which policy to use")
    parser.add_argument("--all-lines", action="store_true", default=False, help="Include an 'all_lines' workload that iterates the entire dataset (default: False)")
    args = parser.parse_args()
    prompts = load_prompts(args.dataset, max_prompts=args.max_prompts)
    if not prompts:
        print("No prompts loaded; exiting")
        sys.exit(1)
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"
    workloads = build_workloads(prompts, args.max_prompts)
    if args.all_lines:
        all_lines_prompts = load_prompts(args.dataset, max_prompts=None)
        workloads.append(("all_lines", all_lines_prompts))
    results = run_benchmarks(base_url, workloads, concurrency=args.concurrency, policy_param=args.policy_param)
    workloads_order = [w[0] for w in workloads]
    visualize_grid(results, workloads_order)
    print("=== Compact Summary ===")
    p = "lru"
    for w in workloads_order:
        m = results[p].get(w, {})
        hits = m.get("hits", 0)
        misses = m.get("misses", 0)
        total = hits + misses
        avg_lat = sum(m.get("latencies", [0])) / max(1, len(m.get("latencies", [])))
        print(f"policy={p:12} workload={w:12} requests={total:6} hit_ratio={hits/max(1,total):.2%} avg_lat={avg_lat:.4f}s")

if __name__ == "__main__":
    main()
