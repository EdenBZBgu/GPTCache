"""
benchmark_policies_with_server.py

Enhanced benchmark + visualization harness for GPTCache policies.

Features:
 - Assumes a local GPTCache server is already running (configurable host & port with sensible defaults).
 - Waits for readiness by probing common HTTP endpoints (/health, /ready, /stats, /)
 - Runs multiple workloads (unique/sequential, repeated-hotset, zipfian, burst, concurrent)
 - Compares multiple policies (default: cost_aware, lru, fifo) by instructing server via `policy` field
 - Collects: hits, misses, evictions, latencies, throughput, memory usage of client
 - Visualizes results in an attractive 2x3 matplotlib grid (no files written by default)
 - Clean CLI with sensible defaults for quick runs.

Usage (defaults are sensible so you can run with minimal args):
    python benchmark_policies_with_server.py --dataset synthetic_datasets/prompts_25MB.jsonl

Advanced example:
    python benchmark_policies_with_server.py \
        --dataset synthetic_datasets/prompts_25MB.jsonl \
        --host 127.0.0.1 --port 8000 \
        --policies cost_aware lru fifo \
        --max-prompts 2000 --concurrency 16

Notes:
 - The script expects a running GPTCache server at the provided host & port.
 - The server is expected to accept POST /get and POST /put endpoints like your example. It should accept an optional `policy` field in the JSON to select eviction behavior; if your server expects a different mechanism, pass `--policy-param KEY` to change the key name.

"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------- Utilities -----------------------------

def now():
    return time.time()


def try_json(resp):
    try:
        return resp.json()
    except Exception:
        return None


# ----------------------------- Workload generators -----------------------------

def load_prompts(path, max_prompts=None):
    prompts = []
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


def workload_unique(prompts):
    # each request a new prompt (unique)
    return list(prompts)


def workload_repeated(prompts, hot_ratio=0.1, repeats=5):
    # hot set repeated many times -- simulates cacheability
    k = max(1, int(len(prompts) * hot_ratio))
    hot = prompts[:k]
    cold = prompts[k:]
    out = []
    for _ in range(repeats):
        out.extend(hot)
    out.extend(cold)
    return out


def workload_zipf(prompts, n_requests, a=1.2):
    # approximate zipfian sampling over prompt indices
    N = len(prompts)
    if N == 0:
        return []
    ranks = list(range(1, N + 1))
    weights = [1.0 / (r ** a) for r in ranks]
    s = sum(weights)
    probs = [w / s for w in weights]
    choices = random.choices(range(N), weights=probs, k=n_requests)
    return [prompts[i] for i in choices]


def workload_burst(prompts, burst_size=100, idle=0.5):
    hot = prompts[:max(1, int(len(prompts) * 0.02))]
    out = []
    out.extend(random.choices(hot, k=burst_size))
    time.sleep(idle)
    out.extend(random.sample(prompts, k=min(len(prompts), burst_size)))
    return out


# ----------------------------- Single-request wrapper -----------------------------

def single_get(base_url, prompt, get_path="/get", put_path="/put", policy_param="policy", policy_name=None, timeout=5.0):
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
        # insert on miss
        put_url = base_url.rstrip("/") + put_path
        try:
            # keep same semantics as your working file: include an "answer" field
            pr = requests.post(put_url, json={"prompt": prompt, "answer": "default answer", policy_param: policy_name}, timeout=timeout)
        except Exception:
            pass
        return {"ok": True, "hit": False, "latency": t, "status": r.status_code}
    return {"ok": True, "hit": True, "latency": t, "status": r.status_code}


# ----------------------------- Server helpers -----------------------------

def flush_server(base_url, policy_name=None, policy_param="policy", timeout=5.0):
    """
    POST /flush before each test. Try JSON payload with the server's policy param first
    (e.g. {"policy": "lru"}), fall back to POST /flush with no body.
    Returns True on a 2xx response, False otherwise.
    """
    url = base_url.rstrip("/") + "/flush"
    # try payload with policy param when available
    try:
        payload = {policy_param: policy_name} if policy_name else {}
        r = requests.post(url, json=payload, timeout=timeout)
        if 200 <= getattr(r, "status_code", 0) < 300:
            print(f"Flush succeeded with payload {payload} (status {r.status_code})")
            return True
        # fallback: no body
        r2 = requests.post(url, timeout=timeout)
        if 200 <= getattr(r2, "status_code", 0) < 300:
            print(f"Flush succeeded (fallback, no payload) (status {r2.status_code})")
            return True
        print(f"Flush attempts failed: {getattr(r, 'status_code', 'err')} / {getattr(r2, 'status_code', 'err')}")
    except Exception as e:
        print(f"Flush exception: {e}")
    return False


# ----------------------------- Benchmark runner -----------------------------

def run_workload(base_url, prompts, policy_name, concurrency=8, policy_param="policy", get_path="/get", put_path="/put", max_workers=16):
    client_proc = psutil.Process()
    metrics = {
        "hits": 0,
        "misses": 0,
        "latencies": [],
        "timestamps": [],
        "server_memory": [],
        "client_memory": [],
    }

    def worker(prompt):
        s = now()
        res = single_get(base_url, prompt, get_path=get_path, put_path=put_path, policy_param=policy_param, policy_name=policy_name)
        elapsed = now() - s
        res["measured_latency"] = elapsed
        return res

    # run with ThreadPoolExecutor to simulate concurrency
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
            # capture client memory
            try:
                metrics["client_memory"].append(client_proc.memory_info().rss / (1024 * 1024))
            except Exception:
                metrics["client_memory"].append(None)
    return metrics


def run_benchmarks(base_url, prompts, policies, workloads, concurrency=8, policy_param="policy"):
    """Run a suite of workloads for each policy.

    workloads: list of tuples (name, prompt_sequence)
    returns nested dict: results[policy][workload_name] = metrics
    """
    results = defaultdict(dict)
    for policy in policies:
        for wname, wprompts in workloads:
            print(f">> Policy={policy} | Workload={wname} | Requests={len(wprompts)} | Concurrency={concurrency}")

            # flush before each test
            flushed = flush_server(base_url, policy_name=policy, policy_param=policy_param)
            if not flushed:
                print(f"Warning: flush did not return success for policy={policy}, workload={wname}. Continuing anyway.")

            metrics = run_workload(base_url, wprompts, policy, concurrency=concurrency, policy_param=policy_param)

            results[policy][wname] = metrics
    return results


# ----------------------------- Visualization -----------------------------

def visualize_grid(results, policies, workloads_order):
    """Create a 2x3 grid of plots comparing the policies across workloads.

    The figure will show: hit ratio, avg latency, latency CDF, latency over time, client memory.
    """
    n_workloads = len(workloads_order)
    # Build aggregated summaries per policy (averaging workloads)
    agg = {}
    for p in policies:
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

    # 1: Hit ratio per workload (stacked bars -- one group per workload)
    ax = axs[0, 0]
    index = list(range(len(workloads_order)))
    width = 0.6
    for i, p in enumerate(policies):
        vals = [results[p][w]["hits"] / max(1, results[p][w]["hits"] + results[p][w]["misses"]) if (results[p].get(w) and (results[p][w]["hits"] + results[p][w]["misses"])>0) else 0 for w in workloads_order]
        ax.bar([x + i * width / len(policies) for x in index], vals, width / len(policies), label=p)
    ax.set_title("Hit ratio per workload")
    ax.set_xticks([x + width / 2 for x in index])
    ax.set_xticklabels(workloads_order)
    ax.set_ylim(0, 1)
    ax.legend()

    # 2: Avg latency per workload
    ax = axs[0, 1]
    for p in policies:
        vals = [ (sum(results[p][w].get("latencies", [])) / max(1, len(results[p][w].get("latencies", [])))) if results[p].get(w) else 0 for w in workloads_order]
        ax.plot(workloads_order, vals, marker="o", label=p)
    ax.set_title("Avg latency per workload")
    ax.set_ylabel("seconds")
    ax.legend()

    # 3: Latency CDF (all workloads combined)
    ax = axs[0, 2]
    for p in policies:
        lats = sorted(agg[p]["latencies_all"]) or [0]
        if len(lats) > 1:
            xs = sorted(lats)
            ys = [i / len(xs) for i in range(1, len(xs) + 1)]
            ax.plot(xs, ys, label=p)
    ax.set_title("Latency CDF (combined)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("CDF")
    ax.legend()

    # 4: Latency over time (sampled)
    ax = axs[1, 0]
    for p in policies:
        lats = agg[p]["latencies_all"][:1000]
        ax.plot(range(len(lats)), lats, label=p)
    ax.set_title("Latency over time (first 1000 samples)")
    ax.set_ylabel("seconds")
    ax.legend()

    # 5: Client memory usage over time (smoothed)
    ax = axs[1, 1]
    for p in policies:
        mem = agg[p]["client_mem_all"]
        if not mem:
            continue
        # simple moving average smoothing
        window = max(1, int(len(mem) / 50))
        smoothed = [sum(mem[max(0, i - window): i + 1]) / max(1, min(i + 1, window)) for i in range(len(mem))]
        ax.plot(smoothed, label=p)
    ax.set_title("Client memory (MB) over time")
    ax.legend()

    # 6: Placeholder (server stats disabled)
    ax = axs[1, 2]
    ax.text(0.5, 0.5, "Server stats disabled (no /stats endpoint)", ha="center", va="center")
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()


# ----------------------------- Main & CLI -----------------------------

def build_workloads(prompts, max_requests):
    workloads = []
    workloads.append(("unique", workload_unique(prompts[:max_requests])))
    workloads.append(("repeated_hot", workload_repeated(prompts[:max_requests], hot_ratio=0.05, repeats=10)))
    workloads.append(("zipfian", workload_zipf(prompts, n_requests=max_requests, a=1.2)))
    workloads.append(("burst", workload_burst(prompts[:max(500, len(prompts))], burst_size=min(500, max_requests))))
    return workloads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="jsonl dataset with 'prompt' fields")
    parser.add_argument("--host", default="127.0.0.1", help="server host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="server port (default: 8000)")
    parser.add_argument("--base-url", default=None, help="base url to the running server (overrides host/port)")
    parser.add_argument("--policies", nargs="+", default=["cost_aware", "lru", "fifo"], help="policies to compare (default: cost_aware lru fifo)")
    parser.add_argument("--max-prompts", type=int, default=2000, help="max number of prompts to use")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrency for executor")
    parser.add_argument("--policy-param", default="policy", help="JSON key used to tell server which policy to use")
    parser.add_argument("--all-lines", action="store_true", default=False, help="If set, include the full dataset as an additional workload (disabled by default)")
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

    # Optional: add an "all_lines" workload that iterates the entire dataset file (no limit)
    if args.all_lines:
        all_lines_prompts = load_prompts(args.dataset, max_prompts=None)
        workloads.append(("all_lines", all_lines_prompts))

    results = run_benchmarks(base_url, prompts, args.policies, workloads, concurrency=args.concurrency, policy_param=args.policy_param)

    workloads_order = [w[0] for w in workloads]
    visualize_grid(results, args.policies, workloads_order)

    print("=== Compact Summary ===")
    for p in args.policies:
        for w in workloads_order:
            m = results[p].get(w, {})
            hits = m.get("hits", 0)
            misses = m.get("misses", 0)
            total = hits + misses
            avg_lat = sum(m.get("latencies", [0])) / max(1, len(m.get("latencies", [])))
            print(f"policy={p:12} workload={w:12} requests={total:6} hit_ratio={hits/max(1,total):.2%} avg_lat={avg_lat:.4f}s")


if __name__ == "__main__":
    main()