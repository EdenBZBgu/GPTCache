#!/usr/bin/env python3
"""
benchmark_weights.py

Benchmark different weight configurations for cost-aware caching policy.
Analyzes how cost_weight, frequency_weight, and recency_weight impact:
- Hit ratio
- Latency
- Memory usage

Based on benchmark_policies_with_server.py infrastructure.
"""

import argparse
import itertools
import sys
from typing import Dict, List, Tuple
from benchmark_policies_with_server import (
    load_prompts,
    build_workloads,
    run_benchmarks,
    visualize_grid
)
import matplotlib.pyplot as plt
import numpy as np

def create_repeated_hot_workload(prompts: List[str], max_requests: int) -> List[Tuple[str, List[str]]]:
    """Create a single repeated-hot workload for consistent testing."""
    hot_ratio = 0.05  # 5% of prompts are "hot"
    repeats = 10      # Number of times to repeat the hot set
    
    k = max(1, int(len(prompts[:max_requests]) * hot_ratio))
    hot = prompts[:k]
    cold = prompts[k:max_requests]
    
    workload_prompts = []
    # Add hot prompts multiple times
    for _ in range(repeats):
        workload_prompts.extend(hot)
    # Add cold prompts once
    workload_prompts.extend(cold)
    
    return [("repeated_hot", workload_prompts)]

def generate_weight_configs(step: float = 0.2) -> List[Dict[str, float]]:
    """Generate different weight configurations to test.
    For each parameter (cost, frequency, recency), we'll vary it from 0 to 1
    while keeping other parameters fixed at their default values."""
    configs = []
    weights = np.arange(0.0, 1.1, step)
    
    # Default values from the original implementation
    default_cost_w = 1.0
    default_freq_w = 0.5
    default_rec_w = 0.3
    
    # Vary cost_weight
    for w in weights:
        configs.append({
            "cost_weight": round(w, 3),
            "frequency_weight": default_freq_w,
            "recency_weight": default_rec_w
        })
    
    # Vary frequency_weight
    for w in weights:
        configs.append({
            "cost_weight": default_cost_w,
            "frequency_weight": round(w, 3),
            "recency_weight": default_rec_w
        })
    
    # Vary recency_weight
    for w in weights:
        configs.append({
            "cost_weight": default_cost_w,
            "frequency_weight": default_freq_w,
            "recency_weight": round(w, 3)
        })
    
    return configs

def run_weight_experiments(
    base_url: str,
    workloads: List[Tuple[str, List[str]]],
    configs: List[Dict[str, float]],
    concurrency: int = 8
) -> Dict:
    """Run benchmarks for each weight configuration."""
    results = {}
    
    for idx, config in enumerate(configs):
        # Create a policy name that encodes the weights
        policy_name = f"cost_{config['cost_weight']:.2f}_freq_{config['frequency_weight']:.2f}_rec_{config['recency_weight']:.2f}"
        print(f"\nTesting configuration {idx + 1}/{len(configs)}:")
        print(f"Cost weight: {config['cost_weight']:.2f}")
        print(f"Frequency weight: {config['frequency_weight']:.2f}")
        print(f"Recency weight: {config['recency_weight']:.2f}")
        
        # Run benchmarks with this configuration
        policy_results = run_benchmarks(
            base_url=base_url,
            policies=[policy_name],
            workloads=workloads,
            concurrency=concurrency,
            policy_param="policy"
        )
        
        results[policy_name] = policy_results[policy_name]
    
    return results

def plot_weight_impact(results: Dict, workloads_order: List[str]):
    """Create visualizations showing the impact of different weights."""
    # Extract all unique weight values for each parameter
    configs = []
    for policy in results.keys():
        parts = policy.split('_')
        configs.append({
            'cost': float(parts[1]),
            'freq': float(parts[3]),
            'rec': float(parts[5])
        })
    
    # Create 3x3 grid of plots
    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Impact of Weights on Cache Performance', fontsize=16)
    
    metrics = ['Hit Ratio', 'Average Latency', 'Memory Usage']
    weights = ['Cost Weight', 'Frequency Weight', 'Recency Weight']
    
    for i, metric in enumerate(metrics):
        for j, weight in enumerate(weights):
            ax = axs[i, j]
            weight_key = weight.lower().split()[0]
            
            # Collect x (weight) and y (metric) values
            x_vals = []
            y_vals = []
            
            for policy, data in results.items():
                parts = policy.split('_')
                weight_val = float(parts[2*j + 1])  # Extract weight value from policy name
                
                # Calculate metric value
                if metric == 'Hit Ratio':
                    total_hits = sum(data[w].get('hits', 0) for w in workloads_order)
                    total_requests = sum((data[w].get('hits', 0) + data[w].get('misses', 0)) for w in workloads_order)
                    metric_val = total_hits / max(1, total_requests)
                elif metric == 'Average Latency':
                    all_latencies = []
                    for w in workloads_order:
                        all_latencies.extend(data[w].get('latencies', []))
                    metric_val = sum(all_latencies) / max(1, len(all_latencies))
                else:  # Memory Usage
                    all_memory = []
                    for w in workloads_order:
                        all_memory.extend(data[w].get('client_memory', []))
                    metric_val = sum(all_memory) / max(1, len(all_memory))
                
                x_vals.append(weight_val)
                y_vals.append(metric_val)
            
            # Plot with trend line
            ax.scatter(x_vals, y_vals, alpha=0.5)
            
            # Add trend line
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            ax.plot(sorted(x_vals), p(sorted(x_vals)), "r--", alpha=0.8)
            
            ax.set_xlabel(weight)
            ax.set_ylabel(metric)
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="jsonl dataset with 'prompt' fields")
    parser.add_argument("--host", default="127.0.0.1", help="server host (default: 127.0.0.1)")
    parser.add_argument("--port", default=8000, type=int, help="server port (default: 8000)")
    parser.add_argument("--base-url", default=None, help="base url to the running server (overrides host/port)")
    parser.add_argument("--max-prompts", type=int, default=2000, help="max number of prompts to use")
    parser.add_argument("--concurrency", type=int, default=8, help="concurrency for executor")
    parser.add_argument("--weight-step", type=float, default=0.2, help="step size for weight values (default: 0.2)")
    args = parser.parse_args()

    # Load prompts
    prompts = load_prompts(args.dataset, max_prompts=args.max_prompts)
    if not prompts:
        print("No prompts loaded; exiting")
        sys.exit(1)

    # Set up base URL
    if args.base_url:
        base_url = args.base_url
    else:
        base_url = f"http://{args.host}:{args.port}"

    # Generate single repeated-hot workload
    workloads = create_repeated_hot_workload(prompts, args.max_prompts)
    workloads_order = ["repeated_hot"]

    # Generate weight configurations to test
    configs = generate_weight_configs(step=args.weight_step)
    print(f"Testing {len(configs)} weight configurations with repeated-hot workload")

    # Run experiments
    results = run_weight_experiments(base_url, workloads, configs, args.concurrency)

    # Visualize results
    plot_weight_impact(results, workloads_order)

    # Print summary
    print("\n=== Weight Configuration Impact Summary ===")
    for policy, data in results.items():
        total_hits = sum(data[w].get('hits', 0) for w in workloads_order)
        total_requests = sum((data[w].get('hits', 0) + data[w].get('misses', 0)) for w in workloads_order)
        avg_hit_ratio = total_hits / max(1, total_requests)
        
        all_latencies = []
        for w in workloads_order:
            all_latencies.extend(data[w].get('latencies', []))
        avg_latency = sum(all_latencies) / max(1, len(all_latencies))
        
        print(f"\nPolicy: {policy}")
        print(f"Overall Hit Ratio: {avg_hit_ratio:.2%}")
        print(f"Average Latency: {avg_latency:.4f}s")

if __name__ == "__main__":
    main()
