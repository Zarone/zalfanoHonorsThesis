"""
Visualize comparison between Autoregressive (GPT2) and Diffusion models.

Creates overlapping graphs with perplexity and HellaSwag accuracy side by side,
organized by dataset size and language.

Colors:
  - Red: Autoregressive (GPT2)
  - Blue: Diffusion

Usage:
    python visualize_comparison.py
    python visualize_comparison.py --metric hellaswag  # Only HellaSwag
    python visualize_comparison.py --metric perplexity  # Only perplexity
    python visualize_comparison.py --lang eng_latn      # Filter by language
    python visualize_comparison.py --size 5mb           # Filter by size
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_eval_records(output_dir: str) -> List[Dict]:
    """Load evaluation records from JSONL file."""
    eval_file = os.path.join(output_dir, 'eval_records.jsonl')
    records = []
    
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Failed to load {eval_file}: {e}")
    
    return records


def extract_lang_size_from_path(model_dir: str) -> Tuple[Optional[str], Optional[str], str, str]:
    """
    Extract language, dataset size, model type from model directory path.
    
    Expected paths like:
    - model_weights/5mb/GPT_eng_latn_5mb
    - model_weights/5mb/dropout_GPT_eng_latn_5mb
    - model_weights_diffusion/10mb/diffusion_fra_latn_10mb
    
    Returns:
        (lang, size, model_type, model_name) where model_type is 'gpt2' or 'diffusion'
    """
    parts = model_dir.rstrip('/').split('/')
    
    if len(parts) < 2:
        return None, None, "unknown", parts[-1]
    
    # Get dataset size (parent directory)
    size = parts[-2]
    if size not in ['5mb', '10mb', '100mb', '1000mb']:
        return None, None, "unknown", parts[-1]
    
    # Get model name (leaf)
    model_name = parts[-1]
    
    # Determine model type
    if 'diffusion' in model_name:
        model_type = 'diffusion'
    else:
        model_type = 'gpt2'
    
    # Extract language from model name
    lang = None
    for lang_code in ['eng_latn', 'fra_latn', 'zho_hans', 'deu_latn', 'jpn_jpan', 'hin_deva']:
        if lang_code in model_name:
            lang = lang_code
            break
    
    if lang is None:
        # Try to extract language code (assumes format: *_LANG_SIZE)
        parts_split = model_name.split('_')
        if len(parts_split) >= 2:
            lang = '_'.join([p for p in parts_split if p not in ['GPT', 'dropout', 'diffusion', size]])
        else:
            lang = model_name
    
    return lang, size, model_type, model_name


def collect_evaluation_logs(
    gpt2_base_dir: str = "model_weights",
    diffusion_base_dir: str = "model_weights_diffusion",
    lang_filter: Optional[str] = None,
    size_filter: Optional[str] = None
) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
    """
    Collect all evaluation records organized by language, size, and model type.
    
    Returns:
        {lang: {size: {model_type: [records]}}}
    """
    logs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for base_dir, model_type in [(gpt2_base_dir, 'gpt2'), (diffusion_base_dir, 'diffusion')]:
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory not found: {base_dir}")
            continue
        
        # Walk through all model directories
        for dataset_size in os.listdir(base_dir):
            size_dir = os.path.join(base_dir, dataset_size)
            
            if not os.path.isdir(size_dir):
                continue
            
            if size_filter and dataset_size != size_filter:
                continue
            
            for model_name in os.listdir(size_dir):
                model_dir = os.path.join(size_dir, model_name)
                
                if not os.path.isdir(model_dir):
                    continue
                
                # Extract language and size
                lang, size, detected_type, full_model_name = extract_lang_size_from_path(model_dir)
                
                if lang is None or size is None:
                    continue
                
                if lang_filter and lang != lang_filter:
                    continue
                
                # Load evaluation records
                records = load_eval_records(model_dir)
                
                if records:
                    logs[lang][size][model_type].extend(records)
    
    return logs


def plot_comparison(
    logs: Dict[str, Dict[str, Dict[str, List[Dict]]]],
    metrics: List[str] = ["perplexity", "hellaswag"],
    output_file: Optional[str] = None
):
    """
    Create comparison plots with perplexity and HellaSwag side by side.
    
    Metrics can include "perplexity" (shows log_ppl) and "hellaswag" (shows accuracy).
    """
    if not logs:
        print("No evaluation records found to plot.")
        return
    
    # Sort languages and sizes for consistent ordering
    languages = sorted(logs.keys())
    
    # Collect all sizes
    all_sizes = set()
    for lang_data in logs.values():
        all_sizes.update(lang_data.keys())
    
    size_order = ['5mb', '10mb', '100mb', '1000mb']
    sizes = sorted(all_sizes, key=lambda x: size_order.index(x) if x in size_order else 999)
    
    # Create figure with subplots: (num_sizes) x (num_metrics)
    num_sizes = len(sizes)
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(num_sizes, num_metrics, figsize=(6 * num_metrics, 5 * num_sizes))
    
    # Handle case with single metric or size - ensure axes is always 2D
    if num_sizes == 1 and num_metrics == 1:
        axes = np.array([[axes]])
    elif num_sizes == 1:
        axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
    elif num_metrics == 1:
        axes = axes.reshape(-1, 1) if hasattr(axes, 'reshape') else np.array([axes]).reshape(-1, 1)
    
    fig.suptitle("Model Comparison: Autoregressive (GPT2) vs Diffusion", fontsize=16, fontweight='bold')
    
    # Iterate through all languages and create subplots
    for lang_idx, lang in enumerate(languages):
        print(f"\nPlotting {lang}...")
        
        for size_idx, size in enumerate(sizes):
            # Get data for this language/size combination
            gpt2_data = logs[lang][size].get('gpt2', [])
            diffusion_data = logs[lang][size].get('diffusion', [])
            
            # Sort by step
            gpt2_data = sorted(gpt2_data, key=lambda x: x.get('step', 0))
            diffusion_data = sorted(diffusion_data, key=lambda x: x.get('step', 0))
            
            for metric_idx, metric in enumerate(metrics):
                ax = axes[size_idx, metric_idx]
                
                # Plot based on metric type
                if metric.lower() == "perplexity":
                    # Plot log_ppl
                    gpt2_steps = [d.get('step', 0) for d in gpt2_data if 'log_ppl' in d]
                    gpt2_ppls = [d.get('log_ppl') for d in gpt2_data if 'log_ppl' in d]
                    
                    diffusion_steps = [d.get('step', 0) for d in diffusion_data if 'log_ppl' in d]
                    diffusion_ppls = [d.get('log_ppl') for d in diffusion_data if 'log_ppl' in d]
                    
                    if gpt2_ppls:
                        ax.plot(gpt2_steps, gpt2_ppls, 'r-o', label='GPT2', linewidth=2, markersize=4)
                    
                    if diffusion_ppls:
                        ax.plot(diffusion_steps, diffusion_ppls, 'b-s', label='Diffusion', linewidth=2, markersize=4)
                    
                    ax.set_ylabel('Log Perplexity', fontweight='bold')
                    ax.set_title(f'{size} - Perplexity ({lang})')
                    
                elif metric.lower() == "hellaswag":
                    # Plot hellaswag_accuracy
                    gpt2_steps = [d.get('step', 0) for d in gpt2_data if 'hellaswag_accuracy' in d]
                    gpt2_accs = [d.get('hellaswag_accuracy') for d in gpt2_data if 'hellaswag_accuracy' in d]
                    
                    diffusion_steps = [d.get('step', 0) for d in diffusion_data if 'hellaswag_accuracy' in d]
                    diffusion_accs = [d.get('hellaswag_accuracy') for d in diffusion_data if 'hellaswag_accuracy' in d]
                    
                    if gpt2_accs:
                        ax.plot(gpt2_steps, gpt2_accs, 'r-o', label='GPT2', linewidth=2, markersize=4)
                    
                    if diffusion_accs:
                        ax.plot(diffusion_steps, diffusion_accs, 'b-s', label='Diffusion', linewidth=2, markersize=4)
                    
                    ax.set_ylabel('HellaSwag Accuracy', fontweight='bold')
                    ax.set_title(f'{size} - HellaSwag ({lang})')
                
                ax.set_xlabel('Training Steps')
                ax.grid(True, alpha=0.3)
                
                # Add legend only to first subplot of each type
                if size_idx == 0 and (lang_idx == 0 or lang_idx % 2 == 0):
                    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    
    plt.show()


def plot_all_languages_grid(
    logs: Dict[str, Dict[str, Dict[str, List[Dict]]]],
    metrics: List[str] = ["perplexity", "hellaswag"],
    output_file: Optional[str] = None
):
    """
    Create a grid where each row is a language and columns are (size, metric) pairs.
    This creates a more comprehensive view of all data.
    """
    if not logs:
        print("No evaluation records found to plot.")
        return
    
    languages = sorted(logs.keys())
    
    # Collect all sizes
    all_sizes = set()
    for lang_data in logs.values():
        all_sizes.update(lang_data.keys())
    
    size_order = ['5mb', '10mb', '100mb', '1000mb']
    sizes = sorted(all_sizes, key=lambda x: size_order.index(x) if x in size_order else 999)
    
    # Create figure: (num_languages) x (num_sizes * num_metrics)
    num_cols = len(sizes) * len(metrics)
    num_rows = len(languages)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4.5 * num_cols, 4 * num_rows))
    
    # Ensure axes is always 2D
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else np.array([axes])
    elif num_cols == 1:
        axes = axes.reshape(-1, 1) if hasattr(axes, 'reshape') else np.array([axes]).reshape(-1, 1)
    
    fig.suptitle("Model Comparison: Autoregressive (GPT2) vs Diffusion\nRed=GPT2, Blue=Diffusion", 
                 fontsize=14, fontweight='bold')
    
    for lang_idx, lang in enumerate(languages):
        print(f"Plotting {lang}...")
        
        for i in range(len(sizes)):
            size = sizes[i]
            # Get data for this language/size combination
            gpt2_data = logs[lang][size].get('gpt2', [])
            diffusion_data = logs[lang][size].get('diffusion', [])
            
            # Sort by step
            gpt2_data = sorted(gpt2_data, key=lambda x: x.get('step', 0))
            diffusion_data = sorted(diffusion_data, key=lambda x: x.get('step', 0))
            
            for j in range(len(metrics)):
                metric = metrics[j]
                print(metric)
                # ax = axes[lang_idx, col_idx]
                col_idx = i * len(metrics) + j
                ax = axes[lang_idx, col_idx]
                
                # Plot based on metric type
                if metric.lower() == "perplexity":
                    print('here')
                    # Plot log_ppl
                    gpt2_steps = [d.get('step', 0) for d in gpt2_data if 'log_ppl' in d]
                    gpt2_ppls = [d.get('log_ppl') for d in gpt2_data if 'log_ppl' in d]
                    
                    diffusion_steps = [d.get('step', 0) for d in diffusion_data if 'log_ppl' in d]
                    diffusion_ppls = [d.get('log_ppl') for d in diffusion_data if 'log_ppl' in d]
                    
                    if gpt2_ppls:
                        ax.plot(gpt2_steps, gpt2_ppls, 'r-o', label='GPT2', linewidth=2, markersize=4)
                    if diffusion_ppls:
                        ax.plot(diffusion_steps, diffusion_ppls, 'b-s', label='Diffusion', linewidth=2, markersize=4)
                    
                    ax.set_ylabel('Log PPL', fontsize=9)
                    ax.set_title(f'{size} PPL', fontsize=10, fontweight='bold')
                    
                elif metric.lower() == "hellaswag":
                    # Plot hellaswag_accuracy
                    gpt2_steps = [d.get('step', 0) for d in gpt2_data if 'hellaswag_accuracy' in d]
                    gpt2_accs = [d.get('hellaswag_accuracy') for d in gpt2_data if 'hellaswag_accuracy' in d]
                    
                    diffusion_steps = [d.get('step', 0) for d in diffusion_data if 'hellaswag_accuracy' in d]
                    diffusion_accs = [d.get('hellaswag_accuracy') for d in diffusion_data if 'hellaswag_accuracy' in d]
                    
                    if gpt2_accs:
                        ax.plot(gpt2_steps, gpt2_accs, 'r-o', label='GPT2', linewidth=2, markersize=4)
                    if diffusion_accs:
                        ax.plot(diffusion_steps, diffusion_accs, 'b-s', label='Diffusion', linewidth=2, markersize=4)
                    
                    ax.set_ylabel('Accuracy', fontsize=9)
                    ax.set_title(f'{size} HellaSwag', fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Steps', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
                
                # Add legend to first row if there's data
                if lang_idx == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(loc='best', fontsize=8)
        
        # Add language label on the left
        fig.text(0.02, 0.5 - lang_idx * (0.9 / num_rows), lang, 
                rotation=90, va='center', fontweight='bold', fontsize=12)
        
        col_idx += len(metrics)
    
    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize comparison between Autoregressive and Diffusion models"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["perplexity", "hellaswag", "both"],
        default="both",
        help="Which metrics to plot"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Filter by language (e.g., eng_latn)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        choices=["5mb", "10mb", "100mb", "1000mb"],
        help="Filter by dataset size"
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["compact", "grid"],
        default="grid",
        help="Plot layout: 'compact' (sizes vs metrics) or 'grid' (languages x sizes x metrics)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save plot to file (e.g., comparison.png)"
    )
    
    args = parser.parse_args()
    
    # Determine which metrics to plot
    if args.metric == "both":
        metrics = ["perplexity", "hellaswag"]
    else:
        metrics = [args.metric]
    
    # Collect evaluation logs
    print("Loading evaluation records...")
    logs = collect_evaluation_logs(lang_filter=args.lang, size_filter=args.size)
    
    if not logs:
        print("No evaluation records found.")
        return
    
    # Check which metrics are actually present in the data
    available_metrics = set()
    for lang_data in logs.values():
        for size_data in lang_data.values():
            for model_data in size_data.values():
                for record in model_data:
                    if 'log_ppl' in record:
                        available_metrics.add('perplexity')
                    if 'hellaswag_accuracy' in record:
                        available_metrics.add('hellaswag')
    
    print(available_metrics)
    
    # Filter requested metrics to only those available
    actual_metrics = [m for m in metrics if m in available_metrics]
    if not actual_metrics:
        print("Warning: No perplexity (log_ppl) data found in records.")
        if 'hellaswag' in available_metrics:
            actual_metrics = ['hellaswag']
        else:
            print("Error: No evaluation metrics found in records.")
            return
    
    metrics = actual_metrics
    
    # Print summary
    print(f"\nFound data for:")
    for lang in sorted(logs.keys()):
        for size in sorted(logs[lang].keys()):
            print(logs[lang][size].get('gpt2', []))
            gpt2_count = len(logs[lang][size].get('gpt2', []))
            diffusion_count = len(logs[lang][size].get('diffusion', []))
            print(f"  {lang} - {size}: GPT2={gpt2_count} records, Diffusion={diffusion_count} records")
    
    # Create plots
    if args.layout == "grid":
        plot_all_languages_grid(logs, metrics=metrics, output_file=args.output)
    else:
        plot_comparison(logs, metrics=metrics, output_file=args.output)


if __name__ == "__main__":
    main()
