"""
Display training evaluation logs for all models by language.

This script loads evaluation records from all trained models and displays both
perplexity and HellaSwag accuracy metrics organized by language and dataset size.

Usage:
    python show_logs.py                          # Show logs for GPT2 models
    python show_logs.py --model diffusion        # Show logs for Diffusion models
    python show_logs.py --model both             # Show logs for both GPT2 and Diffusion
    python show_logs.py --lang eng_latn          # Filter by language
    python show_logs.py --size 100mb             # Filter by dataset size
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


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


def extract_lang_size_from_path(model_dir: str) -> Tuple[str, str, str]:
    """
    Extract language and dataset size from model directory path.
    
    Expected paths like:
    - model_weights/5mb/GPT_eng_latn_5mb
    - model_weights/5mb/dropout_GPT_eng_latn_5mb
    - model_weights_diffusion/10mb/diffusion_fra_latn_10mb
    
    Returns:
        (lang, size, model_name) e.g., ('eng_latn', '5mb', 'GPT_eng_latn_5mb')
    """
    parts = model_dir.rstrip('/').split('/')
    
    if len(parts) < 2:
        return None, None, parts[-1]
    
    # Get dataset size (parent directory)
    size = parts[-2]
    if size not in ['5mb', '10mb', '100mb', '1000mb']:
        return None, None, parts[-1]
    
    # Get model name (leaf)
    model_name = parts[-1]
    
    # Extract language from model name
    # Examples: GPT_eng_latn_5mb, dropout_GPT_eng_latn_5mb, diffusion_eng_latn_10mb
    if 'eng_latn' in model_name:
        lang = 'eng_latn'
    elif 'fra_latn' in model_name:
        lang = 'fra_latn'
    elif 'zho_hans' in model_name:
        lang = 'zho_hans'
    else:
        # Try to extract language code (assumes format: *_LANG_SIZE)
        parts_split = model_name.split('_')
        if len(parts_split) >= 2:
            lang = '_'.join([p for p in parts_split if p not in ['GPT', 'dropout', 'diffusion', size]])
        else:
            lang = model_name
    
    return lang, size, model_name


def collect_evaluation_logs(
    models_base_dir: str,
    lang_filter: str = None,
    size_filter: str = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Collect all evaluation records organized by language and dataset size.
    
    Returns:
        {lang: {size: [records]}}
    """
    logs_by_lang_size = defaultdict(lambda: defaultdict(list))
    
    if not os.path.isdir(models_base_dir):
        print(f"Error: Directory not found: {models_base_dir}")
        return logs_by_lang_size
    
    # Walk through all model directories
    for dataset_size in os.listdir(models_base_dir):
        size_dir = os.path.join(models_base_dir, dataset_size)
        
        if not os.path.isdir(size_dir):
            continue
        
        if size_filter and dataset_size != size_filter:
            continue
        
        for model_name in os.listdir(size_dir):
            model_dir = os.path.join(size_dir, model_name)
            
            if not os.path.isdir(model_dir):
                continue
            
            # Extract language and size
            lang, size, full_model_name = extract_lang_size_from_path(model_dir)
            
            if lang is None or size is None:
                continue
            
            if lang_filter and lang != lang_filter:
                continue
            
            # Load evaluation records
            records = load_eval_records(model_dir)
            
            if records:
                logs_by_lang_size[lang][size].extend(records)
    
    return logs_by_lang_size


def format_float(value, precision=4):
    """Format float value with specified precision."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return "N/A"
    return f"{value:.{precision}f}"


def print_logs_by_language(
    logs: Dict[str, Dict[str, List[Dict]]],
    model_type: str = "GPT2"
):
    """Print evaluation logs organized by language."""
    
    if not logs:
        print(f"No evaluation records found for {model_type}")
        return
    
    print(f"\n{'='*80}")
    print(f"{model_type.upper()} MODEL EVALUATION LOGS")
    print(f"{'='*80}\n")
    
    for lang in sorted(logs.keys()):
        print(f"\n{'-'*80}")
        print(f"Language: {lang}")
        print(f"{'-'*80}")
        
        sizes = logs[lang]
        
        for size in sorted(sizes.keys(), key=lambda x: ['5mb', '10mb', '100mb', '1000mb'].index(x)):
            records = sizes[size]
            
            if not records:
                print(f"  {size}: No records")
                continue
            
            # Sort records by step
            records = sorted(records, key=lambda r: r.get("step", 0))
            
            print(f"\n  {size} dataset ({len(records)} evaluation points):")
            print(f"    {'Step':<8} {'Epoch':<8} {'Examples':<12} {'Log-PPL':<12} {'HellaSwag':<12}")
            print(f"    {'-'*60}")
            
            for record in records:
                step = record.get("step", "N/A")
                epoch = format_float(record.get("epoch"), 1)
                examples = record.get("total_examples", "N/A")
                log_ppl = format_float(record.get("log_ppl"), 4)
                hellaswag = format_float(record.get("hellaswag_accuracy"), 4)
                
                print(f"    {step:<8} {epoch:<8} {examples:<12} {log_ppl:<12} {hellaswag:<12}")
            
            # Print summary statistics
            log_ppls = [r.get("log_ppl") for r in records if "log_ppl" in r]
            hellaswags = [r.get("hellaswag_accuracy") for r in records if "hellaswag_accuracy" in r]
            
            if log_ppls:
                min_ppl = min(log_ppls)
                latest_ppl = log_ppls[-1]
                print(f"    → Perplexity: Min={format_float(min_ppl, 4)}, Latest={format_float(latest_ppl, 4)}")
            
            if hellaswags:
                max_hs = max(hellaswags)
                latest_hs = hellaswags[-1]
                print(f"    → HellaSwag: Max={format_float(max_hs, 4)}, Latest={format_float(latest_hs, 4)}")


def main():
    parser = argparse.ArgumentParser(description="Display training evaluation logs by language")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "diffusion", "both"],
        help="Which model(s) to display logs for"
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
    
    args = parser.parse_args()
    
    # Determine which models to show
    models_to_show = []
    if args.model in ["gpt2", "both"]:
        models_to_show.append(("GPT2", "model_weights"))
    if args.model in ["diffusion", "both"]:
        models_to_show.append(("Diffusion", "model_weights_diffusion"))
    
    # Collect and display logs
    all_logs_empty = True
    
    for model_name, base_dir in models_to_show:
        logs = collect_evaluation_logs(
            base_dir,
            lang_filter=args.lang,
            size_filter=args.size
        )
        
        if logs:
            all_logs_empty = False
            print_logs_by_language(logs, model_name)
    
    if all_logs_empty:
        print("\nNo evaluation records found.")
        if args.lang:
            print(f"  (filtered by language: {args.lang})")
        if args.size:
            print(f"  (filtered by size: {args.size})")
        print("\n  Run training first with: python models/GPT2/train_all.py")


if __name__ == "__main__":
    main()
