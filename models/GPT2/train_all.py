"""
Training script for GPT2 language models across multiple languages and dataset sizes.

Trains GPT2 models with and without dropout for comparison.

To train all models:
    python models/GPT2/train_all.py

With specific evaluation type:
    python models/GPT2/train_all.py --eval_type flores200_perplexity
    python models/GPT2/train_all.py --eval_type hellaswag
"""

import os
import argparse

from util.constants import LANG_SETS
from util.training_utils import unified_train_all, select_model_size
from models.GPT2.train import train


# GPT2-specific constants
MAX_BATCH_PER_DEVICE = 4
DATASET_SIZES = ['5mb', '10mb', '100mb', '1000mb']


def main():
    parser = argparse.ArgumentParser(
        description="Train GPT2 language models across all languages and sizes"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="flores200_perplexity",
        choices=["flores200_perplexity", "hellaswag"],
        help="Type of evaluation to use"
    )
    args = parser.parse_args()
    
    MODELS_OUTDIR = 'model_weights/'
    
    # GPT2-specific callbacks
    def get_gpt2_tokenizer_path(lang, dataset_size, dataset_config):
        """Use 100mb tokenizer for 1000mb models."""
        effective_size = '100mb' if dataset_size == '1000mb' else dataset_size
        return os.path.join(dataset_config.tokenizers_dir, f'{lang}_{effective_size}')
    
    def get_gpt2_grad_accum(batch_size):
        """Calculate gradient accumulation for GPT2."""
        batch_per_device = min(batch_size, MAX_BATCH_PER_DEVICE)
        gradient_accumulation_steps = batch_size // batch_per_device
        assert batch_size % batch_per_device == 0
        return (batch_per_device, gradient_accumulation_steps)
    
    def get_gpt2_train_instances(config):
        """Train both dropout and non-dropout versions."""
        model_dir = os.path.join(MODELS_OUTDIR, config['dataset_size'])
        lang = config['lang']
        dataset_size = config['dataset_size']
        model_size = config['model_size']
        
        dropout_outname = os.path.join(model_dir, f'dropout_GPT_{lang}_{dataset_size}')
        normal_outname = os.path.join(model_dir, f'GPT_{lang}_{dataset_size}')
        
        return [
            (True, dropout_outname, f'dropout_GPT_{lang}'),
            # (False, normal_outname, f'GPT_{lang}'),
        ]
    
    unified_train_all(
        model_type="gpt2",
        models_outdir=MODELS_OUTDIR,
        eval_type=args.eval_type,
        dataset_sizes=DATASET_SIZES,
        train_fn=train,
        lang_sets=LANG_SETS,
        get_tokenizer_path_fn=get_gpt2_tokenizer_path,
        get_grad_accum_fn=get_gpt2_grad_accum,
        get_train_instances_fn=get_gpt2_train_instances,
    )


if __name__ == "__main__":
    main()
