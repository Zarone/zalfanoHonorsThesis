"""
Training script for Continuous Diffusion language models across multiple languages and dataset sizes.

Uses shared training utilities to minimize code duplication with other model types.

To train all models:
    python models/ContinuousDiffusion/train_all.py

With specific evaluation type:
    python models/ContinuousDiffusion/train_all.py --eval_type flores200_perplexity
    python models/ContinuousDiffusion/train_all.py --eval_type hellaswag
"""

import os
import argparse

from util.constants import LANG_SETS
from util.training_utils import unified_train_all
from models.ContinuousDiffusion.train import train


DATASET_SIZES = ['5mb', '10mb', '100mb', '1000mb']


def main():
    parser = argparse.ArgumentParser(
        description="Train Continuous Diffusion language models across all languages and sizes"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="flores200_perplexity",
        choices=["flores200_perplexity", "hellaswag"],
        help="Type of evaluation to use"
    )
    args = parser.parse_args()
    
    MODELS_OUTDIR = 'model_weights_continuous_diffusion/'
    
    # Continuous Diffusion-specific callbacks
    def get_continuous_config_path(model_size):
        """Get path to continuous diffusion config."""
        return f'models/ContinuousDiffusion/continuous_diffusion_{model_size}_config.json'
    
    def get_continuous_diffusion_train_instances(config):
        """Train single Continuous Diffusion instance per language-size."""
        model_dir = os.path.join(MODELS_OUTDIR, config['dataset_size'])
        lang = config['lang']
        dataset_size = config['dataset_size']
        
        outname = os.path.join(model_dir, f'continuous_diffusion_{lang}_{dataset_size}')
        return [(False, outname, f'continuous_diffusion_{lang}')]
    
    unified_train_all(
        model_type="continuous_diffusion",
        models_outdir=MODELS_OUTDIR,
        eval_type=args.eval_type,
        dataset_sizes=DATASET_SIZES,
        train_fn=train,
        lang_sets=LANG_SETS,
        get_config_path_fn=get_continuous_config_path,
        get_train_instances_fn=get_continuous_diffusion_train_instances,
    )


if __name__ == "__main__":
    main()
