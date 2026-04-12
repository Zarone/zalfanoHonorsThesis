"""
Training script for Discrete Diffusion language models across multiple languages and dataset sizes.

Uses shared training utilities to minimise code duplication with other model types.

To train all models:
    python models/DiscreteDiffusion/train_all.py

With a specific evaluation type:
    python models/DiscreteDiffusion/train_all.py --eval_type flores200_perplexity
    python models/DiscreteDiffusion/train_all.py --eval_type hellaswag
"""

import os
import argparse

from util.constants import LANG_SETS
from util.training_utils import unified_train_all
from models.DiscreteDiffusion.train import train


DATASET_SIZES = ["5mb", "10mb", "100mb", "1000mb"]


def main():
    parser = argparse.ArgumentParser(
        description="Train Discrete Diffusion language models across all languages and sizes"
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        default="flores200_perplexity",
        choices=["flores200_perplexity", "hellaswag"],
        help="Type of evaluation to use",
    )
    args = parser.parse_args()

    MODELS_OUTDIR = "model_weights_discrete_diffusion/"

    def get_discrete_config_path(model_size):
        """Get path to discrete diffusion config."""
        return f"models/DiscreteDiffusion/discrete_diffusion_{model_size}_config.json"

    def get_discrete_diffusion_train_instances(config):
        """Train single Discrete Diffusion instance per language-size."""
        model_dir = os.path.join(MODELS_OUTDIR, config["dataset_size"])
        lang = config["lang"]
        dataset_size = config["dataset_size"]

        outname = os.path.join(model_dir, f"discrete_diffusion_{lang}_{dataset_size}")
        return [(False, outname, f"discrete_diffusion_{lang}")]

    unified_train_all(
        model_type="discrete_diffusion",
        models_outdir=MODELS_OUTDIR,
        eval_type=args.eval_type,
        dataset_sizes=DATASET_SIZES,
        train_fn=train,
        lang_sets=LANG_SETS,
        get_config_path_fn=get_discrete_config_path,
        get_train_instances_fn=get_discrete_diffusion_train_instances,
    )


if __name__ == "__main__":
    main()
