import os
import torch
from transformers import AutoTokenizer, AlbertTokenizer

from util.training_utils import unified_train
from util.constants import MAX_SEQ_LEN
from models.DiscreteDiffusion.discrete_diffusion_model import (
    DiscreteDiffusionLanguageModel,
    DiscreteDiffusionConfig,
)

seed = 43


def train(
    output_dir,
    config_name,
    tokenizer_name,
    training_file_path,
    eval_file_path,
    override_n_examples,
    max_steps,
    warmup_steps,
    epochs,
    dropout,
    gradient_accumulation_steps=1,
    batch_size=4,
    eval_steps=40,
    evaluation_type="flores200_perplexity",
):
    """
    Train a Discrete Diffusion Language Model using unified training infrastructure.

    Model-specific: Config and model loading (including mask_token_id wiring).
    All other logic is shared via unified_train.
    """

    def load_config(tokenizer):
        """Load and configure discrete diffusion model config."""
        config = DiscreteDiffusionConfig.from_pretrained(config_name)
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.vocab_size = len(tokenizer)

        # Wire up the mask token — required for discrete diffusion
        if tokenizer.mask_token_id is not None:
            config.mask_token_id = tokenizer.mask_token_id
        else:
            # If the tokenizer has no mask token, add one and use its ID
            tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            config.mask_token_id = tokenizer.mask_token_id
            config.vocab_size = len(tokenizer)  # update after adding token

        return config

    def load_model(config):
        """Create discrete diffusion model from config."""
        model = DiscreteDiffusionLanguageModel(config)
        return model

    unified_train(
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        training_file_path=training_file_path,
        eval_file_path=eval_file_path,
        override_n_examples=override_n_examples,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        epochs=epochs,
        dropout=dropout,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        eval_steps=eval_steps,
        evaluation_type=evaluation_type,
        load_config_fn=load_config,
        load_model_fn=load_model,
        block_size=MAX_SEQ_LEN,
        seed=seed,
    )
