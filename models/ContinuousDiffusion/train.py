import os
import torch
from transformers import AutoTokenizer, AlbertTokenizer

from util.training_utils import unified_train
from util.constants import MAX_SEQ_LEN
from models.ContinuousDiffusion.continuous_diffusion_model import ContinuousDiffusionLanguageModel, ContinuousDiffusionConfig

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
    gradient_accumulation_steps=4,
    batch_size=1,
    eval_steps=40,
    evaluation_type="flores200_perplexity"
):
    """
    Train a Continuous Diffusion Language Model using unified training infrastructure.
    
    Model-specific: Config and model loading. All other logic is shared.
    """
    def load_config(tokenizer):
        """Load and configure continuous diffusion model config."""
        config = ContinuousDiffusionConfig.from_pretrained(config_name)
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.vocab_size = len(tokenizer)
        return config
    
    def load_model(config):
        """Create continuous diffusion model from config."""
        model = ContinuousDiffusionLanguageModel(config)
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
