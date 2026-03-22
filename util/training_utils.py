"""
Shared training utilities for language models (GPT2, Diffusion, etc.)

This module contains common training infrastructure to reduce code duplication
between different model implementations.
"""

import os
import codecs
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable, List, Union
from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    TrainerCallback,
    PreTrainedTokenizer,
    BatchEncoding,
    AutoTokenizer,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    set_seed,
)
from util.constants import MAX_SEQ_LEN
from transformers.trainer_utils import get_last_checkpoint

class RepeatingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, num_epochs):
        self.dataset = dataset
        self.num_epochs = num_epochs

    def __iter__(self):
        for _ in range(self.num_epochs):
            for x in self.dataset:
                yield x

    def __len__(self):
        return len(self.dataset) * self.num_epochs

@dataclass
class TrainingSizes:
    """Training token counts for all dataset sizes and languages."""
    sizes: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_file(cls, path: str) -> "TrainingSizes":
        """Load token counts from TSV file."""
        with codecs.open(path, 'rb', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')[1:]  # Skip header
        sizes = {}
        for line in lines:
            parts = line.split('\t')
            sizes[parts[0]] = int(parts[1])
        
        return cls(sizes=sizes)
    
    def get(self, lang_dataset_key: str, default: int = -1) -> int:
        """Get token count for a language-dataset combination."""
        return self.sizes.get(lang_dataset_key, default)


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and parameters."""
    tokens_per_sequence: int
    datasets_dir: str = 'dataset/raw/tokenized_data_split'
    tokenizers_dir: str = 'dataset/tokenization/monolingual'
    eval_dataset_dir: str = 'dataset/raw/flores200_dataset/tokenized'
    train_sizes_path: str = './train_sizes_tokens.tsv'
    tokenizer_strategy: str = 'simple'  # 'simple' or 'gpt2' (use 100mb for 1000mb)
    
    def get_train_path(self, lang: str, size: str) -> str:
        """Get path to training data."""
        return os.path.join(self.datasets_dir, f'{lang}_{size}.txt')
    
    def get_eval_path(self, lang: str) -> str:
        """Get path to evaluation data."""
        return os.path.join(self.eval_dataset_dir, f'{lang}.txt')
    
    def get_tokenizer_path(self, lang: str, size: str) -> str:
        """Get path to tokenizer."""
        if self.tokenizer_strategy == 'gpt2':
            # GPT2 uses 100mb tokenizer for 1000mb models
            effective_size = '100mb' if size == '1000mb' else size
            return os.path.join(self.tokenizers_dir, f'{lang}_{effective_size}')
        else:
            # Simple: each size has its own tokenizer
            return os.path.join(self.tokenizers_dir, f'{lang}_{size}')


@dataclass
class HyperparametersConfig:
    """Common hyperparameters for training."""
    warmup_proportion: float = 0.10
    epochs: int = 4
    learning_rate: float = 0.0001
    batch_sizes: Dict[str, int] = field(default_factory=lambda: {
        '5mb': 4, '10mb': 8, '100mb': 32, '1000mb': 64
    })


def select_model_size(dataset_size: str) -> str:
    """Select model size (small or base) based on dataset size."""
    if dataset_size in ['5mb', '10mb']:
        return 'small'
    else:
        return 'base'


def calculate_training_steps(
    total_tokens: int,
    tokens_per_batch: int,
    epochs: int = 1,
    warmup_proportion: float = 0.10
) -> Tuple[int, int]:
    """
    Calculate max steps and warmup steps for training.
    
    Args:
        total_tokens: Total number of tokens in training set
        tokens_per_batch: Number of tokens per batch
        epochs: Number of training epochs
        warmup_proportion: Proportion of steps to use for warmup
    
    Returns:
        (max_steps, warmup_steps)
    """
    if total_tokens <= 0:
        # Fallback defaults
        max_steps = 100000
    else:
        max_steps = int((total_tokens * epochs) / tokens_per_batch)
    print(
        "max_steps", max_steps,
        "total_tokens", total_tokens,
        "tokens_per_batch", tokens_per_batch,
        "epochs", epochs
    )
    
    warmup_steps = int(max_steps * warmup_proportion)
    return max_steps, warmup_steps


class LMTrainingCallback(TrainerCallback):
    """Base class for language model training callbacks."""
    
    def __init__(self, args, output_dir: Optional[str] = None):
        self.batch_size = args.per_device_train_batch_size
        self.grad_accum = args.gradient_accumulation_steps
        self.records = []
        self.output_dir = output_dir or getattr(args, 'output_dir', None)
        self.eval_file = None
        if self.output_dir:
            self.eval_file = os.path.join(self.output_dir, 'eval_records.jsonl')
    
    def get_total_examples(self, step: int) -> int:
        """Calculate total training examples seen so far."""
        return step * self.batch_size * self.grad_accum
    
    def save_record(self, record: Dict) -> None:
        """Save evaluation record to file immediately."""
        self.records.append(record)
        if self.eval_file:
            try:
                with open(self.eval_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')
            except Exception as e:
                print(f"Warning: Failed to save evaluation record: {e}")


class PerplexityCallback(LMTrainingCallback):
    """Callback for tracking perplexity during training."""
    
    def __init__(self, args, evaluation_type="loss-based", output_dir: Optional[str] = None):
        super().__init__(args, output_dir=output_dir)
        self.evaluation_type = evaluation_type
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return
        
        step = state.global_step
        epoch = state.epoch
        total_examples = self.get_total_examples(step)
        log_ppl = eval_loss
        
        record = {
            "step": step,
            "epoch": epoch,
            "total_examples": total_examples,
            "log_ppl": log_ppl,
        }
        self.save_record(record)
        
        print(
            f"[Eval] step={step} | epoch={epoch:.2f} | "
            f"total_examples={total_examples} | log-PPL={log_ppl:.4f}"
        )


class HellaSwagCallback(LMTrainingCallback):
    """Callback for tracking HellaSwag accuracy during training."""
    
    def __init__(self, args, tokenizer, device="cuda", eval_steps=500, output_dir: Optional[str] = None):
        super().__init__(args, output_dir=output_dir)
        self.tokenizer = tokenizer
        self.device = device
        self.eval_steps = eval_steps
        self.last_eval_step = 0
        self.evaluation_type = "hellaswag"

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        if state.global_step - self.last_eval_step >= self.eval_steps:
            self._evaluate(state, model)
            self.last_eval_step = state.global_step

    def _evaluate(self, state, model):
        """Run HellaSwag evaluation."""
        try:
            from datasets import load_dataset

            step = state.global_step
            epoch = state.epoch
            total_examples = self.get_total_examples(step)

            dataset = load_dataset("hellaswag", split="validation")
            dataset = dataset.select(range(min(50, len(dataset))))

            model.eval()
            correct = 0

            for example in dataset:
                context = example["ctx"]
                endings = example["endings"]
                label = int(example["label"])
                likelihoods = []

                for ending in endings:
                    input_text = context + " " + ending
                    inputs = self.tokenizer(
                        input_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    ).to(self.device)

                    with torch.no_grad():
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        likelihoods.append(-outputs.loss.cpu().item())

                predicted_idx = np.argmax(likelihoods)
                if predicted_idx == label:
                    correct += 1

            accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
            
            record = {
                "step": step,
                "epoch": epoch,
                "total_examples": total_examples,
                "hellaswag_accuracy": accuracy,
            }
            self.save_record(record)

            print(f"[Eval] step={step} | epoch={epoch:.2f} | HellaSwag Accuracy={accuracy:.4f}")
            model.train()

        except Exception as e:
            print(f"Error computing HellaSwag accuracy: {e}")


def unified_train_all(
    model_type: str,
    models_outdir: str,
    eval_type: str = "flores200_perplexity",
    dataset_sizes: List[str] = None,
    train_fn: Callable = None,
    lang_sets: Dict[str, set] = None,
    # Model-specific callbacks
    get_tokenizer_path_fn: Callable = None,
    get_model_output_dir_fn: Callable = None,
    get_config_path_fn: Callable = None,
    get_grad_accum_fn: Callable = None,
    get_eval_steps_fn: Callable = None,
    get_train_instances_fn: Callable = None,  # Returns list of (dropout, suffix) tuples
) -> None:
    """
    Unified train_all orchestration for both GPT2 and Diffusion models.
    
    Handles all common training_all logic. Model-specific behavior is provided via callbacks.
    
    Args:
        model_type: "gpt2" or "diffusion"
        models_outdir: Output directory for models
        eval_type: Evaluation type ("flores200_perplexity" or "hellaswag")
        dataset_sizes: List of dataset sizes (default: all)
        train_fn: Model-specific train() function to call
        lang_sets: Dictionary mapping dataset_size to set of languages
        get_tokenizer_path_fn: Callable(lang, dataset_size, dataset_config) -> tokenizer_path
        get_model_output_dir_fn: Callable(model_dir, lang, dataset_size) -> output_dir
        get_config_path_fn: Callable(model_size) -> config_path
        get_grad_accum_fn: Callable(batch_size) -> (batch_per_device, grad_accum_steps)
        get_eval_steps_fn: Callable(dataset_size, max_steps, config) -> eval_steps
        get_train_instances_fn: Callable(config) -> [(dropout, model_output_dir, model_name), ...]
    """
    from util.constants import LANG_SETS as DEFAULT_LANG_SETS
    
    os.makedirs(models_outdir, exist_ok=True)
    
    if dataset_sizes is None:
        dataset_sizes = ['5mb', '10mb', '100mb', '1000mb']
    
    if lang_sets is None:
        lang_sets = DEFAULT_LANG_SETS
    
    # Setup configurations
    dataset_config = DatasetConfig(tokens_per_sequence=MAX_SEQ_LEN)
    training_sizes = TrainingSizes.from_file(dataset_config.train_sizes_path)
    hyperparams = HyperparametersConfig()
    
    print(f"Training {model_type.upper()} models with evaluation type: {eval_type}")
    print(f"Output directory: {models_outdir}")
    
    # Use defaults if not provided
    if get_tokenizer_path_fn is None:
        get_tokenizer_path_fn = lambda lang, size, dc: os.path.join(dc.tokenizers_dir, f'{lang}_{size}')
    
    if get_model_output_dir_fn is None:
        get_model_output_dir_fn = lambda md, lang, size: os.path.join(md, f'{model_type}_{lang}')
    
    if get_config_path_fn is None:
        print('model_type', model_type)
        to_model_name = {
            'gpt2': 'GPT2',
            'diffusion': 'Diffusion'
        }
        to_model_type = {
            'gpt2': 'gpt',
            'diffusion': 'diffusion'
        }
        get_config_path_fn = lambda ms: f'models/{to_model_name.get(model_type)}/{to_model_type.get(model_type)}_{ms}_config.json'
    
    if get_grad_accum_fn is None:
        get_grad_accum_fn = lambda bs: (bs, 1)  # Default: no gradient accumulation
    
    if get_eval_steps_fn is None:
        get_eval_steps_fn = lambda size, steps, cfg: get_eval_steps(size, steps)
    
    if get_train_instances_fn is None:
        # Single training instance per config
        get_train_instances_fn = lambda cfg: [(True, cfg.get('output_dir'), 'model')]
    
    # Callback function for each language-dataset combination
    def train_model_callback(config):
        lang = config['lang']
        dataset_size = config['dataset_size']
        model_size = config['model_size']
        batch_size = config['batch_size']
        max_steps = config['max_steps']
        warmup_steps = config['warmup_steps']
        
        model_dir = os.path.join(models_outdir, dataset_size)
        os.makedirs(model_dir, exist_ok=True)
        
        print('model_size', model_size)
        config_path = get_config_path_fn(model_size)
        print('config_path', config_path)
        batch_per_device, gradient_accumulation_steps = get_grad_accum_fn(batch_size)
        eval_steps = get_eval_steps_fn(dataset_size, max_steps, config)
        
        print(f"  {lang}: max_steps={max_steps}, warmup_steps={warmup_steps}, "
              f"batch_size={batch_per_device}, eval_steps={eval_steps}")
        
        # Get training instances (may be multiple for dropout experiments)
        train_instances = get_train_instances_fn(config)
        
        for dropout, model_output_dir, model_name in train_instances:
            print(f"    Training {model_name}...")
            train_fn(
                output_dir=model_output_dir,
                config_name=config_path,
                tokenizer_name=config['tokenizer_path'],
                training_file_path=config['train_data_path'],
                eval_file_path=config['eval_data_path'],
                override_n_examples=config['n_tokens'],
                max_steps=max_steps,
                warmup_steps=warmup_steps,
                epochs=hyperparams.epochs,
                gradient_accumulation_steps=gradient_accumulation_steps,
                batch_size=batch_per_device,
                dropout=dropout,
                eval_steps=eval_steps,
                evaluation_type=eval_type
            )
    
    # Iterate over all language-dataset combinations
    for dataset_size in dataset_sizes:
        model_size = select_model_size(dataset_size)
        langs = lang_sets[dataset_size]
        
        print(f"\n{'='*60}")
        print(f"Processing {model_size.upper()} models for {dataset_size.upper()}")
        print(f"{'='*60}")
        
        for lang in langs:
            batch_size = hyperparams.batch_sizes[dataset_size]
            
            train_data_path = dataset_config.get_train_path(lang, dataset_size)
            eval_data_path = dataset_config.get_eval_path(lang)
            tokenizer_path = get_tokenizer_path_fn(lang, dataset_size, dataset_config)
            
            if not validate_training_config(lang, dataset_size, train_data_path, tokenizer_path=tokenizer_path):
                continue
            
            n_tokens = training_sizes.get(f'{lang}_{dataset_size}', -1)
            tokens_per_batch = batch_size * dataset_config.tokens_per_sequence
            max_steps, warmup_steps = calculate_training_steps(
                total_tokens=n_tokens,
                tokens_per_batch=tokens_per_batch,
                epochs=hyperparams.epochs,
                warmup_proportion=hyperparams.warmup_proportion
            )
            
            config = {
                'lang': lang,
                'dataset_size': dataset_size,
                'model_size': model_size,
                'batch_size': batch_size,
                'max_steps': max_steps,
                'warmup_steps': warmup_steps,
                'n_tokens': n_tokens,
                'train_data_path': train_data_path,
                'eval_data_path': eval_data_path,
                'tokenizer_path': tokenizer_path,
                'eval_dataset_exists': os.path.isfile(eval_data_path),
            }
            
            train_model_callback(config)
    
    print(f"\n✓ {model_type.upper()} model training completed!")



@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    Shared by GPT2, Diffusion, and other LM training scripts.
    - Collates batches of tensors, honoring their tokenizer's pad_token
    - Applies padding and creates attention masks
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = False

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        
        # For language modeling, inputs and labels are the same (shifted handled by model)
        inputs = batch
        labels = batch.clone().detach()
        
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        attention_mask = inputs != self.tokenizer.pad_token_id
        
        if self.tokenizer.pad_token_id is None:
            attention_mask = inputs != -1
            labels[labels == -1] = -100
            inputs[inputs == -1] = 1
        
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        """Convert examples to tensor batch with proper padding."""
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True, padding_value=-1)
            return pad_sequence(
                examples,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )


class IterableTextDataset(IterableDataset):
    """
    Iterable dataset for efficiently streaming pre-tokenized text data.
    Shared by GPT2, Diffusion, and other LM training scripts.
    
    Each line in the input file should be space-separated token IDs.
    [CLS] and [SEP] tokens should already be included in the tokenized data.
    """

    class ExampleIterator:
        """Iterator for reading examples from a file one at a time."""
        
        def __init__(
            self, file_path: str, block_size: int,
            pad_token_id: int, sep_token_id: int
        ):
            self.file_path = file_path
            self.block_size = block_size
            self.pad_token_id = pad_token_id
            self.sep_token_id = sep_token_id
            self.input_file = codecs.open(file_path, 'rb', encoding='utf-8')

        def __iter__(self):
            return self

        def __next__(self):
            return self._next_example()

        def _next_example(self):
            """Read and process one example."""
            if self.input_file is None:
                raise StopIteration
            
            example_string = self.input_file.readline()
            while example_string == '\n':  # Skip blank lines
                example_string = self.input_file.readline()

            if example_string == '':  # End of file
                self.input_file.close()
                self.input_file = None
                raise StopIteration
            
            example_string = example_string.strip()
            example = [int(token_id) for token_id in example_string.split()]
            
            # Truncate to block size
            if len(example) > self.block_size:
                example = example[0:self.block_size]
            
            return {"input_ids": torch.tensor(example, dtype=torch.long)}

    def __init__(
        self,
        file_path: str,
        block_size: int,
        pad_token_id: int,
        sep_token_id: int,
        n_examples: int = -1,
    ):
        super().__init__()
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        
        self.input_filepath = file_path
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        # Count examples if not provided
        if n_examples > 0:
            self.num_examples = n_examples
        else:
            print("Counting examples in training file (this may be slow)...")
            example_count = 0
            with codecs.open(file_path, 'rb', encoding='utf-8') as infile:
                for _ in tqdm(infile):
                    example_count += 1
            self.num_examples = example_count
            print(f"Found {example_count} examples")

    def __iter__(self):
        return self.ExampleIterator(
            self.input_filepath,
            self.block_size,
            self.pad_token_id,
            self.sep_token_id
        )

    def __len__(self):
        return self.num_examples


def validate_paths(paths: Dict[str, str]) -> bool:
    """Validate that all required paths exist."""
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ Missing path: {name} = {path}")
            return False
    return True


def iterate_dataset_languages(
    dataset_sizes: list,
    lang_sets: Dict[str, set],
    dataset_config: DatasetConfig,
    training_sizes: TrainingSizes,
    hyperparams: HyperparametersConfig,
    callback_fn: Callable,
):
    """
    Common iteration pattern for training scripts.
    
    Args:
        dataset_sizes: List of dataset sizes to train on (e.g., ['5mb', '10mb', ...])
        lang_sets: Dictionary mapping size to set of languages
        dataset_config: Dataset configuration object
        training_sizes: Training token counts
        hyperparams: Hyperparameters configuration
        callback_fn: Function called for each (lang, size) combination with computed config
    """
    for dataset_size in dataset_sizes:
        model_size = select_model_size(dataset_size)
        langs = lang_sets[dataset_size]
        
        print(f"\n{'='*60}")
        print(f"Processing {model_size.upper()} models for {dataset_size.upper()}")
        print(f"{'='*60}")
        
        for lang in langs:
            # Get batch size
            batch_size = hyperparams.batch_sizes[dataset_size]
            
            # Get paths
            train_data_path = dataset_config.get_train_path(lang, dataset_size)
            eval_data_path = dataset_config.get_eval_path(lang)
            tokenizer_path = dataset_config.get_tokenizer_path(lang, dataset_size)
            
            # Check paths exist
            if not os.path.isfile(train_data_path):
                print(f"⚠️  Skip {lang} ({dataset_size}): train data not found")
                continue
            
            # Get token count
            n_tokens = training_sizes.get(f'{lang}_{dataset_size}', -1)
            
            # Calculate steps
            tokens_per_batch = batch_size * dataset_config.tokens_per_sequence
            max_steps, warmup_steps = calculate_training_steps(
                total_tokens=n_tokens,
                tokens_per_batch=tokens_per_batch,
                epochs=hyperparams.epochs,
                warmup_proportion=hyperparams.warmup_proportion
            )
            
            # Prepare config dict for callback
            config = {
                'lang': lang,
                'dataset_size': dataset_size,
                'model_size': model_size,
                'batch_size': batch_size,
                'max_steps': max_steps,
                'warmup_steps': warmup_steps,
                'n_tokens': n_tokens,
                'train_data_path': train_data_path,
                'eval_data_path': eval_data_path,
                'tokenizer_path': tokenizer_path,
                'eval_dataset_exists': os.path.isfile(eval_data_path),
            }
            
            # Call the callback with the computed configuration
            callback_fn(config)


def get_eval_steps(dataset_size: str, max_steps: int) -> int:
    """Get evaluation step frequency based on dataset size."""
    # Evaluate more often for smaller datasets
    eval_frequency = 0.025
    # if dataset_size in ['5mb', '10mb']:
    #     eval_frequency = 0.1  # Evaluate 10× per training
    # elif dataset_size == '100mb':
    #     eval_frequency = 0.05  # Evaluate 5× per training
    # else:  # 1000mb
    #     eval_frequency = 0.02  # Evaluate 2× per training
    
    eval_steps = max(10, int(max_steps * eval_frequency))
    return eval_steps


def run_unified_training(
    model_type: str,
    models_outdir: str,
    eval_type: str = "flores200_perplexity",
    dataset_sizes: List[str] = None,
    tokenizers_dir: str = 'dataset/tokenization/spm/',
):
    """
    Unified training orchestration for both GPT2 and Diffusion models.
    
    Reduces code duplication between train_all.py scripts.
    Both models call this with their specific train() function.
    
    Args:
        model_type: "gpt2" or "diffusion"
        models_outdir: Output directory for trained models
        eval_type: Evaluation type ("flores200_perplexity" or "hellaswag")
        dataset_sizes: List of dataset sizes to train
        tokenizers_dir: Directory containing tokenizers
    """
    from util.constants import LANG_SETS, MAX_SEQ_LEN
    import argparse
    
    if dataset_sizes is None:
        dataset_sizes = ['5mb', '10mb', '100mb', '1000mb']
    
    # Setup
    os.makedirs(models_outdir, exist_ok=True)
    
    dataset_config = DatasetConfig(tokens_per_sequence=MAX_SEQ_LEN)
    if model_type == "gpt2":
        dataset_config.tokenizers_dir = tokenizers_dir
    
    training_sizes = TrainingSizes.from_file(dataset_config.train_sizes_path)
    hyperparams = HyperparametersConfig()
    
    print(f"Training {model_type.upper()} models with evaluation type: {eval_type}")
    print(f"Output directory: {models_outdir}")
    
    # Return configuration for use by specific train_all.py scripts
    return {
        'dataset_config': dataset_config,
        'training_sizes': training_sizes,
        'hyperparams': hyperparams,
        'dataset_sizes': dataset_sizes,
        'eval_type': eval_type,
    }


def load_eval_records(output_dir: str) -> List[Dict]:
    """
    Load evaluation records from JSONL file saved by callbacks.
    
    Args:
        output_dir: Training output directory containing eval_records.jsonl
        
    Returns:
        List of evaluation record dictionaries
    """
    eval_file = os.path.join(output_dir, 'eval_records.jsonl')
    records = []
    
    if os.path.exists(eval_file):
        try:
            with open(eval_file, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Failed to load eval records from {eval_file}: {e}")
    
    return records


def validate_training_config(
    lang: str,
    dataset_size: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> bool:
    """
    Validate that all required training files exist.
    
    Args:
        lang: Language code
        dataset_size: Dataset size (e.g., "5mb", "10mb")
        train_data_path: Path to training data
        eval_data_path: Optional path to eval data
        tokenizer_path: Optional path to tokenizer
        
    Returns:
        True if all required files exist, False otherwise
    """
    if not os.path.isfile(train_data_path):
        print(f"⚠️  Skip {lang} ({dataset_size}): train data not found at {train_data_path}")
        return False
    
    if tokenizer_path and not os.path.isdir(tokenizer_path):
        print(f"⚠️  Skip {lang} ({dataset_size}): tokenizer not found at {tokenizer_path}")
        return False
    
    return True


def unified_train(
    output_dir: str,
    tokenizer_name: str,
    training_file_path: str,
    eval_file_path: Optional[str],
    override_n_examples: int,
    max_steps: int,
    warmup_steps: int,
    epochs: int,
    dropout: bool,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 4,
    eval_steps: int = 10,
    evaluation_type: str = "flores200_perplexity",
    # Model-specific callbacks
    load_config_fn: Callable = None,
    load_model_fn: Callable = None,
    block_size: int = 512,
    seed: int = 43,
) -> None:
    """
    Unified training function for language models (GPT2, Diffusion, etc.)
    
    Handles all common training logic. Model-specific code is provided via callbacks.
    This dramatically reduces code duplication between model-specific train.py files.
    
    Args:
        output_dir: Output directory for model checkpoints
        tokenizer_name: Path to tokenizer
        training_file_path: Path to training data
        eval_file_path: Path to evaluation data (optional)
        override_n_examples: Number of training examples
        max_steps: Maximum training steps
        warmup_steps: Number of warmup steps
        epochs: Number training epochs
        dropout: Whether to use dropout
        gradient_accumulation_steps: Gradient accumulation
        batch_size: Batch size per device
        eval_steps: Evaluation frequency
        evaluation_type: "flores200_perplexity" or "hellaswag"
        load_config_fn: Callable(config_path) -> config. Model-specific config loading.
        load_model_fn: Callable(config, dropout) -> model. Model-specific model creation.
        block_size: Sequence length
        seed: Random seed
    """
    # Check if model already exists
    final_model_path = os.path.join(output_dir, 'pytorch_model.bin')
    if os.path.isdir(output_dir) and os.path.isfile(final_model_path):
        print('Model already found: {}'.format(final_model_path))
        return

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and contains no checkpoints. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Set seed
    set_seed(seed)

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Attempting to use local sentencepiece model as tokenizer: {e}")
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_name)

    # Load and configure model (model-specific)
    if load_config_fn is None or load_model_fn is None:
        raise ValueError("load_config_fn and load_model_fn callbacks are required")
    
    config = load_config_fn(tokenizer)
    if not dropout:
        # Disable dropout where available
        for attr in ['attn_pdrop', 'embd_pdrop', 'resid_pdrop', 'hidden_dropout_prob', 'attention_probs_dropout_prob']:
            if hasattr(config, attr):
                setattr(config, attr, 0.0)
    
    model = load_model_fn(config)
    model.resize_token_embeddings(len(tokenizer))
    
    # Ensure model is in training mode
    model.train()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"NUM TRAINABLE MODEL PARAMETERS: {num_params}")

    # Load datasets
    train_dataset = IterableTextDataset(
        training_file_path,
        block_size,
        tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        n_examples=override_n_examples
    )
    train_dataset = RepeatingIterableDataset(train_dataset, epochs)
    
    eval_dataset = None
    if eval_file_path and os.path.isfile(eval_file_path):
        eval_dataset = IterableTextDataset(
            eval_file_path,
            block_size,
            tokenizer.pad_token_id,
            sep_token_id=tokenizer.sep_token_id,
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Determine evaluation strategy
    if evaluation_type == "hellaswag":
        eval_strategy = IntervalStrategy.NO
        eval_data = None
    else:  # flores200_perplexity
        eval_strategy = IntervalStrategy.STEPS
        eval_data = eval_dataset

    print('training args epoch', epochs)
    # Training arguments
    args = TrainingArguments(
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        disable_tqdm=False,
        eval_accumulation_steps=2,
        eval_steps=eval_steps,
        eval_strategy=eval_strategy,
        fp16=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=0.0001,
        log_level="info",
        logging_steps=100,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        max_steps=max_steps,
        num_train_epochs=epochs,
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        remove_unused_columns=False,
        resume_from_checkpoint=last_checkpoint,
        save_steps=500,
        save_strategy=IntervalStrategy.STEPS,
        save_total_limit=2,
        seed=seed,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        do_eval=evaluation_type == "flores200_perplexity",
        use_mps_device=True,
    )

    # Initialize callbacks for BOTH evaluation types
    if torch.backends.mps.is_available():
        device_str = "mps"
    elif torch.cuda.is_available():
        device_str = "cuda"
    else:
        device_str = "cpu"
    
    # Always create both callbacks to save all evaluation types
    hellaswag_callback = HellaSwagCallback(
        args=args,
        tokenizer=tokenizer,
        device=torch.device(device_str),
        eval_steps=eval_steps,
        output_dir=output_dir
    )
    
    perplexity_callback = PerplexityCallback(args, output_dir=output_dir)
    
    # Determine which callbacks to use based on evaluation strategy
    if evaluation_type == "hellaswag":
        # Use HellaSwag for training (perplexity still records if eval data available)
        active_callbacks = [hellaswag_callback, perplexity_callback]
    else:  # flores200_perplexity
        # Use Perplexity for training (HellaSwag still evaluates on its own schedule)
        active_callbacks = [perplexity_callback, hellaswag_callback]

    # Create and run trainer
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_data,
        callbacks=active_callbacks,
    )

    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        train_result = trainer.train()

    print(train_result)
    trainer.save_model()

    # Visualize and save metrics from both callbacks
    all_records = {}
    
    # Collect perplexity records
    if len(perplexity_callback.records) > 0:
        for r in perplexity_callback.records:
            key = r.get("step")
            if key not in all_records:
                all_records[key] = r.copy()
            else:
                all_records[key].update(r)
    
    # Collect hellaswag records
    if len(hellaswag_callback.records) > 0:
        for r in hellaswag_callback.records:
            key = r.get("step")
            if key not in all_records:
                all_records[key] = r.copy()
            else:
                all_records[key].update(r)
    
    # Sort records by step
    records = sorted(all_records.values(), key=lambda x: x.get("step", 0))
    
    if len(records) > 0:
        epochs_list = [r["epoch"] for r in records]
        total_examples_list = [r["total_examples"] for r in records]
        
        # Create figure with subplots for both metrics if both exist
        has_ppl = any("log_ppl" in r for r in records)
        has_hellaswag = any("hellaswag_accuracy" in r for r in records)
        
        if has_ppl and has_hellaswag:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            log_ppl = [r.get("log_ppl") for r in records if "log_ppl" in r]
            ppl_examples = [r["total_examples"] for r in records if "log_ppl" in r]
            ax1.plot(ppl_examples, log_ppl, marker="o")
            ax1.set_ylabel("Log Perplexity")
            ax1.set_xlabel("Total Examples")
            ax1.set_title("Scaling Curve: Examples vs Log-Perplexity")
            ax1.grid(True)
            
            hellaswag_accuracy = [r.get("hellaswag_accuracy") for r in records if "hellaswag_accuracy" in r]
            hs_examples = [r["total_examples"] for r in records if "hellaswag_accuracy" in r]
            ax2.plot(hs_examples, hellaswag_accuracy, marker="o")
            ax2.set_ylabel("HellaSwag Accuracy")
            ax2.set_xlabel("Total Examples")
            ax2.set_title("Scaling Curve: Examples vs HellaSwag Accuracy")
            ax2.grid(True)
            
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'figure.pdf'))
            plt.close(fig)
            
            # Save both metrics to npz
            np.savez(
                os.path.join(output_dir, "scaling_records.npz"),
                epochs=np.array(epochs_list),
                total_examples=np.array(total_examples_list),
                log_ppl=np.array([r.get("log_ppl", np.nan) for r in records]),
                hellaswag_accuracy=np.array([r.get("hellaswag_accuracy", np.nan) for r in records]),
            )
            
            print(f"Saved figure with {len(records)} evaluation points (both metrics)")
        
        elif has_ppl:
            fig, ax = plt.subplots()
            log_ppl = [r["log_ppl"] for r in records]
            ax.plot(total_examples_list, log_ppl, marker="o")
            ax.set_ylabel("Log Perplexity")
            ax.set_title("Scaling Curve: Examples vs Log-Perplexity")
            ax.set_xlabel("Total Examples")
            ax.grid(True)
            fig.savefig(os.path.join(output_dir, 'figure.pdf'))
            plt.close(fig)
            
            np.savez(
                os.path.join(output_dir, "scaling_records.npz"),
                epochs=np.array(epochs_list),
                total_examples=np.array(total_examples_list),
                log_ppl=np.array(log_ppl),
            )
            
            print(f"Saved figure with {len(records)} evaluation points (perplexity only)")
        
        elif has_hellaswag:
            fig, ax = plt.subplots()
            hellaswag_accuracy = [r["hellaswag_accuracy"] for r in records]
            ax.plot(total_examples_list, hellaswag_accuracy, marker="o")
            ax.set_ylabel("HellaSwag Accuracy")
            ax.set_title("Scaling Curve: Examples vs HellaSwag Accuracy")
            ax.set_xlabel("Total Examples")
            ax.grid(True)
            fig.savefig(os.path.join(output_dir, 'figure.pdf'))
            plt.close(fig)
            
            np.savez(
                os.path.join(output_dir, "scaling_records.npz"),
                epochs=np.array(epochs_list),
                total_examples=np.array(total_examples_list),
                hellaswag_accuracy=np.array(hellaswag_accuracy),
            )
            
            print(f"Saved figure with {len(records)} evaluation points (HellaSwag only)")
    else:
        print("WARNING: No evaluation records were collected!")
