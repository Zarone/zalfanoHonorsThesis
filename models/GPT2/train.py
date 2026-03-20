import os
import torch
import matplotlib.pyplot as plt
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    AlbertTokenizer,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizer,
    BatchEncoding,
    set_seed,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
import codecs
from tqdm import tqdm
from torch.utils.data.dataset import IterableDataset
from typing import Dict, List, Union, Tuple
import numpy as np

seed = 43


class EpochsExamplesCallback(TrainerCallback):
    def __init__(self, args):
        self.batch_size = args.per_device_train_batch_size
        self.grad_accum = args.gradient_accumulation_steps
        self.records = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print('on evaluate')

        step = state.global_step
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        epoch = state.epoch
        total_examples = state.global_step * self.batch_size * self.grad_accum
        log_ppl = eval_loss

        self.records.append({
            "step": step,
            "epoch": epoch,
            "total_examples": total_examples,
            "log_ppl": log_ppl,
        })

        print(
            f"[Eval] step={step} | epoch={epoch:.2f} | total_examples={total_examples} | log-PPL={log_ppl:.4f}"
        )

# Data collator for language modeling.
# Includes the attention mask based on pad tokens.
@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[
            Union[
                List[int],
                torch.Tensor, Dict[str, torch.Tensor]
            ]
        ]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        batch = self._tensorize_batch(examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
        else:
            # Note: for GPT-2, the inputs/labels are automatically shifted
            # inside the model for autoregressive language modeling.
            inputs = batch
            labels = batch.clone().detach()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
        attention_mask = inputs != self.tokenizer.pad_token_id
        if self.tokenizer.pad_token_id is None:
            # Replace placeholder pad tokens, which are masked out anyways.
            attention_mask = inputs != -1
            labels[labels == -1] = -100
            # Default to token 1, but will be masked out.
            inputs[inputs == -1] = 1
        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def _tensorize_batch(
        self, examples: List[
            Union[
                List[int],
                torch.Tensor,
                Dict[str, torch.Tensor]
            ]
        ]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(
                x.size(0) == length_of_first for x in examples
            )
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                # Use -1 as a placeholder pad token.
                return pad_sequence(
                    examples, batch_first=True, padding_value=-1
                )
            return pad_sequence(
                examples,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )

    def mask_tokens(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with
        # probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True
            ) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool),
            value=0.0
        )
        if not hasattr(self.tokenizer, '_pad_token') or \
                self.tokenizer._pad_token is None:
            # Placeholder pad token.
            padding_mask = labels.eq(-1)
        else:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input
        # tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, 0.8)
        ).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(labels.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer),
            labels.shape,
            dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep
        # the masked input tokens unchanged
        return inputs, labels


class IterableTextDataset(IterableDataset):
    """
    Iterable version of TextDataset. Data must be preshuffled.
    Each line in the input file should be a string of integers separated by
    spaces.
    [CLS] and [SEP] tokens should already be included.
    Each line should correspond to one example (one or two sentences), but
    padding and truncation is handled automatically.
    """

    # This is the iterator that is returned by the iter() method.
    class ExampleIterator:
        def __init__(self, file_path: str, block_size: int,
                     pad_token_id: int, sep_token_id: int):
            self.file_path = file_path
            self.block_size = block_size
            self.pad_token_id = pad_token_id
            self.sep_token_id = sep_token_id
            # Start the input file from the beginning.
            self.input_file = codecs.open(file_path, 'rb', encoding='utf-8')

        def __iter__(self):
            return self

        def __next__(self):
            return self._next_example()

        # Get one example at a time.
        def _next_example(self):
            # In case this is called after the input file was closed.
            if self.input_file is None:
                print('No input file (or input file has been closed).')
                raise StopIteration
            # Try to read a string of space-separated integers.
            # Each example should be: [CLS] sent_1 [SEP] sent_2 [SEP]
            example_string = self.input_file.readline()
            while example_string == '\n':  # Skip new lines.
                example_string = self.input_file.readline()

            # This only occurs at the end of a file (otherwise there
            # would at least be a newline character).
            if example_string == '':
                self.input_file.close()
                print('Example iterator complete.')
                self.input_file = None
                raise StopIteration
            # Process example.
            example_string = example_string.strip()
            example = [int(token_id) for token_id in example_string.split()]
            # Truncating is done here.
            if len(example) > self.block_size:
                example = example[0:self.block_size]
            # Padding is handled by the collator.
            return {"input_ids": torch.tensor(example, dtype=torch.long)}

    # Init IterableTextDataset.
    def __init__(
        self,
        file_path: str,
        block_size: int,
        pad_token_id: int,
        sep_token_id: int,
        n_examples: int = -1,
    ):
        super(IterableTextDataset).__init__()
        assert os.path.isfile(file_path), \
            f"Input file path {file_path} not found"
        self.input_filepath = file_path
        self.block_size = block_size
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id

        # Initially get total num_examples.
        if n_examples > 0:
            self.num_examples = n_examples
        else:
            print("Counting examples in train file. This can be slow.")
            example_count = 0
            infile = codecs.open(file_path, 'rb', encoding='utf-8')
            for line in tqdm(infile):
                example_count += 1
            infile.close()
            self.num_examples = example_count
            print("Finished counting: {} examples.".format(example_count))

    def __iter__(self):
        return self.ExampleIterator(self.input_filepath, self.block_size,
                                    self.pad_token_id, self.sep_token_id)

    def __len__(self):
        return self.num_examples


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
    eval_steps=40
):
    # Check if model already exists.
    final_model_path = os.path.join(output_dir, 'pytorch_model.bin')
    if os.path.isdir(output_dir) and os.path.isfile(final_model_path):
        print('Model already found: {}'.format(final_model_path))
        return

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already "
                "exists and contains no checkpoints. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
                " To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir`"
                " to train from scratch."
            )

    # Set seed.
    set_seed(seed)

    # Load pretrained model and tokenizer.
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    else:
        print("Must provide model config file.")

    if tokenizer_name:
        tokenizer_path = tokenizer_name
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not"
            "supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        # If passing in a raw tokenizer model file,
        # assume ALBERT sentencepiece model.
        print(
            "Attempting to use local sentencepiece model file as tokenizer.",
            e
        )
        print('tokenizer_path', tokenizer_path)
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)

    # Overwrite special token ids in the configs based on the actual tokenizer
    # ids. This updated config will be saved in the output model directory.
    config.bos_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id
    # Set the vocab_size based on the tokenizer.
    config.vocab_size = len(tokenizer)

    if not dropout:
        config.attn_pdrop = 0
        config.embd_pdrop = 0
        config.resid_pdrop = 0

    # Load models.
    model = AutoModelForCausalLM.from_config(
        config=config
    )

    # By default, weights are tied between the input
    # and output token embeddings.
    model.resize_token_embeddings(config.vocab_size)

    # Print total model parameters.
    # Note that usually output word embeddings are tied to input
    # word embeddings (to save parameters).
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("NUM TRAINABLE MODEL PARAMETERS: {}".format(num_params))

    # Set the max sequence length.
    block_size = config.n_positions

    # Get datasets.
    train_dataset = IterableTextDataset(
        training_file_path,
        block_size,
        tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
        n_examples=override_n_examples
    )
    eval_dataset = IterableTextDataset(
        eval_file_path,
        block_size,
        tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print('output_dir', output_dir)
    args = TrainingArguments(
        adafactor=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-06,
        # gradient_accumulation_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=0.0001,
        max_steps=max_steps,
        num_train_epochs=epochs,
        output_dir=output_dir,
        overwrite_output_dir=False,
        per_device_eval_batch_size=4,
        # per_device_train_batch_size=2,
        per_device_train_batch_size=batch_size,
        save_only_model=False,
        save_safetensors=True,
        save_strategy=IntervalStrategy.NO,
        save_total_limit=None,
        seed=seed,
        use_mps_device=True,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        do_eval=True,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        eval_delay=0,  # Start evaluating immediately
        include_inputs_for_metrics=False,  # Reduce memory usage
        dataloader_num_workers=0
    )

    epochs_callback = EpochsExamplesCallback(args)

    # Initialize our Trainer.
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[epochs_callback],
    )

    _ = trainer.train()

    trainer.save_model()

    trainer.save_state()

    # Compute Epochs vs Log-Perplexity
    records = epochs_callback.records
    print(records)

    if len(records) > 0:
        epochs = [r["epoch"] for r in records]
        total_examples_list = [r["total_examples"] for r in records]
        log_ppl = [r["log_ppl"] for r in records]

        fig, ax = plt.subplots()

        ax.plot(total_examples_list, log_ppl, marker="o")
        # ax.plot(epochs, log_ppl, marker="o")
        # ax.set_xlabel("Epochs")
        ax.set_xlabel("Total Examples")
        ax.set_ylabel("Log Perplexity")
        ax.set_title("Scaling Curve: Epochs vs Log-Perplexity")
        ax.grid(True)

        fig.savefig(os.path.join(output_dir, 'figure.pdf'))
        plt.close(fig)

        np.savez(
            os.path.join(output_dir, "scaling_records.npz"),
            epochs=np.array(epochs),
            total_examples=np.array(total_examples_list),
            log_ppl=np.array(log_ppl),
        )
        print(f"Saved figure with {len(records)} evaluation points")
    else:
        print("WARNING: No evaluation records were collected!")
