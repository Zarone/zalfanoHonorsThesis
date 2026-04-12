import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AlbertTokenizer,
    TrainerCallback
)

from util.training_utils import unified_train
from util.constants import MAX_SEQ_LEN

seed = 43


class HellaSwagCallback(TrainerCallback):
    """GPT2-specific callback for HellaSwag evaluation."""
    def __init__(self, args, tokenizer, device="cuda", eval_steps=500, output_dir=None):
        self.batch_size = args.per_device_train_batch_size
        self.grad_accum = args.gradient_accumulation_steps
        self.records = []
        self.tokenizer = tokenizer
        self.device = device
        self.eval_steps = eval_steps
        self.last_eval_step = 0
        self.evaluation_type = "hellaswag"
        self.output_dir = output_dir
        if output_dir:
            self.eval_file = os.path.join(output_dir, 'eval_records.jsonl')
        else:
            self.eval_file = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step - self.last_eval_step >= self.eval_steps:
            self._evaluate(state, model)
            self.last_eval_step = state.global_step

    def _evaluate(self, state, model):
        try:
            from datasets import load_dataset
            import json
            import numpy as np

            step = state.global_step
            epoch = state.epoch
            total_examples = step * self.batch_size * self.grad_accum

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
            self.records.append(record)
            
            if self.eval_file:
                with open(self.eval_file, 'a') as f:
                    f.write(json.dumps(record) + '\n')

            print(f"[Eval] step={step} | epoch={epoch:.2f} | HellaSwag Accuracy={accuracy:.4f}")
            model.train()

        except Exception as e:
            print(f"Error computing HellaSwag accuracy: {e}")


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
    evaluation_type="flores200_perplexity"
):
    """
    Train a GPT2 language model using unified training infrastructure.
    
    Model-specific: Config and model loading. All other logic is shared.
    """
    def load_config(tokenizer):
        """Load and configure GPT2 model config."""
        print('config_name', config_name)
        config = AutoConfig.from_pretrained(config_name)
        config.bos_token_id = tokenizer.cls_token_id
        config.eos_token_id = tokenizer.sep_token_id
        config.pad_token_id = tokenizer.pad_token_id
        config.vocab_size = len(tokenizer)
        return config
    
    def load_model(config):
        """Create GPT2 model from config."""
        return AutoModelForCausalLM.from_config(config=config)
    
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
        block_size=MAX_SEQ_LEN,  # GPT2 max position embeddings
        seed=seed,
    )
