"""
Evaluate a model on the HellaSwag dataset.
HellaSwag is a dataset of multiple-choice commonsense inference questions.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple
import os


def evaluate_hellaswag(
    model_path: str,
    tokenizer_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
    num_examples: int = None
) -> Dict[str, float]:
    """
    Evaluate a model on the HellaSwag dataset.
    
    Args:
        model_path: Path to the trained model
        tokenizer_path: Path to the tokenizer
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        num_examples: Maximum number of examples to evaluate on (None for all)
    
    Returns:
        Dictionary with evaluation metrics (accuracy, etc.)
    """
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
    model.eval()
    
    print(f"Loading tokenizer from {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load HellaSwag dataset
    print("Loading HellaSwag dataset...")
    dataset = load_dataset("hellaswag", split="validation")
    
    if num_examples is not None:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    correct = 0
    total = 0
    
    print(f"Evaluating on {len(dataset)} examples...")
    
    for example in tqdm(dataset, total=len(dataset)):
        context = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])
        
        # Compute likelihood for each ending
        likelihoods = []
        
        for ending in endings:
            input_text = context + " " + ending
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Get log probabilities
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                # Convert loss to likelihood (lower loss = higher likelihood)
                # Move loss to CPU for item() extraction (MPS safety)
                likelihood = -loss.cpu().item()
            
            likelihoods.append(likelihood)
        
        # Select the ending with highest likelihood
        predicted_idx = np.argmax(likelihoods)
        
        if predicted_idx == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
    
    return results


def compute_hellaswag_accuracy(model, tokenizer, device: str = "cuda"):
    """
    Compute HellaSwag accuracy for a model during training.
    This is a simplified version for use in training callbacks.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        device: Device to run evaluation on
    
    Returns:
        HellaSwag accuracy score
    """
    try:
        dataset = load_dataset("hellaswag", split="validation")
        # Sample a subset for faster evaluation during training
        dataset = dataset.select(range(min(100, len(dataset))))
        
        model.eval()
        correct = 0
        
        for example in dataset:
            context = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])
            
            likelihoods = []
            
            for ending in endings:
                input_text = context + " " + ending
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    # Move loss to CPU for item() extraction (MPS safety)
                    likelihoods.append(-loss.cpu().item())
            
            predicted_idx = np.argmax(likelihoods)
            if predicted_idx == label:
                correct += 1
        
        accuracy = correct / len(dataset) if len(dataset) > 0 else 0.0
        return accuracy
    
    except Exception as e:
        print(f"Error computing HellaSwag accuracy: {e}")
        return 0.0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate a model on HellaSwag"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples to evaluate on")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    
    args = parser.parse_args()
    
    results = evaluate_hellaswag(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size,
        num_examples=args.num_examples
    )
    
    print("\nResults:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")
