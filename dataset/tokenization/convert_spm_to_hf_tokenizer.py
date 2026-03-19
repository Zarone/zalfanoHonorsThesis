"""
From word-acquisition-language-models github
https://github.com/tylerachang/word-acquisition-language-models
scripts/convert_spm_to_hf_tokenizer.py
"""

import os
import sentencepiece as spm
from transformers import AlbertTokenizer


def _parse_spm_vocab(spm_model_path):
    vocab_path = spm_model_path[:-6] + '.vocab' if spm_model_path.endswith('.model') else spm_model_path + '.vocab'
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"SentencePiece vocab file not found: {vocab_path}")
    vocab_scores = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            token = parts[0]
            score = float(parts[1]) if len(parts) > 1 else 0.0
            vocab_scores.append((token, score))
    return vocab_scores


def _build_albert_from_spm(spm_model_path, do_lower_case=False, keep_accents=False):
    print(f"Loading SentencePiece model from {spm_model_path} and building AlbertTokenizer.")
    spm_proc = spm.SentencePieceProcessor(model_file=spm_model_path)
    vocab_scores = []
    for i in range(spm_proc.get_piece_size()):
        piece = spm_proc.id_to_piece(i)
        score = 0.0
        try:
            score = float(spm_proc.get_score(i))
        except Exception:
            pass
        vocab_scores.append((piece, score))

    # Add required Albert special tokens if missing.
    required_tokens = ["[CLS]", "[SEP]", "[MASK]", "<pad>", "<unk>"]
    existing = {t for t, _ in vocab_scores}
    for tok in required_tokens:
        if tok not in existing:
            vocab_scores.append((tok, 0.0))

    tokenizer = AlbertTokenizer(
        vocab=vocab_scores,
        do_lower_case=do_lower_case,
        keep_accents=keep_accents,
    )
    return tokenizer


def convert(args):
    input_path = args.get('input')
    output_dir = args.get('output_dir')
    do_lower_case = args.get('do_lower_case', False)
    keep_accents = args.get('keep_accents', False)
    multiple_of = args.get('multiple_of', 256)

    print("Converting SPM tokenizer to HF tokenizer.")
    print(f"Input: {input_path}")
    if os.path.isdir(input_path):
        tokenizer = AlbertTokenizer.from_pretrained(input_path)
    elif os.path.isfile(input_path) and input_path.endswith('.model'):
        try:
            tokenizer = AlbertTokenizer.from_pretrained(input_path,
                                                        do_lower_case=do_lower_case,
                                                        keep_accents=keep_accents)
        except Exception:
            tokenizer = _build_albert_from_spm(input_path,
                                               do_lower_case=do_lower_case,
                                               keep_accents=keep_accents)
    else:
        tokenizer = AlbertTokenizer.from_pretrained(input_path,
                                                    do_lower_case=do_lower_case,
                                                    keep_accents=keep_accents)

    n_to_add = 0
    if len(tokenizer) % multiple_of != 0:
        n_to_add = multiple_of - (len(tokenizer) % multiple_of)

    special_tokens_list = [f"[XXXXX{i}]" for i in range(n_to_add)]
    if special_tokens_list:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens_list})
    print(f"{n_to_add} tokens added.")
    print(f"Final vocab size: {len(tokenizer)} (this should be set in the language model configs).")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print("Done.")
