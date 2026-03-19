"""
This is code from the Goldfish paper
https://github.com/tylerachang/goldfish/

Tokenizes datasets, after running train_tokenizers.py.

Assumes unshuffled text datasets exist in TEXT_DIR.
"""

import os
from util.constants import TOKENIZED_DATASET_DIR, \
        SHUFFLED_DATASET_DIR, \
        SHUFFLED_TOKENIZED_DATA, \
        MAX_SEQ_LEN, \
        EVAL_DATASET, \
        EVAL_DATASET_TOKENIZED
from dataset.tokenization.script_tokenize_datasets import run_tokenize
import subprocess

TOKENIZERS_DIR = './dataset/tokenization/monolingual'
TEXT_DIR = SHUFFLED_DATASET_DIR
OUTPUT_DIR = TOKENIZED_DATASET_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
fnames = os.listdir(TOKENIZERS_DIR)

fnames.remove('.DS_Store')

for tokenizer_name in sorted(fnames):
    outpath = os.path.join(OUTPUT_DIR, '{}.txt'.format(tokenizer_name))
    if os.path.isfile(outpath):
        print('Already found file: {}'.format(outpath))
        continue
    print('\nTokenizing for tokenizer: {}'.format(tokenizer_name))
    hf_tokenizer_path = os.path.join(TOKENIZERS_DIR, tokenizer_name)
    lang = tokenizer_name[:8]
    inpath = os.path.join(TEXT_DIR, f'{lang}.txt')
    print(f'Tokenizer path: {hf_tokenizer_path}')
    print(f'Text path: {inpath}')

    # command = """
    # python3 dataset/tokenization/word-acquisition-language-models/scripts/tokenize_dataset.py \
    # --tokenizer={0} \
    # --input_file={1} \
    # --output_file={2} \
    # --max_segments=-1 --max_seq_len={3} --max_examples=2_000_000""" \
    # .format(hf_tokenizer_path, inpath, outpath, MAX_SEQ_LEN)
    # result = os.popen(command).read()
    # print(result)

    run_tokenize({
        'tokenizer': hf_tokenizer_path,
        'input_file': inpath,
        'output_file': outpath,
        'max_segments': -1,
        'max_seq_len': MAX_SEQ_LEN,
        'max_examples': 2_000_000_000
    })


    print('Finished for tokenizer: {}'.format(tokenizer_name))

    test_set_inpath = os.path.join(EVAL_DATASET, f'{lang}.txt')
    test_set_outpath = os.path.join(EVAL_DATASET_TOKENIZED, f'{lang}.txt')

    # command = """
    # python3 dataset/tokenization/word-acquisition-language-models/scripts/tokenize_dataset.py \
    # --tokenizer={0} \
    # --input_file={1} \
    # --output_file={2} \
    # --max_segments=-1 --max_seq_len={3} --max_examples=2_000_000""" \
    # .format(hf_tokenizer_path, test_set_inpath, test_set_outpath, MAX_SEQ_LEN)

    # result = os.popen(command).read()
    # print(result)

    run_tokenize({
        'tokenizer': hf_tokenizer_path,
        'input_file': test_set_inpath,
        'output_file': test_set_outpath,
        'max_segments': -1,
        'max_seq_len': MAX_SEQ_LEN,
        'max_examples': 2_000_000_000
    })

    print('Finished for tokenizer: {}'.format(tokenizer_name))

    # Shuffle it
    filepath = outpath
    output_filepath = f'{SHUFFLED_TOKENIZED_DATA}/{tokenizer_name}.txt'
    with open(filepath, 'rb') as infile, \
            open(output_filepath, 'wb') as outfile:
        subprocess.run(
            ['./dataset/fetch_scripts/terashuf/terashuf'],
            stdin=infile,
            stdout=outfile,
            check=True
        )
