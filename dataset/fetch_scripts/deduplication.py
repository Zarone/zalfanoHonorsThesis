from util.constants import all_languages, max_size, BYTES_PER_MB
from util.byte_premiums import byte_premiums
from util.util import get_output_for_lang, get_deduped_output_for_lang
import os
from tqdm import tqdm
import hashlib

"""
I could replace this with the dedup script mentioned in
the Goldfish paper if this is too slow:

git clone https://github.com/google-research/deduplicate-text-datasets
"""


def dedup_file(language: str, max_size_mb: float):
    INSERT_STRIDE = 29

    filepath = get_output_for_lang(language)
    output_filepath = get_deduped_output_for_lang(language)

    # max_elements_bloom = max_size_mb * BYTES_PER_MB / INSERT_STRIDE
    # bloom = BloomFilter(max_elements=max_elements_bloom, error_rate=0.1)
    hash_set = set()

    # This is the amount of bytes added
    total_size = 0

    # This is the maximum number of bytes to write
    max_bytes = max_size_mb * BYTES_PER_MB

    print(f"max_bytes to write: {max_bytes}")

    file_size = os.path.getsize(filepath)

    with open(filepath, 'r', encoding='utf-8') as f, \
            open(output_filepath, 'w+', encoding='utf-8') as dedup, \
            tqdm(
                total=file_size,
                unit="Bytes",
                unit_scale=True,
                desc=f"Deduplicating {language}"
    ) as pbar:

        for line in f:
            text = line.strip()
            if not text:
                continue

            bytes_line = len(line.encode('utf-8'))
            bytes_text = text.encode('utf-8')
            n = len(bytes_text)

            pbar.update(bytes_line)

            # Check if any 100-byte sequence in this line is already seen
            duplicate = False

            seq_hashes = []

            for i in range(0, max(1, n - 99)):
                seq = bytes_text[i:i+100]

                # seq_hash = hash(seq)
                seq_hash = hashlib.blake2b(seq, digest_size=8).digest()

                # We don't need to include all
                # that would be in stride 1.
                if i % INSERT_STRIDE == 0:
                    seq_hashes.append(seq_hash)

                if seq_hash in hash_set:
                    duplicate = True
                    break

            if duplicate:
                continue

            total_size += bytes_line

            if total_size >= max_bytes:
                break

            # Add all 100-byte sequences from this line to seen
            for seq_hash in seq_hashes:
                hash_set.add(seq_hash)

            dedup.write(line)

    if total_size < max_bytes:
        print("Did not have enough data to fill up file")


def main():
    # dedup and limit to max_size (with & without language scaling)
    # "Because there is likely significant overlap between different dataset
    # sources, we deduplicate repeated sequences of 100 UTF-8 bytes for each
    # language (Lee et al., 2022)."
    for language in all_languages:
        byte_premium = byte_premiums.get(language)

        # You can only use 512 tokens per line, so we cut off a lot of data.
        # Because of that, we need to ensure there's suffient size after that.
        safety_margin = 3

        necessary_capacity = safety_margin * max(
            max_size, max_size*byte_premium
        )
        print(
            f'language {language} has necessary capacity {necessary_capacity}'
        )
        dedup_file(language, necessary_capacity)


if __name__ == '__main__':
    main()

