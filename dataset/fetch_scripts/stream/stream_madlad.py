import gzip
import json
import random
from huggingface_hub import list_repo_files, hf_hub_download
from util.util import utf_size


def get_madlad_shards(language: str):
    """
    Get all the shard files for MADLAD-400

    Args:
        language (str): Language code ISO-639-3

    Returns:
        shard_files: list of shard files
    """
    # List all files in the dataset
    files = list_repo_files("allenai/MADLAD-400", repo_type="dataset")

    # Find shards for this language/split
    shard_files = [
        f for f in files
        if f.startswith(f"data/{language}/{language}_clean_")
        and f.endswith(".jsonl.gz")
    ]

    if not shard_files:
        print(f"No shards found for {language}/clean")
        return

    # Shuffle the order of shards for additional randomness
    random.shuffle(shard_files)
    return shard_files


def stream_madlad(
    language="eng_Lat",
    max_size=1_000
):
    """
    Stream examples from MADLAD-400 dataset.

    Args:
        language (str): Language code ISO-639-3
        max_size (int): Maximum size to fetch. None for all.
        buffer_size (int): Size of buffer for shuffling.

    Yields:
        dict: Example with 'text' key
    """
    # currently returned size in mb
    mb_written = 0

    buffer = []
    current_buffer_size = 0
    buffer_size = 1_000

    shard_files = get_madlad_shards(language)

    def yield_and_check_size(item) -> bool:
        """
        Returns true if the size exceeds the max size
        """
        nonlocal mb_written
        yield item, utf_size(item['text'])
        mb_written += len(item['text'].encode('utf-8')) / 1_000_000
        if max_size and mb_written >= max_size:
            return True
        return False

    def empty_buffer() -> bool:
        """
        Returns true if the size exceeds the max size
        """
        nonlocal buffer
        if buffer_size > 0:
            random.shuffle(buffer)
        for item in buffer:
            reached = yield from yield_and_check_size(item)
            if reached:
                return True
        buffer = []
        return False

    for shard_file in shard_files:
        try:
            print("Buffer Size", len(buffer))
            filepath = hf_hub_download(
                "allenai/MADLAD-400",
                shard_file,
                repo_type="dataset"
            )

            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    example = json.loads(line)
                    buffer.append(example)
                    current_buffer_size += utf_size(example['text'])

                    if current_buffer_size >= buffer_size:
                        reached = yield from empty_buffer()
                        if reached:
                            return

        except Exception as e:
            print(f"Error loading {shard_file}: {e}")
            continue

    # Yield remaining examples in the last buffer
    if buffer:
        yield from empty_buffer()
