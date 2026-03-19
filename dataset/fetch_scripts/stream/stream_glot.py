from datasets import load_dataset
from util.util import utf_size


def stream_glot(
    language='eng_Lat',
    max_size=1_000
):
    """
    Stream examples from Glot500 dataset.

    Args:
        language (str): Language code ISO-639-3
        max_size (int): Maximum size to fetch. None for all.
        buffer_size (int): Size of buffer for shuffling.

    Yields:
        dict: Example with 'text' key
    """
    buffer_size = 100
    try:
        dataset = load_dataset(
            'cis-lmu/glot500',
            language,
            streaming=True,
        )['train']
        shuffled_dataset = dataset.shuffle(seed=42, buffer_size=buffer_size)
        for example in shuffled_dataset:
            if 'text' in example:
                yield example, utf_size(example['text'])
    except Exception as e:
        print(
            f"Warning: Could not download GLOT500 for {language}: {e}"
        )
        return 0
