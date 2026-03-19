from util.util import get_deduped_output_for_lang, get_shuffled_output_for_lang
import subprocess
from util.constants import all_languages


def shuffle_language(language: str):
    filepath = get_deduped_output_for_lang(language)
    output_filepath = get_shuffled_output_for_lang(language)
    with open(filepath, 'rb') as infile, \
            open(output_filepath, 'wb') as outfile:
        subprocess.run(
            ['./dataset/fetch_scripts/terashuf/terashuf'],
            stdin=infile,
            stdout=outfile,
            check=True
        )


def main():
    # shuffle datasets
    for language in all_languages:
        shuffle_language(language)


if __name__ == '__main__':
    main()
