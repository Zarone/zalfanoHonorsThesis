from dataset.fetch_scripts.data_source import DataSource
from util.util import cap_script_iso639, iso639_3_to_iso639_1, \
        get_output_for_lang
from util.constants import all_languages, max_size, BYTES_PER_MB
from dataset.fetch_scripts.stream.stream_madlad import stream_madlad
from dataset.fetch_scripts.stream.stream_glot import stream_glot
from util.byte_premiums import byte_premiums


def merge_in_order(
    language: str,
    sources: type(DataSource),
    max_size_mb: float,
    byte_premium: float
):
    this_output_file = get_output_for_lang(language)

    max_size_mb_adjusted = max(max_size_mb, max_size_mb*byte_premium)
    max_size_bytes = max_size_mb_adjusted * BYTES_PER_MB

    print(
      f"Started writing to {this_output_file}, \
with anticipated size {max_size_bytes}..."
    )
    with open(this_output_file, 'w', encoding='utf-8') as f:

        # in bytes
        written_size = 0

        for source in sources:
            print("Started viewing source with stream:", source.stream)
            source.set_stream(language, max_size_mb_adjusted)
            for line, _ in source:
                new_text = line['text'].strip()+'\n'
                size = len(new_text.encode('utf-8'))
                f.write(new_text)
                written_size += size
                if written_size >= max_size_bytes:
                    return
        print(
            f"Finished source {source.stream} with written_size {written_size}"
        )


def main():
    # describe sources, maybe as their own class, and they take in a
    # formatting function from iso639-3. use streaming and buffered shuffling
    sources = [
        # DataSource(nllb_script, stream_nllb),
        DataSource(iso639_3_to_iso639_1, stream_madlad),
        DataSource(cap_script_iso639, stream_glot),
    ]

    # merge in the order: nllb, Glot500, MADLAD-400
    for language in all_languages:
        byte_premium = byte_premiums.get(language)
        print(f'language {language} has byte premium {byte_premium}')

        # Since we dedup later, this is to ensure the final size
        # is large enough. We also limit the size per line later,
        # so this has to be pretty high
        safety_margin = 3.5

        merge_in_order(
            language,
            sources,
            max_size*safety_margin,
            byte_premium
        )


if __name__ == '__main__':
    main()
