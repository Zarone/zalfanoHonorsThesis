## Get Data

To get the raw data for training:

python -m dataset.fetch_scripts.fetch_all_languages
python -m dataset.fetch_scripts.deduplication
python -m dataset.fetch_scripts.shuffle_dataset

## Tokenization

To get a usable tokenized dataset for our model, we do:

python -m dataset.tokenization.create_tokenizer_datasets
python -m dataset.tokenization.train_tokenizers

cd ./dataset/raw/flores200_dataset/
chmod +x merge.sh
./merge.sh

python -m dataset.tokenization.tokenize_datasets
python -m dataset.tokenization.sample_tokenized_datasets

