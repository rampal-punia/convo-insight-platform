import pandas as pd
import json
from pathlib import Path
from preprocess import preprocess_rsics, preprocess_twitter, preprocess_3k_conversations


def process_rsics_data(input_path, output_dir):
    df = pd.read_csv(input_path)
    # Perform cleaning and preprocessing
    processed_df = preprocess_rsics(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(output_dir / 'conversations.csv', index=False)


def process_3k_conversations(input_path, output_dir):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Perform cleaning and preprocessing
    processed_data = preprocess_3k_conversations(data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'conversations.json', 'w') as f:
        json.dump(processed_data, f)


def process_twitter_data(input_path, output_dir):
    df = pd.read_csv(input_path)
    # Perform cleaning and preprocessing
    processed_df = preprocess_twitter(df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(output_dir / 'tweets_and_replies.csv', index=False)


# Run the processing
process_rsics_data('raw_data/rsics.csv', 'processed_data/rsics')
process_3k_conversations('raw_data/3k_conversations.json',
                         'processed_data/3k_conversations')
process_twitter_data('raw_data/twitter_support.csv', 'processed_data/twitter')

# Create metadata
metadata = {
    'rsics': {
        'files': ['conversations.csv'],
        'format': 'CSV',
        'description': 'Processed RSiCS dataset'
    },
    '3k_conversations': {
        'files': ['conversations.json'],
        'format': 'JSON',
        'description': 'Processed 3K Conversations dataset'
    },
    'twitter': {
        'files': ['tweets_and_replies.csv'],
        'format': 'CSV',
        'description': 'Processed Twitter Customer Support dataset'
    }
}

with open('processed_data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
