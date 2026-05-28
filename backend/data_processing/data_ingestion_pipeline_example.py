"""
Data ingestion pipeline example.

Reads raw conversation datasets, preprocesses them via preprocess.py,
and writes cleaned outputs to the processed_data/ directory.

Usage:
    cd backend
    python -m data_processing.data_ingestion_pipeline_example
"""

import json
from pathlib import Path

import pandas as pd

from .preprocess import preprocess_3k_conversations, preprocess_rsics, preprocess_twitter

logger = __import__('logging').getLogger(__name__)


def process_rsics_data(input_path: str, output_dir: str) -> None:
    """Load, preprocess, and save RSICS conversation data."""
    df = pd.read_csv(input_path)
    processed_df = preprocess_rsics(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out / 'conversations.csv', index=False)
    logger.info('RSICS data saved to %s', out / 'conversations.csv')


def process_3k_conversations_data(input_path: str, output_dir: str) -> None:
    """Load, preprocess, and save 3K conversations data."""
    with open(input_path) as f:
        data = json.load(f)

    processed = preprocess_3k_conversations(data)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'conversations.json', 'w') as f:
        json.dump(processed, f, indent=2)
    logger.info('3K conversations data saved to %s', out / 'conversations.json')


def process_twitter_data(input_path: str, output_dir: str) -> None:
    """Load, preprocess, and save Twitter customer support data."""
    df = pd.read_csv(input_path)
    processed_df = preprocess_twitter(df)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out / 'tweets_and_replies.csv', index=False)
    logger.info('Twitter data saved to %s', out / 'tweets_and_replies.csv')


def create_metadata(output_base: str) -> None:
    """Write metadata.json describing all processed datasets."""
    metadata = {
        'rsics': {
            'files': ['conversations.csv'],
            'format': 'CSV',
            'description': 'Processed RSiCS dataset',
        },
        '3k_conversations': {
            'files': ['conversations.json'],
            'format': 'JSON',
            'description': 'Processed 3K Conversations dataset',
        },
        'twitter': {
            'files': ['tweets_and_replies.csv'],
            'format': 'CSV',
            'description': 'Processed Twitter Customer Support dataset',
        },
    }
    out = Path(output_base)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info('Metadata saved to %s', out / 'metadata.json')


def run_pipeline(base_dir: str = 'data_processing') -> None:
    """
    Run the full ingestion pipeline.

    Looks for raw data in <base_dir>/raw_data/ and writes to <base_dir>/processed_data/.
    """
    raw = Path(base_dir) / 'raw_data'
    processed = Path(base_dir) / 'processed_data'

    datasets = [
        (raw / 'rsics.csv', processed / 'rsics', process_rsics_data),
        (raw / '3k_conversations.json', processed / '3k_conversations', process_3k_conversations_data),
        (raw / 'twitter_support.csv', processed / 'twitter', process_twitter_data),
    ]

    for input_path, output_path, processor in datasets:
        if input_path.exists():
            logger.info('Processing %s...', input_path.name)
            processor(str(input_path), str(output_path))
        else:
            logger.warning('Skipping %s — file not found at %s', input_path.name, input_path)

    create_metadata(str(processed))
    logger.info('Pipeline complete.')


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_pipeline()
