"""
Data preprocessing utilities.

This module provides data preprocessing pipeline for ingesting and cleaning
customer conversation data before NLP analysis.

Supports three data sources:
- RSICS: Retail customer service conversation logs (CSV)
- 3K Conversations: General conversation dataset (JSON)
- Twitter: Customer support tweets and replies (CSV)
"""

import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Common text cleaning patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
MULTI_SPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s.,!?\'"-]')
REPEATED_CHAR_PATTERN = re.compile(r'(.)\1{2,}')


def clean_text(text: str) -> str:
    """Clean a single text string: normalize whitespace, remove URLs/emails/special chars."""
    if not isinstance(text, str):
        return ''
    text = URL_PATTERN.sub('', text)
    text = EMAIL_PATTERN.sub('', text)
    text = REPEATED_CHAR_PATTERN.sub(r'\1\1', text)
    text = SPECIAL_CHAR_PATTERN.sub(' ', text)
    text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
    return text


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names: lowercase, strip, replace spaces/special chars with underscores."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9_]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df


def drop_empty_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Drop rows where all values (or subset columns) are NaN/empty."""
    if subset:
        df = df.dropna(subset=subset, how='all')
    else:
        df = df.dropna(how='all')
    return df


# ---------------------------------------------------------------------------
# RSICS Preprocessing
# ---------------------------------------------------------------------------

def preprocess_rsics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess RSICS (Retail Service Industry Customer Support) conversation data.

    Expected CSV columns: conversation_id, speaker, message, timestamp, topic, sentiment
    Outputs cleaned text, normalized columns, deduplicated rows.
    """
    logger.info('Preprocessing RSICS data: %d rows', len(df))

    df = normalize_column_names(df)
    df = drop_empty_rows(df)

    text_col = _find_text_column(df, ['message', 'text', 'content', 'body'])
    if text_col:
        df['cleaned_text'] = df[text_col].apply(clean_text)
        df['text_length'] = df['cleaned_text'].str.len()
        df = df[df['text_length'] > 0]

    df = df.drop_duplicates()

    logger.info('RSICS preprocessing complete: %d rows after cleaning', len(df))
    return df


# ---------------------------------------------------------------------------
# 3K Conversations Preprocessing
# ---------------------------------------------------------------------------

def preprocess_3k_conversations(data: list[dict] | dict) -> list[dict]:
    """
    Preprocess 3K Conversations dataset (JSON format).

    Expected structure: list of conversation objects, each with a 'messages' key
    containing message objects with 'text', 'speaker', 'timestamp' fields.
    Handles both list-of-conversations and single-conversation-wrapping-dict formats.
    """
    logger.info('Preprocessing 3K conversations data')

    if isinstance(data, dict):
        conversations = data.get('conversations', data.get('data', [data]))
    else:
        conversations = data

    processed = []
    for conv in conversations:
        messages = conv.get('messages', [])
        if not messages:
            continue

        cleaned_messages = []
        for msg in messages:
            text = msg.get('text', '')
            cleaned = clean_text(text)
            if not cleaned:
                continue
            cleaned_msg = {**msg, 'cleaned_text': cleaned}
            cleaned_messages.append(cleaned_msg)

        if cleaned_messages:
            processed.append({
                'conversation_id': conv.get('conversation_id', conv.get('id', '')),
                'messages': cleaned_messages,
                'message_count': len(cleaned_messages),
            })

    logger.info('3K conversations preprocessing complete: %d conversations', len(processed))
    return processed


# ---------------------------------------------------------------------------
# Twitter Preprocessing
# ---------------------------------------------------------------------------

def preprocess_twitter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess Twitter customer support conversation data.

    Expected CSV columns: tweet_id, text, in_response_to_tweet_id, author_id, created_at
    Outputs cleaned text, normalized columns, removed retweets and empty tweets.
    """
    logger.info('Preprocessing Twitter data: %d rows', len(df))

    df = normalize_column_names(df)
    df = drop_empty_rows(df)

    text_col = _find_text_column(df, ['text', 'tweet_text', 'content', 'body'])
    if text_col:
        # Remove retweets
        rt_mask = df[text_col].str.startswith('RT ', na=False)
        df = df[~rt_mask]

        # Clean text (strip @mentions to plain text, remove hashtags symbol)
        df['cleaned_text'] = df[text_col].apply(clean_text)
        df['cleaned_text'] = df['cleaned_text'].str.replace(r'@\w+', '', regex=True).str.strip()
        df['cleaned_text'] = df['cleaned_text'].str.replace(r'#', '', regex=False)
        df['text_length'] = df['cleaned_text'].str.len()
        df = df[df['text_length'] > 0]

    df = df.drop_duplicates()

    logger.info('Twitter preprocessing complete: %d rows after cleaning', len(df))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_text_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching text column from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
