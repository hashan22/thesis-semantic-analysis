
# ============================================================
# THESIS: Semantic Similarity, Sentiment and Emotion Analysis
# Models: ChatGPT and Mistral
# Reference: Baseline narrative
# ============================================================

# Install dependencies in a separate notebook cell if needed:
# %pip install pandas nltk sentence-transformers nrclex

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError

import pandas as pd
import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHATGPT_DIR = Path("Prompts/Chatgpt")
MISTRAL_DIR = Path("Prompts/Mistral")
BASELINE_PATH = Path("Prompts/Baseline.txt")

OUTPUT_CSV = Path("semantic_sentiment_analysis_results_2.csv")
METADATA_JSON = Path("semantic_sentiment_analysis_metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_PROMPTS = range(1, 81)

# Chunking parameters.
# The model's configured maximum sequence length is checked below.
CHUNK_OVERLAP_TOKENS = 40

# A consistent set of NRC categories for every response.
EMOTION_CATEGORIES = [
    "fear", "anger", "anticipation", "trust",
    "surprise", "positive", "negative", "sadness",
    "disgust", "joy"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# ============================================================
# 2. DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("vader_lexicon", quiet=True)


# ============================================================
# 3. VALIDATE INPUTS
# ============================================================

if not BASELINE_PATH.is_file():
    raise FileNotFoundError(
        f"Baseline file not found: {BASELINE_PATH.resolve()}"
    )

with BASELINE_PATH.open("r", encoding="utf-8-sig") as file:
    baseline_text = file.read().strip()

if not baseline_text:
    raise ValueError("The baseline narrative is empty.")

if not CHATGPT_DIR.is_dir():
    raise FileNotFoundError(
        f"ChatGPT directory not found: {CHATGPT_DIR.resolve()}"
    )

if not MISTRAL_DIR.is_dir():
    raise FileNotFoundError(
        f"Mistral directory not found: {MISTRAL_DIR.resolve()}"
    )


# ============================================================
# 4. LOAD MODELS ONCE
# ============================================================

logging.info("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

sia = SentimentIntensityAnalyzer()

tokenizer = model.tokenizer
MAX_SEQUENCE_LENGTH = model.max_seq_length

# Reserve room for special tokens.
SPECIAL_TOKENS = tokenizer.num_special_tokens_to_add(
    pair=False
)

MAX_CONTENT_TOKENS = MAX_SEQUENCE_LENGTH - SPECIAL_TOKENS

if not 0 <= CHUNK_OVERLAP_TOKENS < MAX_CONTENT_TOKENS:
    raise ValueError(
        "Chunk overlap must be non-negative and smaller "
        "than the maximum content tokens."
    )

logging.info(
    "Model loaded. Maximum sequence length: %s",
    MAX_SEQUENCE_LENGTH
)


# ============================================================
# 5. SPLIT LONG TEXTS INTO OVERLAPPING TOKEN CHUNKS
# ============================================================

def split_into_chunks(text):
    """
    Split text into overlapping token chunks.

    This prevents long documents from being represented only
    by their initial tokens. Chunking can split sentences.
    """

    token_ids = tokenizer.backend_tokenizer.encode(
        text,
        add_special_tokens=False
    ).ids

    if not token_ids:
        return []

    step = MAX_CONTENT_TOKENS - CHUNK_OVERLAP_TOKENS
    chunks = []

    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[
            start:start + MAX_CONTENT_TOKENS
        ]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if start + MAX_CONTENT_TOKENS >= len(token_ids):
            break

    return chunks


# ============================================================
# 6. DOCUMENT-LEVEL SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_similarity(reference_text, response_text):
    """
    Calculate symmetric mean best-match chunk similarity.

    Each reference chunk is matched with its most similar
    response chunk, and vice versa. The two directional
    means are averaged.

    This is a chunk-level coverage metric, NOT the same as
    cosine similarity between two single document embeddings.
    """

    reference_chunks = split_into_chunks(reference_text)
    response_chunks = split_into_chunks(response_text)

    if not reference_chunks or not response_chunks:
        return float("nan"), len(reference_chunks), len(response_chunks)

    reference_embeddings = model.encode(
        reference_chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    response_embeddings = model.encode(
        response_chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Rows: reference chunks. Columns: response chunks.
    similarity_matrix = util.cos_sim(
        reference_embeddings,
        response_embeddings
    )

    # Best response match for each reference chunk.
    reference_to_response = similarity_matrix.max(dim=1).values.mean()

    # Best reference match for each response chunk.
    response_to_reference = similarity_matrix.max(dim=0).values.mean()

    # Symmetric coverage score.
    similarity = (
        reference_to_response + response_to_reference
    ) / 2

    return (
        float(similarity.item()),
        len(reference_chunks),
        len(response_chunks)
    )


# Cache the baseline chunks and embeddings for reuse.
logging.info("Preparing baseline embeddings...")

baseline_chunks = split_into_chunks(baseline_text)

if not baseline_chunks:
    raise ValueError("The baseline produced no usable text chunks.")

baseline_embeddings = model.encode(
    baseline_chunks,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=False
)


def similarity_to_cached_baseline(response_text):
    """Compare a response with the cached baseline embeddings."""

    response_chunks = split_into_chunks(response_text)

    if not response_chunks:
        return float("nan"), 0

    response_embeddings = model.encode(
        response_chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    matrix = util.cos_sim(
        baseline_embeddings,
        response_embeddings
    )

    baseline_to_response = matrix.max(dim=1).values.mean()
    response_to_baseline = matrix.max(dim=0).values.mean()

    score = (baseline_to_response + response_to_baseline) / 2

    return float(score.item()), len(response_chunks)


# ============================================================
# 7. SENTIMENT ANALYSIS
# ============================================================

def get_sentiment_scores(text):
    """
    VADER returns:
    negative, neutral, positive and compound scores.
    """

    return sia.polarity_scores(text)


# ============================================================
# 8. EMOTION ANALYSIS
# ============================================================

def get_emotion_scores(text):
    """
    Return raw NRC category counts and counts per 1,000 words.

    NRC categories can overlap. These are lexicon-based
    word associations, not calibrated emotion probabilities.
    """

    word_count = len(text.split())

    raw_scores = NRCLex(text).raw_emotion_scores

    scores = {}

    for category in EMOTION_CATEGORIES:
        count = int(raw_scores.get(category, 0))

        scores[f"Raw_{category}"] = count

        scores[f"Per1000Words_{category}"] = (
            count * 1000 / word_count
            if word_count > 0
            else 0.0
        )

    return scores


# ============================================================
# 9. PROCESS ALL PROMPT FILES
# ============================================================

results = []
missing_files = []
empty_files = []

for i in EXPECTED_PROMPTS:

    file_name = f"Prompt{i:02d}.txt"

    chatgpt_path = CHATGPT_DIR / file_name
    mistral_path = MISTRAL_DIR / file_name

    # Check both files.
    if not chatgpt_path.is_file():
        missing_files.append(str(chatgpt_path))
    if not mistral_path.is_file():
        missing_files.append(str(mistral_path))

    if not chatgpt_path.is_file() or not mistral_path.is_file():
        logging.warning("Skipping %s: missing response file", file_name)
        continue

    # Read responses.
    try:
        with chatgpt_path.open(
            "r", encoding="utf-8-sig"
        ) as file:
            chatgpt_response = file.read().strip()

        with mistral_path.open(
            "r", encoding="utf-8-sig"
        ) as file:
            mistral_response = file.read().strip()

    except (OSError, UnicodeError) as error:
        logging.error("Could not read %s: %s", file_name, error)
        continue

    if not chatgpt_response or not mistral_response:
        empty_files.append(file_name)
        logging.warning("Skipping %s: empty response", file_name)
        continue

    logging.info("Processing %s", file_name)

    # Semantic similarity.
    similarity_chatgpt, chatgpt_chunk_count = (
        similarity_to_cached_baseline(chatgpt_response)
    )

    similarity_mistral, mistral_chunk_count = (
        similarity_to_cached_baseline(mistral_response)
    )

    # Sentiment.
    sentiment_chatgpt = get_sentiment_scores(chatgpt_response)
    sentiment_mistral = get_sentiment_scores(mistral_response)

    # Emotions.
    emotions_chatgpt = get_emotion_scores(chatgpt_response)
    emotions_mistral = get_emotion_scores(mistral_response)

    # Assemble one row per prompt.
    row = {
        "Prompt": file_name,

        "Similarity_ChatGPT": similarity_chatgpt,
        "Similarity_Mistral": similarity_mistral,

        "Words_ChatGPT": len(chatgpt_response.split()),
        "Words_Mistral": len(mistral_response.split()),

        "Chunks_ChatGPT": chatgpt_chunk_count,
        "Chunks_Mistral": mistral_chunk_count,

        **{
            f"ChatGPT_Sent_{key}": value
            for key, value in sentiment_chatgpt.items()
        },

        **{
            f"Mistral_Sent_{key}": value
            for key, value in sentiment_mistral.items()
        },

        **{
            f"ChatGPT_Emo_{key}": value
            for key, value in emotions_chatgpt.items()
        },

        **{
            f"Mistral_Emo_{key}": value
            for key, value in emotions_mistral.items()
        }
    }

    results.append(row)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

if not results:
    raise RuntimeError(
        "No complete prompt pairs were processed. "
        "Check your input directories and filenames."
    )

df = pd.DataFrame(results)

# Keep prompts in numeric order.
df["_sort"] = df["Prompt"].str.extract(
    r"(\d+)"
).astype(int)

df = (
    df.sort_values("_sort")
    .drop(columns="_sort")
    .reset_index(drop=True)
)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
    float_format="%.6f"
)


# ============================================================
# 11. SAVE REPRODUCIBILITY METADATA
# ============================================================

def package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


metadata = {
    "analysis_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "python_version": sys.version,
    "embedding_model": MODEL_NAME,
    "model_max_sequence_length": MAX_SEQUENCE_LENGTH,
    "max_content_tokens_per_chunk": MAX_CONTENT_TOKENS,
    "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,

    "similarity_method": (
        "Symmetric mean best-match chunk cosine similarity"
    ),

    "emotion_method": (
        "NRClex raw category counts and counts per 1000 words"
    ),

    "baseline_path": str(BASELINE_PATH),
    "baseline_chunks": len(baseline_chunks),

    "processed_prompt_count": len(df),
    "missing_files": missing_files,
    "empty_prompt_pairs": empty_files,

    "package_versions": {
        "pandas": package_version("pandas"),
        "nltk": package_version("nltk"),
        "sentence-transformers": package_version(
            "sentence-transformers"
        ),
        "nrclex": package_version("NRCLex"),
        "torch": package_version("torch")
    }
}

with METADATA_JSON.open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2, ensure_ascii=False)


# ============================================================
# 12. REPORT
# ============================================================

print("\nAnalysis complete.")
print(f"Prompts processed: {len(df)} / {len(EXPECTED_PROMPTS)}")
print(f"Missing files: {len(missing_files)}")
print(f"Empty prompt pairs: {len(empty_files)}")
print(f"Results saved to: {OUTPUT_CSV.resolve()}")
print(f"Metadata saved to: {METADATA_JSON.resolve()}")
