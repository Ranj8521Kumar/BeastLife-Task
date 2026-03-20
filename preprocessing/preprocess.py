"""
=============================================================
  AI Customer Intelligence System
  PHASE 3: Data Preprocessing Pipeline
=============================================================
  Steps:
    1. Text Cleaning       — lowercase, punctuation, whitespace
    2. Stopword Removal    — remove common noise words
    3. Entity Extraction   — order IDs, amounts, phone numbers
    4. Language Detection  — detect Hindi vs English
    5. Normalization       — stem/lemmatize, slang handling
    6. Feature Engineering — query length, word count, urgency
    7. Export              — preprocessed_queries.csv
=============================================================
"""

import pandas as pd
import re
import string
import os

# ── Optional NLP libraries (graceful fallback if not installed) ──
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("[INFO] nltk not installed. Using basic preprocessing fallback.")

try:
    from langdetect import detect as lang_detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("[INFO] langdetect not installed. Language detection will use CSV 'language' column.")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV  = os.path.join(BASE_DIR, "data", "customer_queries.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "preprocessed_queries.csv")

# Slang / informal → normalized form
SLANG_MAP = {
    "ur": "your",
    "u": "you",
    "plz": "please",
    "pls": "please",
    "asap": "as soon as possible",
    "cant": "cannot",
    "wont": "will not",
    "didnt": "did not",
    "dont": "do not",
    "ive": "i have",
    "im": "i am",
    "its": "it is",
    "thats": "that is",
    "havent": "have not",
    "hasnt": "has not",
    "isnt": "is not",
    "wasnt": "was not",
    "arent": "are not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "lemme": "let me",
    "gimme": "give me",
    "kinda": "kind of",
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "smh": "shaking my head",
    "btw": "by the way",
    "omg": "oh my god",
    "wtf": "what the",
    "idk": "i do not know",
    "imo": "in my opinion",
    "fyi": "for your information",
}

# Urgency keywords → weight high priority
URGENCY_KEYWORDS = [
    "immediately", "urgent", "asap", "emergency", "right now",
    "still waiting", "unacceptable", "terrible", "broken",
    "deducted", "failed", "wrong", "never", "again", "delayed",
    "not received", "not arrived", "missing", "refund", "damaged"
]

# Basic English stopwords fallback (if nltk unavailable)
BASIC_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "with", "this", "that", "my", "your",
    "i", "me", "we", "you", "he", "she", "they", "was", "are",
    "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might",
    "can", "not", "no", "so", "if", "as", "by", "from", "up",
    "out", "about", "into", "through", "during", "its", "their"
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: TEXT CLEANING
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Lowercase, remove special characters, extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)
    # Remove punctuation (keep digits for entity extraction later)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("#", "")))
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# STEP 2: ENTITY EXTRACTION (before stopword removal)
# ═══════════════════════════════════════════════════════════════

def extract_entities(text: str) -> dict:
    """Extract structured entities from raw query text."""
    entities = {
        "order_ids"   : re.findall(r"(?:order\s*(?:#|id|no|num|number)?[\s:]*)([\w\-]+)", text, re.IGNORECASE),
        "amounts"     : re.findall(r"(?:₹|rs\.?|inr|usd|\$)\s?([\d,]+)", text, re.IGNORECASE),
        "phone"       : re.findall(r"\b(?:\+91|0)?[6-9]\d{9}\b", text),
        "pin_codes"   : re.findall(r"\b[1-9]\d{5}\b", text),
        "product_refs": re.findall(r"\b(?:order|item|product|parcel|package)\b", text, re.IGNORECASE),
    }
    # Flatten lists to comma-separated strings for CSV storage
    return {k: ", ".join(v) if v else "" for k, v in entities.items()}


# ═══════════════════════════════════════════════════════════════
# STEP 3: SLANG NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_slang(text: str) -> str:
    """Replace informal words with formal equivalents."""
    words = text.split()
    return " ".join(SLANG_MAP.get(w, w) for w in words)


# ═══════════════════════════════════════════════════════════════
# STEP 4: STOPWORD REMOVAL
# ═══════════════════════════════════════════════════════════════

def remove_stopwords(text: str) -> str:
    """Remove stopwords from text."""
    if NLTK_AVAILABLE:
        sw = set(stopwords.words("english"))
    else:
        sw = BASIC_STOPWORDS
    return " ".join(w for w in text.split() if w not in sw)


# ═══════════════════════════════════════════════════════════════
# STEP 5: LEMMATIZATION
# ═══════════════════════════════════════════════════════════════

def lemmatize_text(text: str) -> str:
    """Reduce words to their root form."""
    if not NLTK_AVAILABLE:
        return text  # skip if unavailable
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(w) for w in text.split())


# ═══════════════════════════════════════════════════════════════
# STEP 6: LANGUAGE DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_language(text: str, fallback: str = "English") -> str:
    """Auto-detect query language."""
    if LANGDETECT_AVAILABLE:
        try:
            code = lang_detect(text)
            lang_map = {"en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil"}
            return lang_map.get(code, code.upper())
        except Exception:
            return fallback
    # Fallback: check for common Hindi words
    hindi_words = {"mera", "meri", "kab", "kyun", "nahi", "aaya", "chahiye", "wapas", "abhi"}
    if any(w in text.lower().split() for w in hindi_words):
        return "Hindi"
    return fallback


# ═══════════════════════════════════════════════════════════════
# STEP 7: URGENCY SCORING
# ═══════════════════════════════════════════════════════════════

def compute_urgency_score(text: str) -> int:
    """Count urgency keyword matches (0–10 scale)."""
    text_lower = text.lower()
    hits = sum(1 for kw in URGENCY_KEYWORDS if kw in text_lower)
    return min(hits, 10)


# ═══════════════════════════════════════════════════════════════
# STEP 8: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_features(raw_text: str, cleaned_text: str) -> dict:
    """Generate statistical features from text."""
    words = raw_text.split()
    return {
        "word_count"        : len(words),
        "char_count"        : len(raw_text),
        "has_order_ref"     : int(bool(re.search(r"order|ord|#\d", raw_text, re.IGNORECASE))),
        "has_amount_ref"    : int(bool(re.search(r"₹|rs\.?|inr|\$|\d{4,}", raw_text, re.IGNORECASE))),
        "is_question"       : int("?" in raw_text),
        "urgency_score"     : compute_urgency_score(raw_text),
        "exclamation_count" : raw_text.count("!"),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to the dataset."""

    print(f"\n{'='*55}")
    print("  STARTING PREPROCESSING PIPELINE")
    print(f"{'='*55}")
    print(f"  Input rows  : {len(df)}")
    print(f"  Columns     : {list(df.columns)}\n")

    results = []

    for idx, row in df.iterrows():
        raw_text = str(row["query_text"])

        # ── Step 1: Entity extraction (before cleaning) ──
        entities = extract_entities(raw_text)

        # ── Step 2: Language detection ──
        detected_lang = detect_language(raw_text, fallback=str(row.get("language", "English")))

        # ── Step 3: Clean text ──
        cleaned = clean_text(raw_text)

        # ── Step 4: Slang normalization ──
        cleaned = normalize_slang(cleaned)

        # ── Step 5: Stopword removal (only for English) ──
        if detected_lang == "English":
            cleaned_no_sw = remove_stopwords(cleaned)
        else:
            cleaned_no_sw = cleaned  # preserve Hindi queries

        # ── Step 6: Lemmatization ──
        final_text = lemmatize_text(cleaned_no_sw)

        # ── Step 7: Feature engineering ──
        features = engineer_features(raw_text, final_text)

        # ── Assemble output row ──
        result = {
            # Original fields
            "query_id"         : row["query_id"],
            "channel"          : row["channel"],
            "category"         : row["category"],
            "sentiment"        : row["sentiment"],
            "priority"         : row["priority"],
            # Text columns
            "raw_query"        : raw_text,
            "cleaned_query"    : final_text,
            # Detected/confirmed language
            "detected_language": detected_lang,
            # Extracted entities
            "extracted_order_ids"   : entities["order_ids"],
            "extracted_amounts"     : entities["amounts"],
            "extracted_phone"       : entities["phone"],
            "extracted_pin_codes"   : entities["pin_codes"],
            # Engineered features
            "word_count"       : features["word_count"],
            "char_count"       : features["char_count"],
            "has_order_ref"    : features["has_order_ref"],
            "has_amount_ref"   : features["has_amount_ref"],
            "is_question"      : features["is_question"],
            "urgency_score"    : features["urgency_score"],
            "exclamation_count": features["exclamation_count"],
        }
        results.append(result)

        # Show first 3 rows as transformation examples
        if idx < 3:
            print(f"  [EXAMPLE {idx+1}]")
            print(f"    RAW     : {raw_text}")
            print(f"    CLEANED : {final_text}")
            print(f"    ENTITIES: {entities}")
            print(f"    FEATURES: {features}")
            print()

    processed_df = pd.DataFrame(results)
    print(f"{'='*55}")
    print(f"  Preprocessing COMPLETE")
    print(f"  Output rows  : {len(processed_df)}")
    print(f"  Output cols  : {len(processed_df.columns)}")
    print(f"  Saved to     : {OUTPUT_CSV}")
    print(f"{'='*55}\n")
    return processed_df


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load dataset
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input file not found: {INPUT_CSV}")
        exit(1)

    df_raw = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Loaded {len(df_raw)} rows from {INPUT_CSV}")

    # Run pipeline
    df_processed = preprocess_pipeline(df_raw)

    # Save output
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_processed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[INFO] Preprocessed dataset saved → {OUTPUT_CSV}")

    # Show summary statistics
    print("\n📊 CATEGORY DISTRIBUTION (after preprocessing):")
    print(df_processed["category"].value_counts().to_string())

    print("\n📊 CHANNEL DISTRIBUTION:")
    print(df_processed["channel"].value_counts().to_string())

    print("\n📊 AVG URGENCY SCORE BY CATEGORY:")
    print(df_processed.groupby("category")["urgency_score"].mean().round(2).to_string())

    print("\n📊 SENTIMENT SPLIT:")
    print(df_processed["sentiment"].value_counts().to_string())
