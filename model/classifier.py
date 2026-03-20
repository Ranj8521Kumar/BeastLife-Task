"""
=============================================================
  AI Customer Intelligence System
  PHASE 5: Hybrid Classifier — Model Implementation
=============================================================
  Architecture:
    Text Input
        ↓
    TF-IDF Vectorizer
        ↓
    Logistic Regression (multi-class)
        ↓
    Confidence Score
        ↓
    Decision Router:
      ≥ 0.80  →  Auto-Reply      (High Confidence)
      0.50-0.79 → Ticket Creation (Medium Confidence)
      < 0.50  →  Human Escalation (Low Confidence)
=============================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline          import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model      import LogisticRegression
from sklearn.multiclass        import OneVsRestClassifier
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       accuracy_score)
from sklearn.preprocessing     import LabelEncoder

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "preprocessed_queries.csv")
RAW_DATA_PATH  = os.path.join(BASE_DIR, "data", "customer_queries.csv")
MODEL_DIR      = os.path.join(BASE_DIR, "model")
MODEL_PATH     = os.path.join(MODEL_DIR, "hybrid_classifier.pkl")
REPORT_PATH    = os.path.join(MODEL_DIR, "evaluation_report.json")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "data", "predictions.csv")

# ── Confidence Thresholds (tuned for small dataset) ────────────
HIGH_CONFIDENCE_THRESHOLD   = 0.50   # Auto-Reply
MEDIUM_CONFIDENCE_THRESHOLD = 0.30   # Ticket Creation
# Below 0.30 → Human Escalation

# ── Category Labels ────────────────────────────────────────────
CATEGORIES = [
    "Order Status",
    "Delivery Delay",
    "Refund Request",
    "General Inquiry",
    "Payment Issue",
    "Product Complaint",
]

# ── Auto-Reply Templates ───────────────────────────────────────
AUTO_REPLIES = {
    "Order Status": (
        "Hi! Thanks for reaching out. Your order is currently being processed. "
        "You can track it here: [tracking_link]. Expected delivery: 2–5 business days."
    ),
    "Delivery Delay": (
        "We sincerely apologize for the delay. Our logistics team is working to "
        "expedite your delivery. You'll receive an update within 24 hours."
    ),
    "Refund Request": (
        "We've received your refund request. It will be processed within 5–7 "
        "business days to your original payment method. Ref ID: [ref_id]."
    ),
    "General Inquiry": (
        "Thanks for your query! Our support team is happy to help. "
        "Please visit our FAQ page or reply with more details and we'll assist you."
    ),
    "Payment Issue": (
        "We're sorry for the payment inconvenience. Please check your bank statement. "
        "If the amount was debited, it will be refunded within 3–5 business days."
    ),
    "Product Complaint": (
        "We apologize for the experience with your product. Please share photos of "
        "the issue and we'll arrange a replacement or full refund immediately."
    ),
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load preprocessed or raw dataset — prefer raw_query for richer features."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Use raw_query (full text) for better classification on small datasets
        text_col = "raw_query" if "raw_query" in df.columns else "cleaned_query"
        print(f"[INFO] Loaded preprocessed data: {len(df)} rows | using column: '{text_col}'")
    else:
        df = pd.read_csv(RAW_DATA_PATH)
        text_col = "query_text"
        print(f"[INFO] Loaded raw data (fallback): {len(df)} rows")

    assert text_col in df.columns, f"Column '{text_col}' not found!"
    assert "category" in df.columns, "Column 'category' not found!"

    df = df.dropna(subset=[text_col, "category"])
    df[text_col] = df[text_col].astype(str)
    return df, text_col


# ═══════════════════════════════════════════════════════════════
# STEP 2: BUILD PIPELINE
# ═══════════════════════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """Assemble TF-IDF + Logistic Regression pipeline."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),      # unigrams + bigrams + trigrams
            max_features=8000,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=5.0,                   # less regularization for small dataset
            solver="lbfgs",
            random_state=42,
            class_weight="balanced", # handle slight class imbalance
        )),
    ])
    return pipeline


# ═══════════════════════════════════════════════════════════════
# STEP 3: TRAIN & EVALUATE
# ═══════════════════════════════════════════════════════════════

def train_and_evaluate(df: pd.DataFrame, text_col: str):
    """Train the pipeline and print evaluation metrics."""
    X = df[text_col].values
    y = df["category"].values

    print(f"\n{'='*55}")
    print("  MODEL TRAINING")
    print(f"{'='*55}")
    print(f"  Features    : {text_col}")
    print(f"  Samples     : {len(X)}")
    print(f"  Categories  : {sorted(set(y))}\n")

    # Split — stratified so each class is represented in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train size  : {len(X_train)}")
    print(f"  Test  size  : {len(X_test)}\n")

    # Train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)
    confidence  = np.max(y_prob, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    print(f"  [OK] Test Accuracy  : {acc*100:.1f}%")

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  [OK] CV Accuracy    : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    print(f"  CV Fold Scores  : {[f'{s*100:.1f}%' for s in cv_scores]}\n")

    # Full report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("  📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    cm_df = pd.DataFrame(cm, index=pipeline.classes_, columns=pipeline.classes_)
    print("  📊 Confusion Matrix:")
    print(cm_df.to_string())

    # Confidence distribution
    high   = np.sum(confidence >= HIGH_CONFIDENCE_THRESHOLD)
    medium = np.sum((confidence >= MEDIUM_CONFIDENCE_THRESHOLD) &
                    (confidence <  HIGH_CONFIDENCE_THRESHOLD))
    low    = np.sum(confidence < MEDIUM_CONFIDENCE_THRESHOLD)
    print(f"\n  📊 Confidence Distribution (test set):")
    print(f"     🟢 High   (≥{HIGH_CONFIDENCE_THRESHOLD}) : {high:2d} queries → Auto-Reply")
    print(f"     🟡 Medium ({MEDIUM_CONFIDENCE_THRESHOLD}–{HIGH_CONFIDENCE_THRESHOLD-0.01:.2f}): {medium:2d} queries → Ticket")
    print(f"     🔴 Low    (<{MEDIUM_CONFIDENCE_THRESHOLD}) : {low:2d} queries → Human")

    # Save report
    eval_report = {
        "accuracy"      : round(acc, 4),
        "cv_mean"       : round(cv_scores.mean(), 4),
        "cv_std"        : round(cv_scores.std(), 4),
        "cv_folds"      : [round(s, 4) for s in cv_scores.tolist()],
        "classes"       : list(pipeline.classes_),
        "classification_report": report,
        "confidence_distribution": {
            "high_count"  : int(high),
            "medium_count": int(medium),
            "low_count"   : int(low),
        },
        "thresholds": {
            "high"  : HIGH_CONFIDENCE_THRESHOLD,
            "medium": MEDIUM_CONFIDENCE_THRESHOLD,
        }
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  [INFO] Evaluation report saved → {REPORT_PATH}")

    return pipeline, eval_report


# ═══════════════════════════════════════════════════════════════
# STEP 4: CONFIDENCE ROUTER
# ═══════════════════════════════════════════════════════════════

def route_decision(category: str, confidence: float, query_text: str) -> dict:
    """
    Route the prediction to an action based on confidence score.
    Returns a decision dict with action, reply, and metadata.
    """
    text_lower = query_text.lower()

    # Keyword overrides — force High priority regardless of confidence
    urgent_keywords = [
        "urgent", "immediately", "payment failed", "deducted",
        "broken", "wrong item", "not received", "never received",
        "fraud", "scam", "police", "legal", "missing",
    ]
    is_urgent = any(kw in text_lower for kw in urgent_keywords)

    if is_urgent or confidence >= HIGH_CONFIDENCE_THRESHOLD:
        action    = "Auto-Reply"
        tier      = "High"
        reply     = AUTO_REPLIES.get(category, "We'll get back to you shortly.")
        escalate  = False

    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        action    = "Ticket Creation"
        tier      = "Medium"
        reply     = (f"We've logged your query under '{category}'. "
                     f"A support agent will respond within 2–4 hours.")
        escalate  = False

    else:
        action    = "Human Escalation"
        tier      = "Low"
        reply     = ("We're routing your query to a live agent for personalized help. "
                     "Please hold — estimated wait time: 5 minutes.")
        escalate  = True

    return {
        "predicted_category": category,
        "confidence_score"  : round(float(confidence), 4),
        "confidence_tier"   : tier,
        "action"            : action,
        "auto_reply"        : reply,
        "escalate_to_human" : escalate,
        "urgent_override"   : is_urgent,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 5: PREDICT ON FULL DATASET
# ═══════════════════════════════════════════════════════════════

def predict_full_dataset(pipeline, df: pd.DataFrame, text_col: str):
    """Run inference on full dataset and save predictions."""
    texts      = df[text_col].values
    categories = pipeline.predict(texts)
    probs      = pipeline.predict_proba(texts)
    confidences = np.max(probs, axis=1)

    raw_texts = (df["raw_query"].values
                 if "raw_query" in df.columns
                 else df[text_col].values)

    rows = []
    for i, (text, cat, conf, raw) in enumerate(
            zip(texts, categories, confidences, raw_texts)):
        decision = route_decision(cat, conf, str(raw))
        rows.append({
            "query_id"          : df["query_id"].iloc[i] if "query_id" in df.columns else i,
            "channel"           : df["channel"].iloc[i]  if "channel"  in df.columns else "",
            "raw_query"         : raw,
            "cleaned_query"     : text,
            "true_category"     : df["category"].iloc[i],
            "predicted_category": decision["predicted_category"],
            "correct"           : df["category"].iloc[i] == decision["predicted_category"],
            "confidence_score"  : decision["confidence_score"],
            "confidence_tier"   : decision["confidence_tier"],
            "action"            : decision["action"],
            "escalate_to_human" : decision["escalate_to_human"],
            "urgent_override"   : decision["urgent_override"],
            "auto_reply_preview": decision["auto_reply"][:80] + "...",
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\n  [INFO] Predictions saved → {PREDICTIONS_PATH}")

    # Summary
    print(f"\n  📊 FULL DATASET PREDICTION SUMMARY:")
    print(f"     Total Queries : {len(pred_df)}")
    print(f"     Overall Match : {pred_df['correct'].sum()}/{len(pred_df)} "
          f"({pred_df['correct'].mean()*100:.1f}%)")
    print(f"\n     Action Distribution:")
    print(pred_df["action"].value_counts().to_string())
    print(f"\n     Confidence Tier Split:")
    print(pred_df["confidence_tier"].value_counts().to_string())

    return pred_df


# ═══════════════════════════════════════════════════════════════
# STEP 6: SAVE & LOAD MODEL
# ═══════════════════════════════════════════════════════════════

def save_model(pipeline):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n  [INFO] Model saved → {MODEL_PATH}")


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════
# STEP 7: LIVE INFERENCE (single query)
# ═══════════════════════════════════════════════════════════════

def predict_single(query_text: str, pipeline=None) -> dict:
    """
    Predict a single query and return full decision.
    Useful for API integration or live testing.
    """
    if pipeline is None:
        pipeline = load_model()

    # Basic cleaning
    import re, string
    text = query_text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    probs      = pipeline.predict_proba([text])[0]
    category   = pipeline.classes_[np.argmax(probs)]
    confidence = float(np.max(probs))

    decision = route_decision(category, confidence, query_text)
    decision["input_query"] = query_text
    return decision


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI CUSTOMER INTELLIGENCE — HYBRID CLASSIFIER")
    print("="*55)

    # 1. Load data
    df, text_col = load_data()

    # 2. Train + Evaluate
    pipeline, report = train_and_evaluate(df, text_col)

    # 3. Save model
    save_model(pipeline)

    # 4. Predict on full dataset
    pred_df = predict_full_dataset(pipeline, df, text_col)

    # 5. Live sample tests
    print("\n" + "="*55)
    print("  🔬 LIVE SAMPLE TESTS (5 queries)")
    print("="*55)

    test_queries = [
        "Where is my order? It was supposed to arrive yesterday!",   # Order Status
        "My payment was deducted but order not placed. URGENT!",      # Payment Issue
        "I want a refund for my damaged product.",                    # Refund Request
        "Do you offer free shipping?",                               # General Inquiry
        "blah blah random gibberish xyz abc",                        # Low confidence
    ]

    for q in test_queries:
        result = predict_single(q, pipeline)
        print(f"\n  Query      : {q}")
        print(f"  Category   : {result['predicted_category']}")
        print(f"  Confidence : {result['confidence_score']} ({result['confidence_tier']})")
        print(f"  Action     : {result['action']}")
        if result["urgent_override"]:
            print(f"  [WARN]  URGENT KEYWORD OVERRIDE TRIGGERED")
        print(f"  Reply      : {result['auto_reply'][:100]}...")

    print(f"\n{'='*55}")
    print("  [OK] Phase 5 Complete — Model ready for automation layer")
    print(f"{'='*55}\n")
