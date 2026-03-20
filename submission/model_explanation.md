# Model Explanation (Phase 9 Deliverable)

## 1) What the model does
The system classifies each customer query into one of 6 categories, then uses the model’s confidence score to decide how to handle the query:
- `Auto-Reply` (high confidence)
- `Ticket Creation` (medium confidence)
- `Human Escalation` (low confidence)

Urgent keywords can override the confidence-based routing and force `High` priority.

## 2) Category labels
The classifier predicts one of:
- `Order Status`
- `Delivery Delay`
- `Refund Request`
- `General Inquiry`
- `Payment Issue`
- `Product Complaint`

## 3) Hybrid classifier design (TF-IDF + Logistic Regression)
The training pipeline is implemented in `model/classifier.py` and uses:
1. `TfidfVectorizer` over text
   - word n-grams: `1–3`
   - `max_features=8000`
   - `sublinear_tf=True`
2. `LogisticRegression` (multi-class)
   - `C=5.0`
   - `class_weight="balanced"`
   - `max_iter=2000`

This combination is “hybrid” in the sense that it uses feature engineering (TF‑IDF) plus a probabilistic classifier (logistic regression), enabling confidence scores (`predict_proba`).

## 4) Confidence routing rules
Implemented via thresholds in both `model/classifier.py` and `automation/response_engine.py`:
- `High` confidence: `>= 0.50` → `Auto-Reply`
- `Medium` confidence: `>= 0.30` and `< 0.50` → `Ticket Creation`
- `Low` confidence: `< 0.30` → `Human Escalation`

## 5) Urgent keyword override
If the raw query contains any urgent keywords/phrases (examples include: `urgent`, `immediately`, `emergency`, `asap`, `payment failed`, `deducted`, `fraud`, `scam`, `police`, `legal`, `missing`, `broken`, `never received`, `wrong item`, `defective`), the routing is treated as `High` priority regardless of confidence.

## 6) Auto-reply templates
When routed to `Auto-Reply`, the engine selects a template per predicted category and fills placeholders like:
- `[tracking_link]` (generated reference link)
- `[ref_id]`
- `[faq_link]`

Channel-aware greeting/closing are applied (WhatsApp / Email / Instagram DM / Website Chat).

## 7) Evaluation (from `evaluation_report.json`)
The saved evaluation report indicates:
- Cross-validation mean accuracy (`cv_mean`): `0.58`
- Cross-validation std (`cv_std`): `0.0872`
- Test accuracy (`accuracy`): `0.50`

The report also includes per-class precision/recall/F1 and confidence-tier distribution.

## 8) Where to find artifacts
- Trained model: `model/hybrid_classifier.pkl`
- Evaluation report: `model/evaluation_report.json` (also copied to this `submission/` folder)
- Inference/predictions: `data/predictions.csv`
- Routing + automation: `automation/response_engine.py`

