# AI Automation & Customer Intelligence — Workflow Architecture (Phase 8)

## System Goal
Classify incoming customer queries into predefined issue categories and automatically route them into:
- `Auto-Reply` (high confidence)
- `Ticket Creation` (medium confidence)
- `Human Escalation` (low confidence or urgent keywords)

The system also assigns SLA expectations and records an audit trail for later dashboard insights.

## End-to-End Flow (Input → Processing → Output)

```mermaid
flowchart TD
  A[Customer Query Input<br/>(query_text, channel, priority, query_id)] --> B[Preprocessing (lightweight)]
  B --> C[Urgency Keyword Override Check]
  C --> D[Category Prediction (Hybrid Classifier)]
  D --> E{Routing Decision}
  E -->|Auto-Reply| F[Generate Channel-Specific Reply]
  E -->|Ticket Creation| G[Create Ticket + SLA Deadlines]
  E -->|Human Escalation| H[Escalate to Live Agent + SLA Deadlines]
  F --> I[Write Audit Log Record]
  G --> I[Write Audit Log Record]
  H --> I[Write Audit Log Record]
  I --> J[Dashboard Inputs]
  J --> K[Compute Metrics & Visualizations]
```

## Components (as implemented in this repo)

### 1) Data & Features (Phases 2–3)
- Dataset: `data/customer_queries.csv`
- Preprocessing pipeline: `preprocessing/preprocess.py`
  - Cleans text, normalizes slang, detects language
  - Extracts entities (order IDs, amounts, phone/pin codes)
  - Adds engineered numeric features
  - Exports: `data/preprocessed_queries.csv`

### 2) Model (Phases 4–5)
- Model training + evaluation: `model/classifier.py`
  - TF-IDF vectorizer (word n-grams) + Logistic Regression (multi-class)
  - Saves trained model: `model/hybrid_classifier.pkl`
  - Exports predictions for analysis: `data/predictions.csv`

### 3) Automation Logic (Phase 6)
- Response engine: `automation/response_engine.py`
  - Loads trained model `model/hybrid_classifier.pkl`
  - Lightweight cleaning of live text
  - Detects urgent keywords (keyword override)
  - Predicts category + confidence via model
  - Applies confidence routing thresholds:
    - High (>= 0.50): `Auto-Reply`
    - Medium (>= 0.30 and < 0.50): `Ticket Creation`
    - Low (< 0.30): `Human Escalation`
  - Assigns SLA deadlines based on `priority` (and urgent override)
  - Ticket storage:
    - `outputs/tickets.json` (created/updated by `TicketManager`)
  - Audit logging:
    - `outputs/automation_log.csv` (written via `save_log()` if used)

### 4) Dashboard (Phase 7)
- Dashboard generator: `dashboard/dashboard.py`
  - Reads `data/predictions.csv`
  - Computes metrics (routing mix, confidence tiers, accuracy, top confused pairs, etc.)
  - Exports submission artifacts:
    - `outputs/dashboard_metrics.json`
    - `outputs/dashboard.html`

## Routing Rules (Operational Detail)
1. **Urgent keyword override**
   - If urgent keywords are found in the raw query text, routing is treated as `High` priority regardless of model confidence.
2. **Confidence-based routing**
   - `High confidence` → `Auto-Reply`
   - `Medium confidence` → `Ticket Creation`
   - `Low confidence` → `Human Escalation`
3. **Channel-specific message formatting**
   - `WhatsApp`, `Email`, `Instagram DM`, `Website Chat` each get different greetings/closings.
4. **SLA assignment**
   - First response and resolution windows are assigned using the configured SLA table.

## Batch / Automation Workflow (How it runs in practice)
1. Load the dataset of incoming queries (or live queries from an integration).
2. Call `ResponseEngine.process_query()` for each query (single) or `process_batch()` (batch).
3. For `Ticket Creation` and `Human Escalation`, `TicketManager` creates ticket records with SLA deadlines.
4. Optionally call `engine.save_log()` to persist `automation_log.csv`.
5. Run `dashboard/dashboard.py` to generate the dashboard from `data/predictions.csv` (and later, optionally extend to include logs/tickets).

## Outputs Produced by This System
- Automated customer response message (`auto_reply`)
- Support ticket record (`ticket_id`, status, SLA deadlines)
- Audit log record (routing decision details, confidence, urgency override)
- Dashboard artifacts:
  - `outputs/dashboard_metrics.json`
  - `outputs/dashboard.html`

