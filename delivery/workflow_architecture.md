# AI Automation & Customer Intelligence — Workflow Architecture (Phase 8)

## Architecture Diagram
![Architecture Diagram](architecture_diagram.png)

## System Goal
Classify incoming customer queries into predefined issue categories and automatically route them into:
- `Auto-Reply` (high confidence)
- `Ticket Creation` (medium confidence)
- `Human Escalation` (low confidence or urgent keywords)

The system also assigns SLA expectations, supports FAQ-assisted replies, and records an audit trail for later dashboard insights.

## End-to-End Flow (Input → Processing → Output)

- Input: customer query enters with `query_text`, `channel`, `priority`, and `query_id`
- Processing:
  - Lightweight preprocessing and urgent keyword check
  - Category prediction with hybrid classifier
  - Routing into auto-reply, ticket creation, or human escalation
  - SLA assignment and audit logging
- Output:
  - Customer-facing response
  - Ticket record (if applicable)
  - Audit + dashboard-ready metrics

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
  - Applies dynamic FAQ matching for suitable general inquiries
  - Supports lightweight context-aware responses using recent conversation turns
  - Applies confidence routing thresholds:
    - High (>= 0.50): `Auto-Reply`
    - Medium (>= 0.30 and < 0.50): `Ticket Creation`
    - Low (< 0.30): `Human Escalation`
  - Assigns SLA deadlines based on `priority` (and urgent override)
  - Ticket storage:
    - `data/tickets.json` (created/updated by `TicketManager`)
  - Audit logging:
    - `data/automation_log.csv` (written via `save_log()`)

### 4) Dashboard (Phase 7)
- Dashboard generator: `dashboard/dashboard.py`
  - Reads `data/predictions.csv`
  - Computes problem-distribution metrics required by the challenge:
    - % of total queries by category
    - most common customer problems
    - weekly/monthly trend view (timeline proxy from query order)
  - Exports submission artifacts:
    - `outputs/dashboard_metrics.json`
    - `outputs/dashboard.html` (self-contained HTML: no external chart images required)

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
5. **FAQ and context-aware enhancement**
   - For matched general inquiries, a relevant FAQ answer is inserted into the response.
   - If prior conversation turns are provided, response text includes context awareness.

## Batch / Automation Workflow (How it runs in practice)
1. Load the dataset of incoming queries (or live queries from an integration).
2. Call `ResponseEngine.process_query()` for each query (single) or `process_batch()` (batch). Optionally pass `conversation_history` for contextual responses.
3. For `Ticket Creation` and `Human Escalation`, `TicketManager` creates ticket records with SLA deadlines.
4. Call `engine.save_log()` to persist `data/automation_log.csv`.
5. Run `dashboard/dashboard.py` to generate the dashboard from `data/predictions.csv` (and later, optionally extend to include logs/tickets).

## Outputs Produced by This System
- Automated customer response message (`auto_reply`)
- Support ticket record (`ticket_id`, status, SLA deadlines)
- Audit log record (routing decision details, confidence, urgency override)
- FAQ-assisted response snippets for relevant general inquiries
- Dashboard artifacts:
  - `outputs/dashboard_metrics.json`
  - `outputs/dashboard.html`

