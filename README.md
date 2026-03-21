# AI Automation & Customer Intelligence System

An end-to-end AI-driven customer support automation project for e-commerce support operations.

This project demonstrates how to:
- classify incoming customer queries into issue categories,
- automate low-risk responses,
- escalate uncertain/high-risk cases to human support,
- and convert support data into dashboard-ready business insights.

---

## Why This Project Matters (Interview Lens)

This project is built to show practical product + ML + automation thinking:
- **Business value:** faster response time, reduced support load, better issue visibility.
- **ML value:** confidence-aware prediction and routing, not just classification.
- **Operations value:** SLA-aware ticketing and audit logging.
- **Execution value:** complete pipeline from dataset to working dashboard.

If you are presenting this in interviews, this is a strong example of:
- solving a real workflow problem end-to-end,
- balancing automation with safe human escalation,
- and shipping usable deliverables.

---

## Problem Statement

Customer queries arrive from channels like WhatsApp, Email, Instagram DM, and Website Chat.  
The support team needs a system that can:
1. Understand the query intent (category),
2. Decide whether to auto-respond, create a ticket, or escalate,
3. Track outcomes and generate actionable support insights.

---

## Core Features

### 1) Query Categorization (AI)
- Model: **TF-IDF (1-3 gram) + Logistic Regression**
- Multi-class categories:
  - Order Status
  - Delivery Delay
  - Refund Request
  - General Inquiry
  - Payment Issue
  - Product Complaint

### 2) Confidence-Based Automation
- High confidence (`>= 0.50`) -> **Auto-Reply**
- Medium confidence (`>= 0.30` and `< 0.50`) -> **Ticket Creation**
- Low confidence (`< 0.30`) -> **Human Escalation**
- Urgent keyword override for critical terms (fraud, legal, missing, etc.)

### 3) Smart Response Layer
- Channel-specific response formatting
- FAQ-assisted dynamic responses for general inquiries
- Context-aware response enhancement via optional conversation history

### 4) Operational Tracking
- Ticket generation with SLA windows
- Audit logs for traceability and review
- Batch processing support

### 5) Problem Distribution Dashboard
- `% of total queries by category`
- `Most common customer problems`
- `Weekly/monthly trend` view (timeline proxy from query order)
- Self-contained `dashboard.html` suitable for submission/demo

---

## Project Structure

- `data/` -> datasets, predictions, logs, tickets
- `preprocessing/preprocess.py` -> text cleaning + feature engineering
- `model/classifier.py` -> training, evaluation, prediction export
- `automation/response_engine.py` -> routing logic + reply/ticket engine
- `dashboard/dashboard.py` -> dashboard metrics + self-contained HTML output
- `outputs/` -> generated dashboard artifacts
- `workflow_architecture.md` -> workflow explanation document
- `task.md` -> phase-wise progress tracker

---

## How It Works (End-to-End)

1. Input queries are ingested with metadata (`query_text`, `channel`, `priority`).
2. Preprocessing standardizes and enriches text.
3. Classifier predicts category + confidence.
4. Router decides action (auto-reply / ticket / escalation).
5. Ticket and audit outputs are written for operations.
6. Dashboard generation converts model outputs into business insights.

---

## Quick Start

From project root:

1. Preprocess data
```powershell
python "preprocessing/preprocess.py"
```

2. Train and evaluate model
```powershell
python "model/classifier.py"
```

3. Run automation engine demo
```powershell
python "automation/response_engine.py"
```

4. Generate dashboard
```powershell
python "dashboard/dashboard.py"
```

5. Open dashboard
- `outputs/dashboard.html`

---

## Interview Talking Points

Use these points when presenting:
- "I designed confidence-aware automation, not blind auto-reply."
- "I implemented a safe fallback path: uncertain/urgent cases go to humans."
- "I connected model outputs to business metrics via a working dashboard."
- "I focused on production behavior: SLA logic, ticketing, auditability."
- "I can scale this design with queue workers, DB persistence, and retraining loops."

---

## Measurable Outcomes You Can Claim

- Reduced manual triage by routing high-confidence cases automatically.
- Better support visibility via category distribution and trend metrics.
- Safer automation through threshold controls + urgent overrides.
- Clear path to production with scaling and reliability plan.

---

## Future Improvements

- Use true timestamps for trend charts (instead of sequence proxy)
- Add per-category threshold tuning
- Add database-backed persistence for tickets/logs
- Add active learning loop from human-corrected labels
- Add API/webhook layer for real-time channel integration

---

## Deliverables Completed

- Workflow explanation document
- Sample dataset
- AI categorization and automation logic
- Working dashboard
- Scaling and optimization plan

This repository is ready to be presented as a practical AI automation case study for interviews and job applications.

