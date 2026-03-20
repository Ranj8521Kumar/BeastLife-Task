# Scaling & Optimization Plan (Phase 9/10 Deliverable)

## 1) Performance improvements
1. Replace re-training with offline training + online inference
   - Train the hybrid TF-IDF + Logistic Regression model offline (CI/CD pipeline or scheduled job).
   - Load the serialized model (`model/hybrid_classifier.pkl`) for all runtime scoring.
2. Avoid heavy preprocessing at runtime
   - The current `automation/response_engine.py` uses lightweight cleaning.
   - For production, reuse the same preprocessing logic used during training (or ensure training/runtime text transforms match).
3. Batch inference
   - Use `process_batch()` for large message volumes.
   - Queue messages and score in batches to reduce per-request overhead.
4. Cache repeated queries / fingerprints
   - Cache by `(channel, normalized_query_text_hash)` to reuse predictions for duplicates.

## 2) Cost control
1. Use the classifier for the majority of queries
   - Confidence routing is designed to keep most traffic on `Auto-Reply`.
   - Human escalation should be the minority path.
2. Limit “human in the loop”
   - Keep a clear confidence threshold policy and enforce it in the router.
3. Reduce model compute footprint
   - TF-IDF + Logistic Regression is lightweight compared to LLM inference.
   - It can run on CPU instances without specialized GPUs.

## 3) Reliability & correctness
1. Monitoring & audit logs
   - Continue writing audit records (ticket id, action, confidence, urgent override).
   - Track drift: distribution changes in `confidence_tier`, categories, and urgent overrides.
2. Data validation
   - Validate required fields (`query_text`, `channel`, `priority`, `query_id`) before scoring.
3. Fallback behavior
   - If the model file is missing/unreadable, route to `Human Escalation` safely.
4. SLA policy enforcement
   - The `TicketManager` uses SLA windows by priority.
   - Ensure ticket creation is idempotent (avoid duplicate tickets for retries).

## 4) Scalability architecture (recommended)
1. Ingress
   - WhatsApp/Email/Chat webhook receives messages and normalizes them into an internal request object.
2. Orchestration
   - A worker service runs `ResponseEngine.process_query()` or `process_batch()`.
3. Persistence
   - Tickets + audit events stored in a database (replace `tickets.json` with a DB for concurrency).
4. Analytics
   - Dashboard reads from stored audit/prediction tables to compute KPIs at any time.

## 5) Next steps to reach production readiness
1. Implement persistent storage
   - Replace `outputs/tickets.json` with a relational DB or document store.
2. Add evaluation retraining loop
   - Periodically label new queries and retrain to reduce misclassification hotspots.
3. Expand automation templates
   - Improve auto-reply placeholders (more robust ref id / tracking links).
4. Add more robust urgency detection
   - Use both keyword overrides and an urgency classifier signal (optional hybrid approach).

