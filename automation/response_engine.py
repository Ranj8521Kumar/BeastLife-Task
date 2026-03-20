"""
=============================================================
  AI Customer Intelligence System
  PHASE 6: Automation Logic — Response Engine
=============================================================
  Handles:
    1. Auto-Reply Generation     (High Confidence >= 0.50)
    2. Ticket Creation           (Medium Confidence 0.30-0.49)
    3. Human Escalation          (Low Confidence  < 0.30)
    4. SLA Timer Assignment      (priority-based deadlines)
    5. Keyword Override Rules    (urgent words bypass scoring)
    6. Channel-specific Routing  (WhatsApp / Email / Instagram / Chat)
    7. Audit Logging             (full trace per query)
=============================================================
"""

import os
import sys
import json
import uuid
import pickle
import datetime
import numpy as np
import pandas as pd

# ── Add project root to path ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

MODEL_PATH   = os.path.join(BASE_DIR, "model", "hybrid_classifier.pkl")
DATA_PATH    = os.path.join(BASE_DIR, "data", "preprocessed_queries.csv")
RAW_PATH     = os.path.join(BASE_DIR, "data", "customer_queries.csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data")
TICKETS_PATH = os.path.join(OUTPUT_DIR, "tickets.json")
LOG_PATH     = os.path.join(OUTPUT_DIR, "automation_log.csv")


# ═══════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Confidence thresholds (match classifier.py)
THRESHOLD_HIGH   = 0.50   # Auto-Reply
THRESHOLD_MEDIUM = 0.30   # Ticket Creation
# Below 0.30 => Human Escalation

# SLA response deadlines (in hours) by priority
SLA_CONFIG = {
    "High"   : {"first_response": 1,   "resolution": 4  },
    "Medium" : {"first_response": 4,   "resolution": 24 },
    "Low"    : {"first_response": 24,  "resolution": 72 },
}

# Urgency keyword overrides — triggers immediate High routing
URGENT_OVERRIDE_KEYWORDS = [
    "urgent", "immediately", "emergency", "asap",
    "payment failed", "deducted", "fraud", "scam",
    "police", "legal", "lawsuit", "missing", "broken",
    "never received", "wrong item", "defective",
]

# Channel-specific prefixes for replies
CHANNEL_GREETINGS = {
    "WhatsApp"    : "Hi! Thanks for messaging us on WhatsApp.",
    "Email"       : "Dear Customer,\n\nThank you for reaching out via email.",
    "Instagram DM": "Hey! Thanks for DMing us on Instagram.",
    "Website Chat": "Hello! Thanks for contacting us on our website.",
}

CHANNEL_CLOSINGS = {
    "WhatsApp"    : "\n\nReply anytime — we're here 24/7! 💬",
    "Email"       : "\n\nBest regards,\nCustomer Support Team\nsupport@beastlife.com",
    "Instagram DM": "\n\nFeel free to DM us anytime! 📩",
    "Website Chat": "\n\nIs there anything else I can help you with today?",
}


# ═══════════════════════════════════════════════════════════════
# SECTION 2: AUTO-REPLY TEMPLATES
# ═══════════════════════════════════════════════════════════════

AUTO_REPLY_TEMPLATES = {
    "Order Status": {
        "subject" : "Your Order Status Update",
        "body"    : (
            "Your order is currently being processed by our fulfillment team. "
            "You can track your order in real-time at: [tracking_link]. "
            "Expected delivery: 2-5 business days. "
            "If you need further assistance, our team is available 24/7."
        ),
    },
    "Delivery Delay": {
        "subject" : "Update Regarding Your Delivery",
        "body"    : (
            "We sincerely apologize for the inconvenience caused by the delivery delay. "
            "Our logistics team has been notified and is actively working to expedite your shipment. "
            "You will receive a detailed update within the next 24 hours. "
            "As a token of apology, a discount coupon has been added to your account."
        ),
    },
    "Refund Request": {
        "subject" : "Refund Request Received — Ref: [ref_id]",
        "body"    : (
            "We have successfully received your refund request (Reference: [ref_id]). "
            "Your refund of [amount] will be credited to your original payment method "
            "within 5-7 business days. "
            "You will receive a confirmation email once the refund has been initiated."
        ),
    },
    "General Inquiry": {
        "subject" : "Thanks for Reaching Out!",
        "body"    : (
            "Thank you for your query. Our support team has received your message "
            "and will get back to you with a detailed response shortly. "
            "In the meantime, you can visit our FAQ at: [faq_link] "
            "for quick answers to common questions."
        ),
    },
    "Payment Issue": {
        "subject" : "Payment Issue — Immediate Assistance",
        "body"    : (
            "We're sorry to hear about the payment issue you've experienced. "
            "Our payment team has been alerted and is investigating this urgently. "
            "If an amount was deducted, it will be automatically refunded within 3-5 business days. "
            "Your transaction reference will be shared via email within 2 hours."
        ),
    },
    "Product Complaint": {
        "subject" : "We're Sorry About Your Product Experience",
        "body"    : (
            "We sincerely apologize for the experience with your product. "
            "This is not the standard we hold ourselves to. "
            "Please share photos/videos of the issue by replying to this message, "
            "and we will arrange an immediate replacement or full refund — "
            "whichever you prefer."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════
# SECTION 3: TICKET CREATION
# ═══════════════════════════════════════════════════════════════

class TicketManager:
    """Creates and manages support tickets."""

    def __init__(self, tickets_path: str = TICKETS_PATH):
        self.path = tickets_path
        self.tickets = self._load()

    def _load(self) -> list:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.tickets, f, indent=2, default=str)

    def create_ticket(self, query_id: str, channel: str, category: str,
                      raw_query: str, confidence: float, priority: str,
                      agent: str = "AUTO") -> dict:
        """Create a new support ticket."""
        now = datetime.datetime.now()
        sla = SLA_CONFIG.get(priority, SLA_CONFIG["Medium"])

        ticket = {
            "ticket_id"         : f"TKT-{uuid.uuid4().hex[:8].upper()}",
            "query_id"          : query_id,
            "channel"           : channel,
            "category"          : category,
            "raw_query"         : raw_query,
            "confidence_score"  : round(confidence, 4),
            "priority"          : priority,
            "status"            : "Open",
            "assigned_to"       : agent,
            "created_at"        : now.isoformat(),
            "first_response_by" : (now + datetime.timedelta(hours=sla["first_response"])).isoformat(),
            "resolution_by"     : (now + datetime.timedelta(hours=sla["resolution"])).isoformat(),
            "sla_first_response": f"{sla['first_response']}h",
            "sla_resolution"    : f"{sla['resolution']}h",
            "notes"             : [],
        }
        self.tickets.append(ticket)
        self._save()
        return ticket

    def get_open_tickets(self) -> list:
        return [t for t in self.tickets if t["status"] == "Open"]

    def summary(self) -> dict:
        total = len(self.tickets)
        by_status   = {}
        by_category = {}
        by_priority = {}
        for t in self.tickets:
            by_status[t["status"]]     = by_status.get(t["status"], 0) + 1
            by_category[t["category"]] = by_category.get(t["category"], 0) + 1
            by_priority[t["priority"]] = by_priority.get(t["priority"], 0) + 1
        return {
            "total"          : total,
            "by_status"      : by_status,
            "by_category"    : by_category,
            "by_priority"    : by_priority,
        }


# ═══════════════════════════════════════════════════════════════
# SECTION 4: RESPONSE ENGINE (CORE)
# ═══════════════════════════════════════════════════════════════

class ResponseEngine:
    """
    Core automation engine.
    Loads the trained model and routes each query to the right action.
    """

    def __init__(self):
        self.model          = self._load_model()
        self.ticket_manager = TicketManager()
        self.log_records    = []

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run model/classifier.py first."
            )
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # ── Preprocessing (lightweight, mirrors classifier) ─────────
    def _clean(self, text: str) -> str:
        import re, string
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", text).strip()

    # ── Urgency keyword check ───────────────────────────────────
    def _check_urgent(self, raw_text: str) -> bool:
        lower = raw_text.lower()
        return any(kw in lower for kw in URGENT_OVERRIDE_KEYWORDS)

    # ── Build personalized reply ─────────────────────────────────
    def _build_reply(self, category: str, channel: str,
                     ref_id: str, action: str) -> str:
        greeting = CHANNEL_GREETINGS.get(channel, "Hello!")
        closing  = CHANNEL_CLOSINGS.get(channel, "\n\nThank you.")
        template = AUTO_REPLY_TEMPLATES.get(category, {})
        body     = template.get("body", "We have received your query and will respond shortly.")

        # Fill placeholders
        body = body.replace("[ref_id]", ref_id)
        body = body.replace("[tracking_link]", "https://track.beastlife.com/" + ref_id)
        body = body.replace("[faq_link]", "https://help.beastlife.com/faq")
        body = body.replace("[amount]", "the debited amount")

        if action == "Auto-Reply":
            return f"{greeting}\n\n{body}{closing}"
        elif action == "Ticket Creation":
            return (
                f"{greeting}\n\nThank you for reaching out. "
                f"We've created a support ticket for your query (Ref: {ref_id}). "
                f"A dedicated agent will respond within the SLA window.{closing}"
            )
        else:  # Human Escalation
            return (
                f"{greeting}\n\nYour query requires personalized attention. "
                f"We're connecting you with a live support agent right now. "
                f"Estimated wait time: 3-5 minutes. Your reference: {ref_id}.{closing}"
            )

    # ── MAIN ROUTING FUNCTION ────────────────────────────────────
    def process_query(self, query_id: str, raw_query: str,
                      channel: str = "Website Chat",
                      priority: str = "Medium") -> dict:
        """
        Process a single customer query end-to-end.
        Returns a full decision record.
        """
        ref_id    = f"REF-{uuid.uuid4().hex[:6].upper()}"
        cleaned   = self._clean(raw_query)
        is_urgent = self._check_urgent(raw_query)

        # Model prediction
        probs      = self.model.predict_proba([cleaned])[0]
        category   = self.model.classes_[int(np.argmax(probs))]
        confidence = float(np.max(probs))

        # Determine confidence tier
        if is_urgent or confidence >= THRESHOLD_HIGH:
            tier   = "High"
            action = "Auto-Reply"
        elif confidence >= THRESHOLD_MEDIUM:
            tier   = "Medium"
            action = "Ticket Creation"
        else:
            tier   = "Low"
            action = "Human Escalation"

        # Escalation overrides priority to High if urgent
        effective_priority = "High" if is_urgent else priority

        # SLA
        sla = SLA_CONFIG[effective_priority]

        # Build reply
        reply = self._build_reply(category, channel, ref_id, action)

        # Create ticket (for Medium + Low)
        ticket = None
        if action in ("Ticket Creation", "Human Escalation"):
            agent  = "Human Agent" if action == "Human Escalation" else "Bot-Tier2"
            ticket = self.ticket_manager.create_ticket(
                query_id   = query_id,
                channel    = channel,
                category   = category,
                raw_query  = raw_query,
                confidence = confidence,
                priority   = effective_priority,
                agent      = agent,
            )

        # Assemble record
        record = {
            "query_id"          : query_id,
            "ref_id"            : ref_id,
            "channel"           : channel,
            "raw_query"         : raw_query,
            "predicted_category": category,
            "confidence_score"  : round(confidence, 4),
            "confidence_tier"   : tier,
            "urgent_override"   : is_urgent,
            "action"            : action,
            "effective_priority": effective_priority,
            "sla_first_response": f"{sla['first_response']}h",
            "sla_resolution"    : f"{sla['resolution']}h",
            "auto_reply"        : reply,
            "ticket_id"         : ticket["ticket_id"] if ticket else "N/A",
            "timestamp"         : datetime.datetime.now().isoformat(),
        }
        self.log_records.append(record)
        return record

    # ── BATCH PROCESSING ─────────────────────────────────────────
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all queries in a DataFrame."""
        results = []
        for _, row in df.iterrows():
            qid      = str(row.get("query_id", f"Q{_:03d}"))
            query    = str(row.get("query_text", row.get("raw_query", "")))
            channel  = str(row.get("channel",  "Website Chat"))
            priority = str(row.get("priority", "Medium"))
            record   = self.process_query(qid, query, channel, priority)
            results.append(record)
        return pd.DataFrame(results)

    # ── SAVE LOG ─────────────────────────────────────────────────
    def save_log(self, path: str = LOG_PATH):
        if self.log_records:
            pd.DataFrame(self.log_records).to_csv(path, index=False, encoding="utf-8")
            print(f"  [INFO] Automation log saved -> {path}")


# ═══════════════════════════════════════════════════════════════
# SECTION 5: DEMO RUNNER
# ═══════════════════════════════════════════════════════════════

def print_result(r: dict):
    """Pretty-print a single routing result."""
    tier_icon = {"High": "[HIGH]", "Medium": "[MED]", "Low": "[LOW]"}.get(r["confidence_tier"], "")
    print(f"\n  {'='*52}")
    print(f"  Query ID   : {r['query_id']}")
    print(f"  Channel    : {r['channel']}")
    print(f"  Query      : {r['raw_query'][:70]}...")
    print(f"  Category   : {r['predicted_category']}")
    print(f"  Confidence : {r['confidence_score']:.4f}  {tier_icon}")
    if r["urgent_override"]:
        print(f"  ** URGENT KEYWORD OVERRIDE **")
    print(f"  Action     : {r['action']}")
    print(f"  Priority   : {r['effective_priority']}")
    print(f"  SLA        : Response in {r['sla_first_response']} | Resolve in {r['sla_resolution']}")
    print(f"  Ticket     : {r['ticket_id']}")
    print(f"  Reply Preview:")
    print(f"  {'-'*50}")
    for line in r["auto_reply"].split("\n")[:5]:
        print(f"    {line}")
    print(f"  {'-'*50}")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI CUSTOMER INTELLIGENCE - RESPONSE ENGINE")
    print("="*55)

    engine = ResponseEngine()

    # --- DEMO 1: Live single-query tests -------------------------
    print("\n[SECTION 1] LIVE ROUTING TESTS (6 sample queries)\n")
    sample_cases = [
        ("Q001", "Where is my order #45231? It was supposed to arrive yesterday!",
         "WhatsApp", "High"),
        ("Q002", "My payment failed but money was deducted from my account URGENT",
         "WhatsApp", "High"),
        ("Q003", "I want a refund for my damaged product please.",
         "Email", "Medium"),
        ("Q004", "Do you offer free shipping on orders above 500?",
         "Website Chat", "Low"),
        ("Q005", "Product delivered was completely wrong. I got someone else order.",
         "Instagram DM", "High"),
        ("Q006", "kya aap mujhe delivery date bata sakte hain?",
         "WhatsApp", "Low"),
    ]

    for qid, query, channel, priority in sample_cases:
        result = engine.process_query(qid, query, channel, priority)
        print_result(result)

    # --- DEMO 2: Batch processing from raw CSV -------------------
    print("\n\n[SECTION 2] BATCH PROCESSING (full dataset)\n")
    raw_df = pd.read_csv(RAW_PATH)
    batch_df = engine.process_batch(raw_df)
    batch_df.to_csv(os.path.join(OUTPUT_DIR, "automation_results.csv"),
                    index=False, encoding="utf-8")

    print(f"  Processed   : {len(batch_df)} queries")
    print(f"\n  Action Distribution:")
    print(batch_df["action"].value_counts().to_string())
    print(f"\n  Confidence Tier Split:")
    print(batch_df["confidence_tier"].value_counts().to_string())
    print(f"\n  Top Categories Predicted:")
    print(batch_df["predicted_category"].value_counts().to_string())

    # --- DEMO 3: Ticket summary ----------------------------------
    print("\n\n[SECTION 3] TICKET MANAGER SUMMARY\n")
    summary = engine.ticket_manager.summary()
    print(f"  Total Tickets   : {summary['total']}")
    print(f"  By Status       : {summary['by_status']}")
    print(f"  By Priority     : {summary['by_priority']}")
    print(f"  By Category     : {json.dumps(summary['by_category'], indent=4)}")

    # --- Save log -----------------------------------------------
    engine.save_log()
    print(f"\n  Automation results -> {OUTPUT_DIR}/automation_results.csv")
    print(f"  Tickets database   -> {OUTPUT_DIR}/tickets.json")
    print(f"  Audit log          -> {OUTPUT_DIR}/automation_log.csv")

    print("\n" + "="*55)
    print("  [OK] Phase 6 Complete - Automation Engine Ready")
    print("="*55 + "\n")
