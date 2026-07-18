"""
expert_validation.py — Backend workflow for Expert Validation
================================================================
Flow required by review feedback (poin #4):

    Threshold Alert -> Pending Review -> Expert Approve/Reject
                     -> Save Label -> Training Dataset

Purpose: `get_threshold_alerts()` (astra_anomaly_detection.py) may flag
a reading as `is_labeling_candidate=True` — a sensor pattern the
threshold rules consider abnormal, which may or may not be something
the AI model already understands. This module turns that flag into a
reviewable queue item, lets a human expert confirm/reject the true
condition, and — only on approval — appends a properly labeled row to
a growing training dataset that `retrain_pipeline.py` can pick up.

Scope: BACKEND ONLY. No frontend/UI here — El Shaddai wires these
functions to routes/buttons. Storage is a simple JSON-Lines "queue"
file + CSV append for approved labels, so it works with zero extra
infrastructure (no DB required) and is trivially swappable for a real
database later (see `NOTE: swap storage` comments below).
"""

import os
import json
import uuid
import csv
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional, List, Dict

from astra_config import SENSOR_COLS, LABEL_ORDER, FAULT_ORDER

# ------------------------------------------------------------------
# Storage locations
# ------------------------------------------------------------------
# NOTE: swap storage — replace these two file-backed stores with a
# real table (e.g. `review_queue`, `expert_labeled_data`) in
# Postgres/MySQL when the app moves off flat files. The function
# signatures below are written so that swap only touches the
# _load_queue/_save_queue/_append_labeled_row internals.
DATA_DIR          = os.environ.get("ASTRA_DATA_DIR", "expert_validation_data")
QUEUE_PATH        = os.path.join(DATA_DIR, "review_queue.jsonl")
LABELED_DATA_PATH = os.path.join(DATA_DIR, "expert_labeled_data.csv")

VALID_STATUSES = ("pending", "approved", "rejected")


def _ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(QUEUE_PATH):
        open(QUEUE_PATH, "a").close()
    if not os.path.exists(LABELED_DATA_PATH):
        with open(LABELED_DATA_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Timestamp", "Motor_ID"] + SENSOR_COLS +
                ["Motor_State", "Fault_Type_True", "Expert_ID",
                 "Reviewed_At", "Review_Notes", "Source_Review_ID"]
            )


@dataclass
class ReviewItem:
    """One row in the Expert Validation queue."""
    review_id:    str
    motor_id:     Optional[str]
    timestamp:    Optional[str]
    sensor_data:  Dict[str, float]           # raw SENSOR_COLS values
    threshold_alert: Dict                    # full get_threshold_alerts() output
    status:       str = "pending"            # pending / approved / rejected
    expert_id:    Optional[str] = None
    expert_label: Optional[str] = None       # Motor_State chosen by expert
    expert_fault_type: Optional[str] = None  # Fault_Type_True chosen by expert
    notes:        Optional[str] = None
    created_at:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_at:  Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------
# Low-level storage helpers (JSON-Lines queue)
# ------------------------------------------------------------------

def _load_queue() -> List[dict]:
    _ensure_storage()
    items = []
    with open(QUEUE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _save_queue(items: List[dict]):
    _ensure_storage()
    with open(QUEUE_PATH, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _append_labeled_row(item: dict):
    _ensure_storage()
    with open(LABELED_DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        sd = item["sensor_data"]
        writer.writerow(
            [item.get("timestamp"), item.get("motor_id")] +
            [sd.get(c) for c in SENSOR_COLS] +
            [item["expert_label"], item["expert_fault_type"],
             item["expert_id"], item["reviewed_at"], item.get("notes") or "",
             item["review_id"]]
        )


# ------------------------------------------------------------------
# STEP 1 — Threshold Alert -> Pending Review
# ------------------------------------------------------------------

def submit_for_review(threshold_alert_result: dict, sensor_data: dict) -> ReviewItem:
    """
    Called right after `get_threshold_alerts()` returns
    `is_labeling_candidate=True`. Creates a new PENDING review item.

    Typical call site (in Flask, after hitting /api/threshold-alerts):
        alert = get_threshold_alerts(sensor_data, motor_id, timestamp)
        if alert["is_labeling_candidate"]:
            review = submit_for_review(alert, sensor_data)

    Returns the created ReviewItem (status='pending').
    """
    if not threshold_alert_result.get("is_labeling_candidate"):
        raise ValueError(
            "Only alerts with is_labeling_candidate=True should be "
            "submitted for review (source: get_threshold_alerts())."
        )

    item = ReviewItem(
        review_id=str(uuid.uuid4()),
        motor_id=threshold_alert_result.get("motor_id"),
        timestamp=threshold_alert_result.get("timestamp"),
        sensor_data={k: sensor_data.get(k) for k in SENSOR_COLS},
        threshold_alert=threshold_alert_result,
        status="pending",
    )
    items = _load_queue()
    items.append(item.to_dict())
    _save_queue(items)
    return item


# ------------------------------------------------------------------
# STEP 2 — Read the queue
# ------------------------------------------------------------------

def list_reviews(status: Optional[str] = "pending",
                  motor_id: Optional[str] = None) -> List[ReviewItem]:
    """
    List review items, optionally filtered by status ('pending',
    'approved', 'rejected', or None for all) and/or motor_id.

    Used by El's admin "Condition Alerts / Review Queue" page to
    populate the review list.
    """
    items = _load_queue()
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}")
        items = [i for i in items if i["status"] == status]
    if motor_id is not None:
        items = [i for i in items if i["motor_id"] == motor_id]
    return [ReviewItem(**i) for i in items]


def get_review(review_id: str) -> Optional[ReviewItem]:
    items = _load_queue()
    for i in items:
        if i["review_id"] == review_id:
            return ReviewItem(**i)
    return None


# ------------------------------------------------------------------
# STEP 3 — Expert Approve / Reject -> Save Label -> Training Dataset
# ------------------------------------------------------------------

def approve_review(review_id: str, expert_id: str,
                    expert_label: str, expert_fault_type: str,
                    notes: Optional[str] = None) -> ReviewItem:
    """
    Expert confirms the reading and assigns the TRUE condition label.

    On approval:
      1. Queue item status -> 'approved'
      2. A fully labeled row (raw sensors + Motor_State + Fault_Type_True)
         is appended to expert_labeled_data.csv
      3. That CSV is picked up by retrain_pipeline.py's
         `load_combined_training_data()` on the next retrain run.

    Args:
        review_id: id from submit_for_review() / list_reviews()
        expert_id: identifier of the reviewing expert (for audit trail)
        expert_label: one of LABEL_ORDER ('Normal'/'Warning'/'Critical'/'Failure')
        expert_fault_type: one of FAULT_ORDER
        notes: optional free-text justification
    """
    if expert_label not in LABEL_ORDER:
        raise ValueError(f"expert_label must be one of {LABEL_ORDER}")
    if expert_fault_type not in FAULT_ORDER:
        raise ValueError(f"expert_fault_type must be one of {FAULT_ORDER}")

    items = _load_queue()
    for i in items:
        if i["review_id"] == review_id:
            if i["status"] != "pending":
                raise ValueError(
                    f"Review {review_id} already {i['status']}, cannot re-approve."
                )
            i["status"]            = "approved"
            i["expert_id"]         = expert_id
            i["expert_label"]      = expert_label
            i["expert_fault_type"] = expert_fault_type
            i["notes"]             = notes
            i["reviewed_at"]       = datetime.now(timezone.utc).isoformat()
            _save_queue(items)
            _append_labeled_row(i)
            return ReviewItem(**i)
    raise KeyError(f"Review {review_id} not found")


def reject_review(review_id: str, expert_id: str,
                   notes: Optional[str] = None) -> ReviewItem:
    """
    Expert reviews the reading and determines it is NOT a valid new
    pattern (e.g. sensor glitch, noise, already-known normal variation).

    Rejected items are kept in the queue (status='rejected') for audit
    purposes but are NOT added to the training dataset.
    """
    items = _load_queue()
    for i in items:
        if i["review_id"] == review_id:
            if i["status"] != "pending":
                raise ValueError(
                    f"Review {review_id} already {i['status']}, cannot re-reject."
                )
            i["status"]      = "rejected"
            i["expert_id"]   = expert_id
            i["notes"]       = notes
            i["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            _save_queue(items)
            return ReviewItem(**i)
    raise KeyError(f"Review {review_id} not found")


def review_stats() -> dict:
    """Quick counts for an admin dashboard widget."""
    items = _load_queue()
    stats = {s: 0 for s in VALID_STATUSES}
    for i in items:
        stats[i["status"]] = stats.get(i["status"], 0) + 1
    stats["total"] = len(items)
    return stats


# ------------------------------------------------------------------
# Self-test (run: python expert_validation.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)

    sample_sensor = {
        'Voltage_L1': 401.2, 'Voltage_L2': 399.8, 'Voltage_L3': 400.5,
        'Current_L1': 8.4,   'Current_L2': 8.35,  'Current_L3': 8.5,
        'Frequency': 50.02, 'Power_Factor': 1.0, 'Temperature': 63.0,
        'Vibration_X': 1.4, 'Vibration_Y': 1.35, 'Vibration_Z': 1.2,
        'Rotational_Speed': 1471.0,
    }
    fake_threshold_alert = {
        "motor_id": "MTR-007", "timestamp": "2026-07-16T10:00:00Z",
        "condition_label": "Critical", "status_color": "orange",
        "violations": [{"parameter": "Current_L1", "tier": "critical",
                         "actual_value": 8.4, "threshold": 7.9}],
        "total_violations": 1, "is_labeling_candidate": True,
        "source": "threshold_rule",
    }

    print("1) Submit for review...")
    review = submit_for_review(fake_threshold_alert, sample_sensor)
    print(f"   review_id={review.review_id} status={review.status}")

    print("2) List pending reviews...")
    pending = list_reviews(status="pending")
    print(f"   pending count = {len(pending)}")
    assert len(pending) == 1

    print("3) Expert approves with true label...")
    approved = approve_review(
        review.review_id, expert_id="expert_wahyu",
        expert_label="Critical", expert_fault_type="Stator Winding",
        notes="Confirmed via thermal camera + megger test.",
    )
    print(f"   status={approved.status} reviewed_at={approved.reviewed_at}")

    print("4) Check training dataset got the row...")
    with open(LABELED_DATA_PATH) as f:
        rows = f.readlines()
    print(f"   {len(rows)-1} labeled row(s) in {LABELED_DATA_PATH}")
    assert len(rows) == 2  # header + 1 row

    print("5) Stats...")
    print("  ", review_stats())

    print("\nAll self-tests passed.")
