"""
retrain_pipeline.py — Versioned retraining with gated deployment
====================================================================
Design (per review feedback poin #5):

    1. Combine original training data + expert-approved labels
       (expert_validation.py -> expert_labeled_data.csv)
    2. Train a CANDIDATE model (new version, not yet live)
    3. Evaluate candidate against PREDEFINED minimum metrics
    4. Deploy ONLY if candidate passes the gate; otherwise keep the
       currently-deployed (production) version untouched
    5. Every trained candidate — pass or fail — is versioned and
       archived under models/vN/ with a metadata.json for audit trail

Reuses the SAME feature engineering / thresholds as
astra_anomaly_detection.py via astra_config.py (see astra_config.py
docstring) so the retrained model is guaranteed consistent with the
main pipeline's feature space — no drift between "how the model was
first trained" and "how it gets retrained".

This module does NOT touch astra_anomaly_detection.py's own training
run; it is a separate, callable pipeline (e.g. triggered from an admin
"Retrain" button via Flask, or a scheduled job).
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ['OMP_NUM_THREADS'] = '1'

import json
import shutil
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

from astra_config import (
    SENSOR_COLS, FEATURE_COLS, LABEL_COL, LABEL_ORDER,
    FAULT_COL, FAULT_ORDER, add_engineered_features,
)

# ------------------------------------------------------------------
# Registry layout
# ------------------------------------------------------------------
# models/
#   registry.json          <- points to the currently DEPLOYED version
#   v1/  metadata.json, condition_classifier.joblib, fault_type_classifier.joblib,
#        isolation_forest.joblib, condition_scaler.joblib
#   v2/  ... (candidate that failed the gate stays here too, for audit,
#             but registry.json is NOT updated to point at it)
MODELS_DIR    = os.environ.get("ASTRA_MODELS_DIR", "models")
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")

# ------------------------------------------------------------------
# DEPLOYMENT GATE — predefined minimum metrics.
# A candidate is deployed ONLY if it clears ALL of these versus the
# currently deployed model's metrics (or absolute floors on first run).
# Adjust with care — these are the quality bar for going live.
# ------------------------------------------------------------------
DEPLOYMENT_GATE = {
    # Absolute floors (must always be met, regardless of current model)
    "min_condition_f1_macro": 0.75,
    "min_fault_f1_macro":     0.80,
    "min_condition_accuracy": 0.85,
    # Relative requirement: candidate must not regress vs current
    # deployed model by more than this tolerance (protects against a
    # "technically passes the floor but is worse than before" deploy)
    "max_f1_macro_regression": 0.01,
}


def _ensure_registry():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w") as f:
            json.dump({"deployed_version": None, "history": []}, f, indent=2)


def _load_registry() -> dict:
    _ensure_registry()
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _save_registry(reg: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def _next_version(reg: dict) -> int:
    existing = [h["version"] for h in reg["history"]]
    return (max(existing) + 1) if existing else 1


# ------------------------------------------------------------------
# STEP 1 — Combine original + expert-approved data
# ------------------------------------------------------------------

def load_combined_training_data(
    base_csv: str = "client_training_dataset.csv",
    expert_csv: str = os.path.join("expert_validation_data", "expert_labeled_data.csv"),
) -> pd.DataFrame:
    """
    Merge the original client dataset with any expert-approved labels
    collected via expert_validation.py's Approve workflow.

    Expert rows use the same SENSOR_COLS + Motor_State + Fault_Type_True
    schema (see expert_validation.LABELED_DATA_PATH). Only SENSOR_COLS +
    LABEL_COL + FAULT_COL are required for training.
    """
    base = pd.read_csv(base_csv)
    n_base = len(base)
    keep_cols = SENSOR_COLS + [LABEL_COL, FAULT_COL]

    if os.path.exists(expert_csv) and os.path.getsize(expert_csv) > 0:
        expert = pd.read_csv(expert_csv)
        expert = expert[[c for c in keep_cols if c in expert.columns]]
        combined = pd.concat([base[keep_cols], expert], ignore_index=True)
        n_expert = len(expert)
    else:
        combined = base[keep_cols].copy()
        n_expert = 0

    print(f"  Base dataset rows         : {n_base:,}")
    print(f"  Expert-approved rows added: {n_expert:,}")
    print(f"  Combined training rows    : {len(combined):,}")
    return combined


# ------------------------------------------------------------------
# STEP 2 — Train a candidate model + evaluate
# ------------------------------------------------------------------

def train_candidate(df: pd.DataFrame) -> dict:
    """
    Train condition classifier + fault-type classifier on `df` using
    the SAME feature engineering as astra_anomaly_detection.py.

    Uses one shared train/test split + one scaler for BOTH classifiers
    (same leakage-safe design used in astra_anomaly_detection.py's
    ASTRA-10 section — see that file for the rationale).

    Returns a dict with fitted models, scaler, and evaluation metrics
    (accuracy, precision/recall/F1 macro, confusion matrices).
    """
    df = add_engineered_features(df)
    X = df[FEATURE_COLS].copy()
    y_cond  = df[LABEL_COL].copy()
    y_fault = df[FAULT_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cond, test_size=0.2, random_state=42, stratify=y_cond
    )
    yf_train = y_fault.loc[X_train.index]
    yf_test  = y_fault.loc[X_test.index]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    cond_clf = RandomForestClassifier(
        n_estimators=30, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=1
    )
    cond_clf.fit(X_train_sc, y_train)
    cond_pred = cond_clf.predict(X_test_sc)

    fault_clf = RandomForestClassifier(
        n_estimators=40, max_depth=18, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=1
    )
    fault_clf.fit(X_train_sc, yf_train)
    fault_pred = fault_clf.predict(X_test_sc)

    contam = min(0.499, round((y_train != 'Normal').sum() / len(y_train), 3))
    iso = IsolationForest(contamination=contam, n_estimators=100,
                          random_state=42, n_jobs=1)
    iso.fit(X_train_sc[y_train == 'Normal'])

    metrics = {
        "condition": {
            "accuracy":   round(accuracy_score(y_test, cond_pred), 4),
            "f1_macro":   round(f1_score(y_test, cond_pred, average="macro", zero_division=0), 4),
            "precision_macro": round(precision_score(y_test, cond_pred, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(y_test, cond_pred, average="macro", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, cond_pred, labels=LABEL_ORDER).tolist(),
            "classification_report": classification_report(
                y_test, cond_pred, labels=LABEL_ORDER, target_names=LABEL_ORDER,
                zero_division=0, output_dict=True),
        },
        "fault_type": {
            "accuracy":   round(accuracy_score(yf_test, fault_pred), 4),
            "f1_macro":   round(f1_score(yf_test, fault_pred, average="macro", zero_division=0), 4),
            "precision_macro": round(precision_score(yf_test, fault_pred, average="macro", zero_division=0), 4),
            "recall_macro":    round(recall_score(yf_test, fault_pred, average="macro", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(yf_test, fault_pred, labels=FAULT_ORDER).tolist(),
            "classification_report": classification_report(
                yf_test, fault_pred, labels=FAULT_ORDER, target_names=FAULT_ORDER,
                zero_division=0, output_dict=True),
        },
        "n_train": len(X_train), "n_test": len(X_test),
    }

    return {
        "condition_classifier": cond_clf,
        "fault_type_classifier": fault_clf,
        "isolation_forest": iso,
        "scaler": scaler,
        "metrics": metrics,
    }


# ------------------------------------------------------------------
# STEP 3 — Gate check
# ------------------------------------------------------------------

def passes_gate(candidate_metrics: dict, deployed_metrics: Optional[dict],
                 gate: dict = DEPLOYMENT_GATE):
    """
    Returns (passed: bool, reasons: list[str]).
    `reasons` explains every failed check (empty list if passed).
    """
    reasons = []
    c = candidate_metrics["condition"]
    f = candidate_metrics["fault_type"]

    if c["f1_macro"] < gate["min_condition_f1_macro"]:
        reasons.append(
            f"condition F1 macro {c['f1_macro']} < floor {gate['min_condition_f1_macro']}")
    if c["accuracy"] < gate["min_condition_accuracy"]:
        reasons.append(
            f"condition accuracy {c['accuracy']} < floor {gate['min_condition_accuracy']}")
    if f["f1_macro"] < gate["min_fault_f1_macro"]:
        reasons.append(
            f"fault-type F1 macro {f['f1_macro']} < floor {gate['min_fault_f1_macro']}")

    if deployed_metrics is not None:
        prev_f1 = deployed_metrics["condition"]["f1_macro"]
        drop = prev_f1 - c["f1_macro"]
        if drop > gate["max_f1_macro_regression"]:
            reasons.append(
                f"condition F1 macro regressed by {drop:.4f} "
                f"(deployed={prev_f1}, candidate={c['f1_macro']}), "
                f"tolerance={gate['max_f1_macro_regression']}")

    return (len(reasons) == 0), reasons


# ------------------------------------------------------------------
# STEP 4 — Save candidate (always) + deploy (only if it passes)
# ------------------------------------------------------------------

def _save_candidate(version: int, trained: dict) -> str:
    vdir = os.path.join(MODELS_DIR, f"v{version}")
    os.makedirs(vdir, exist_ok=True)
    joblib.dump(trained["condition_classifier"], os.path.join(vdir, "condition_classifier.joblib"))
    joblib.dump(trained["fault_type_classifier"], os.path.join(vdir, "fault_type_classifier.joblib"))
    joblib.dump(trained["isolation_forest"], os.path.join(vdir, "isolation_forest.joblib"))
    joblib.dump(trained["scaler"], os.path.join(vdir, "condition_scaler.joblib"))
    return vdir


def run_retraining(base_csv: str = "client_training_dataset.csv",
                    expert_csv: str = os.path.join("expert_validation_data", "expert_labeled_data.csv"),
                    triggered_by: str = "manual") -> dict:
    """
    Full pipeline entry point — call this from the admin "Retrain"
    endpoint. Trains a candidate, evaluates it, versions it, and
    deploys it ONLY if it clears DEPLOYMENT_GATE.

    Returns a summary dict suitable for returning as JSON from Flask:
        {version, deployed (bool), gate_failure_reasons, metrics, model_dir}
    """
    reg = _load_registry()
    version = _next_version(reg)
    print(f"[retrain_pipeline] Starting retraining -> candidate v{version} "
          f"(triggered_by={triggered_by})")

    t0 = time.time()
    df = load_combined_training_data(base_csv, expert_csv)
    trained = train_candidate(df)
    train_time = round(time.time() - t0, 1)
    print(f"[retrain_pipeline] Candidate v{version} trained in {train_time}s")
    print(f"  Condition : acc={trained['metrics']['condition']['accuracy']} "
          f"f1_macro={trained['metrics']['condition']['f1_macro']}")
    print(f"  FaultType : acc={trained['metrics']['fault_type']['accuracy']} "
          f"f1_macro={trained['metrics']['fault_type']['f1_macro']}")

    deployed_version = reg["deployed_version"]
    deployed_metrics = None
    if deployed_version is not None:
        for h in reg["history"]:
            if h["version"] == deployed_version:
                deployed_metrics = h["metrics"]
                break

    passed, reasons = passes_gate(trained["metrics"], deployed_metrics)

    vdir = _save_candidate(version, trained)

    # NEW: save the exact combined dataframe used for this training run
    snapshot_path = os.path.join(vdir, "training_data_snapshot.csv")
    df.to_csv(snapshot_path, index=False)
    print(f"[retrain_pipeline] Training data snapshot saved -> {snapshot_path}")

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "triggered_by": triggered_by,
        "n_train_rows": trained["metrics"]["n_train"],
        "n_test_rows": trained["metrics"]["n_test"],
        "metrics": trained["metrics"],
        "gate": DEPLOYMENT_GATE,
        "passed_gate": passed,
        "gate_failure_reasons": reasons,
        "deployed": False,   # updated below if it passes
    }

    if passed:
        metadata["deployed"] = True
        reg["deployed_version"] = version
        print(f"[retrain_pipeline] v{version} PASSED gate -> DEPLOYED "
              f"(previous deployed: v{deployed_version})")
        
        # Append expert data to base_csv and clear expert_csv
        if os.path.exists(expert_csv) and os.path.getsize(expert_csv) > 0:
            base_df = pd.read_csv(base_csv)
            expert_df = pd.read_csv(expert_csv)
            common_cols = [c for c in expert_df.columns if c in base_df.columns]
            base_df = pd.concat([base_df, expert_df[common_cols]], ignore_index=True)
            base_df.to_csv(base_csv, index=False)
            expert_df.iloc[0:0].to_csv(expert_csv, index=False)
            print(f"[retrain_pipeline] Appended {len(expert_df)} rows to {base_csv} and cleared {expert_csv}")
    else:
        print(f"[retrain_pipeline] v{version} FAILED gate -> NOT deployed. "
              f"Deployed model remains v{deployed_version}.")
        for r in reasons:
            print(f"    - {r}")

    with open(os.path.join(vdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    reg["history"].append({"version": version, "metrics": trained["metrics"],
                           "passed_gate": passed, "created_at": metadata["created_at"]})
    _save_registry(reg)

    return {
        "version": version,
        "deployed": passed,
        "deployed_version_after": reg["deployed_version"],
        "gate_failure_reasons": reasons,
        "metrics": trained["metrics"],
        "model_dir": vdir,
        "train_time_sec": train_time,
    }


def get_deployed_model_dir() -> Optional[str]:
    """Path to the currently deployed model's version folder, or None."""
    reg = _load_registry()
    if reg["deployed_version"] is None:
        return None
    return os.path.join(MODELS_DIR, f"v{reg['deployed_version']}")


def rollback_to(version: int) -> dict:
    """
    Manual rollback: point registry.deployed_version at an older
    (already-trained) version without retraining. Useful if a deployed
    model turns out to misbehave in production despite passing the
    gate offline.
    """
    reg = _load_registry()
    vdir = os.path.join(MODELS_DIR, f"v{version}")
    if not os.path.isdir(vdir):
        raise ValueError(f"Version v{version} not found in {MODELS_DIR}")
    old = reg["deployed_version"]
    reg["deployed_version"] = version
    _save_registry(reg)
    print(f"[retrain_pipeline] Rolled back deployed model: v{old} -> v{version}")
    return {"previous_version": old, "deployed_version": version}


# ------------------------------------------------------------------
# Self-test (run: python retrain_pipeline.py)
# ------------------------------------------------------------------
if __name__ == "__main__":
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)

    print("=== Retraining run #1 (base dataset only) ===")
    result1 = run_retraining(triggered_by="self_test_1")
    print(json.dumps({k: v for k, v in result1.items() if k != "metrics"}, indent=2))

    print("\n=== Retraining run #2 (simulate a second retrain) ===")
    result2 = run_retraining(triggered_by="self_test_2")
    print(json.dumps({k: v for k, v in result2.items() if k != "metrics"}, indent=2))

    print("\n=== Registry state ===")
    print(json.dumps(_load_registry()["deployed_version"], indent=2))
    print("Deployed model dir:", get_deployed_model_dir())
