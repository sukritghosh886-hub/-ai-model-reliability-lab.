
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="AI Model Reliability Lab",
    description=(
        "A lightweight ML evaluation API for classification metrics, "
        "prediction errors, and confidence analysis."
    ),
    version="1.0.0",
)


class EvaluationInput(BaseModel):
    true_labels: List[int] = Field(
        ..., min_length=1, description="Actual binary labels: 0 or 1"
    )
    predicted_labels: List[int] = Field(
        ..., min_length=1, description="Predicted binary labels: 0 or 1"
    )
    confidences: Optional[List[float]] = Field(
        None,
        description=(
            "Optional confidence in each supplied prediction, "
            "between 0 and 1."
        ),
    )


def evaluate_predictions(true_labels, predicted_labels, confidences=None):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Both label lists must have equal lengths.")

    if not true_labels:
        raise ValueError("Provide at least one prediction.")

    if any(label not in (0, 1) for label in true_labels + predicted_labels):
        raise ValueError("This version supports binary labels 0 and 1 only.")

    if confidences is not None:
        if len(confidences) != len(true_labels):
            raise ValueError("Confidence count must match the label count.")

        if any(not 0 <= c <= 1 for c in confidences):
            raise ValueError("Every confidence must be between 0 and 1.")

    errors = [
        i for i, (actual, predicted)
        in enumerate(zip(true_labels, predicted_labels))
        if actual != predicted
    ]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, zero_division=0
    )
    recall = recall_score(
        true_labels, predicted_labels, zero_division=0
    )
    f1 = f1_score(
        true_labels, predicted_labels, zero_division=0
    )

    result = {
        "sample_count": len(true_labels),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(
            true_labels, predicted_labels, labels=[0, 1]
        ).tolist(),
        "error_count": len(errors),
        "error_indices_zero_based": errors,
        "recommendations": [],
    }

    if recall < 0.80:
        result["recommendations"].append(
            "Recall is below 0.80. Investigate false negatives."
        )

    if precision < 0.80:
        result["recommendations"].append(
            "Precision is below 0.80. Investigate false positives."
        )

    if accuracy < 0.80:
        result["recommendations"].append(
            "Accuracy is below 0.80. Review data quality and baselines."
        )

    if confidences is not None:
        high_confidence_errors = [
            i for i in errors if confidences[i] >= 0.80
        ]

        result["confidence_analysis"] = {
            "mean_confidence": round(
                float(np.mean(confidences)), 4
            ),
            "high_confidence_wrong_count": len(high_confidence_errors),
            "high_confidence_wrong_indices_zero_based":
                high_confidence_errors,
            "threshold": 0.80,
            "note": (
                "Confidence values are supplied by the caller. "
                "This analysis does not establish calibration."
            ),
        }

        if high_confidence_errors:
            result["recommendations"].append(
                "Investigate incorrect predictions with confidence >= 0.80."
            )

    if not result["recommendations"]:
        result["recommendations"].append(
            "No basic metric warning triggered. Test on independent data."
        )

    return result


@app.get("/")
def home():
    return {
        "project": "AI Model Reliability Lab",
        "status": "running",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(data: EvaluationInput):
    try:
        return evaluate_predictions(
            data.true_labels,
            data.predicted_labels,
            data.confidences,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/demo")
def run_demo():
    # Uses a small dataset bundled with scikit-learn.
    # No separate dataset download is needed.
    dataset = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.25,
        random_state=42,
        stratify=dataset.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=42),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    result = evaluate_predictions(
        y_test.tolist(),
        predictions.tolist(),
        confidence.tolist(),
    )

    result["benchmark"] = {
        "dataset": "Scikit-learn Breast Cancer Wisconsin",
        "model": "StandardScaler + LogisticRegression",
        "brier_score_positive_class": round(
            float(
                brier_score_loss(
                    y_test, probabilities[:, 1]
                )
            ),
            4,
        ),
        "brier_score_note": (
            "Lower is better for probabilistic predictions; "
            "this single score does not establish calibration."
        ),
        "limitation": (
            "Educational benchmark only; not a medical or clinical tool."
        ),
    }

    return result
