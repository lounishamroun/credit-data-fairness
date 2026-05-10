from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def cost_matrix_score(y_true, y_pred, false_negative_cost: int = 5, false_positive_cost: int = 1) -> int:
    """Project cost where missing a bad-credit case is more expensive."""
    cm = confusion_matrix(y_true, y_pred)
    return int(cm[1, 0] * false_negative_cost + cm[0, 1] * false_positive_cost)


def classification_summary(name: str, model, X_test, y_test) -> dict[str, float | str]:
    """Return the main performance metrics used in the notebook."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "auc": roc_auc_score(y_test, y_prob),
        "cost": cost_matrix_score(y_test, y_pred),
    }


def favorable_outcome_audit(y_true, y_pred, sensitive) -> dict[str, float | dict[int, float]]:
    """Audit fairness for the favorable credit outcome: prediction == good credit (0)."""
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": sensitive})
    frame["approved_pred"] = (frame["y_pred"] == 0).astype(int)

    approval_by_group = frame.groupby("group")["approved_pred"].mean()
    good_credit = frame["y_true"] == 0
    good_credit_tpr = frame[good_credit].groupby("group")["approved_pred"].mean()

    return {
        "approval_gap_max_min": float(approval_by_group.max() - approval_by_group.min()),
        "good_credit_tpr_gap_max_min": float(good_credit_tpr.max() - good_credit_tpr.min()),
        "approval_rates": approval_by_group.to_dict(),
        "good_credit_tpr": good_credit_tpr.to_dict(),
    }
