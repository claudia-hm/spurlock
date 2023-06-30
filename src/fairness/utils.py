"""Utils file for fairness folder.

Author: Claudia Herron Mulet
"""
import numpy as np
import pandas as pd
from aif360.sklearn.metrics import (
    average_odds_error,
    difference,
    equal_opportunity_difference,
)
from imblearn.metrics import specificity_score
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from src.config.utils import readYML


def FPR_difference(
    y_true, y_pred, *, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None
):
    """Difference in FPR for unpriveleged and privileged.

    Args:
        y_true: true labels.
        y_pred: predicitions of classifier.
        prot_attr: protected attribute colname.
        priv_group: label for privileged.
        pos_label: pos_label from aif360.
        sample_weight: sample_weight from aif360.
    """
    fpr_diff = -difference(
        specificity_score,
        y_true,
        y_pred,
        prot_attr=prot_attr,
        priv_group=priv_group,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )
    return fpr_diff


def createBinaryFairnessFeatures(df):
    """Binarize fairness attributes into privileged and non privileged.

    Args:
        df: dataframe to prepare.
    """
    privileged_groups = readYML("src/config/privileged_groups.yml")

    for feature in privileged_groups:
        try:
            df[f"PA_{feature}"] = (
                df[feature].isin(privileged_groups[feature]).astype(int)
            )
        except Exception:
            val = privileged_groups[feature]
            if val < 1:
                perc = np.nanpercentile(df[feature], val * 100)
                df[f"PA_{feature}"] = (df[feature] < perc).astype(int)
            else:
                if feature == "F34":
                    age = 2006 - df[feature]
                    df[f"PA_{feature}"] = (age < val).astype(int)
                else:
                    df[f"PA_{feature}"] = (df[feature] < val).astype(int)
    return df


def getEvalMetrics(y_pred, y_test, name):
    """Get fairness evaluation metrics.

    Args:
        y_pred: predictions.
        y_test: ground truth.
        name: model name.
    """
    metrics = {}
    metrics["name"] = name
    metrics["F1"] = f1_score(y_test, y_pred)
    metrics["BA"] = balanced_accuracy_score(y_test, y_pred)
    metrics["MCC"] = matthews_corrcoef(y_test, y_pred)
    metrics["AOE Sex"] = average_odds_error(y_test, y_pred, prot_attr="PA_F31")
    metrics["AOE BMI"] = average_odds_error(y_test, y_pred, prot_attr="PA_F21001")
    metrics["AOE Age"] = average_odds_error(y_test, y_pred, prot_attr="PA_F34")
    metrics["EOD Sex"] = equal_opportunity_difference(
        y_test, y_pred, prot_attr="PA_F31"
    )
    metrics["FPRD BMI"] = FPR_difference(y_test, y_pred, prot_attr="PA_F21001")
    metrics["FPRD Age"] = FPR_difference(y_test, y_pred, prot_attr="PA_F34")
    return metrics


def getFPR(y_true, y_pred):
    """Get FPR given true and predicted.

    Args:
        y_true: true labels.
        y_pred: predicted labels.
    """
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fpr = fp / (fp + tn)
    return fpr


class SeparationPostprocessor:
    def __init__(self, estimator, prot_attr):
        self.thresholds = {}
        self.estimator = estimator
        self.prot_attr = prot_attr

    def fit(self, X, y, desired_fpr_per_group=0.1):
        """Find threshold for target fpr."""
        self.positive_class = np.where(self.estimator.classes_ == 1)[0][0]
        y_proba = self.estimator.predict_proba(X)[:, self.positive_class]

        groups = X.index.unique(level=self.prot_attr)

        for group in groups:
            group_indices = X.index.get_level_values(level=self.prot_attr) == group
            group_proba = y_proba[group_indices]
            group_true = y[group_indices]
            group_threshold = self._find_threshold_for_fpr(
                group_proba, group_true, desired_fpr_per_group
            )
            self.thresholds[group] = group_threshold

    def predict(self, X):
        """Apply threshold for classifying."""
        y_proba = self.estimator.predict_proba(X)[:, self.positive_class]

        y_pred = pd.Series(index=X.index)
        groups = X.index.unique(level=self.prot_attr)

        for group in groups:
            group_indices = X.index.get_level_values(self.prot_attr) == group
            group_proba = y_proba[group_indices]
            group_threshold = self.thresholds[group]
            group_prediction = (group_proba >= group_threshold).astype(int)
            y_pred[group_indices] = group_prediction

        return y_pred

    def _find_threshold_for_fpr(self, y_proba, y_true, desired_fpr):
        """Find group thresholds."""
        thresholds = np.linspace(
            0, 1, 1000
        )  # Adjust the number of thresholds as needed

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            fpr = getFPR(y_true, y_pred)

            if fpr <= desired_fpr:
                return threshold

        return 0.5  # If no threshold achieves the desired FPR rate, return a default threshold
