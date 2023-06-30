"""Script with miscellaneous ML util functions."""

import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import paths
from src.config.utils import readYML


def getMLConfiguration():
    """Get ML configuration."""
    config = readYML("src/config/ml.yml")
    return config


def generateFilenameGrid(
    grid: Dict,
    icd10: str,
    nfeatures: int,
    name: str,
    seed: int,
    opt_score: str,
) -> str:
    """Generate a filename to save grid search based on number\
     of parameters and seed.

    Args:
        grid: grid dictionary.
        icd10: icd10 code for disease.
        nfeatures: number of independent features.
        name: model name.
        seed: random seed.
        opt_score: score optimized in grid search.
    """
    count = 0
    for x in grid:
        if isinstance(grid[x], list):
            count += len(grid[x])
    return (
        f"{icd10}_grid_{opt_score}_{nfeatures}_features_{name}_{count}_params"
        f""
        f"_{seed}_seed.pk1"
    )


def generateFilenameCustomModel(
    icd10: str,
    nfeatures: int,
    name: str,
    seed: int,
) -> str:
    """Generate a filename to save grid search based on number\
     of parameters and seed.

    Args:
        icd10: icd10 code for disease.
        nfeatures: number of independent features.
        name: model name.
        seed: random seed.
    """
    return f"{icd10}_{nfeatures}_features_{name}_{seed}_seed.pk1"


def getGridSearchFeaturenames(grid_search: GridSearchCV) -> List:
    """Retrieve Feature names resulted after preprocessing.

    Args:
        grid_search: fitted grid search object.
    """
    return grid_search.best_estimator_.named_steps.preprocessor.get_feature_names_out()


def getPositiveClassIndex(best_grid_search):
    """Get the index of the postive class.

    Args:
        best_grid_search: grid search to get.
        best_model_name: model name.
    """
    return np.where(best_grid_search.classes_)[0][0]


def getTopFeatures(model, feature_names, k=10):
    """Returns the top k most important features\
     for a given scikit-learn model.

    Args:
        model: scikit-learn estimator object. The \
        trained model for which to get the feature importances.
        feature_names: list. The list of feature names in \
        the same order as the features used to train the model.
        k : the number of top features to return.
    """
    if hasattr(model, "coef_"):
        # for linear models like Logistic Regression
        feature_importances = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        # for tree-based models like Random Forest
        feature_importances = model.feature_importances_
    else:
        print(
            'The given model does not have attribute "coef_" or "feature_importances_".'
        )
        return []

    # get the indices of the top k features
    top_k_idx = np.argsort(feature_importances)[::-1][:k]

    # get the feature names corresponding to the top k indices
    top_features = [feature_names[idx] for idx in top_k_idx]

    return top_features


def getScorer(opt_score):
    """Get scored based on name.

    Args:
        opt_score: name of scoring function.
    """
    if opt_score == "f1":
        scorer = make_scorer(f1_score, pos_label=1)
    elif opt_score == "f2":
        scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    else:
        scorer = opt_score
    return scorer


def resample(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Resample imbalanced data.

    Args:
        X_train: independent variables.
        y_train: dependent variables.
        random_state: random seed.
    """
    logging.info("Perform resampling")
    # define the under/over sampling pipeline
    sampling_pipeline = make_pipeline(RandomUnderSampler(random_state=random_state))

    return sampling_pipeline.fit_resample(X_train, y_train)


def findOptimalThreshold(y_probs, y_true, measure="f1"):
    """This function takes in a classification probability vector,\
    a ground truth binary label vector, and a measure \
    (e.g. accuracy, F1 score, etc.) and returns the optimal threshold\
    for classifying data points.

    Args:
        y_probs: probabilities prediction.
        y_true: ground truth.
        measure: measure to optimize.
    """
    thresholds = np.arange(0, 1, 0.01)
    scores = []
    for t in thresholds:
        predicted_label = np.where(y_probs >= t, True, False)
        if measure == "accuracy":
            score = accuracy_score(y_true, predicted_label)
        if measure == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, predicted_label)
        elif measure == "f1":
            score = f1_score(y_true, predicted_label)
        elif measure == "f2":
            score = fbeta_score(y_true, predicted_label, beta=2)
        elif measure == "f1_weighted":
            score = f1_score(y_true, predicted_label, average="weighted")
        elif measure == "precision":
            score = precision_score(y_true, predicted_label)
        elif measure == "recall":
            score = recall_score(y_true, predicted_label)
        else:
            raise ValueError(
                "Invalid measure specified. Please choose from 'accuracy', 'f1', 'precision', or 'recall'."
            )
        scores.append(score)
    optimal_threshold = thresholds[np.argmax(scores)]

    return optimal_threshold, np.max(scores)


def getPreprocessingPipelines(
    cat_features: List,
    num_features: List,
    num_features_special_na: List,
    tree: bool = True,
):
    """Function to define preprocessing pipelines.

    Args:
        cat_features: list of categorical features.
        num_features: list of numerical features.
        num_features_special_na: list of features that have -1 nans.
        tree: preprocessing for tree model or not.
    """
    logging.info("Define preprocessing pipeline steps")
    # define the pipeline for categorical features
    if not tree:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist", drop="if_binary"
                    ),
                ),
            ]
        )
        # define the pipeline for numerical features
        num_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )
        num_pipeline_special_na = Pipeline(
            [
                ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
                ("scaler", MinMaxScaler()),
            ]
        )
    else:
        # alternative encoding
        cat_pipeline = Pipeline(
            [
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=np.nan
                    ),
                ),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        # define the pipeline for numerical features
        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        num_pipeline_special_na = Pipeline(
            [("imputer", SimpleImputer(missing_values=-1, strategy="median"))]
        )
    # combine the pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(
        [
            ("cat", cat_pipeline, cat_features),
            ("num", num_pipeline, num_features),
            ("num_na", num_pipeline_special_na, num_features_special_na),
        ],
        verbose_feature_names_out=False,
        remainder="passthrough",
        sparse_threshold=0,
    )

    return preprocessor


def evaluate_models(models, X_val, y_val, opt_score):
    """Function to evaluate a dictionary of ML models in validation.

    Args:
        models: dictionary of name to fitted model.
        X_val: validation data.
        y_val: validation labels.
        opt_score: metric to choose threshold.
    """
    metrics = {"roc_auc": [], "f1": [], "balanced_accuracy": [], "mcc": []}

    for model_name, model in models.items():
        positive_class = getPositiveClassIndex(model)
        y_probs = model.predict_proba(X_val)[:, positive_class]
        th, max_score = findOptimalThreshold(y_probs, y_val, opt_score)
        y_pred = y_probs > th

        auc_roc = roc_auc_score(y_val, y_probs)
        f1 = f1_score(y_val, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)

        metrics["roc_auc"].append(auc_roc)
        metrics["f1"].append(f1)
        metrics["balanced_accuracy"].append(balanced_accuracy)
        metrics["mcc"].append(mcc)

    df = pd.DataFrame(metrics, index=models.keys())
    return df


def saveGridSearchResults(
    models: Dict,
    opt_score: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metadata: pd.DataFrame,
    icd10: str,
):
    """Save the results of a grid search to file.

    Args:
        models: dictionary of models with names.
        opt_score: optimized score
        X_val: test data.
        y_val: test labels.
        metadata: dataframe with metadata info.
        icd10: disease being predicted.
    """
    # Get best model
    eval = evaluate_models(models, X_val, y_val, opt_score)
    best_model_name = eval[opt_score].idxmax()
    logging.info(f"Best model: {best_model_name}")
    best_grid_search = models[best_model_name]
    feature_names = getGridSearchFeaturenames(best_grid_search)

    # optimize threshold based on validation data
    positive_class = getPositiveClassIndex(best_grid_search)
    y_probs_val = best_grid_search.predict_proba(X_val)[:, positive_class]
    th, max_score = findOptimalThreshold(y_probs_val, y_val, opt_score)
    y_pred = y_probs_val > th

    # retrieve best features
    top10_features = getTopFeatures(
        best_grid_search.best_estimator_.named_steps.classifier, feature_names, k=10
    )

    # generate the classification report
    report = classification_report(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_probs_val)
    sensitivity = calculateSensitivity(y_val, y_pred)
    specificity = calculateSpecificity(y_val, y_pred)
    precision = calculatePrecision(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_pred, y_val)
    mc = matthews_corrcoef(y_pred, y_val)

    logging.info(f"Validation ROCAUC score: {roc_auc:.4f}")

    # write the classification report to a file
    filename = os.path.join(
        paths.REPORTS_ML_DIR, f"model_selection_report_{datetime.now()}.txt"
    )
    with open(
        filename,
        "w",
    ) as f:
        f.write("===GRID SEARCH REPORT===\n")
        f.write(f"ICD10: {icd10}\n")
        f.write(eval.to_string())
        f.write("\n")
        f.write("\n")
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"Best parameters found: {best_grid_search.best_params_}\n")
        f.write("\n")
        f.write("===BEST MODEL EVALUATION===\n")
        f.write(report)
        f.write("\n")
        f.write(
            f"Best training cross validation {opt_score}:"
            f" {best_grid_search.best_score_:.4f}\n"
        )
        f.write(f"{opt_score}-optimized threshold: {th}\n")
        f.write("\n")
        f.write(f"Validation ROC AUC score: {roc_auc:.4f}\n")
        f.write(f"Validation F1: {f1:.4f}\n")
        f.write(f"Validation balanced accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"Validation mattheus correlation: {mc:.4f}\n")
        f.write(f"Validation Sensitivity/Recall: {sensitivity:.4f}\n")
        f.write(f"Validation Specificity: {specificity:.4f}\n")
        f.write(f"Validation Precision: {precision:.4f}\n")
        f.write("\n")
        f.write("Top-10 most important features:\n")
        for i, feature in enumerate(top10_features):
            if "_" in feature:
                feature = feature.split("_")[0]
            feature_name = metadata.loc[feature, "Name"]
            feature_category = metadata.loc[feature, "cat_name"]
            if len(feature.split("_")) > 1:
                category = feature.split("_")[1]
                f.write(
                    f"\t {i}. {feature}-{feature_name} ({category}). "
                    f"{feature_category}. \n"
                )
            else:
                f.write(f"\t {i}. {feature}-{feature_name}. {feature_category}\n")


# metrics
def calculateSensitivity(y_test, y_pred):
    """Calculates the sensitivity (true positive rate) \
    of a binary classifier.

    Args:
    y_test (array-like): True labels of the test set.
    y_pred (array-like): Predicted labels of the test set.
    """
    tp = sum((y_test == 1) & (y_pred == 1))
    fn = sum((y_test == 1) & (y_pred == 0))
    sensitivity = tp / (tp + fn)
    return sensitivity


def calculateSpecificity(y_test, y_pred):
    """Calculates the specificity (true negative rate)\
     of a binary classifier.

    Args:
    y_test (array-like): True labels of the test set.
    y_pred (array-like): Predicted labels of the test set.
    """
    tn = sum((y_test == 0) & (y_pred == 0))
    fp = sum((y_test == 0) & (y_pred == 1))
    specificity = tn / (tn + fp)
    return specificity


def calculatePrecision(y_test, y_pred):
    """Calculates the precision of a binary classifier.

    Args:
    y_test (array-like): True labels of the test set.
    y_pred (array-like): Predicted labels of the test set.
    """
    tp = sum((y_test == 1) & (y_pred == 1))
    fp = sum((y_test == 0) & (y_pred == 1))
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError as e:
        precision = -1
        print(e)
    return precision
