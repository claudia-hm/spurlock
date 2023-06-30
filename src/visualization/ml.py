"""Script with visualization functions for ml."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import paths

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

CB_color_cycle = [
    "#88CCEE",
    "#117733",
    "#332288",
    "#DDCC77",
    "#999933",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]

color1 = "#44AA99"
color2 = "#CC6677"
color3 = "#DDCC77"
color4 = "#eef8f6"


def plotProbs(
    y_probs_positive, y_probs_negative, dataset, model="catboost", icd10="I10"
):
    """Plot probability distributions for negative and positive classes.

    Args:
        y_probs_positive: probs of positive class.
        y_probs_negative: probs of negative class.
        dataset: dataset name.
        model: model name to save file.
        icd10: icd10 code of the disease.
    """
    plt.figure(figsize=(5, 4.5))
    sns.histplot(
        x=y_probs_positive,
        bins=100,
        kde=True,
        stat="percent",
        label="Positive " "class",
        color=color1,
    )
    sns.histplot(
        x=y_probs_negative,
        bins=100,
        kde=True,
        stat="percent",
        label="Negative " "class",
        color=color2,
    )
    plt.title(f"Distribution of probabilities per class {dataset}")
    plt.xlabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/probability_distribution_{dataset}_{model}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_precision_recall(y_true, y_probs, dataset, model="catboost", icd10="I10"):
    """Plot precision and recall line given truth and probabilities.

    Args:
        y_true: true labels.
        y_probs: model probabilities.
        dataset: dataset name (train/test).
        model: model name to save file.
        icd10: icd10 code of the disease.
    """
    # Calculate precision and recall values at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    auc_pr = average_precision_score(y_true, y_probs)

    # Create the plot
    plt.figure(figsize=(5, 4.4))
    plt.plot(
        recall[:-1],
        precision[:-1],
        label="PR curve (AUC-PR = %0.2f)" % auc_pr,
        color=color1,
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    # plot the no skill precision-recall curve
    plt.plot(
        [0, 1],
        [no_skill, no_skill],
        linestyle="--",
        color=color2,
        label="Random " "guessing",
    )
    plt.legend()
    plt.title(f"Precision recall curve for {dataset}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/precision_recall_curve_{dataset}_{model}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_roc_curve(y_true, y_score, dataset, model="catboost", icd10="I10"):
    """Plot the ROC curve.

    Args:
        y_true: true labels.
        y_score: model scores.
        dataset: name of dataset.
        model: model name to save file.
        icd10: icd10 code of the disease.
    """
    # Calculate the false positive rate (FPR) and true positive
    # rate (TPR) at different thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate the area under the ROC curve (AUC)
    auc = roc_auc_score(y_true, y_score)

    # Plot the ROC curve
    plt.figure(figsize=(5, 4.4))
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % auc, color=color1)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guessing", color=color2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic (ROC) Curve {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/rocauc_curve_{dataset}_" f"{model}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_confusion_matrix(
    y_true, y_pred, labels, dataset, model="catboost", cmap="Greens", icd10="I10"
):
    """Plot a confusion matrix using seaborn.

    Args:
       y_true: true labels.
       y_pred: predicted labels.
       labels: list of label names.
       dataset: name of dataset.
       model: model name to save file.
       cmap: colormap to use.
       icd10: icd10 code of the disease.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    colors = [color4, color1]
    nodes = [0.0, 1]
    cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    ax = plt.subplot()
    sns.heatmap(
        cm,
        annot=True,
        ax=ax,
        cmap=cmap,
        fmt="d",
        linewidths=0.5,
        annot_kws={"fontsize": 16},
    )

    # set labels, title and ticks
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_title(f"Confusion Matrix {model}", fontsize=16)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/confusion_matrix_{dataset}_" f"{model}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_kde_histograms(data1, data2, label1, label2, title):
    """Plot kde histogram for two datasets together.

    Args:
        data1: first dataframe.
        data2: second dataframe.
        label1: first dataframe label.
        label2: second dataframe label.
        title: title of the plot.
    """
    plt.clf()
    sns.set_style("whitegrid")
    sns.kdeplot(data1, color="blue", alpha=0.7, label=label1)
    sns.kdeplot(data2, color="red", alpha=0.7, label=label2)
    plt.xlabel("X Label")
    plt.ylabel("Density")
    plt.legend()
    plt.title(title)
    plt.show()
    plt.clf()


def plot_density_histogram(data, title, x_label, y_label, icd10="I10"):
    """Plot a density histogram using seaborn.

    Args:
        data : The input data for the plot.
        x_col: The name of the column to use for the x-axis.
        title : The title of the plot.
        x_label : The label for the x-axis.
        y_label : The label for the y-axis.
        icd10: icd10 code of the disease.
    """
    # Set style
    sns.set_style("darkgrid")

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot density histogram
    sns.histplot(
        data, color="darkblue", stat="percent", kde=True, alpha=0.7, ax=ax, bins=50
    )

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Remove top and right spines
    sns.despine()

    # Show plot
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/probability_distribution_train_catboost.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_metrics_vs_threshold(
    y_test, y_probs, dataset, th=None, model="catboost", icd10="I10"
):
    """Plot metrics as a function of threshold.

    Args:
        y_test: ground truth.
        y_probs: model probabilities.
        dataset: train/test dataset string.
        th: current threshold.
        model: modelname.
        icd10: icd10 code of the disease.
    """
    thresholds = np.linspace(0, 1, num=100)  # Set the number of thresholds

    balanced_accuracies = []
    f1_scores = []
    mat_corrs = []
    for threshold in thresholds:
        y_pred = np.where(y_probs >= threshold, 1, 0)  # Apply thresholding
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        mat_corrs.append(matthews_corrcoef(y_test, y_pred))

    # Plotting the results
    plt.figure(figsize=(5, 4.4))
    plt.plot(thresholds, balanced_accuracies, label="Balanced accuracy", color=color1)
    plt.plot(thresholds, f1_scores, label="F1 score", color=color2)
    plt.plot(thresholds, mat_corrs, label="Matthews correlation", color=color3)
    if th:
        plt.axvline(th, label="Selected threshold", c="black", linestyle="--")
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Metrics")
    plt.title(f"Metrics vs. Threshold ({dataset})")
    plt.tight_layout()
    # Show plot
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR, f"{icd10}/metrics_vs_threshold_{dataset}_{model}.png"
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()


def plot_shap(model, X, dataset, metadata=None, icd10="I10", beeswarm: bool = True):
    """Plot SHAP values in waterfall plot.

    Args:
        model: ml model pipeline.
        X: data.
        dataset: dataset name.
        metadata: dataframe with metadata info.
        icd10: icd10 code.
        beeswarm: beeswarm plot or not (else barplot).
    """
    prep = model.named_steps.preprocessor
    classifier = model.named_steps.classifier
    model_name = classifier.__class__.__name__
    colors = [color1, color2]
    nodes = [0.0, 1]
    cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

    if metadata is not None:
        features = prep.get_feature_names_out()
        feature_names = [metadata.loc[feature, "Name"] for feature in features]
        X_prep = pd.DataFrame(
            prep.transform(X),
            columns=feature_names,
            index=X.index,
        )
        classifier.feature_names = feature_names
    else:
        X_prep = pd.DataFrame(
            prep.transform(X),
            columns=prep.get_feature_names_out(),
            index=X.index,
        )
    explainer = shap.Explainer(classifier)
    shap_values = explainer(X_prep)
    if beeswarm:
        shap.plots.beeswarm(shap_values, show=False, color=cmap)
    else:
        shap.plots.bar(shap_values, show=False)

    plt.gcf().set_size_inches(13, 6)
    plt.title(f"SHAP beeswarm plot ({dataset})")
    plt.tight_layout()
    # Show plot
    plt.savefig(
        os.path.join(paths.FIGURES_DIR, f"{icd10}/shap_{dataset}_{model_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.gcf().set_size_inches(6.4, 4.8)
