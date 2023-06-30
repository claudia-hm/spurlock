"""Model for plotting of fairness.

Authors: Claudia Herron Mulet.
"""
import os
from typing import List

import seaborn as sns
from aif360.sklearn.utils import check_groups
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import paths

sns.set(font_scale=1.8)

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=12)  # legend fontsize
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


def disagregatedROC(
    y_true, y_scores, attribute, filename, prot_attr=None, priv_group=1, dataset="Test"
):
    """Plot the ROC curve.

    Args:
        y_true: true labels.
        y_scores: scores of classifier.
        attribute: protected attribute human-readable name.
        filename: filename for saving plot.
        prot_attr: protected attribute colname.
        priv_group: label for privileged.
        dataset: name of dataset.
    """
    # Calculate the false positive rate (FPR) and true positive
    # rate (TPR) at different thresholds
    plt.figure(figsize=(5, 4.4))
    groups, _ = check_groups(y_true, prot_attr)
    idx = groups == priv_group
    unpriv = [y[~idx] for y in (y_true, y_scores) if y is not None]
    priv = [y[idx] for y in (y_true, y_scores) if y is not None]

    fpr_p, tpr_p, thresholds_p = roc_curve(*priv)
    fpr_u, tpr_u, thresholds_u = roc_curve(*unpriv)

    # Calculate the area under the ROC curve (AUC)
    auc_p = roc_auc_score(*priv)
    auc_u = roc_auc_score(*unpriv)

    # Plot the ROC curve
    plt.plot(fpr_p, tpr_p, label="ROC (AUC = %0.2f) privileged" % auc_p, color=color1)
    plt.plot(fpr_u, tpr_u, label="ROC (AUC = %0.2f) unprivileged" % auc_u, color=color2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if attribute.startswith("Body"):
        attribute = "BMI"
    plt.title(f"Disagregated ROC ({dataset}) for {attribute}")
    plt.legend()
    plt.savefig(
        os.path.join(
            paths.FIGURES_DIR,
            filename,
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.clf()
    plt.close()


def generatePalette(names: List):
    """Give a list of names, return a seaborn palette that\
     associates each name with a color.

    Args:
        names: names to give color.
    """
    num_names = len(names)
    # color_palette = sns.color_palette("colorblind", num_names)
    color_palette = CB_color_cycle[:num_names]

    name_palette = dict(zip(names, color_palette))

    return name_palette


def plotBars(
    metrics,
    title="",
    ref="",
    filename="",
    palette: dict = None,
    hideticks: bool = False,
):
    """Bar plot for fairness metric.

    Args:
        metrics: dictionary with metrics per protected group.
        title: plot title.
        ref: reference line label.
        filename: filename where to save plot.
        palette: color palette.
        hideticks: wether to hide the y ticks or not.
    """
    figure = plt.figure()
    metrics = dict(sorted(metrics.items(), key=lambda item: item[1], reverse=True))
    if not palette:
        sns.barplot(y=list(metrics.keys()), x=list(metrics.values()), color=color1)
    else:
        sns.barplot(
            y=list(metrics.keys()),
            x=list(metrics.values()),
            palette=palette,
        )
    plt.title(title)
    if ref:
        plt.axvline(1, color="black", linestyle="--", label=ref)
        plt.legend()
    if hideticks:
        figure.set_size_inches(6, 6)
        # Create a legend with custom colors
        legend_labels = list(palette.keys())
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[color]) for color in palette
        ]

        plt.legend(legend_handles, legend_labels)

        plt.yticks([])
    else:
        figure.set_size_inches(9, 6)

    plt.tight_layout()

    plt.savefig(os.path.join(paths.FIGURES_DIR, filename), dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
