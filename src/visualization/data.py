"""Script for visualization functions used in the data folder.

Author: Claudia Herron Mulet.
"""

import os

import seaborn as sns
from matplotlib import pyplot as plt

from src.config import paths

color1 = "#44AA99"


def plotHistogram(
    x, xlabel: str = "", title: str = "", th: int = None, bins: int = 100
):
    """Basic histogram plot with custom formatting.

    Args:
        x: array with values to histogram.
        xlabel: xlabel of plot.
        title: title of the plot.
        th: threshold for vertical line.
        bins: number of bins.
        title: plot title.
    """
    sns.displot(x, stat="percent", bins=bins, kde=True, color=color1)
    if th is not None:
        plt.axvline(th, c="black")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(
        os.path.join(paths.FIGURES_DIR, f"data_analysis_{xlabel}_distribution.png")
    )
    plt.clf()
