"""Script to do basic data analysis.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 26/03/2023
"""
import numpy as np

from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getSamePatients
from src.visualization.data import plotHistogram

if __name__ == "__main__":
    # 1. Load data
    X = getProcessedFields()
    y = getDisease("G20")
    metadata = getMetadataWithCategories()
    X, y = getSamePatients(X, y)

    # 2. Create age variable
    X["Age"] = 2006 - X.F34

    # 3. Basic plotting
    plotHistogram(
        X.Age.values,
        xlabel="Age",
        title="Age distribution",
        th=65,
        bins=len(np.unique(X.Age.values)),
    )
    plotHistogram(
        X.F189.values,
        xlabel="Townsend Deprivation Index",
        title="Townsend Deprvation Index distribution",
        th=np.nanpercentile(X.F189.values, 0.8),
    )
    plotHistogram(X.F21001.values, xlabel="BMI", title="BMI distribution", th=30)
