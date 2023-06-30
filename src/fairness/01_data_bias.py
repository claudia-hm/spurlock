"""Script to identify biases in training data.

Author: Claudia Herron Mulet (clherrom12@alumnes.ub.edu)
Date: 01/03/2023
"""
import logging
import os

import pandas as pd
from aif360.sklearn.metrics import disparate_impact_ratio
from sklearn.model_selection import train_test_split

from src.config import paths
from src.data.loading import getDisease, getMetadataWithCategories, getProcessedFields
from src.data.utils import getFeaturesByDatatype, getSamePatients
from src.fairness.utils import createBinaryFairnessFeatures
from src.ml.utils import getMLConfiguration
from src.visualization.fairness import generatePalette, plotBars

if __name__ == "__main__":
    # 1. Configuration
    logging.basicConfig(
        filename=os.path.join(paths.LOGS_DIR, "data_bias.log"),
        filemode="w",
        encoding="utf-8",
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    logging.info("Starting data bias script")
    config = getMLConfiguration()
    seed = config["SEED"]
    icd10 = config["ICD10"]

    data_bias_report_filename = os.path.join(
        paths.REPORTS_FAIRNESS, f"{icd10}_data_bias.txt"
    )

    # 2. Load data
    logging.info("Load data")
    metadata = getMetadataWithCategories()
    X = getProcessedFields()
    y = getDisease(icd10)
    X, y = getSamePatients(X, y)
    cat_features, num_features, num_features_special = getFeaturesByDatatype(
        X, cat_idx=False
    )

    # 3. Create fairness multiindex
    logging.info("Create fairness multiindex")
    X = createBinaryFairnessFeatures(X)
    PA_features = [col for col in X.columns if col.startswith("PA")]
    PA_names = [metadata.loc[col.split("_")[1]].Name for col in PA_features]
    palette = generatePalette(PA_names)
    multi_index = pd.MultiIndex.from_frame(
        X[PA_features].reset_index(), names=["patient"] + PA_features
    )
    X.set_index(multi_index, inplace=True)
    X.drop(columns=PA_features, inplace=True)
    y.index = multi_index

    # 3. Split data
    logging.info("Split train, test, val data")
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y, stratify=y, random_state=seed
    )

    # 4. FAIRNESS EVALUATION
    # 4. 1. Measure disparate impact
    with open(data_bias_report_filename, "w") as f:
        f.write(" === Data fairness report ===\n")
        logging.info("Measure DIR")
        f.write("1. Disparate impact analysis\n")
        disparate_impact_dict = {}
        for feature, name in zip(PA_features, PA_names):
            di = disparate_impact_ratio(y_train, prot_attr=feature, priv_group=1)
            disparate_impact_dict[name] = di
            f.write(f"Disparate impact ratio for {name} ({feature}): {di:.4f}\n")
        f.write("\n")
        f.write("-----------------------------\n")

        # 2. Correlations
        logging.info("Report potential proxies for protected attributes")
        f.write("2. Highly correlated features to protected attributes\n")
        corr_matrix = X_train.reset_index().drop(columns="patient").corr()
        for feature, name in zip(PA_features, PA_names):
            f.write(f"Top-10 most correlated attributes to {name} ({feature})\n")
            corrs = abs(corr_matrix[feature]).sort_values(ascending=False).drop(feature)
            if feature.split("_")[1] in corrs:
                corrs.drop(feature.split("_")[1], inplace=True)
            topk = corrs.head(10)
            topk_index = [
                metadata.loc[col, "Name"]
                if not col.startswith("PA")
                else metadata.loc[col.split("_")[1], "Name"]
                for col in topk.index
            ]
            topk.index = pd.Index(topk_index)
            f.write(str(topk))
            f.write("\n")
            plotBars(
                topk.to_dict(),
                title=f"Most correlated features with " f"{name}",
                filename=f"{icd10}/data_bias_top10_corr_{feature}.png",
            )

        f.write("\n")
        f.write("-----------------------------\n")
        # 3. Support
        # compute number of privileged
        logging.info("Compute support for groups")
        f.write("3.1 Support: percentage of privileged class per protected attribute\n")
        support = (
            X_train.index.droplevel().to_frame().reset_index(drop=True).sum(axis=0)
        )
        # compute total train number of samples
        N = len(X_train)
        # compute percentage of privileged
        perc_privileged = (support / N) * 100
        f.write(str(perc_privileged))

        f.write("3.2 Support: disease distribution over protected attributes\n")
        df_pa = X_train.index.to_frame().reset_index(drop=True).set_index("patient")
        df_pa[icd10] = y_train.reset_index(drop=True).values
        for pa in PA_features:
            f.write(str(df_pa.groupby([pa, icd10]).size() / N * 100))
            f.write("\n")

    # 5. Fairness plots
    plotBars(
        disparate_impact_dict,
        title="Disparate impact ratio per " "protected attribute",
        ref="Equal base rates",
        filename=f"{icd10}/data_bias_disparate_impact.png",
        palette=palette,
    )
