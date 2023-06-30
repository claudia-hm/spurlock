"""Script with miscellaneous util functions."""
import logging
import os
import traceback
from datetime import date
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.errors import IntCastingNaNError
from tqdm import tqdm

from src.config import paths
from src.data.loading import (
    getAssessmentDate,
    getCodings,
    getDataDictShowcase,
    getDiagnosisDates,
    getDiseases,
    getField,
    getMetadata,
    getSpecialNaNFeatures,
)
from src.data.UKBiobank_scraper import build_data_dict, get_value_type, open_url


# DATA WRANGLING
def encodeFromDataDict(
    data_dict: pd.DataFrame, fid: int, df_field: pd.DataFrame
) -> Tuple:
    """Encode field with encoding dictionary and rename column.

    Args:
        data_dict: dataframe with information for each field.
        fid: field id.
        df_field: dataframe with field data.
    """
    codings = getCodings()
    # get name and coding for field
    name = data_dict.loc[fid, "Field"]
    coding = data_dict.loc[fid, "Coding"]
    data_type = data_dict.loc[fid, "ValueType"]

    # filter first visit column
    if any([col.endswith("0.0") for col in df_field.columns]):
        first_visit_col = f"f.{fid}.0.0"
        df_field = df_field[[first_visit_col]]

        # if there is encoding, replace with coding
        if not np.isnan(coding):
            code = zip(codings.loc[coding].Value, codings.loc[coding].Meaning)
            for value, meaning in code:
                try:
                    df_field.replace(int(value), meaning, inplace=True)
                except ValueError:
                    df_field.replace(value, meaning, inplace=True)
        else:
            code = None

        # get only first column, first assessment visit
        metadata = {
            "Field": f"F{fid}",
            "Name": name,
            "Data_Type": data_type,
            "Data coding": code,
        }
        col = df_field.iloc[:, 0].rename(f"F{fid}")
    else:
        logging.info(f"There was no first visit data for {fid}: {name}")
        col, metadata = None, None
    return col, metadata


def encodeFromScrapping(fid: int, df_field: pd.DataFrame) -> Tuple:
    """Encode field with encoding obtained through scrapping.

    Args:
        fid: field id.
        df_field: dataframe with field data.
    """
    column = str(fid) + "-0.0"
    html = open_url(column)
    # check if there is html response and first assessment visit
    if html and any([col.endswith("0.0") for col in df_field.columns]):
        first_visit_col = f"f.{fid}.0.0"
        df_field = df_field[[first_visit_col]]
        field_dict = build_data_dict(html, column)
        name = field_dict["description"]
        data_type = field_dict["value_type"]
        if get_value_type(html, column).startswith("Categorical"):
            coding = field_dict["data_coding"]["data_coding_types"]
            for value, meaning in coding.items():
                try:
                    df_field.replace(int(value), meaning, inplace=True)
                except ValueError:
                    df_field.replace(value, meaning, inplace=True)
        else:
            coding = []
        col = df_field.iloc[:, 0].rename(f"F{fid}")
        metadata = {
            "Field": f"F{fid}",
            "Name": name,
            "Data_Type": data_type,
            "Data coding": coding,
        }
        return col, metadata
    else:
        logging.warning(f"Impossible to scrap field {fid}")
        return None, None


def createDatasetFromFields(from_file: bool = False):
    """Create dataset from field files.

    Args:
        from_file: select only fields in listed in file.
    """
    data_dict = getDataDictShowcase()
    if from_file:
        # read list of fields
        logging.info("Reading field filenames to load")
        field_files = readList(
            filename=os.path.join(paths.DATA_DIR_EXTERNAL, "fields.txt")
        )
    else:
        # read all field files in field directory
        field_files = sorted(os.listdir(paths.DATA_DIR_FIELDS))
    data, metadata = [], []
    for field_file in tqdm(field_files):
        if field_file.startswith("f."):
            # read field data
            df_field = getField(field_file)
            # get field id from filename
            fid = int(os.path.splitext(field_file)[0].split(".")[1])

            # check if fied is in data dictionary or we have to scrap it
            if fid in list(data_dict.index):
                col, meta = encodeFromDataDict(data_dict, fid, df_field)
                if (col is not None) and (meta is not None):
                    metadata.append(meta)
                    data.append(col)
            else:
                col, meta = encodeFromScrapping(fid, df_field)
                if (col is not None) and (meta is not None):
                    metadata.append(meta)
                    data.append(col)
    # join all fields in a file and save
    df_metadata = pd.DataFrame.from_records(data=metadata)
    df_metadata.to_csv(
        os.path.join(paths.DATA_DIR_INTERIM, "metadata_fields.csv"), index=False
    )
    df = pd.concat(data, axis=1)
    df.to_csv(os.path.join(paths.DATA_DIR_INTERIM, "fields.csv"))


def checkICD10(row, icd10) -> bool:
    """For every row in the diseases dataset, \
    check if the queried icd10 code is present.

    Args:
        row: row in diseases dataframe.
        icd10: icd10 code that we want to query.
    """
    all_visits = row.dropna().values
    return any([x.startswith(icd10) for x in all_visits])


def getFirstDiagnosisCol(row) -> str:
    """Get first column with true value, i.e.,\
     first diagnosis for disease.

    Args:
        row: row from the diagnosis dates dataframe.
    """
    return row[row].index.tolist()[0]


def getFirstDiagnosisDate(row, first_diagnosis) -> date:
    """Get date of first diagnosis.

    Args:
        row: a row from the diagnosis date dataframe.
        first_diagnosis: the column where the first \
        diagnosis was detected.
    """
    # retrieve field name
    feid = row.name
    # retrieve column of first diagnosis
    col = first_diagnosis[feid]
    # retrieve visit number of diagnose
    # visit = col[-1]
    visit = col.split(".")[-1]
    # concatenate date field with visit number
    col_date = "f.41280.0." + visit
    # return corresponding value in the dataframe
    return row[col_date]


def filterPriorDiagnosis(label, diseases, icd10) -> pd.Series:
    """Filter out patients that were diagnosed previously\
     to the visit to the assessment visit center.

    Args:
        label: initial boolean labelling per patient.
        diseases: dataframe with all icd10 codes.
        icd10: icd10 code that we want to filter previous diagnosis.
    """
    # filter patients with disease
    diagnosed_patients = label[label].index.tolist()

    diagnosis_dates = getDiagnosisDates()
    dates_patients = diagnosis_dates.index.tolist()

    assessment_dates = getAssessmentDate()
    assessment_patients = assessment_dates.index.tolist()

    # check patients in all files
    patients = list(
        set(diagnosed_patients) & set(dates_patients) & set(assessment_patients)
    )

    if len(patients) > 0:
        # filter out patients with no information in all fields
        diagnosed_diseases = diseases.loc[patients]
        diagnosis_dates = diagnosis_dates.loc[patients]
        assessment_dates = assessment_dates.loc[patients]

        # retrieve column where disease is diagnosed
        mask = diagnosed_diseases.applymap(
            lambda x: x.startswith(icd10) if isinstance(x, str) else False
        )  # (diagnosed_diseases == icd10)
        first_diagnosis = mask.apply(getFirstDiagnosisCol, axis=1)
        first_diagnosis_date = diagnosis_dates.apply(
            lambda x: getFirstDiagnosisDate(x, first_diagnosis), axis=1
        )

        # check if the diagnosis was previous to the data collection
        is_prior_diagnosis = first_diagnosis_date.le(assessment_dates)
        if any(is_prior_diagnosis):
            logging.info(f"Removing prior diagnosis for {icd10}")
            prior_diagnosis_patients = is_prior_diagnosis[
                is_prior_diagnosis
            ].index.tolist()
            return label.drop(prior_diagnosis_patients)
        else:
            return label

    return label


def createDiseaseLabel():
    """Obtain a boolean label for each patient for each icd10 code."""
    # retrieve disease file
    diseases = getDiseases()
    # get list of ic10 codes to label
    icd10_codes = readList(os.path.join(paths.DATA_DIR_EXTERNAL, "icd10.txt"))

    for icd10 in icd10_codes:
        logging.info(f"Processing ICD10 code {icd10}")
        # get boolean indicating if disease is recorded per patient
        label = diseases.apply(lambda row: checkICD10(row, icd10), axis=1)
        if any(label):
            # remove patients that were diagnosed previous
            # to the first assessment
            label = filterPriorDiagnosis(label, diseases, icd10)
            label.to_csv(os.path.join(paths.DATA_DIR_INTERIM, f"{icd10}.csv"))
        else:
            logging.debug(f"There are no diagnosis for ICD10 code {icd10}")


# CLEANING
def integerCleaningNaNs(series: pd.Series) -> pd.Series:
    """Replace na by -1 value."""
    return series.fillna(-1)


def cleanCustom(col, series) -> pd.Series:
    """Apply custom cleaning depending on field."""
    if col == "F1438":
        return series.replace(
            {
                "Less than one": 0,
                np.NAN: -1,
                "Prefer not to answer": -1,
                "Do not know": -1,
            }
        ).astype(float)
    elif col == "F400":
        return series.replace({np.NAN: -1, "T": -1}).astype(float)
    else:
        return series


def setDataTypes(X: pd.DataFrame) -> pd.DataFrame:
    """Set correct data type for all fields.

    Args:
        X: dataframe with all fields.
    """
    dtypes = pd.read_csv(
        os.path.join(paths.DATA_DIR_EXTERNAL, "dtypes.csv"), sep=";", index_col=0
    )
    metadata = getMetadata()
    nan_replacing = []

    for col in tqdm(X.columns):
        data_type_raw = metadata.loc[col, "Data_Type"]
        try:
            X[col] = X[col].astype(dtypes.loc[data_type_raw, "Type"])
        except IntCastingNaNError:
            logging.info(f"Replacing nans by -1 for: {col}")
            nan_replacing.append(col)
            X[col] = integerCleaningNaNs(X[col])
            X[col] = X[col].astype(dtypes.loc[data_type_raw, "Type"])
        except ValueError:
            logging.info(f"Correcting value errors for: {col}")
            X[col] = cleanCustom(col, X[col])
            X[col] = X[col].astype(dtypes.loc[data_type_raw, "Type"])
        except Exception:
            logging.info(f"Unknown error for column: {col}")
            logging.error(traceback.format_exc())

    pd.DataFrame(nan_replacing, columns=["Field"]).to_csv(
        paths.DATA_DIR_INTERIM + "/replaced_nan.csv", index=False
    )
    return X


def dropAxisWithManyNaN(
    X: pd.DataFrame, axis: int = 1, perc: float = 0.9
) -> pd.DataFrame:
    """Drop columns or rows that contain more than perc\
    of NaN values.

    Args:
        X: dataframe to clean.
        axis: axis to apply cleaning. Default are columns.
        perc: threhold to drop axis.
    """
    before_shape = X.shape
    X = X.dropna(thresh=X.shape[int(not axis)] * perc, axis=axis)
    after_shape = X.shape

    diff = before_shape[axis] - after_shape[axis]
    logging.info(f"Dropped {diff} vectors from axis {axis}")
    return X


# MISCELLANEOUS
def readList(filename) -> List:
    """Read txt file with list.

    Args:
        filename: filename to open.
    """
    file = open(filename, "r")

    # reading the file
    data = file.read()

    # replacing end splitting the text
    # when newline ('\n') is seen.
    data_list = data.split("\n")
    file.close()

    return data_list


def getSamePatients(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Ensure that X and y refer to the same patients.

    Args:
        X: independent variables.
        y: dependent variables.
    """
    patients = set(X.index) & set(y.index)
    return X.loc[list(patients)], y.loc[list(patients)].astype(int)


def getFeaturesByDatatype(X: pd.DataFrame, cat_idx: bool = True) -> Tuple:
    """Get column names based on datatype.

    Args:
        X:  independent variables.
        cat_idx: wether to return categorical indexes.
    """
    # define the categorical and numerical features
    cat_features = list(X.select_dtypes(include=["category"]).columns)
    num_features = set(X.select_dtypes(include=["int64", "float64"]).columns)
    num_features_special_all = set(getSpecialNaNFeatures())
    num_features_special = set(X.columns.tolist()).intersection(
        num_features_special_all
    )
    num_features = num_features - num_features_special
    if cat_idx:
        cat_indices = [X.columns.get_loc(i) for i in cat_features]
        return cat_features, list(num_features), list(num_features_special), cat_indices
    else:
        return cat_features, list(num_features), list(num_features_special)
