
Towards Fair Machine learning in Healthcare: Ensuring Non-Discrimination for Disease Prediction
==============================

This repository contains the code, data, and documentation associated with the master thesis titled *Towards Fair Machine learning in Healthcare: Ensuring Non-Discrimination for Disease Prediction* conducted by *Claudia Herron Mulet*, and supervised by Dr. Polyxeni Gkontra and Dr. Karim Lekadir.

## Abstract

Over the past few years, there has been a rise in the utilization of information and communication technologies (ICTs) and electronic health records (EHRs) within the healthcare system. This increase has led to a substantial gathering of medical data, opening up promising prospects for personalized medicine. Notably, one promising application is the creation of disease risk assessment tools, designed to precisely estimate an individual’s predisposition to developing certain illnessess. These innovative tools empower healthcare professionals to conduct more targeted trials, closely monitor high-risk subjects, and implement timely interventions. However, as these systems start to be tested in real world scenarios, recent studies reveal that they might worsen off the situation of historically underprivileged groups in our society. These discriminatory biases might be caused by many reasons: unequal access to healthcare, false beliefs about biological differences, non-diverse datasets, machine learning (ML) models optimizing for the majority and disregarding underrepresented communities, etc. As a result, it becomes crucial to design and implement metrics and techniques to quantify and mitigate discriminatory biases.

In this work, we propose a comprehensive methodology that encompasses data wrangling, model evaluation, and the monitoring of both model performance and potential disparities. Building upon existing research on fairness in machine learning, we aim to adapt the fairness framework specifically for disease prediction, con- sidering that some of the protected features also contribute to increased disease risk. Furthermore, we apply both in-processing and post-processing mitigation techniques to a classifier trained on a large-scale dataset. By experimenting with two diseases of increasing prevalence, Primary Hypertension and Parkinson’s Disease, we seek to assess the effectiveness of these techniques in reducing discriminatory biases and ensuring equitable outcomes.

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │     └── diagnosis        <- The original diagnosis files.
    │     └── fields            <- The original field files.
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── ml             <- Generated ML result reports
    │   └── ml             <- Generated fairness result reports
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── config           <- Configuration scripts, ex., global vars.
    │   ├── data            <- Scripts to perform basic ETL
    │   ├── fairness        <- Scripts to perform fairness analysis and bias mitigation
    │   ├── ml              <- Scripts to train and evaluate ML models
    │   └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │
    └── .gitignore                  <- gitignore
    └── .pre-commit-config.yaml      <- pre commit configuration
    └── poetry.lock                 <- poetry lock file
    └── pyproject.toml              <- poetry toml file
    └── tox.ini                     <- tox file with settings 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


## Installation
Make sure you have Poetry installed. If you don't have it installed, you can follow the instructions at [Poetry Installation Guide](https://python-poetry.org/docs/).
1. Clone the repository:
```bash
git clone git@github.com:claudia-hm/spurlock.git
```
2. Navigate to the project directory:
```bash
cd spurlock
```
 3. Set up the Python environment and install the project dependencies using Poetry:
```bash
poetry install
```
This command will create a virtual environment for the project and install all the required dependencies specified in the pyproject.toml file.

4. Activate the virtual environment:
```bash
poetry shell
```

Activating the virtual environment ensures that you're working within the project's isolated environment.

Now you're ready to run the project and utilize the provided functionality.

## Data

This project uses UK Biobank, a large-scale biomedical research database. It is one of the most significant health research initiatives worldwide and aims to improve the prevention, diagnosis, and treatment of various diseases. This dataset is not publicly available. If you have acces to it, please place field field files under the directory `data/raw/fields` and diagnosis files in `data/raw/diagnosis`. 

Also, please add the publicly available coding files in the folder `data/external`: [Codings.csv](https://biobank.ctsu.ox.ac.uk/~bbdatan/Codings.csv) and [Data_Dictionary_Showcase.csv](https://biobank.ctsu.ox.ac.uk/~bbdatan/Data_Dictionary_Showcase.csv). 

## Source code 
### `config`: project configuration files
* `ml.yml`: machine learning configuration 
* `paths.py`: python file specifying paths
* `privileged_groups.yml`: configuration file for specifying privileged groups.
* `utils.py`: miscellaneous configuration helpers.

### `data`: scripts related to basic ETL
* `01_wrangling.py`: script to perform data wrangling. 
* `02_wrangling.py`: script to perform basic cleaning.
* `03_create_categories.py`: script to specify medical categories.
* `04_analysis.py`: script to perform custom data analysis.
* `loading.py`: data loading functions.
* `UKBiobank_scraper.yml`: UK Biobank scrapping functions. Script retrieved from https://github.com/lwaw/UK-Biobank-scraper
* `utils.py`: functions for performing ETL.

### `fairness`: scripts related to fairness evaluation and bias mitigation.
* `01_data_bias.py`: data bias evaluation protocol
* `02_model_fairness.py`: ML model fairness evaluation.
* `03_mitigation_inprocessing.py`: script to perform inprocessing mitigation.
* `04_mitigation_postprocessing.py`: script to perform postprocessing mitigation.
* `utils.py`: functions for performing ETL.


### `ml`: scripts related to training and performance, explainability analysis.
* `01_model_selection.py`: train models and perform hyperparameter search. Selects best model.
* `01a_custom_model_train.py`: train custom model, without performing model comparison. Supports resampling strategies.
* `02_model_analysis.py`: produce performance evaluation report and figures.
* `ml_models.py`: machine learning model definitions and grids for hyperparameter search.
* `utils.py`: functions related to ML.

### `visualization`: scripts for plotting.
* `data.py`: plots used in the `data` module.
* `fairness.py`: plots used in the `fairness` module.
* `ml.py`: plots used in the `ml` module.

## Why Spurlock?

Jeanne Spurlock was a psychiatrist who broke barriers as the first African American and first woman to receive the Edward A. Strecker M.D. Award. Despite coming from a low-income background, she pursued her passion for medicine and overcame financial obstacles to obtain her education. She made significant contributions to the field of psychiatry, completing her residency and fellowship in Chicago and holding prestigious positions at institutions like the Neuropsychiatric Institute and Meharry Medical College. Spurlock's research focused on the impact of discrimination and inequality on health and childhood development, and she actively advocated for funding and representation for minorities in medicine. Her legacy as a compassionate and pioneering psychiatrist continues to inspire and make a lasting impact on the field.

Source: [wikipedia](https://en.wikipedia.org/wiki/Jeanne_Spurlock)
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.


## Contact Information

If you have any questions or need further assistance, please don't hesitate to reach out to me:

GitHub: [claudia-hm](https://github.com/claudia-hm)
Linkedin: [Claudia Herron Mulet](https://www.linkedin.com/in/claudiaherronmulet/)
