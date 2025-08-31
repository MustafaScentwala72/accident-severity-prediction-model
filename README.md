# Accident Severity Prediction Model

## Overview
This repository contains an end-to-end workflow to predict accident severity from road, environment, and traffic features. It includes two parts: building a clean, modeling-ready dataset and training a supervised classifier with clear, class-wise metrics. Everything is delivered via two Jupyter notebooks and a minimal Python stack.

## Notebooks
- `notebooks/dataset-creation/dataset-creation-road-safety.ipynb` - builds the modeling-ready dataset (ingest, clean, harmonise columns, handle missing values, encode fields).
- `notebooks/model-selection-metrics/model-selection-and-metrics.ipynb` - trains and compares models, reports confusion matrix and class-wise precision/recall/F1, with emphasis on the severe class.

## Goals
- Prepare a tidy dataset: consolidate raw CSVs, harmonise columns, handle missing values, and encode fields.
- Train and compare baseline and tree-based models for accident severity prediction.
- Report confusion matrix, precision, recall, and F1 with a focus on the severe class.
- Keep the work reproducible and easy to follow for reviewers and collaborators.

## Data (high level)
- Tabular CSVs only. Exact filenames are omitted here.
- If you plan to publish raw data, review the original license and remove any sensitive attributes.
- Place CSVs under `data/` and update paths in the notebooks if needed.

## Environment
```
python 3.9+
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost  # optional, used in model comparison
```

Install:
```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

## How to run
1) Dataset creation
```
jupyter notebook notebooks/dataset-creation/dataset-creation-road-safety.ipynb
```
2) Model selection and metrics
```
jupyter notebook notebooks/model-selection-metrics/model-selection-and-metrics.ipynb
```
Run cells top to bottom. Adjust any file paths to your local `data/` folder if necessary.

## Method in brief
1. EDA: class balance, missing values, and key feature distributions.
2. Prepare data: consistent types, imputation, encoding, simple feature engineering.
3. Split: train/test with a fixed random seed.
4. Models: baseline (e.g., Logistic Regression) and tree-based models (Decision Tree, Random Forest, optionally XGBoost).
5. Validation: cross-validation on train where useful; final metrics on the held-out test set.
6. Reporting: confusion matrix and class-wise precision, recall, F1. Emphasis on recall for the severe class, with threshold tuning notes if needed.

## Results (where to look)
- The final notebook cells print the confusion matrix and class-wise metrics on the test set. Treat those outputs as the source of truth.
- If threshold tuning is performed, note the operating point used for the reported metrics.

## Business view
- Why this matters: better identification of severe cases supports faster response, targeted interventions, and resource planning.
- Trade-off: improve recall on the severe class while keeping precision acceptable for operational review.
- Next steps: use cost-aware thresholding (review cost vs. missed severe case cost) and monitor metrics by segment (road type, weather, time).

## Repository layout
.
- data/
- notebooks/
  - dataset-creation/
    - dataset-creation-road-safety.ipynb
  - model-selection-metrics/
    - model-selection-and-metrics.ipynb
- README.md

## Notes and next steps
- Add a simple model card to the repo (intended use, metrics, risks, and limitations).
- Save the trained model with joblib and provide a small predict script for batch scoring.
- For larger data, store samples and document how to fetch the full datasets.
- Consider temporal splits and stronger validation if deploying beyond coursework.

## License
MIT

## Author
Mustafa Scentwala
