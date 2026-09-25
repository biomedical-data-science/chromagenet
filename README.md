# ChromAgeNet

Code accompanying the publication:

> **Deep learning predicts haematopoietic stem cell ageing from 3D chromatin images**
> *Aging Cell* (in press).

ChromAgeNet is a convolutional neural network that classifies haematopoietic stem cells (HSCs) as young or aged from 3D nuclear chromatin images. The repository also contains the radiomics-based analysis used for comparison and interpretation.

**Scope:** this repository is meant to reproduce the analyses and figures shown in the paper. Please note that it is not a maintained software package, and the code is provided as-is.

## Repository structure

```
envs/        Conda environment files (one per analysis stage)
methods/     Python modules used by the notebooks
models/      Trained ChromAgeNet models (see "Trained models" below)
notebooks/   Jupyter notebooks, numbered in the order they should be run
results/     Metadata tables, and intermediate results written by the notebooks
```

The notebooks expect the raw and preprocessed images in a `data/` folder placed **next to** the repository (`../data` from the repository root).

`results/` includes two metadata tables:

- `data_folder_structure.csv`: condition, year, microscopist, cell type and sex of each image folder (input to notebook 01).
- `czi_metadata_filt.csv`: image metadata after outlier removal (output of notebook 01, used by notebooks 02, 05 and 09–12).

All other files in `results/` are created when running the notebooks.

## Installation

Create environments you need with [conda](https://docs.conda.io):

```bash
# Preprocessing and radiomics (notebooks 01–03, 08–13)
conda env create -f envs/preprocessing_radiomics.yml
conda activate chromagenet-radiomics
pip install --no-build-isolation pyradiomics==3.0.1

# CNN training and evaluation (notebooks 04–06)
conda env create -f envs/deep_learning.yml

# Explainability (notebook 07)
conda env create -f envs/xai.yml
conda activate chromagenet-xai
pip install --no-deps xplique==1.4.0
```

The extra `pip` steps are explained in the header of each environment file. Training the CNN requires a GPU.

## Notebooks

| Notebook | Description | Environment |
|---|---|---|
| `01_czi_exploration_and_outlier_detection` | Reads the raw `.czi` images and metadata, and detects outliers | radiomics |
| `02_dataset_summary_tables` | Summary tables of the dataset | radiomics |
| `03_czi_preprocessing` | Nucleus segmentation, resampling and export of 3D volumes and 2D slices | radiomics |
| `04_chromagenet_training_crossvalidation` | Trains ChromAgeNet with 5-fold cross-validation, and calibrates it | dl |
| `05_chromagenet_metadata_evaluation` | Relates model predictions to image metadata | dl |
| `06_chromagenet_compounds_evaluation` | Applies the model to HSCs treated with rejuvenating compounds | dl |
| `07_chromagenet_explainability` | Explainability analysis of the CNN (SHAP, Xplique) | xai |
| `08_radiomics_feature_extraction` | Extracts 3D radiomics features with PyRadiomics | radiomics |
| `09_radiomics_classification` | Feature selection, machine learning benchmark and SHAP analysis | radiomics |
| `10_radiomics_classification_binarized` | Same analysis as 09, on binarized nuclear masks | radiomics |
| `11_radiomics_compounds_comparison` | Radiomics features of compound-treated HSCs | radiomics |
| `12_radiomics_filters_visual_comparison` | Visual comparison of radiomics image filters | radiomics |
| `13_radiomics_supplementary_tables` | Supplementary tables for the radiomics benchmark | radiomics |

Run the notebooks from inside the `notebooks/` folder. Notebooks 09–13 depend on the features from 08.

## Data

The imaging data can be found in the following link:
https://doi.org/10.34810/data2372

## Trained models

```
models/chromagenet/   calmodel_fold_1.keras        Trained model (fold 1)
                      calmodel_fold_*.joblib       Beta calibrators (folds 1–5)
models/checkpoints/   fold_1/44-0.692.weights.h5   Best checkpoint weights (fold 1)
```

Notebooks 05–07 use the fold 1 model, so they can be run without retraining. The models and checkpoints of the remaining folds will be added shortly; they can also be regenerated with notebook 04.


## License

This project is released under the [MIT License](LICENSE).
