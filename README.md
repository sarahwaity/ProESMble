# ProESMble

**ProESMble** is a Python pipeline for predicting the functional effects of protein mutations using state-of-the-art transformer-based protein embeddings (ESM2) and an ensemble of machine learning models. It is designed for researchers in computational biology and protein engineering who want to integrate deep sequence representations with interpretable, high-performance predictive models.

---

## Features

- **Protein Embedding**: Leverages ESM2 models to extract per-sequence embeddings from protein FASTA files.
- **Mutation Effect Integration**: Joins sequence embeddings with experimental effect data for model training.
- **Stacked Ensembling**: Trains a combination of Random Forest, K-Nearest Neighbors, and Gaussian Process regressors with an XGBoost meta-model.
- **Prediction of Novel Sequences**: Evaluates unseen variants and predicts their mutation effects.
- **Auto-Saving**: Automatically saves results to a dated results directory, including predictions and intermediate data.

---

## Installation

This project requires the following dependencies:

```bash
pip install pandas scikit-learn numpy xgboost biopython tqdm torch esm
```

> **Note**: You must install [Facebook's ESM](https://github.com/facebookresearch/esm) package and download the required ESM2 model weights.

---

## Input Files

- `known_library_file_name.fasta`: Protein sequences with known mutation effects.
- `mutation_effect_file_name.csv`: A CSV file with at least two columns: `ID` and `Effect`.
- `novel_library_file_name.fasta`: Novel protein sequences to be predicted.

---

## Function Overview

### `ProESMble(known_library_file_name, mutation_effect_file_name, novel_library_file_name, esm_model_name='esm2_t30_150M_UR50D')`

This is the main function that orchestrates the pipeline:

#### Inputs:
- `known_library_file_name`: Path to FASTA file of known variants.
- `mutation_effect_file_name`: Path to CSV with mutation effect data.
- `novel_library_file_name`: Path to FASTA file of novel sequences.
- `esm_model_name`: ESM2 model variant to use (default: `'esm2_t30_150M_UR50D'`).

#### Outputs:
Returns 3 pandas DataFrames:
- `train_predictions`: Training set predictions (base + ensemble).
- `test_predictions`: Test set predictions.
- `novel_predictions`: Predictions on novel protein sequences.

All results are saved to a directory:
```
<date>_ProESMble_Results/
├── Data/
│   ├── mutation_embeddings_<esm_model>.csv
│   ├── novel_embeddings_<esm_model>.csv
├── Predictions/
│   ├── train_set_predictions_<esm_model>.csv
│   ├── test_set_predictions_<esm_model>.csv
│   ├── novel_set_predictions_<esm_model>.csv
```

---

## Example

```python
from ProESMble import ProESMble

train_preds, test_preds, novel_preds = ProESMble(
    known_library_file_name="mutant_sequences.fasta",
    mutation_effect_file_name="mutation_effects.csv",
    novel_library_file_name="novel_variants.fasta"
)
```

---

## Model Architecture

- **Embedding**: ESM2 (via Facebook's pretrained models)
- **Preprocessing**: StandardScaler + PCA
- **Base Models**:
  - Random Forest Regressor
  - K-Nearest Neighbors Regressor
  - Gaussian Process Regressor
- **Meta Model**: XGBoost Regressor

---

## Contact

For questions, feel free to open an issue or contact the repository maintainer (sarah.j.wait@mgmail.com).

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

Happy modeling!
