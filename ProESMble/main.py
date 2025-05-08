#general use imports
import numpy as np
import pandas as pd

#results saving
import datetime
from datetime import date
import os.path
import glob
from pathlib import Path

#embedding
import torch
import Bio
from Bio import SeqIO
import esm 
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import tqdm
from tqdm import tqdm

#ML-training
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def ProESMble(known_library_file_name, mutation_effect_file_name, novel_library_file_name, esm_model_name = 'esm2_t30_150M_UR50D'):

    """
    Predicts mutation effects on protein function by embedding sequences using a pre-trained ESM model
    and training an ensemble model on known data to predict effects on novel variants.

    This function performs the following steps:
        1. Creates a results directory with the current date.
        2. Loads mutated and novel protein sequences from FASTA files.
        3. Loads experimental mutation effect data from a CSV file.
        4. Extracts protein embeddings from a specified ESM2 model.
        5. Trains an ensemble model on the known mutations and their effects.
        6. Predicts effects on a test set and novel sequence set.
        7. Saves all intermediate and final results to disk.

    Args:
        known_library_file_name (str): Path to FASTA file containing known mutant sequences.
        mutation_effect_file_name (str): Path to CSV file containing mutation effect data (must include an 'ID' column).
        novel_library_file_name (str): Path to FASTA file containing novel mutant sequences for prediction.
        esm_model_name (str, optional): Name of the ESM2 model to use for sequence embedding. 
                                        Defaults to 'esm2_t30_150M_UR50D'.

    Returns:
        tuple:
            - train_predictions (pd.DataFrame): Predictions on the training set.
            - test_predictions (pd.DataFrame): Predictions on the held-out test set.
            - novel_predictions (pd.DataFrame): Predictions for the novel mutant sequences.

    Raises:
        AssertionError: If the FASTA sequence IDs in `known_library_file_name` do not match the IDs in `mutation_effect_file_name`.

    Example:
        >>> ProESMble("known.fasta", "mutation_effects.csv", "novel.fasta")
        Embedding Known Sequences:
        Embedding Novel Sequences:
        Beginning Model Training:
        Files Saved Under : /path/to/ProESMble_Results/
    """

    #Create a saving Directory
    date = datetime.date.today().isoformat()
    folder = os.path.abspath(os.curdir)
    if os.path.exists(folder+"/" + date + "_" + 'ProESMble_Results') == False:
        Path(folder+"/" + date + "_" + 'ProESMble_Results').mkdir(parents=True, exist_ok=True)
        Path(folder+"/" + date + "_" + 'ProESMble_Results' + "/Data").mkdir(parents=True, exist_ok=True)
        Path(folder+"/" + date + "_" + 'ProESMble_Results' + "/Predictions").mkdir(parents=True, exist_ok=True)

    #Read In Data
    # Load mutated sequences from FASTA (assume IDs as column headers)
    mutant_sequences = {record.id: str(record.seq) for record in SeqIO.parse(known_library_file_name, "fasta")}
    novel_sequences = {record.description: str(record.seq) for record in SeqIO.parse(novel_library_file_name, "fasta")}

    # Load mutation effect data (assumes CSV with columns: ID, Effect)
    metadata_df = pd.read_csv(mutation_effect_file_name, index_col = 0)

    # Ensure the FASTA IDs match the metadata primary key
    assert set(mutant_sequences.keys()).issubset(set(metadata_df["ID"])), "Mismatch between FASTA and metadata IDs"

    #Embed Sequences
    print('Embedding Known Sequences:')
    mutant_embeddings = embed_sequences(mutant_sequences, esm_model_name)
    mutant_embedding_df = pd.DataFrame.from_dict(mutant_embeddings, orient="index")
    mutant_embedding_df.index.name = "ID"
    merged_df = metadata_df.merge(mutant_embedding_df, on="ID")

    print('Embedding Novel Sequences:')
    novel_embeddings = embed_sequences(novel_sequences, esm_model_name)
    novel_embedding_df = pd.DataFrame.from_dict(novel_embeddings, orient="index")
    novel_embedding_df.index.name = "ID"

 
    print('Beginning Model Training:')
    train_predictions, test_predictions, novel_predictions = model_training_and_predictions(merged_df,novel_embedding_df)

    # Save for further analysis
    merged_df.to_csv(folder+"/" + date + "_" + 'ProESMble_Results'+ "/Data/mutation_embeddings_"+esm_model_name+".csv", index=False)
    novel_embedding_df.to_csv(folder+"/" + date + "_" + 'ProESMble_Results'+ "/Data/novel_embeddings_"+esm_model_name+".csv", index=False)
    train_predictions.to_csv(folder+"/" + date + "_" + 'ProESMble_Results'+ "/Predictions/train_set_predictions_"+esm_model_name+".csv", index=False)
    test_predictions.to_csv(folder+"/" + date + "_" + 'ProESMble_Results'+ "/Predictions/test_set_predictions_"+esm_model_name+".csv", index=False)
    novel_predictions.to_csv(folder+"/" + date + "_" + 'ProESMble_Results'+ "/Predictions/novel_set_predictions_"+esm_model_name+".csv", index=False)

    print('Files Saved Under : '+folder+"/" + date + "_" + 'ProESMble_Results/')

    return train_predictions,test_predictions,novel_predictions





def extract_numbers(text):
    """
    Extract numeric characters from a string and return them as an integer.

    Args:
        text (str): Input string that may contain numeric characters.

    Returns:
        int: Integer composed of all digit characters found in the input string.
    """
    numbers = "".join([char for char in text if char.isdigit()])
    if not numbers:
        raise ValueError(f"No numeric characters found in the input: {text}")
    return int(numbers)




def embed_sequences(sequences, esm_model_name):
    """
    Generate ESM-based protein sequence embeddings.

    This function takes a dictionary of protein sequences and an ESM model name,
    then computes the mean per-residue embeddings for each sequence using the
    specified pretrained ESM model.

    Args:
        sequences (dict): A dictionary where keys are sequence IDs and values are amino acid sequences.
        esm_model_name (str): The name of the ESM pretrained model to use (e.g., "esm2_t33_650M_UR50D").

    Returns:
        dict: A dictionary mapping each sequence ID to its corresponding embedding (as a NumPy array).
    """
    if len(esm_model_name.split('_')) != 4:
        raise ValueError("Unknown ESM2 model: must be one of DEFAULT:'esm2_t30_150M_UR50D', 'esm2_t6_8M_UR50D', " \
        "'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D','esm2_t36_3B_UR50D','esm2_t48_15B_UR50D'")

    number_of_layers = extract_numbers(esm_model_name.split('_')[1])
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for seq_id, sequence in tqdm(sequences.items(), desc = 'Rows Embedded'):
            data = [(seq_id, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            
            results = model(batch_tokens, repr_layers=[number_of_layers], return_contacts=False)
            token_reps = results["representations"][number_of_layers]

            # Exclude CLS and EOS (if present)
            seq_embedding = token_reps[0, 1:len(sequence)+1].mean(dim=0).numpy()

            embeddings[seq_id] = seq_embedding

    return embeddings




def model_training_and_predictions(merged_df,novel_embedding_df):

    """
    Trains a stacked ensemble model to predict protein mutation effects and generates predictions for training, 
    test, and novel variant sets.

    This function performs the following steps:
        1. Splits the provided merged DataFrame (with known mutation effects and embeddings) into training and test sets.
        2. Applies standard scaling and PCA to reduce dimensionality.
        3. Trains and tunes three base regressors: Random Forest (RFR), K-Nearest Neighbors (KNR), and Gaussian Process (GPR).
        4. Uses cross-validated predictions from these base models to train a meta-model using XGBoost.
        5. Applies the ensemble to predict mutation effects for both test and novel sequences.
        6. Returns DataFrames with prediction breakdowns for each set.

    Args:
        merged_df (pd.DataFrame): DataFrame containing known mutation embeddings and mutation effect values. 
                                  Must include columns "ID" and "Effect".
        novel_embedding_df (pd.DataFrame): DataFrame containing ESM embeddings for novel sequences.

    Returns:
        tuple:
            - train_predictions (pd.DataFrame): Predictions on the training set, including outputs from each base model,
                                                the meta-model prediction, true values, and sequence IDs.
            - test_predictions (pd.DataFrame): Same structure as `train_predictions` for the test set.
            - novel_predictions (pd.DataFrame): Predictions for novel sequences, with base and meta-model outputs and IDs.

    Raises:
        ValueError: If any model training or transformation fails due to unexpected input dimensions or missing values.

    Example:
        >>> train_df, test_df, novel_df = model_training_and_predictions(merged_df, novel_embedding_df)
        RFR Tuned.
        KNR Tuned.
        GPR Tuned.
        R² score XGB (test): 0.754
        R² score XGB (train): 0.891
    """

    #Prepare Data For Model Training
    # Merge with mutation effect metadata
    X = merged_df.drop(columns=["ID", "Effect"])
    y = merged_df["Effect"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify= y)

    #Data_Preprocessing#################################################################################
    std_scale = StandardScaler(with_mean=True, with_std=True).fit(X_train)
    X_train_scaled = std_scale.transform(X_train)
    X_test_scaled = std_scale.transform(X_test)

    pca = PCA(n_components=0.95)  # keep 95% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    #Base-Model Hyper-Parameter Tuning###################################################################
    cv_method = KFold(n_splits=5, shuffle=True)
    scoring = 'neg_mean_squared_error'

    #########Gaussian Process Regressor########
    print('Training RFR...')

    pipeline = Pipeline([
        ('model', RandomForestRegressor(random_state = 42))
    ])

    param_grid = {
        'model__n_estimators': [100, 300, 500],            # More trees = better learning
        'model__max_depth': [10, 30, 50, None],            # None = full depth, try deeper
        'model__min_samples_split': [2, 5, 10],            # Lower = more splits = higher complexity
        'model__min_samples_leaf': [1, 2, 4],              # Smaller = more complexity
        'model__max_features': ['sqrt', 0.5, 1.0]          # Controls randomness, try more options
    }

    hyperparameter_grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv_method,
        n_jobs=-1,
        verbose=1
    )

    hyperparameter_grid.fit(X_train_pca, y_train)

    best_params = {
        key.replace('model__', ''): val
        for key, val in hyperparameter_grid.best_params_.items()
        if key.startswith('model_')
    }

    model1 = RandomForestRegressor(**best_params,random_state = 42).fit(X_train_pca,y_train)

    print('RFR Tuned.')


    #########K-Neighbors Regressor########
    print('Training KNR...')

    pipeline = Pipeline([
        ('KNR', KNeighborsRegressor())
    ])


    param_grid = {
        'model__weights': ['uniform', 'distance'],
        'model__algorithm': ['auto', 'ball_tree',],
        'model__p': [1,2],
        'model__n_neighbors': [3, 5, 10, 20],
    }

    hyperparameter_grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv_method,
        n_jobs=-1,
        verbose=1
    )

    hyperparameter_grid.fit(X_train_pca, y_train)

    best_params = {
        key.replace('model__', ''): val
        for key, val in hyperparameter_grid.best_params_.items()
        if key.startswith('model_')
    }

    model2 = KNeighborsRegressor(**best_params).fit(X_train_pca,y_train)

    print('KNR Tuned.')


    #########Gaussian Process Regressor########
    print('Training GPR...')

    pipeline = Pipeline([
        ('model', GaussianProcessRegressor(random_state = 42))
    ])

    param_grid = {
        'model__kernel': [
            1.0 * RBF(length_scale=1.0),
            1.0 * Matern(length_scale=1.0, nu=1.5),
            RationalQuadratic(length_scale=1.0, alpha=1.0),
        ],
        'model__alpha': [1e-5, 1e-3],
        'model__normalize_y': [True]
    }

    hyperparameter_grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=scoring,
        cv=cv_method,
        n_jobs=-1,
        verbose=1
    )

    hyperparameter_grid.fit(X_train_pca, y_train)

    best_params = {
        key.replace('model__', ''): val
        for key, val in hyperparameter_grid.best_params_.items()
        if key.startswith('model_')
    }

    model3 = GaussianProcessRegressor(**best_params,random_state = 42).fit(X_train_pca,y_train)

    #Base-Model Predicttions#############################################################################

    model1_preds = cross_val_predict(model1, X_train_pca, y_train, cv=5)
    model2_preds = cross_val_predict(model2, X_train_pca, y_train, cv=5)
    model3_preds = cross_val_predict(model3, X_train_pca, y_train, cv=5)

    X_meta_train = np.stack([model1_preds, model2_preds, model3_preds], axis=1)

    model1_test_pred = model1.predict(X_test_pca)
    model2_test_pred = model2.predict(X_test_pca)
    model3_test_pred = model3.predict(X_test_pca)

    X_meta_test = np.stack([model1_test_pred, model2_test_pred, model3_test_pred], axis=1)

    #Meta-Model Training and Predictions################################################################

    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
        X_meta_train, y_train, test_size=0.2, random_state=42
    )


    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=20,
    )

    # Fit with early stopping
    xgb.fit(
        X_train_meta, y_train_meta,
        eval_set=[(X_val_meta, y_val_meta)],
        verbose=False
    )

    y_train_predict = xgb.predict(X_meta_train)
    y_test_predict = xgb.predict(X_meta_test)

    ### Novel 
    X_novel_scaled = std_scale.transform(novel_embedding_df)
    X_novel_pca = pca.transform(X_novel_scaled)

    # Forming Predictions##############################################################################
    model1_novel_pred =  model1.predict(X_novel_pca)
    model2_novel_pred =  model2.predict(X_novel_pca)
    model3_novel_pred =  model3.predict(X_novel_pca)

    X_meta_novel= np.stack([model1_novel_pred, model2_novel_pred, model3_novel_pred], axis=1)

    # Meta Predictions##################################################################################

    y_novel_predict = xgb.predict(X_meta_novel)


    #### Saving Block ###############

    train_predictions = pd.DataFrame(data = X_meta_train, columns = ['RFR', 'KNR', 'GPR'])
    train_predictions['XGBoost'] = y_train_predict
    train_predictions['True Value'] = y_train.values
    train_predictions['ID'] = merged_df.loc[X_train.index, "ID"].values
    train_predictions = train_predictions.sort_values(by = 'XGBoost', ascending = False)

    test_predictions = pd.DataFrame(data = X_meta_test, columns = ['RFR', 'KNR', 'GPR'])
    test_predictions['XGBoost'] = y_test_predict
    test_predictions['True Value'] = y_test.values
    test_predictions['ID'] = merged_df.loc[X_test.index, "ID"].values
    test_predictions = test_predictions.sort_values(by = 'XGBoost', ascending = False)

    novel_predictions = pd.DataFrame(data = X_meta_novel, columns = ['RFR', 'KNR', 'GPR'])
    novel_predictions['XGBoost'] = y_novel_predict
    novel_predictions['ID'] = novel_embedding_df.index.values
    novel_predictions = novel_predictions.sort_values(by = 'XGBoost', ascending = False)


    print(f"R\u00b2 score XGB (test): {r2_score(y_test,y_test_predict):.3f}")
    print(f"R\u00b2 score XGB (train): {r2_score(y_train,y_train_predict):.3f}")

    return train_predictions,test_predictions,novel_predictions

