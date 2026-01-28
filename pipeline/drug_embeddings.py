import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import interp1d
from typing import List, Tuple

# ----------------------------- EMBEDDING METHODS -----------------------------

def morgan(
    smiles_list: list[str],
    vocab_size: int,
    radius: int = 2,
    fpSize: int = 1024
) -> np.ndarray:
    """
    Compute Morgan fingerprints for a list of SMILES strings.

    Parameters:
        smiles_list (list[str]): List of SMILES strings (may include None).
        vocab_size (int): Number of unique drugs (length of embedding matrix).
        radius (int): Morgan fingerprint radius.
        fpSize (int): Size of the fingerprint vector.

    Returns:
        np.ndarray: A (vocab_size, fpSize) array of molecular fingerprints.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError as e:
        raise ImportError("RDKit is required for 'morgan'. Install with `pip install rdkit-pypi`.") from e

    generator = GetMorganGenerator(radius=radius, fpSize=fpSize)
    fingerprints = np.zeros((vocab_size, fpSize), dtype=np.float32)

    for i, smi in enumerate(smiles_list):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((fpSize,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints[i] = arr

    return fingerprints

def chembert(
    smiles_list: list[str],
    vocab_size: int,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 128
) -> np.ndarray:
    """
    Generate ChemBERTa embeddings for a list of SMILES strings.

    Parameters:
        smiles_list (list[str]): List of SMILES strings.
        vocab_size (int): Number of unique drugs.
        model_name (str): HuggingFace model name.
        batch_size (int): Batch size for embedding inference.

    Returns:
        np.ndarray: A (vocab_size, embedding_dim) array of embeddings.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
    except ImportError as e:
        raise ImportError("Transformers and PyTorch required. Install with `pip install transformers torch`.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embedding_dim = model.config.hidden_size
    embeddings = torch.zeros((vocab_size, embedding_dim), dtype=torch.float32)

    valid_indices = []
    valid_smiles = []

    for i, smi in enumerate(smiles_list):
        if smi is not None:
            valid_indices.append(i)
            valid_smiles.append(smi)

    for i in range(0, len(valid_smiles), batch_size):
        batch = valid_smiles[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        for j, idx in enumerate(valid_indices[i:i + batch_size]):
            embeddings[idx] = batch_embeddings[j]

    return embeddings.numpy()


METHOD_DICT = {
    "morgan": morgan,
    "chembert": chembert,
}

# ----------------------------- MAIN FUNCTION -----------------------------

def drug_embeddings(
    smiles_list: list[str],
    vocab_size: int,
    drug_embedding_method: str,
    use_cache: bool,
    cache_path: str = None,
    **drug_embedding_params
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Generate or load drug embeddings and encode drug columns in dataframe.

    Parameters:
        vocab_size (int): Total number of unique drugs.
        drug_embedding_method (str): Embedding method name ('morgan' or 'chembert').
        use_cache (bool): Whether to use cached results if available.
        cache_path (str): Directory to store or load cached results.
        **drug_embedding_params: Additional parameters for embedding method.

    Returns:
        tuple:
            df_encoded (pd.DataFrame): Dataframe with _smile and _no_smile columns.
            drug_embedding_matrix (np.ndarray): Drug embeddings array.
            drug_encoder (LabelEncoder): Fitted encoder from drug ID to index.
    """
    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)["embeddings"]

    if drug_embedding_method not in METHOD_DICT:
        raise ValueError(f"Unknown embedding method: {drug_embedding_method}")
    
    drug_embedding_matrix = METHOD_DICT[drug_embedding_method](
        smiles_list, vocab_size=vocab_size, **drug_embedding_params
    )

    if use_cache:
        np.savez_compressed(cache_path, embeddings=drug_embedding_matrix)

    return drug_embedding_matrix


def drug_response_auc_representation(
    combi_drugs: pd.DataFrame,
    study_sample_pairs: list[tuple[int, int]],
    model,
    batch_size: int = 2048,
    kind: str = "pair",
    use_cache: bool = False,
    cache_path: str = None,
):
    """
    Run cold-start projection for (study_id, sample_id) × drug combinations.

    Parameters
    ----------
    df_drugs : pd.DataFrame
        DataFrame with drug1_smile, ..., drugN_smile, drug1_no_smile, ..., drugN_no_smile.
    study_sample_pairs : List[Tuple[int, int]]
        List of (study_id, sample_id) pairs to test.
    model_predict : Callable
        Function that takes a batch of inputs and returns model predictions (np.ndarray).
    output_path : str
        Path to save the projection results as a Feather file.
    batch_size : int, optional
        Batch size for inference.
    """
    if use_cache and os.path.exists(cache_path):
        return pd.read_feather(cache_path)
    
    study_sample_arr = np.array(study_sample_pairs, dtype=np.int32)

    print(study_sample_arr.shape, combi_drugs.shape)
    
    inputs = []
    preds = []
    
    for k in tqdm(np.arange(len(combi_drugs)), desc="Computing AUCs"):
        drug_feature_array = np.tile(combi_drugs.values[k], (study_sample_arr.shape[0], 1))
        batch = np.concatenate([study_sample_arr, drug_feature_array], axis=1)
        
        yp = model(tf.identity(batch)).numpy()
        if kind == "unique":
            auc = yp[:, 0, :, :].mean((1, 2))
        elif kind == "pair":
            rank = yp.shape[2]
            auc = (1./rank) * np.matmul(yp[:, 0, :, :].transpose((0, 2, 1)),
                                        yp[:, 1, :, :]).mean((1, 2))
        else:
            auc = (1./rank) * np.einsum('bti, btj, btk -> bijk', yp[:, 0], yp[:, 1], yp[:, 2]).mean((1, 2, 3))
        
        inputs.append(batch)
        preds.append(auc)

    inputs = np.concatenate(inputs)
    auc_values = np.concatenate(preds)
    
    if kind == "unique":
        colnames = ["study_id", "sample_id"] + ["drug1"] + ["auc"]
        inputs = inputs[:, :3]
    elif kind == "pair":
        colnames = ["study_id", "sample_id"] + ["drug1", "drug2"] + ["auc"]
        inputs = inputs[:, :4]
    else:
        colnames = ["study_id", "sample_id"] + ["drug1", "drug2", "drug3"] + ["auc"]
        inputs = inputs[:, :5]
    
    full_data = np.column_stack((inputs, auc_values))
    
    result_df = pd.DataFrame(full_data, columns=colnames)
    
    if use_cache:
        result_df.to_feather(cache_path)
        print(f"[✓] Cold-start projection saved to {cache_path}")
    
    return result_df


def compute_drug_importance_scores(
    model,
    study_sample_pairs: List[Tuple[int, int]],
    drug_smile_labels: List[int],
    drug_no_smile_labels: List[int],
    num_drugs: int,
    batch_size: int = 2048,
    diff_order: int = 2,
    quantile: float = 0.9,
    use_cache: bool = False,
    cache_path: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute drug response predictions and importance scores for a list of drugs over 
    (study_id, sample_id) pairs using a trained TensorFlow model.

    Args:
        model: Trained TensorFlow model that takes input of shape (B, 2 + 2*num_drugs)
               and returns predictions of shape (B, D, N, F).
        study_sample_pairs: List of (study_id, sample_id) pairs representing samples to evaluate.
        drug_smile_labels: List of drug IDs using SMILES encoding.
        drug_no_smile_labels: List of drug IDs without SMILES encoding.
        num_drugs: Number of drugs considered per combination (D).
        batch_size: Number of samples per prediction batch.
        diff_order: Order of finite difference used to estimate drug response change.
        quantile: Quantile of the finite difference magnitudes used to determine importance.
        use_cache: If True, attempts to load/save computed results from/to cache_path.
        cache_path: File path to load/save cached results when use_cache is True.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - importance_scores: Array of shape (max_drug_id + 1, N), where each row contains
              quantile-based importance scores per dose for a drug.
            - avg_response: Array of shape (max_drug_id + 1, N), where each row contains
              the average predicted drug response per dose.
    """
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data["importance_scores"], data["avg_response"]

    # Predict in batches
    resp_list = []
    diff_list = []

    study_sample_array = np.array(study_sample_pairs)
    drug_feature_array = np.zeros((study_sample_array.shape[0], 2*num_drugs))

    pad_width = ((1,1),)
    
    for k in tqdm(np.arange(len(drug_smile_labels)), desc="Computing importance scores"):
        drug_feature_array[:, 0] = drug_smile_labels[k]
        drug_feature_array[:, num_drugs] = drug_no_smile_labels[k]
        batch = tf.identity(np.concatenate([study_sample_array, drug_feature_array], axis=1))
        
        yp = model(batch).numpy()
        assert yp.shape[1] == num_drugs

        prod = yp[:, 0, :, :]
        for d in range(1, num_drugs):
            next_yp = yp[:, d, :, :]
            next_yp = np.expand_dims(next_yp[:, :, 0], -1)
            prod *= next_yp
        drug_resp = prod.mean(axis=1)
        diff = np.diff(drug_resp, n=diff_order, axis=1)

        diff = np.pad(np.quantile(np.abs(diff), quantile, axis=0), pad_width, mode='constant')
        
        resp_list.append(drug_resp.mean(axis=0))
        diff_list.append(diff)

    avg_response = np.zeros((max(drug_smile_labels)+1, resp_list[0].shape[0]))
    importance_scores = np.zeros((max(drug_smile_labels)+1, resp_list[0].shape[0]))
    
    avg_response[drug_smile_labels] = np.stack(resp_list, axis=0)
    importance_scores[drug_smile_labels] = np.stack(diff_list, axis=0)

    if use_cache:
        np.savez(cache_path, importance_scores=importance_scores, avg_response=avg_response)

    return importance_scores, avg_response