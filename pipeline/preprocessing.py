import pandas as pd
import os
import numpy as np
from typing import Optional, Tuple, Dict, Any

def preprocessing(
    df: pd.DataFrame,
    drug_df: pd.DataFrame,
    n_drugs: int = 3,
    n_doses: int = 100,
    min_val: float = -6.0,
    max_val: float = 4.0,
    use_cache: bool = False,
    cache_path: str = None
) -> Tuple[pd.DataFrame, Optional[Dict[str, Dict[Any, int]]]]:
    """
    Preprocess the drug response dataset in-place:
    - Log-scale and discretize dose columns into bins.
    - Clip float_value between 0 and 1.
    - Encode study_id, sample_id, and drug IDs with integer labels.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame. Must contain dose columns, 'float_value', 'study_id', and 'sample_id'.
    n_drugs : int
        Number of drug columns (e.g., drug1, drug2, ...).
    n_doses : int
        Number of bins for dose discretization.
    min_val : float
        Minimum log10 dose value for clipping.
    max_val : float
        Maximum log10 dose value for clipping.

    Returns
    -------
    pd.DataFrame
        The modified input DataFrame (in-place).
    Optional[dict]
        If return_encoders is True, returns a dictionary of label encodings.
    """
    if use_cache and os.path.exists(os.path.join(cache_path, "preprocessed_data.feather")):
        dff = pd.read_feather(os.path.join(cache_path, "preprocessed_data.feather"))
        enc_dff = pd.read_feather(os.path.join(cache_path, "label_encoders.feather"))
        return dff, enc_dff
    
    # Get SMILE dict
    drug_id_to_smile = drug_df.set_index("id")["smiles"].to_dict()
    drug_id_to_smile[""] = None
    
    drug_columns = [f"drug{i+1}" for i in range(n_drugs)]
    dose_columns = [f"dose{i+1}" for i in range(n_drugs)]

    for c in drug_columns + dose_columns:
        if c not in df.columns:
            df[c] = None

    # Replace NaNs and log-transform doses
    df[drug_columns] = df[drug_columns].fillna("")
    df[dose_columns] = df[dose_columns].fillna(10**min_val)
    df[dose_columns] = np.log10(df[dose_columns])

    # Clip and discretize doses
    df[dose_columns] = df[dose_columns].clip(lower=min_val, upper=max_val)
    df[dose_columns] = ((df[dose_columns] - min_val) / (max_val - min_val) * (n_doses - 1)).round().astype("Int64")

    # Clip float values
    df["float_value"] = df["float_value"].clip(lower=0.0, upper=1.0)

    # Encode categorical columns
    study_classes = np.sort(df["study_id"].unique())
    study_encoder = {v: i for i, v in enumerate(study_classes)}
    df["study_id"] = df["study_id"].map(study_encoder)

    sample_classes = np.sort(df["sample_id"].unique())
    sample_encoder = {v: i for i, v in enumerate(sample_classes)}
    df["sample_id"] = df["sample_id"].map(sample_encoder)

    drug_classes = np.sort(np.unique(df[drug_columns].values.ravel()))
    drug_smile_encoder = {v: i for i, v in enumerate(drug_classes)}
    drug_no_smile_encoder = {k: 0 if drug_id_to_smile[k] else v for k, v in drug_smile_encoder.items()}
    for col in drug_columns:
        df[f"{col}_smile"] = df[col].map(drug_smile_encoder)
    for col in drug_columns:
        df[f"{col}_no_smile"] = df[col].map(drug_no_smile_encoder)
    
    df = df.drop(drug_columns, axis=1)

    enc_df = pd.concat({
        "study_id": pd.Series({v: k for k, v in study_encoder.items()}),
        "sample_id": pd.Series({v: k for k, v in sample_encoder.items()}),
        "drug_smile_id": pd.Series({v: k for k, v in drug_smile_encoder.items()}),
        "drug_no_smile_id": pd.Series({v: k for k, v in drug_no_smile_encoder.items() if v != 0 or k == ""}),
        "smiles": pd.Series({v: drug_id_to_smile[k] for k, v in drug_smile_encoder.items()})
    }, axis=1).where(pd.notnull, None)
    enc_df["label_id"] = enc_df.index

    assert enc_df.index.is_monotonic_increasing, "Index is not sorted in ascending order"

    if use_cache:
        os.makedirs(cache_path, exist_ok=True)
        df.to_feather(os.path.join(cache_path, "preprocessed_data.feather"))
        enc_df.to_feather(os.path.join(cache_path, "label_encoders.feather"))
    
    return df, enc_df


def preprocessing_fm():
    max_length = int(emb.sum(1).max() + 1)

    tokens_ids = np.full((emb.shape[0], max_length), 1025, dtype=np.int32)
    tokens_ids[:, 0] = 1024
    attention_mask = np.zeros((emb.shape[0], max_length), dtype=np.int32)
    
    for i in range(emb.shape[0]):
        args = np.argwhere(emb[i]).ravel()
        tokens_ids[i, 1:len(args)+1] = args
        attention_mask[i, :len(args)+1] = 1