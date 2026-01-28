import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from pan_preclinical_data_models.containers import ResultSet
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def split_dose_drug(df: pd.DataFrame) -> pd.DataFrame:
    """
    Splits the 'doses' and 'drug_ids' list-columns into separate fixed columns.

    Parameters:
    - df: Input DataFrame with 'doses' and 'drug_ids' as list-like columns.

    Returns:
    - DataFrame with new columns dose1, dose2, ..., drug1, drug2, ...
      replacing the original list columns.
    """
    max_num_drugs = df['drug_ids'].apply(len).max()

    # Explode both columns at the same time
    df_exploded = df[['doses', 'drug_ids']].explode(['doses', 'drug_ids'])
    
    # Create a unique index for each original row before exploding
    df_exploded['orig_index'] = df_exploded.index
    
    # Reset index to remove MultiIndex issues
    df_exploded = df_exploded.reset_index(drop=True)
    
    # Pivot to get 4 columns per original row
    df_final = df_exploded.groupby('orig_index').agg(lambda x: x.tolist() + [None] * (max_num_drugs - len(x)))

    # Flatten the lists into separate columns
    df_final = pd.concat((pd.DataFrame(df_final['doses'].to_list(), columns=[f'dose{i+1}' for i in range(max_num_drugs)]),
                          pd.DataFrame(df_final['drug_ids'].to_list(), columns=[f'drug{i+1}' for i in range(max_num_drugs)])),
                          axis=1, ignore_index=False)
    
    df.reset_index(drop=True, inplace=True)
    for i in range(max_num_drugs):
        df[f'drug{i+1}'] = df_final[f'drug{i+1}']
    for i in range(max_num_drugs):
        df[f'dose{i+1}'] = df_final[f'dose{i+1}']
    df.drop(columns=['doses', 'drug_ids'], inplace=True)
    
    return df, max_num_drugs


def fit_isotonic(
    df: pd.DataFrame,
    result_col: str,
    max_num_drugs: int = 2
) -> pd.DataFrame:
    """
    Fits isotonic regression on dose1 vs float_value for each group of drugs and samples.

    Parameters:
    - df: Input DataFrame with dose and drug columns.
    - result_col: Name for the column to store isotonic predictions.
    - max_num_drugs: Number of drugs considered per sample.

    Returns:
    - DataFrame with an additional column containing isotonic regression predictions.
    """
    group_cols = ['study_id', 'sample_id', 'drug_treatment_id', 'drug1']
    group_cols += [f'drug{i}' for i in range(2, max_num_drugs+1)]
    group_cols += [f'dose{i}' for i in range(2, max_num_drugs+1)]

    grouped = df.groupby(group_cols, dropna=False)
    results = {"id": [], result_col: []}

    for _, group in tqdm(grouped, desc="Fitting isotonic regression", unit="group"):
        x = group.dose1.values
        y = group.float_value.values

        ir = IsotonicRegression(out_of_bounds="clip", increasing=False)
        yp = ir.fit_transform(x, y)

        results["id"].extend(group.id.values)
        results[result_col].extend(yp)

    iso_df = pd.DataFrame(results).set_index("id")
    print("iso_df created")
    df[result_col] = df["id"].map(dict(zip(results["id"], results[result_col])))
    print("result col created")
    return df


def remove_noisy_data(
    df: pd.DataFrame,
    result_col: str = "isotonic_pred",
    threshold: float = 0.2,
    max_num_drugs: int = 2,
    remove_study_ids: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Cleans the dataset by removing groups with high average absolute isotonic regression error 
    and optionally removing rows from specified studies.

    The grouping is based on drug and dose columns.

    Parameters:
    - df: DataFrame to clean.
    - result_col: Column name with isotonic regression predictions.
    - threshold: Maximum allowed average absolute error per group.
    - max_num_drugs: Number of drugs considered per sample.
    - remove_study_ids: Optional list of study IDs to remove entirely.

    Returns:
    - Filtered DataFrame with only groups below the error threshold.
    """
    group_cols = ['study_id', 'sample_id', 'drug_treatment_id', 'drug1']
    group_cols += [f'drug{i}' for i in range(2, max_num_drugs+1)]
    group_cols += [f'dose{i}' for i in range(2, max_num_drugs+1)]

    # Remove specified studies if any
    if remove_study_ids:
        df = df.loc[~df["study_id"].isin(remove_study_ids)]

    df["diff"] = (df["float_value"] - df[result_col]).abs()
    df["avg_diff"] = df.groupby(group_cols, dropna=False)['diff'].transform('mean')

    df = df.loc[df.avg_diff < threshold]
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['diff', 'avg_diff'], inplace=True)
    return df


def clean_data(
    df: pd.DataFrame,
    remove_study_ids: Optional[List[str]] = None,
    result_col: str = "isotonic_pred",
    max_isotonic_error: float = 0.2,
    use_cache_isotonic: bool = True,
    use_cache_clean: bool = True,
    isotonic_save_path: str = "data/isotonic/default_isotonic.feather",
    clean_save_path: str = "data/cleaned/default_cleaned.feather",
) -> pd.DataFrame:
    """
    Full data cleaning pipeline.

    Steps:
    1. Take raw dataset df.
    2. If `use_cache_clean` is True and `clean_save_path` exists, load and return cached cleaned data.
    3. Otherwise, perform isotonic regression:
        - Load from `isotonic_save_path` if `use_cache_isotonic` is True and file exists.
        - Else compute and save isotonic regression results.
    4. Clean the dataset:
        - Remove groups with isotonic error > `max_isotonic_error`.
        - Drop any entries from `remove_study_ids` if provided.
    5. Expand list-based `doses` and `drug_ids` columns into fixed columns (`dose1`, `drug1`, etc.).
    6. Save cleaned dataset to `clean_save_path`.

    Parameters:
    - df: Dataframe.
    - remove_study_ids: Optional list of study IDs to exclude from the data.
    - result_col: Name of the column storing isotonic regression predictions.
    - max_isotonic_error: Threshold for filtering groups with poor isotonic regression fit.
    - use_cache_isotonic: Whether to load isotonic regression results from cache if available.
    - use_cache_clean: Whether to load cleaned data from cache if available.
    - isotonic_save_path: Path to save or load isotonic regression output.
    - clean_save_path: Path to save or load cleaned dataset.

    Returns:
    - A cleaned `pd.DataFrame` ready for downstream processing.
    """
    if use_cache_clean and Path(clean_save_path).exists():
        print(f"Loading cached cleaned data from {clean_save_path}")
        return pd.read_feather(clean_save_path)
    
    # Load cache if available and allowed
    if use_cache_isotonic and Path(isotonic_save_path).exists():
        print(f"Loading cached cleaned data from {isotonic_save_path}")
        df = pd.read_feather(isotonic_save_path)
        max_num_drugs = len(["drug%i"%i for i in range(10) if "drug%i"%i in df.columns])
    
    else:
        # Drop unused columns
        df.drop(columns=[
            "assay_measurement_ids",
            "source_file_id",
            "study_viability_measurement_id"
        ], inplace=True)
        
        # Split doses and drugs into fixed columns
        _, max_num_drugs = split_dose_drug(df)
        
        # --- Fit isotonic regression ---
        print("Fitting isotonic regression...")
        fit_isotonic(df, result_col, max_num_drugs=max_num_drugs)
    
        # Save isotonic results
        if use_cache_isotonic:
            Path(isotonic_save_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_feather(isotonic_save_path)
            print(f"Saved isotonic predictions to {isotonic_save_path}")
    
    # --- Clean data ---
    print(f"Dataset shape before cleaning: {df.shape}")
    df = remove_noisy_data(
        df,
        result_col=result_col,
        threshold=max_isotonic_error,
        max_num_drugs=max_num_drugs,
        remove_study_ids=remove_study_ids,
    )
    print(f"Dataset shape after cleaning: {df.shape}")

    # Save cleaned data
    if use_cache_clean:
        Path(clean_save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(clean_save_path)
        print(f"Saved cleaned data to {clean_save_path}")

    return df