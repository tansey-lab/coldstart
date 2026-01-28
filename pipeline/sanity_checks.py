import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt


def show_dataframe_head(
    df: pd.DataFrame,
    drop_cols: list = None,
    truncate_len: int = 10,
    head_rows: int = 5,
    title: str = "DataFrame Head"
):
    """
    Print a cleaned and truncated head of the DataFrame.

    Parameters:
    - df: Input DataFrame
    - drop_cols: List of columns to drop (default: None)
    - truncate_len: Number of characters to keep in string fields and column names (default: 10)
    - head_rows: Number of rows to show from the head (default: 5)
    - title: Title to print before table (default: "DataFrame Head:")
    """
    # Take the head first for efficiency
    df_to_show = df.head(head_rows).copy()

    # Drop unwanted columns
    if drop_cols:
        df_to_show = df_to_show.drop(columns=drop_cols, errors='ignore')

    # Convert all columns to strings and truncate
    for col in df_to_show.columns:
        df_to_show[col] = df_to_show[col].astype(str).str[:truncate_len]

    # Truncate column names
    truncated_columns = {col: col[:truncate_len] for col in df_to_show.columns}
    df_to_show.rename(columns=truncated_columns, inplace=True)

    # Display
    print(f"\n{title}:\n{tabulate(df_to_show, headers='keys', tablefmt='psql')}\n")


def plot_isotonic_diagnostics(
    df,
    remove_study_ids=None,
    result_col="isotonic_pred",
    save_path="data/sanity_checks/"
):
    """
    Plot isotonic regression fits for visual sanity checks.
    """
    if remove_study_ids:
        assert len(set(remove_study_ids) & set(df.study_id.unique())) == 0
    
    np.random.seed(42)
    indexes = np.argsort((df["float_value"] - df[result_col]).abs()).ravel()[::-1]
    
    count = 0
    for num, k in enumerate(indexes):
        row = df.iloc[k]

        mask = (
            (df.study_id == row.study_id) &
            (df.sample_id == row.sample_id) &
            (df.drug_treatment_id == row.drug_treatment_id)
        )
        for i in range(1, 10):
            col = "drug%i" % i
            if col in df.columns:
                val = row[col]
                mask &= (df[col] == val) | (df[col].isna() & pd.isna(val))
        for i in range(2, 10):
            col = "dose%i" % i
            if col in df.columns:
                val = row[col]
                mask &= (df[col] == val) | (df[col].isna() & pd.isna(val))

        dff = df.loc[mask]
        
        if dff.shape[0] > 2:
            x = np.log10(dff.dose1.values)
            args = np.argsort(x).ravel()
            x = x[args]
            y = dff.float_value.values[args]
            yp = dff[result_col].values[args]
    
            mae = np.mean(np.abs(y - yp))
    
            study = dff.study_id.values[0]
    
            plt.figure(figsize=(7, 5))
            plt.plot(
                x, y, "o", c="w",
                markeredgecolor=f"C3",
                markersize=6,
                label="Data"
            )
    
            plt.plot(x, yp, c="C0", label="Isotonic Regression", lw=2)
            plt.title(f'IR MAE: {mae:.4f} - Study: {study}', fontsize=15)
            plt.xlabel('Log10 Dose', fontsize=13)
            plt.ylabel('Viability', fontsize=13)
            plt.legend(loc="upper right", fontsize=12, bbox_to_anchor=(1.4, 1.0))
            plt.tight_layout()
    
            if save_path:
                filename = f"{save_path}/ir_group_{count}.png"
                plt.savefig(filename, dpi=150)
                plt.clf()
                plt.close()

            count += 1
            if count == 10:
                break
            