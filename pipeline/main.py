import os
import time
import pipeline
import pandas as pd
import itertools
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from pan_preclinical_data_models.containers import ResultSet


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


def print_section(title, char="=", width=60, major=False):
    print("\n" + char * width)
    if major:
        print(char * width)
    print(f"{title.center(width)}")
    if major:
        print(char * width)
    print(char * width + "\n")
    

def print_df_info(df, name, char="-", width=60):
    print(f"\n{name} - shape: {df.shape}")
    df.info()
    show_dataframe_head(df, title=name)

def print_end_section(start, char="=", width=60):
    print(f"Step done in {time.time()-start:.2f} seconds")
    # print(char * width + "\n")

def run(cfg):
    pipeline_steps = cfg.pipeline_steps
    
    print_section("MODEL PRE-TRAINING", char="*", major=True)
    
    if "load_dataset" in pipeline_steps:
        print_section("STEP: Load Dataset")
        start = time.time()
        if cfg.data_cleaning.use_cache_clean and os.path.exists(cfg.data_cleaning.clean_save_path):
            rs = ResultSet.read(cfg.data.path)
            df = pd.DataFrame([])
            drug_df = rs.drug_df.reset_index(drop=True)
            print_df_info(drug_df, "Drug Info")
        else:
            if ".feather" == cfg.data.path[-8:]:
                df = pd.read_feather(cfg.data.path)
                drug_df = pd.read_feather(cfg.data.drug_df_path)
            else:
                rs = ResultSet.read(cfg.data.path)
                df = rs.viability_measurement_df
                df.reset_index(drop=True, inplace=True)
                drug_df = rs.drug_df.reset_index(drop=True)
            print_df_info(df, "Raw Data")
            print_df_info(drug_df, "Drug Info")
        print_end_section(start)
        
    
    if "clean_data" in pipeline_steps:
        print_section("STEP: Clean Data")
        start = time.time()
        df = pipeline.data_cleaning.clean_data(
            df,
            remove_study_ids=cfg.data_cleaning.remove_study_ids,
            result_col=cfg.data_cleaning.isotonic_result_column,
            max_isotonic_error=cfg.data_cleaning.max_isotonic_error,
            use_cache_isotonic=cfg.data_cleaning.use_cache_isotonic,
            use_cache_clean=cfg.data_cleaning.use_cache_clean,
            isotonic_save_path=cfg.data_cleaning.isotonic_save_path,
            clean_save_path=cfg.data_cleaning.clean_save_path,
        )
        print_df_info(df, "Cleaned Data")
        print_end_section(start)
    
    if "preprocessing" in pipeline_steps:
        print_section("STEP: Preprocessing")
        start = time.time()
        df, label_encoders = pipeline.preprocessing.preprocessing(
            df,
            drug_df,
            n_drugs=cfg.model.n_drugs,
            n_doses=cfg.model.n_dose_bins,
            min_val=cfg.preprocessing.min_val,
            max_val=cfg.preprocessing.max_val,
            use_cache=cfg.preprocessing.use_cache,
            cache_path=cfg.preprocessing.cache_path
        )
        print_df_info(df, "Preprocessed Data")
        print_df_info(label_encoders, "Label Encoders")
        print_end_section(start)
    
    if "drug_embeddings" in pipeline_steps:
        print_section("STEP: Drug SMILE Embeddings")
        start = time.time()
        drug_embs = pipeline.drug_embeddings.drug_embeddings(
            smiles_list=list(label_encoders["smiles"].values),
            vocab_size=cfg.model.drug_vocab_size,
            drug_embedding_method=cfg.drug_embeddings.name,
            use_cache=cfg.drug_embeddings.use_cache,
            cache_path=cfg.drug_embeddings.cache_path,
            **cfg.drug_embeddings.params,
        )
        print(f"Drug Embeddings Shape: {drug_embs.shape}")
        print_end_section(start)
    else:
        drug_embs = None
    
    if "create_model" in pipeline_steps:
        print_section("STEP: Create Model")
        start = time.time()
        model, model_predict = pipeline.model.create_deep_model(
            drug_embeddings_weights=drug_embs,
            study_vocab_size=cfg.model.study_vocab_size,
            sample_vocab_size=cfg.model.sample_vocab_size,
            drug_vocab_size=cfg.model.drug_vocab_size,
            n_doses=cfg.model.n_dose_bins,
            n_drugs=cfg.model.n_drugs,
            emb_dim=cfg.model.emb_dim,
            drug_emb_dim=cfg.model.drug_emb_dim,
            n_layers=cfg.model.n_layers,
            units=cfg.model.units,
            rank=cfg.model.rank,
            semi_window_size=cfg.model.semi_window_size,
            gamma=cfg.model.gamma,
            activation=cfg.model.activation,
        )
        model.summary()
        print_end_section(start)
    
    if "extract_drug_info" in pipeline_steps:
        print_section("STEP: Load Drug Library")
        start = time.time()
        drug_lib = pipeline.drug_library.extract_drug_info(
            raw_path=cfg.drug_library.raw_path,
            read_params=cfg.drug_library.read_params,
            column_name=cfg.drug_library.column_name,
            cache_path=cfg.drug_library.cache_path,
            use_cache=cfg.drug_library.use_cache,
        )

        drug_to_remove = drug_lib.loc[drug_lib.smiles.isna()].name
        print(f"Drug LIb shape: {drug_lib.shape}")
        print(f"Remove drug without SMILE: Nb of drug removed {len(drug_to_remove)}")
        for name in drug_to_remove:
            print(f"  - {name}")

        drug_lib = drug_lib.loc[~drug_lib.smiles.isna()]
        
        print_df_info(drug_lib, f"Drug Library: {cfg.drug_library.name}")
        print_end_section(start)
    
    if "train_model" in pipeline_steps:
        print_section("STEP: Training")
        start = time.time()
        compound_ids = drug_lib["pubchem_compound_id"].dropna().unique().astype(float)
        substance_ids = drug_lib["pubchem_substance_id"].dropna().unique().astype(float)
    
        mask = (
            drug_df["pubchem_compound_id"].astype(float).isin(compound_ids) |
            drug_df["pubchem_substance_id"].astype(float).isin(substance_ids)
        )
    
        matching_ids = drug_df.loc[mask, "id"].unique()
        drugs_to_keep = label_encoders.loc[label_encoders.drug_smile_id.isin(matching_ids), "label_id"].unique()

        print(f"Drugs to keep: {drugs_to_keep.shape}")
        
        pipeline.training.train_pipeline(
            df=df,
            model=model,
            n_drugs=cfg.model.n_drugs,
            save_path=cfg.training.save_path,
            batch_size=cfg.training.batch_size_train,
            eval_batch_size=cfg.training.batch_size_test,
            steps_per_epoch=cfg.training.steps_per_epoch,
            epochs=cfg.training.epochs,
            n_held_out_drugs=cfg.training.super_test_n_drugs,
            history_path=cfg.training.training_history,
            optimizer=cfg.training.optimizer,
            optimizer_params=cfg.training.optimizer_params,
            drugs_to_keep=drugs_to_keep,
            use_cache=cfg.training.use_cache
        )
        print("Training Done!")
        print_end_section(start)
    
    print_section("COLD-START AL", char="*", major=True)
    
    if "cold_start_embeddings" in pipeline_steps:
        print_section("STEP: New Drug SMILE Embedding")
        start = time.time()
        drug_embs_for_drug_lib = pipeline.drug_embeddings.drug_embeddings(
            smiles_list=list(drug_lib.smiles.where(pd.notnull, None).values),
            vocab_size=cfg.model.drug_vocab_size,
            drug_embedding_method=cfg.drug_embeddings.name,
            use_cache=False,
            **cfg.drug_embeddings.params,
        )
        print(f"New Drug Embeddings Shape: {drug_embs_for_drug_lib.shape}")
    
        new_embeddings = np.zeros(model.get_layer("drug_embedding_smile").embeddings.shape)
        new_embeddings[:len(drug_embs_for_drug_lib)] = drug_embs_for_drug_lib
        model.get_layer("drug_embedding_smile").embeddings.assign(tf.identity(new_embeddings))

        print("Model Embedding Updated!")
        print_end_section(start)
    
    
    if "drug_auc" in pipeline_steps:
        print_section("STEP: Drug AUC Representation")
        start = time.time()
        study_sample_pairs = [(x[0], x[1]) for x in df[['study_id', 'sample_id']].drop_duplicates().values]
    
        pairs = list(itertools.combinations(drug_lib.index.values, 2))
        combi_drugs = pd.DataFrame(pairs, columns=["drug1_smile", "drug2_smile"])
        for i in range(3, cfg.model.n_drugs + 1):
            combi_drugs[f"drug{i}_smile"] = 0
        for i in range(1, cfg.model.n_drugs + 1):
            combi_drugs[f"drug{i}_no_smile"] = 0

        print("Number of Combos", len(pairs))
        
        drug_auc_emb = pipeline.drug_embeddings.drug_response_auc_representation(
            combi_drugs=combi_drugs,
            study_sample_pairs=study_sample_pairs,
            model=model,
            kind=cfg.cold_start.combo_drug_selection.kind,
            batch_size=cfg.cold_start.combo_drug_selection.batch_size,
            use_cache=cfg.cold_start.combo_drug_selection.use_cache,
            cache_path=cfg.cold_start.combo_drug_selection.cache_path,
        )
        print_df_info(drug_auc_emb, f"Drug AUC")
        print_end_section(start)
    

    if "importance_scores" in pipeline_steps:
        print_section("STEP: Importance Score")
        start = time.time()
        importance_scores, avg_response = pipeline.drug_embeddings.compute_drug_importance_scores(
            model,
            study_sample_pairs=study_sample_pairs,
            drug_smile_labels=drug_lib.index.values,
            drug_no_smile_labels=np.zeros(len(drug_lib.index.values)),
            num_drugs=cfg.model.n_drugs,
            batch_size=cfg.cold_start.dose_selection.batch_size,
            diff_order=cfg.cold_start.dose_selection.diff_order,
            quantile=cfg.cold_start.dose_selection.quantile,
            use_cache=cfg.cold_start.dose_selection.use_cache,
            cache_path=cfg.cold_start.dose_selection.cache_path,
        )
        print(f"Importance Scores Shape: {importance_scores.shape}")
        print_end_section(start)

    if "kmedoids_selection" in pipeline_steps:
        print_section("STEP: Cold-Start Selection")
        start = time.time()
        
        print("Dose Range:")
        max_val = cfg.preprocessing.max_val
        min_val = cfg.preprocessing.min_val
        n_doses = cfg.model.n_dose_bins
        range_doses = np.arange(n_doses) * (max_val - min_val) / (n_doses-1) + min_val
        mask = (range_doses >= -4) & (range_doses <= 2)

        range_doses_bins = ((range_doses[mask] - min_val) * (n_doses-1) / (max_val - min_val)).round().astype(int)
        print(range_doses_bins)

        print("Drug Combi")
        no_smiles = drug_lib.loc[drug_lib.smiles.isna()].index
        drug_auc_emb = drug_auc_emb.loc[(~drug_auc_emb.drug1.isin(no_smiles)) & (~drug_auc_emb.drug2.isin(no_smiles))]
        
        selected_drug_combi, selected_dose_combi = pipeline.cold_start.cold_start_kmedoids(
            drug_rpz_df=drug_auc_emb,
            importance_scores=importance_scores,
            unique_dose=range_doses_bins,
            n_combinations=cfg.cold_start.budget,
            distance_metric=cfg.cold_start.combo_drug_selection.method_params.metric,
            save_path=cfg.cold_start.save_path
        )

        mapping_drug = drug_lib['name'].to_dict()
        mapping_dose = {k: 10**range_doses[k] for k in range(len(range_doses))}
        
        mapped_drug_combi = np.vectorize(mapping_drug.get)(selected_drug_combi)
        mapped_dose_combi = np.vectorize(mapping_dose.get)(selected_dose_combi)
        
        cols=["drug1", "drug2", "dose1", "dose2"]
        cols += [c+"_index" for c in cols]
        pd.DataFrame(np.concatenate((
            mapped_drug_combi,
            mapped_dose_combi,
            selected_drug_combi,
            selected_dose_combi,
        ), axis=1),
                     columns=cols
        ).to_csv(cfg.cold_start.save_path)
        
        print_end_section(start)