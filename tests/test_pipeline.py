from omegaconf import OmegaConf

from pipeline.main import run

config = {
    "data": {
        "name": "test_data",
        "path": "/data1/tanseyw/projects/deepadaptx/deepadaptx/data/sample_raw_data_10k.feather",
        "drug_df_path": "/data1/tanseyw/projects/deepadaptx/deepadaptx/data/sample_drugs_df.feather",
    },
    "data_cleaning": {
        "name": "default",
        "isotonic_result_column": "isotonic_pred",
        "use_cache_isotonic": False,
        "isotonic_save_path": None,
        "remove_study_ids": ["golub_2020"],
        "max_isotonic_error": 0.2,
        "use_cache_clean": False,
        "clean_save_path": None
    },
    "preprocessing": {
        "name": "default_preprocessing",
        "min_val": -6,
        "max_val": 4,
        "n_drugs": 2,
        "n_dose_bins": 100,
        "use_cache": False,
        "cache_path": None
    },
    "drug_embeddings": {
        "name": "morgan",
        "drug_embedding_dim": 1024,
        "params": {
            "radius": 2,
            "fpSize": 1024
        },
        "use_cache": False,
        "cache_path": None
    },
    "model": {
        "name": "model1",
        "n_drugs": 2,
        "n_dose_bins": 100,
        "drug_emb_dim": 1024,
        "study_vocab_size": 40,
        "sample_vocab_size": 4000,
        "drug_vocab_size": 6000,
        "emb_dim": 128,
        "n_layers": 5,
        "units": 256,
        "rank": 30,
        "activation": "relu",
        "semi_window_size": 5,
        "gamma": 0.2
    },
    "training": {
        "name": "default",
        "test_size": 0.01,
        "super_test_n_drugs": 20,
        "batch_size_train": 512,
        "batch_size_test": 2048,
        "buffer_size": 1024,
        "epochs": 10,
        "steps_per_epoch": None,
        "optimizer": "adam",
        "optimizer_params": {
            "learning_rate": 0.001
        },
        "use_cache": False,
        "cache_path": None,
        "save_path": None,
        "training_history": None
    },
    "drug_library": {
        "name": "fake_drug_lib",
        "raw_path": None,
        "read_params": {},
        "column_name": "Drug Name",
        "use_cache": True,
        "cache_path": "data/drug_library/fake_drug_lib.csv"
    },
    "cold_start": {
        "name": "kmedoids",
        "budget": 3,
        "use_cache": True,
        "drug_preselection": {
            "drug_emb": "auc",
            "samples": "ex-vivo",
            "n_unique_selected": 2,
            "batch_size": 2048,
            "use_cache": False,
            "cache_path": None,
            "method": "kmedoids",
            "method_params": {
                "algorithm": "fasterpam",
                "metric": "euclidean_distance"
            }
        },
        "combo_drug_selection": {
            "drug_emb": "auc",
            "samples": "ex-vivo",
            "kind": "pair",
            "batch_size": 2048,
            "use_cache": False,
            "cache_path": None,
            "method": "kmedoids",
            "method_params": {
                "algorithm": "fasterpam",
                "metric": "euclidean_distance"
            }
        },
        "dose_selection": {
            "samples": "ex-vivo",
            "importance": True,
            "diff_order": 2,
            "quantile": 0.9,
            "batch_size": 2048,
            "use_cache": False,
            "cache_path": None
        },
        "save_path": "data/cold_start/test.csv"
    },
    "pipeline_steps": [
        "load_dataset",
        "clean_data",
        "preprocessing",
        "drug_embeddings",
        "create_model",
        "extract_drug_info",
        "train_model",
        "cold_start_embeddings",
        "drug_auc",
        "importance_scores",
        "kmedoids_selection"
    ]
}

def test_pipe():
    cfg = OmegaConf.create(config)
    run(cfg)
    