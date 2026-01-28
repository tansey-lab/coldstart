import pandas as pd
import os
from tqdm import tqdm

from pan_preclinical_data_models.models import Study, SourceFile, SourceFileType
from pan_preclinical_etl.drugs import drug_object_from_name


def extract_drug_info(
    raw_path: str,
    read_params: dict = {},
    column_name: str = "Drug Name",
    cache_path: str = "drug_info.csv",
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Extract structured drug metadata from unique drug names in a dataframe.

    If use_cache is True and the file exists, loads data from cache instead.

    Parameters
    ----------
    source_file : str
        Source file metadata object.
    cache_path : str
        Path to save/load the cached drug info CSV.
    use_cache : bool
        Whether to use cached version if available.

    Returns
    -------
    pd.DataFrame with drug metadata.
    """
    if use_cache and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    if raw_path.split(".")[-1] == "xlsx":
        dff = pd.read_excel(raw_path, **read_params)
    elif raw_path.split(".")[-1] == "csv":
        dff = pd.read_csv(raw_path, **read_params)
    else:
        raise ValueError("xlsx or csv")
    
    source_file = SourceFile(md5sum="", study_id="", type="MICROARRAY", path="", provided_by_authors=False)
    study = Study(id="", year=2016, paper_title="")

    records = []
    for name in tqdm(dff[column_name].dropna().unique(), desc="Extracting drug info"):
        obj = drug_object_from_name(name, study, source_file)
        if obj is None:
            continue

        records.append({
            "id": obj.id,
            "study_id": obj.study_id,
            "name": obj.name,
            "source_file_id": obj.source_file_id,
            "pubchem_name": obj.pubchem_name,
            "pubchem_substance_id": obj.pubchem_substance_id,
            "pubchem_compound_id": obj.pubchem_compound_id,
            "pubchem_parent_compound_id": obj.pubchem_parent_compound_id,
            "smiles": obj.smiles,
            "fda_approval": obj.fda_approval.value if obj.fda_approval is not None else None,
        })

    df = pd.DataFrame(records)

    if use_cache:
        df.to_csv(cache_path, index=False)
    
    return df