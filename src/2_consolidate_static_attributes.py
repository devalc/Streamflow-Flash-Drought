import pandas as pd
import os
import glob
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# FINAL TARGETED ATTRIBUTE CONSOLIDATION
# ==============================================================================
"""
Enforces the user's requirement that 'STAID' is the only index column, and 
fixes the previous failure by explicitly including 'STAID' in the `usecols` 
list for files with specific column requests.
"""
# ==============================================================================

# --- Configuration ---
ATTRIBUTES_DIR = './data/v2/attributes/' 
OUTPUT_STATIC_ATTRIBUTES = './data/CAMELSH_STATIC_ATTRIBUTES.parquet'
INDEX_COL = 'STAID'

# FINAL CONSOLIDATED MAP
FINAL_FILE_MAP = {
    'attributes_nldas2_climate.csv': 'ALL',
    'attributes_hydroATLAS.csv': 'ALL',
    'attributes_gageii_Bas_Classif.csv': 'AGGECOREGION,HYDRO_DISTURB_INDX',
    'attributes_gageii_BasinID.csv': 'DRAIN_SQKM',
   # 'attributes_gageii_Climate_Ppt_Annual.csv': 'ALL',
    #'attributes_gageii_Climate_Tmp_Annual.csv': 'ALL',
    'attributes_gageii_Geology.csv': 'GEOL_REEDBUSH_DOM',
    'attributes_gageii_HydroMod_Dams.csv': 'RAW_DIS_NEAREST_DAM,RAW_AVG_DIS_ALLDAMS,RAW_DIS_NEAREST_MAJ_DAM,RAW_AVG_DIS_ALL_MAJ_DAMS',
    'attributes_gageii_HydroMod_Other.csv': 'FRESHW_WITHDRAWAL,PCT_IRRIG_AG',
    'attributes_gageii_Landscape_Pat.csv': 'ALL',
    'attributes_gageii_Soils.csv': 'ALL',
    'attributes_gageii_Topo.csv': 'ALL'
}


def load_and_consolidate_attributes(attr_directory: str, file_map: Dict[str, str], index_col: str) -> pd.DataFrame:
    """
    Loads files based on map, selects only specified columns, and merges.
    """
    
    master_df: pd.DataFrame = pd.DataFrame()
    files_to_process: List[Tuple[str, str, List[str]]] = []

    if not os.path.isdir(attr_directory):
        print(f"CRITICAL: Attribute directory not found at '{attr_directory}'. Check your path.")
        return master_df
    
    # Compile the list of all files, delimiters, and variables
    for file_name_in_map, variables in file_map.items():
        separator = '\t' if 'hydroATLAS' in file_name_in_map else ','
        variables_list = [v.strip().upper() for v in variables.split(',')] if variables.upper() != 'ALL' else ['ALL']
        
        file_path = os.path.join(attr_directory, file_name_in_map)
        if os.path.exists(file_path):
            files_to_process.append((file_name_in_map, separator, variables_list))

    print(f"Starting consolidation of {len(files_to_process)} attribute files...")
    
    # Process and merge each file
    for file_name, separator, variables_list in files_to_process:
        file_path = os.path.join(attr_directory, file_name)
        
        try:
            # Determine usecols:
            usecols = None
            if 'ALL' not in variables_list:
                # Start with the user's requested variables
                usecols = [v for v in variables_list if v] 
                # CRITICAL FIX: Explicitly include 'STAID' to ensure it's loaded 
                usecols.append(index_col.upper()) 
                usecols = list(set(usecols))
            
            # Load data
            df = pd.read_csv(file_path, sep=separator, usecols=usecols)
            
            # Standardize column names (uppercase, clean symbols)
            df.columns = df.columns.str.strip().str.replace(r'[\s\.\-\(\)]', '_', regex=True).str.upper()
            
            # Check for STAID column
            if index_col.upper() not in df.columns: 
                 print(f"ERROR: Column '{index_col.upper()}' not found in {file_name} after loading. Skipping.")
                 continue
            
            # Set the index
            df = df.set_index(index_col.upper())
            df.index.name = index_col # Ensure the index name is standardized
            
            # Filter columns to only those requested (after index is set)
            if 'ALL' not in variables_list:
                cols_to_keep = [col for col in df.columns if col in variables_list]
                df = df[cols_to_keep]

            # Merge into master DataFrame
            if master_df.empty:
                master_df = df
            else:
                master_df = master_df.join(df, how='outer', rsuffix='_DUP')
                cols_to_drop = [col for col in master_df.columns if col.endswith('_DUP')]
                master_df = master_df.drop(columns=cols_to_drop)

            print(f"Processed {file_name}. Total unique attributes now: {len(master_df.columns)}")

        except Exception as e:
            print(f"Error reading or processing file {file_name}: {e}. Check file format/variables.")

    # Final cleanup (HUC02)
    master_df = master_df[~master_df.index.isna()]
    if 'HUC02' in master_df.columns:
        master_df['HUC02'] = master_df['HUC02'].fillna(0).astype(int).astype(str).str.zfill(2)

    return master_df

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_STATIC_ATTRIBUTES), exist_ok=True)
    
    try:
        final_static_features = load_and_consolidate_attributes(ATTRIBUTES_DIR, FINAL_FILE_MAP, INDEX_COL)
        
        final_static_features.to_parquet(OUTPUT_STATIC_ATTRIBUTES)
        
        print(f"\nData Consolidation Complete: {len(final_static_features.columns)} final static attributes consolidated and saved to {OUTPUT_STATIC_ATTRIBUTES}")
        print(f"Head of Final Static Attributes (X_stat):")
        print(final_static_features.head())

    except Exception as e:
        print(f"\nData consolidation failed: {e}")