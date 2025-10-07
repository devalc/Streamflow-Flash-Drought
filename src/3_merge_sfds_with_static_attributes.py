import pandas as pd
from pathlib import Path
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- INPUT FILES ---
# 1. Output from the SFD analysis script
SFD_EVENTS_CSV_PATH = Path('./data/sfd_classified_daily_resampled/conus_daily_resampled_streamflow_flash_droughts.csv')

# 2. Output from the static attribute consolidation script
STATIC_ATTRIBUTES_PARQUET_PATH = Path('./data/CAMELSH_STATIC_ATTRIBUTES.parquet')

# --- OUTPUT FILE ---
FINAL_MERGED_PARQUET_PATH = Path('./data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

# --- JOIN COLUMNS ---
SFD_JOIN_COLUMN = 'Station_ID' 
ATTR_JOIN_INDEX_NAME = 'STAID' # The index name of the Parquet file

# ==============================================================================
# MERGE PROCESS
# ==============================================================================
def merge_events_with_attributes(
    events_path: Path, 
    attributes_path: Path, 
    output_path: Path
) -> pd.DataFrame:
    
    print("--- Starting Data Merge ---")
    
    # Load SFD Events Data (Time-Series Data)
    if not events_path.exists():
        print(f"CRITICAL ERROR: SFD Events file not found at {events_path}")
        return pd.DataFrame()
        
    df_events = pd.read_csv(events_path)
    print(f"Loaded SFD Events: {len(df_events):,} rows.")
    
    # Ensure the SFD events column is a string
    df_events[SFD_JOIN_COLUMN] = df_events[SFD_JOIN_COLUMN].astype(str)
    
    # Load Static Attributes Data
    if not attributes_path.exists():
        print(f"CRITICAL ERROR: Static Attributes file not found at {attributes_path}")
        return pd.DataFrame()
        
    df_attributes = pd.read_parquet(attributes_path)
    print(f"Loaded Static Attributes: {len(df_attributes):,} rows and {len(df_attributes.columns):,} features.")
    
  
    try:
        df_attributes.index = df_attributes.index.astype(str)
        # Ensure the index name matches the intended join name for clarity, though not strictly required
        df_attributes.index.name = ATTR_JOIN_INDEX_NAME
    except Exception as e:
        print(f"Error converting attribute index to string: {e}")
        return pd.DataFrame()
    
    # --- Perform the Merge (Left Join) ---
    df_merged = df_events.merge(
        df_attributes, 
        left_on=SFD_JOIN_COLUMN, 
        right_index=True, 
        how='left'
    )
    
    # 3. Final Cleanup and Output
    # Create a list of columns that should *not* be all NaN after a successful merge
    cols_to_check = [col for col in df_attributes.columns if col not in df_events.columns]
    
    initial_rows = len(df_merged)
    # Drop rows where all the newly merged attribute columns are missing
    df_merged = df_merged.dropna(subset=cols_to_check, how='all')
    
    if initial_rows - len(df_merged) > 0:
        print(f"WARNING: Dropped {initial_rows - len(df_merged)} SFD events for stations not found in the static attributes file.")

    # Save the final merged DataFrame as a Parquet file
    os.makedirs(output_path.parent, exist_ok=True)
    df_merged.to_parquet(output_path, index=False)
    
    print("\n--- Merge Complete ---")
    print(f"Final Merged Data Shape: {df_merged.shape}")
    print(f"Saved merged dataset to: {output_path.resolve()}")
    
    return df_merged

if __name__ == '__main__':
    # 
    os.makedirs('./data', exist_ok=True)
    
    final_data = merge_events_with_attributes(
        SFD_EVENTS_CSV_PATH, 
        STATIC_ATTRIBUTES_PARQUET_PATH, 
        FINAL_MERGED_PARQUET_PATH
    )
    
    if not final_data.empty:
        print("\nHead of Final Dataset (Time-series data + Static Attributes):")
        # Display the first few rows and columns to confirm merge
        display_cols = ['Station_ID', 'Onset_Time', 'Duration_Days', 'DRAIN_SQKM', 'GEOL_REEDBUSH_DOM']
        # Safely select columns, handling cases where the attributes might not have loaded
        cols_to_display = [col for col in display_cols if col in final_data.columns]
        print(final_data[cols_to_display].head())