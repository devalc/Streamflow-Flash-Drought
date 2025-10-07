"""
STREAMFLOW FLASH DROUGHT (SFD) CLASSIFICATION CRITERIA

This script identifies Streamflow Flash Droughts (SFDs) using a two-phase, 
three-threshold method adapted from flash drought literature. Specifically the code below follows the method outlined by  Singh and Mishra (2024).
Singh and Mishra (2024): https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2024WR037036
All analysis is performed on daily-resampled streamflow data.

SFD Identification Phases:
--------------------------

1. ONSET PHASE (The Rapid Drop)
    A potential SFD begins when the flow drops rapidly from a normal/wet state 
    to a drought state.
    
    * **Start Condition:** Streamflow must be **ABOVE P_ONSET_START (P40)**.
    * **Onset Trigger:** Streamflow must then fall **BELOW P_ONSET_END (P20)**.
    * **Onset Rate Constraint:** The time taken for this drop must be less 
        than or equal to **MAX_ONSET_DAYS** (15 days).

2. PERSISTENCE & SEVERITY PHASE (The Drought State)
    Once the flow drops below P20, the event is considered an SFD, and its 
    duration and severity are checked.
    
    * **Duration:** The drought persists as long as the streamflow remains 
        **BELOW P_ONSET_START (P40)**.
    * **Minimum Duration Constraint:** The total duration (from P20 drop to P40 recovery)
        must be greater than or equal to **MIN_DURATION_DAYS** (15 days).
    * **Severity Check:** The final event is only classified as a confirmed SFD 
        if the **Mean_SFD_Flow_Percentile** (the average flow percentile during the 
        entire P20-to-P40 recovery period) is **BELOW P_SEVERITY_CHECK (P25)**.

The SFD event is terminated the day before streamflow recovers back above the P40 threshold.
"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import percentileofscore
import warnings

# This warning occurs when calculating quantiles (P40, P20) for days with no historical data.
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
# Also keep the existing filter for np.nanmean on empty slice
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


# =============================================================================
# CONSTANTS & THRESHOLDS
# =============================================================================

FLOW_VAR = 'Streamflow' 
TIME_DIM = 'DateTime' 

P_ONSET_START = 0.40      # P40: Flow must start ABOVE this percentile (40%)
P_ONSET_END = 0.20        # P20: Flow must drop BELOW this percentile (20%) to signal onset
P_SEVERITY_CHECK = 0.25   # P25: Mean flow percentile during the event must be BELOW this (25%)

# --- DAILY TIME SERIES PARAMETERS ---
MAX_ONSET_DAYS = 15.0  # Max time allowed for flow to drop from P40 to P20 or below
MIN_DURATION_DAYS = 15.0 # Minimum duration (days) the drought must persist below P40
# Time factor used for converting timedelta to time unit (seconds per day)
TIME_FACTOR = 86400.0 

# =============================================================================
# PLOTTING FUNCTION 
# =============================================================================

def create_and_save_single_plot(
    station_id: str, 
    df_plot: pd.DataFrame, 
    sfd_events_df: pd.DataFrame,
    output_dir: Path, 
    flow_var: str
):
    """Generates and saves a plot for a single station, including SFD events."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot Flow and Thresholds
        ax.plot(df_plot.index, df_plot['flow'], label=flow_var, color='#1f77b4', linewidth=0.8)
        ax.plot(df_plot['P40'], color='#ff7f0e', linestyle='--', linewidth=1.0, label=f'P{P_ONSET_START*100:.0f} Threshold')
        ax.plot(df_plot['P20'], color='#d62728', linestyle='--', linewidth=1.0, label=f'P{P_ONSET_END*100:.0f} Threshold')
        
        # Plot SFD Events
        if not sfd_events_df.empty:
            sfd_events_df['Onset_Time'] = pd.to_datetime(sfd_events_df['Onset_Time'])
            sfd_events_df['Termination_Time'] = pd.to_datetime(sfd_events_df['Termination_Time'])
            
            for i, row in sfd_events_df.iterrows():
                # Onset Time (Cyan/Green, solid line)
                ax.axvline(
                    row['Onset_Time'], 
                    color='#00bfa5', 
                    linestyle='-', 
                    linewidth=1.5, 
                    alpha=0.6, 
                    label='SFD Onset' if i == 0 else "" 
                )
                # Termination Time (Black, dashed line)
                ax.axvline(
                    row['Termination_Time'], 
                    color='k', 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.5, 
                    label='SFD Termination' if i == 0 else "" 
                )

        # Finalize Plot 
        ax.set_title(f'Station {station_id} Daily Resampled Time Series and Thresholds', fontsize=14)
        ax.set_ylabel(f'{flow_var} (Daily Avg)')
        ax.set_xlabel(f'{TIME_DIM}')
        
        # Ensure only unique labels are in the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        plot_path = output_dir / f'{station_id}_daily_resampled_plot.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig) 
        
        return f"Plot saved for {station_id} to {plot_path.name}"
    
    except Exception as e:
        return f"Error plotting station {station_id}: {e}"

# =============================================================================
# CORE SFD ANALYSIS FUNCTIONS
# =============================================================================

def calculate_percentile_rank(flow_value: float, climatology_data: np.ndarray) -> float:
    """Calculates the percentile rank of a single flow value."""
    climatology_data = climatology_data[~np.isnan(climatology_data)]
    if climatology_data.size == 0:
        return np.nan 
    return percentileofscore(climatology_data, flow_value, kind='weak')

def identify_station_sfd_exact(nc_file_path: Path, output_dir: Path, is_test_mode: bool = False) -> pd.DataFrame:
    """
    Worker function: Reads high-frequency data, resamples to daily, cleans data, 
    and runs Streamflow Flash Drought (SFD) analysis.
    """
    
    station_id = nc_file_path.stem
    DOY_COORD_NAME = 'doy_str' 

    try:
        with xr.open_dataset(nc_file_path) as ds:
            if FLOW_VAR not in ds or ds[TIME_DIM].size == 0:
                 print(f"Error: Variable '{FLOW_VAR}' not found or time dim empty in {station_id}")
                 return pd.DataFrame()

            da_flow_original = ds[FLOW_VAR]
            
            # RESAMPLE TO DAILY MEAN
            da_flow_daily = da_flow_original.resample(DateTime='D').mean()
            
            # DATA CLEANING: SET NEGATIVE VALUES TO ZERO
            # Use .clip(min=0) to ensure non-physical negative flows are zeroed
            da_flow_daily = da_flow_daily.clip(min=0)
            
            flow_series = da_flow_daily.to_series().dropna()
            
            if flow_series.empty:
                print(f"[{station_id}] Flow series is empty after daily resampling and cleaning.")
                return pd.DataFrame()

            # ---  Calculate Climatology and Thresholds ---
            
            doy_values = da_flow_daily[TIME_DIM].dt.strftime('%j').rename(DOY_COORD_NAME) 
            climatology_groups = da_flow_daily.groupby(doy_values)
            
            # This is the block that triggers the All-NaN slice warning when a DOY group is empty
            percentile_thresholds = climatology_groups.quantile(
                [P_ONSET_START, P_ONSET_END], 
                dim=TIME_DIM, 
                keep_attrs=True
            ).rename({doy_values.name: DOY_COORD_NAME})

            P40_val_series = percentile_thresholds.sel(quantile=P_ONSET_START).to_pandas()
            P20_val_series = percentile_thresholds.sel(quantile=P_ONSET_END).to_pandas()
            
            doy_series_for_lookup = doy_values.to_series()
            P40_threshold = doy_series_for_lookup.map(P40_val_series)
            P20_threshold = doy_series_for_lookup.map(P20_val_series)
            
            # Interpolate and fill gaps
            P40_threshold = P40_threshold.ffill().bfill()
            P20_threshold = P20_threshold.ffill().bfill()

            # --- Prepare Comparison DataFrame ---
            aligned_P40 = P40_threshold.reindex(flow_series.index)
            aligned_P20 = P20_threshold.reindex(flow_series.index)

            df_comp_raw = pd.DataFrame({
                'flow': flow_series,
                'P40': aligned_P40,
                'P20': aligned_P20
            })
            
            df_comp = df_comp_raw.dropna() 
            df_comp_len = len(df_comp)

            if df_comp_len == 0:
                return pd.DataFrame()
            
            df_comp['is_above_P40'] = df_comp['flow'] > df_comp['P40']
            df_comp['is_below_P20'] = df_comp['flow'] < df_comp['P20']
            
            doy_series = doy_values.to_series() 
            df_comp['DOY'] = doy_series.reindex(df_comp.index) 

            # --- Event Identification Loop (SFD Detection) ---
            sfd_events = []
            i = 0
            n = len(df_comp)
            
            while i < n:
                if df_comp.iloc[i]['is_above_P40']:
                    t_start_i = i 
                    t_start = df_comp.index[i]
                    j = i + 1
                    event_found = False 
                    
                    # Onset time limit check (MAX_ONSET_DAYS)
                    while j < n and (df_comp.index[j] - t_start).total_seconds() / TIME_FACTOR <= MAX_ONSET_DAYS:
                        if df_comp.iloc[j]['is_below_P20']:
                            t_onset = df_comp.index[j]
                            
                            k = j
                            while k < n and df_comp.iloc[k]['is_above_P40'] == False:
                                k += 1
                            
                            t_termination = df_comp.index[k-1]
                            
                            duration_days = (t_termination - t_onset).total_seconds() / TIME_FACTOR

                            if duration_days >= MIN_DURATION_DAYS:
                                event_slice = df_comp.loc[t_onset:t_termination]
                                
                                percentile_ranks = []
                                for _, row in event_slice.iterrows():
                                    current_flow = row['flow']
                                    current_doy_str = row['DOY']
                                    
                                    if current_doy_str in climatology_groups.groups:
                                        climatology_dist = climatology_groups[current_doy_str].values
                                        p_rank = calculate_percentile_rank(current_flow, climatology_dist)
                                        percentile_ranks.append(p_rank)
                                    else:
                                        percentile_ranks.append(np.nan) 
                                
                                mean_percentile = np.nanmean(percentile_ranks)
                                
                                # Final Severity Check
                                if mean_percentile < P_SEVERITY_CHECK * 100: 
                                    onset_rate_days = (t_onset - t_start).total_seconds() / TIME_FACTOR
                                    
                                    sfd_events.append({
                                        'Station_ID': station_id,
                                        'Onset_Time': t_onset,
                                        'Termination_Time': t_termination,
                                        'Duration_Days': duration_days,
                                        'Onset_Rate_Days': onset_rate_days,
                                        'Mean_SFD_Flow_Percentile': mean_percentile
                                    })
                                    i = k 
                                    event_found = True
                                    break 
                                else:
                                    j += 1
                            else:
                                j += 1
                        else:
                            j += 1
                    
                    if not event_found:
                        i = t_start_i + 1
                else:
                    i += 1
            
            sfd_events_df = pd.DataFrame(sfd_events)
            
            # --- PLOTTING STEP ---
            if is_test_mode and df_comp_len > 0:
                plot_df = df_comp[['flow', 'P40', 'P20']].copy()
                plot_report = create_and_save_single_plot(
                    station_id, 
                    plot_df, 
                    sfd_events_df, 
                    output_dir, 
                    FLOW_VAR
                )
                print(f"[PLOT] {plot_report}") 


            return sfd_events_df

    except Exception as e:
        print(f"Error processing station {station_id}: {e}") 
        return pd.DataFrame()


# =============================================================================
# Run in PARALLEL 
# =============================================================================

def parallel_sfd_analysis(nc_files: list[Path], output_dir: Path, max_workers: int = 10, is_test_mode: bool = False) -> pd.DataFrame:
    """
    Coordinates the parallel analysis of all NetCDF files.
    """
    mode_str = "TEST (Parallel Analysis & Plotting)" if is_test_mode else "FULL PRODUCTION (Parallel Analysis)"
    print(f"Starting {mode_str} on {len(nc_files)} files...")
    all_results = []
    processed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(identify_station_sfd_exact, file, output_dir, is_test_mode): file 
            for file in nc_files
        }
        
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            processed_count += 1
            
            try:
                result_df = future.result() 
                if not result_df.empty:
                    all_results.append(result_df)
                
                # Check for completion every 10 files
                if processed_count % 10 == 0:
                    print(f"Completed analysis for {processed_count} / {len(nc_files)} stations.")
                    
            except Exception as exc:
                print(f"Error retrieving result for station {file.stem}: {exc}")
                
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


# =============================================================================
# EXECUTION BLOCK
# =============================================================================

if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # --- CONFIGURATION ---
    # -------------------------------------------------------------------------
    
    # Path to input data folder 
   
    FOLDER_WITH_NETCDF_FILES = Path('./data/v2/Data/CAMELSH/timeseries/')
    
    # Define the output directory and ensure it exists
    OUTPUT_DIRECTORY = Path('./data/sfd_classified_daily_resampled').resolve() 
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    # Core count 
    NUM_WORKERS = 10 
    
    # SET TEST (Number of files you want to run)/FULL RUN MODE (None):
    FILE_LIMIT_FOR_TESTING = None 
    # -------------------------------------------------------------------------
    
    print("--- Starting Streamflow Flash Drought Classification (DAILY RESAMPLED & CLEANED) ---")
    print(f"Output directory (ABS PATH): {OUTPUT_DIRECTORY}")
    
    if not FOLDER_WITH_NETCDF_FILES.is_dir():
        print(f"ERROR: Input folder not found: {FOLDER_WITH_NETCDF_FILES}")
        exit()

    all_nc_files = [Path(p) for p in glob.glob(f"{FOLDER_WITH_NETCDF_FILES}/*.nc")]
    
    if not all_nc_files:
        print(f"ERROR: No NetCDF files found in {FOLDER_WITH_NETCDF_FILES}")
        exit()

    if FILE_LIMIT_FOR_TESTING is not None and FILE_LIMIT_FOR_TESTING > 0:
        print(f"!!! WARNING: Running in TEST MODE on {FILE_LIMIT_FOR_TESTING} files !!!")
        print("Data is RESAMPLED to Daily Mean, and NEGATIVE VALUES ARE SET TO 0.")
        print(f"Duration threshold: {MIN_DURATION_DAYS} days | Onset window: {MAX_ONSET_DAYS} days")
        nc_files_to_process = all_nc_files[:FILE_LIMIT_FOR_TESTING]
        output_file_name = f'sfd_events_daily_resampled_test_{FILE_LIMIT_FOR_TESTING}.csv'

        final_sfd_events = parallel_sfd_analysis(
            nc_files_to_process, 
            OUTPUT_DIRECTORY, 
            max_workers=NUM_WORKERS, 
            is_test_mode=True
        )

    else:
        print("Running in FULL PRODUCTION MODE on all files.")
        print("Data is RESAMPLED to Daily Mean, and NEGATIVE VALUES ARE SET TO 0.")
        nc_files_to_process = all_nc_files
        output_file_name = 'conus_daily_resampled_streamflow_flash_droughts.csv'
        
        final_sfd_events = parallel_sfd_analysis(
            nc_files_to_process, 
            OUTPUT_DIRECTORY, 
            max_workers=NUM_WORKERS, 
            is_test_mode=False
        )


    if not final_sfd_events.empty:
        print("\n--- FINAL ANALYSIS REPORT ---")
        print(f"Total SFD events identified: {len(final_sfd_events)}")
        
        output_path = OUTPUT_DIRECTORY / output_file_name
        final_sfd_events.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    else:
        print("\n--- FINAL ANALYSIS REPORT ---")
        print("No SFD events were successfully identified or processed.")