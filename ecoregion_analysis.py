#!/usr/bin/env python3
"""
Streamflow Drought Events: Ecoregion Analysis
Analysis of temporal trends across different ecoregions for streamflow drought events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data(file_path):
    """Load parquet file and prepare temporal features."""
    print("Loading data...")
    df = pd.read_parquet(file_path)
    
    # Convert Onset_Time to datetime and extract temporal features
    df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
    df['Month'] = df['Onset_Time'].dt.month
    df['Month_Name'] = df['Onset_Time'].dt.month_name()
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    return df

def basic_ecoregion_stats(df):
    """Generate basic statistics for ecoregions."""
    print('=== ECOREGION BASIC STATISTICS ===\n')
    
    # Basic info about the AGGECOREGION column
    print('1. ECOREGION COLUMN INFO:')
    print(f'   Total records: {len(df)}')
    print(f'   Non-null values: {df["AGGECOREGION"].notna().sum()}')
    print(f'   Null values: {df["AGGECOREGION"].isna().sum()}')
    print(f'   Unique ecoregions: {df["AGGECOREGION"].nunique()}')
    print()
    
    # Value counts for each ecoregion
    print('2. ECOREGION DISTRIBUTION:')
    ecoregion_counts = df['AGGECOREGION'].value_counts()
    print(ecoregion_counts)
    print()
    
    # Percentage distribution
    print('3. ECOREGION PERCENTAGE DISTRIBUTION:')
    ecoregion_pct = df['AGGECOREGION'].value_counts(normalize=True) * 100
    for region, pct in ecoregion_pct.items():
        print(f'   {region}: {pct:.2f}%')
    print()
    
    # Summary statistics
    print('4. SUMMARY STATISTICS:')
    print(f'   Most common ecoregion: {ecoregion_counts.index[0]} ({ecoregion_counts.iloc[0]} events)')
    print(f'   Least common ecoregion: {ecoregion_counts.index[-1]} ({ecoregion_counts.iloc[-1]} events)')
    print(f'   Mean events per ecoregion: {ecoregion_counts.mean():.1f}')
    print(f'   Median events per ecoregion: {ecoregion_counts.median():.1f}')
    print(f'   Standard deviation: {ecoregion_counts.std():.1f}')
    print()

def monthly_seasonal_analysis(df):
    """Analyze monthly and seasonal trends."""
    print('=== MONTHLY AND SEASONAL TRENDS BY ECOREGION ===\n')
    
    # Variables to analyze
    variables = ['Mean_SFD_Flow_Percentile', 'Duration_Days', 'Onset_Rate_Days']
    
    # Monthly analysis by ecoregion
    for var in variables:
        print(f'--- {var} by Month and Ecoregion ---')
        monthly_stats = df.groupby(['AGGECOREGION', 'Month_Name'])[var].agg(['mean', 'std', 'count']).round(2)
        print(monthly_stats.head(20))  # Show first 20 rows to avoid overwhelming output
        print('...')
        print()
    
    # Seasonal analysis
    print('--- SEASONAL SUMMARY ---')
    for var in variables:
        print(f'\n{var} by Season and Ecoregion:')
        seasonal_stats = df.groupby(['AGGECOREGION', 'Season'])[var].agg(['mean', 'std', 'count']).round(2)
        print(seasonal_stats)

def create_visualizations(df, save_plots=True):
    """Create comprehensive visualizations."""
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette('tab10')
    
    # Variables to analyze
    variables = ['Mean_SFD_Flow_Percentile', 'Duration_Days', 'Onset_Rate_Days']
    var_labels = ['Mean SFD Flow Percentile', 'Duration (Days)', 'Onset Rate (Days)']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Streamflow Drought Events: Monthly and Seasonal Trends by Ecoregion', 
                 fontsize=16, fontweight='bold')
    
    # Month and season order for proper plotting
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    
    for i, (var, label) in enumerate(zip(variables, var_labels)):
        # Monthly trends (left column)
        ax1 = axes[i, 0]
        monthly_data = df.groupby(['AGGECOREGION', 'Month_Name'])[var].mean().reset_index()
        monthly_pivot = monthly_data.pivot(index='Month_Name', columns='AGGECOREGION', values=var)
        monthly_pivot = monthly_pivot.reindex(month_order)
        
        for ecoregion in monthly_pivot.columns:
            ax1.plot(range(12), monthly_pivot[ecoregion], marker='o', linewidth=2, label=ecoregion)
        
        ax1.set_title(f'{label} - Monthly Trends', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel(label)
        ax1.set_xticks(range(12))
        ax1.set_xticklabels([m[:3] for m in month_order], rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Seasonal trends (right column)
        ax2 = axes[i, 1]
        seasonal_data = df.groupby(['AGGECOREGION', 'Season'])[var].mean().reset_index()
        seasonal_pivot = seasonal_data.pivot(index='Season', columns='AGGECOREGION', values=var)
        seasonal_pivot = seasonal_pivot.reindex(season_order)
        
        seasonal_pivot.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title(f'{label} - Seasonal Trends', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Season')
        ax2.set_ylabel(label)
        ax2.set_xticklabels(season_order, rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('ecoregion_temporal_trends.png', dpi=300, bbox_inches='tight')
        print('Plot saved as: ecoregion_temporal_trends.png')
    
    plt.show()

def main():
    """Main analysis function."""
    # File path
    file_path = 'data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet'
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(file_path)
        
        # Basic statistics
        basic_ecoregion_stats(df)
        
        # Monthly and seasonal analysis
        monthly_seasonal_analysis(df)
        
        # Create visualizations
        create_visualizations(df)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()