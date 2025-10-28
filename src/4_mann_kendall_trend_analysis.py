#!/usr/bin/env python3
"""
Mann-Kendall Trend Analysis for Streamflow Flash Drought Data
Analyzes trends in onset time, duration, and flow percentile across years and seasons
Uses 90% confidence level (p < 0.10) for significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def mann_kendall_test(data):
    """
    Perform Mann-Kendall test for trend detection
    Returns: tau (Kendall's tau), p-value, trend direction
    """
    n = len(data)
    if n < 3:
        return np.nan, np.nan, 'insufficient_data'
    
    # Calculate S statistic
    S = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if data[j] > data[i]:
                S += 1
            elif data[j] < data[i]:
                S -= 1
    
    # Calculate variance
    var_S = n * (n - 1) * (2 * n + 5) / 18
    
    # Calculate Z statistic
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0
    
    # Calculate p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    
    # Calculate Kendall's tau
    tau = S / (n * (n - 1) / 2)
    
    # Determine trend direction (90% confidence level)
    if p_value < 0.10:
        if tau > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no_trend'
    
    return tau, p_value, trend

def get_season(date):
    """Get season from date"""
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def analyze_trends():
    """Main analysis function"""
    print("Loading data...")
    df = pd.read_csv('data/sfd_classified_daily_resampled/conus_daily_resampled_streamflow_flash_droughts.csv')
    
    # Convert date columns
    df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
    df['Termination_Time'] = pd.to_datetime(df['Termination_Time'])
    
    # Extract year and season
    df['Year'] = df['Onset_Time'].dt.year
    df['Season'] = df['Onset_Time'].apply(get_season)
    
    print(f"Data loaded: {len(df)} records for {df['Station_ID'].nunique()} unique stations")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    
    # Initialize results storage
    results = []
    
    # Get unique stations
    stations = df['Station_ID'].unique()
    
    print("Analyzing trends for each station...")
    print("Filtering stations with at least 10 years of data...")
    
    stations_processed = 0
    stations_filtered = 0
    
    for i, station in enumerate(stations):
        if i % 100 == 0:
            print(f"Processing station {i+1}/{len(stations)} (Processed: {stations_processed}, Filtered: {stations_filtered})")
        
        station_data = df[df['Station_ID'] == station].copy()
        
        if len(station_data) < 3:
            continue
        
        # Annual aggregation for each metric
        annual_data = station_data.groupby('Year').agg({
            'Duration_Days': 'mean',
            'Mean_SFD_Flow_Percentile': 'mean',
            'Onset_Time': 'count'  # Count of events per year
        }).reset_index()
        annual_data.rename(columns={'Onset_Time': 'Event_Count'}, inplace=True)
        
        # Filter stations with at least 10 years of data
        years_span = station_data['Year'].max() - station_data['Year'].min() + 1
        if len(annual_data) < 10 or years_span < 10:
            stations_filtered += 1
            continue
        
        stations_processed += 1
        
        # Overall trend analysis (across all years)
        if len(annual_data) >= 10:
            # Duration trend
            tau_dur, p_dur, trend_dur = mann_kendall_test(annual_data['Duration_Days'].values)
            
            # Flow percentile trend
            tau_flow, p_flow, trend_flow = mann_kendall_test(annual_data['Mean_SFD_Flow_Percentile'].values)
            
            # Event count trend
            tau_count, p_count, trend_count = mann_kendall_test(annual_data['Event_Count'].values)
        else:
            tau_dur = p_dur = tau_flow = p_flow = tau_count = p_count = np.nan
            trend_dur = trend_flow = trend_count = 'insufficient_data'
        
        # Seasonal trend analysis
        seasonal_trends = {}
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            season_data = station_data[station_data['Season'] == season]
            
            if len(season_data) >= 3:
                season_annual = season_data.groupby('Year').agg({
                    'Duration_Days': 'mean',
                    'Mean_SFD_Flow_Percentile': 'mean',
                    'Onset_Time': 'count'
                }).reset_index()
                
                if len(season_annual) >= 3:
                    tau_s_dur, p_s_dur, trend_s_dur = mann_kendall_test(season_annual['Duration_Days'].values)
                    tau_s_flow, p_s_flow, trend_s_flow = mann_kendall_test(season_annual['Mean_SFD_Flow_Percentile'].values)
                    tau_s_count, p_s_count, trend_s_count = mann_kendall_test(season_annual['Onset_Time'].values)
                else:
                    tau_s_dur = p_s_dur = tau_s_flow = p_s_flow = tau_s_count = p_s_count = np.nan
                    trend_s_dur = trend_s_flow = trend_s_count = 'insufficient_data'
            else:
                tau_s_dur = p_s_dur = tau_s_flow = p_s_flow = tau_s_count = p_s_count = np.nan
                trend_s_dur = trend_s_flow = trend_s_count = 'insufficient_data'
            
            seasonal_trends[season] = {
                'duration': (tau_s_dur, p_s_dur, trend_s_dur),
                'flow': (tau_s_flow, p_s_flow, trend_s_flow),
                'count': (tau_s_count, p_s_count, trend_s_count)
            }
        
        # Store results
        result = {
            'Station_ID': station,
            'Total_Events': len(station_data),
            'Years_Covered': len(annual_data),
            'Year_Range': f"{station_data['Year'].min()}-{station_data['Year'].max()}",
            
            # Overall trends
            'Duration_Tau': tau_dur,
            'Duration_P_Value': p_dur,
            'Duration_Trend': trend_dur,
            
            'Flow_Percentile_Tau': tau_flow,
            'Flow_Percentile_P_Value': p_flow,
            'Flow_Percentile_Trend': trend_flow,
            
            'Event_Count_Tau': tau_count,
            'Event_Count_P_Value': p_count,
            'Event_Count_Trend': trend_count,
        }
        
        # Add seasonal trends
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            for metric in ['duration', 'flow', 'count']:
                tau, p_val, trend = seasonal_trends[season][metric]
                result[f'{season}_{metric.title()}_Tau'] = tau
                result[f'{season}_{metric.title()}_P_Value'] = p_val
                result[f'{season}_{metric.title()}_Trend'] = trend
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('results/mann_kendall_trend_results.csv', index=False)
    print(f"Results saved to results/mann_kendall_trend_results.csv")
    print(f"Total stations processed: {stations_processed}")
    print(f"Stations filtered out (< 10 years): {stations_filtered}")
    
    return results_df, df

if __name__ == "__main__":
    results_df, original_df = analyze_trends()

def generate_summary_and_plots(results_df, original_df):
    """Generate summary statistics and plots"""
    print("\n" + "="*60)
    print("MANN-KENDALL TREND ANALYSIS SUMMARY")
    print("="*60)
    
    # Overall trend summary
    print("\n1. OVERALL TRENDS ACROSS ALL YEARS:")
    print("-" * 40)
    
    # Duration trends
    duration_trends = results_df['Duration_Trend'].value_counts()
    print(f"\nDuration Trends:")
    for trend, count in duration_trends.items():
        print(f"  {trend.replace('_', ' ').title()}: {count} stations")
    
    # Flow percentile trends
    flow_trends = results_df['Flow_Percentile_Trend'].value_counts()
    print(f"\nFlow Percentile Trends:")
    for trend, count in flow_trends.items():
        print(f"  {trend.replace('_', ' ').title()}: {count} stations")
    
    # Event count trends
    count_trends = results_df['Event_Count_Trend'].value_counts()
    print(f"\nEvent Count Trends:")
    for trend, count in count_trends.items():
        print(f"  {trend.replace('_', ' ').title()}: {count} stations")
    
    # Seasonal trends summary
    print("\n2. SEASONAL TRENDS:")
    print("-" * 40)
    
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    metrics = ['Duration', 'Flow', 'Count']
    
    for season in seasons:
        print(f"\n{season}:")
        for metric in metrics:
            col_name = f'{season}_{metric}_Trend'
            if col_name in results_df.columns:
                seasonal_trends = results_df[col_name].value_counts()
                print(f"  {metric} - ", end="")
                trend_summary = []
                for trend, count in seasonal_trends.items():
                    trend_summary.append(f"{trend.replace('_', ' ').title()}: {count}")
                print(", ".join(trend_summary))
    
    # Create plots
    print("\n3. GENERATING PLOTS...")
    print("-" * 40)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get top stations for each trend type - DURATION (using p-values for selection)
    significant_increase_duration = results_df[
        (results_df['Duration_Trend'] == 'increasing') & 
        (results_df['Duration_P_Value'] < 0.10)
    ].nsmallest(10, 'Duration_P_Value')  # Most significant (lowest p-values)
    
    significant_decrease_duration = results_df[
        (results_df['Duration_Trend'] == 'decreasing') & 
        (results_df['Duration_P_Value'] < 0.10)
    ].nsmallest(10, 'Duration_P_Value')  # Most significant (lowest p-values)
    
    no_trend_duration = results_df[
        results_df['Duration_Trend'] == 'no_trend'
    ].nlargest(10, 'Duration_P_Value')  # Highest p-values (most non-significant)
    
    # Get top stations for each trend type - FLOW PERCENTILE (using p-values for selection)
    significant_increase_flow = results_df[
        (results_df['Flow_Percentile_Trend'] == 'increasing') & 
        (results_df['Flow_Percentile_P_Value'] < 0.10)
    ].nsmallest(10, 'Flow_Percentile_P_Value')  # Most significant (lowest p-values)
    
    significant_decrease_flow = results_df[
        (results_df['Flow_Percentile_Trend'] == 'decreasing') & 
        (results_df['Flow_Percentile_P_Value'] < 0.10)
    ].nsmallest(10, 'Flow_Percentile_P_Value')  # Most significant (lowest p-values)
    
    no_trend_flow = results_df[
        results_df['Flow_Percentile_Trend'] == 'no_trend'
    ].nlargest(10, 'Flow_Percentile_P_Value')  # Highest p-values (most non-significant)
    
    # Get top stations for each trend type - EVENT COUNT (using p-values for selection)
    significant_increase_count = results_df[
        (results_df['Event_Count_Trend'] == 'increasing') & 
        (results_df['Event_Count_P_Value'] < 0.10)
    ].nsmallest(10, 'Event_Count_P_Value')  # Most significant (lowest p-values)
    
    significant_decrease_count = results_df[
        (results_df['Event_Count_Trend'] == 'decreasing') & 
        (results_df['Event_Count_P_Value'] < 0.10)
    ].nsmallest(10, 'Event_Count_P_Value')  # Most significant (lowest p-values)
    
    no_trend_count = results_df[
        results_df['Event_Count_Trend'] == 'no_trend'
    ].nlargest(10, 'Event_Count_P_Value')  # Highest p-values (most non-significant)
    
    # Plot 1: Duration trends for significant increase stations
    if len(significant_increase_duration) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Increasing Duration Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_increase_duration.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year')['Duration_Days'].mean()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7)
            axes[row, col].set_title(f'{station_id}\np={station_row["Duration_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Duration (days)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_increase_duration), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/duration_increasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Duration increasing trends plot saved")
    
    # Plot 2: Duration trends for significant decrease stations
    if len(significant_decrease_duration) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Decreasing Duration Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_decrease_duration.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year')['Duration_Days'].mean()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='red')
            axes[row, col].set_title(f'{station_id}\np={station_row["Duration_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Duration (days)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_decrease_duration), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/duration_decreasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Duration decreasing trends plot saved")
    
    # Plot 3: Duration trends for no trend stations
    if len(no_trend_duration) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('10 Stations with No Significant Duration Trends (Highest P-Values)', fontsize=16)
        
        for i, (_, station_row) in enumerate(no_trend_duration.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year')['Duration_Days'].mean()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='gray')
            axes[row, col].set_title(f'{station_id}\np={station_row["Duration_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Duration (days)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(no_trend_duration), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/duration_no_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Duration no trends plot saved")
    
    # Plot 4: Flow Percentile trends for significant increase stations
    if len(significant_increase_flow) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Increasing Flow Percentile Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_increase_flow.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year')['Mean_SFD_Flow_Percentile'].mean()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='green')
            axes[row, col].set_title(f'{station_id}\np={station_row["Flow_Percentile_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Flow Percentile (%)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_increase_flow), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/flow_percentile_increasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Flow percentile increasing trends plot saved")
    
    # Plot 5: Flow Percentile trends for significant decrease stations
    if len(significant_decrease_flow) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Decreasing Flow Percentile Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_decrease_flow.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year')['Mean_SFD_Flow_Percentile'].mean()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='orange')
            axes[row, col].set_title(f'{station_id}\np={station_row["Flow_Percentile_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Flow Percentile (%)')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_decrease_flow), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/flow_percentile_decreasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Flow percentile decreasing trends plot saved")
    
    # Plot 6: Event Count trends for significant increase stations
    if len(significant_increase_count) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Increasing Event Count Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_increase_count.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year').size()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='purple')
            axes[row, col].set_title(f'{station_id}\np={station_row["Event_Count_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Event Count')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_increase_count), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/event_count_increasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Event count increasing trends plot saved")
    
    # Plot 7: Event Count trends for significant decrease stations
    if len(significant_decrease_count) > 0:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Top 10 Most Significant Decreasing Event Count Trends (90% Confidence, p<0.10)', fontsize=16)
        
        for i, (_, station_row) in enumerate(significant_decrease_count.iterrows()):
            if i >= 10:
                break
            
            row, col = i // 5, i % 5
            station_id = station_row['Station_ID']
            
            # Get station data
            station_data = original_df[original_df['Station_ID'] == station_id]
            annual_data = station_data.groupby('Year').size()
            
            axes[row, col].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='brown')
            axes[row, col].set_title(f'{station_id}\np={station_row["Event_Count_P_Value"]:.4f}', fontsize=10)
            axes[row, col].set_xlabel('Year')
            axes[row, col].set_ylabel('Event Count')
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(significant_decrease_count), 10):
            row, col = i // 5, i % 5
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/event_count_decreasing_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Event count decreasing trends plot saved")
    
    # SEASONAL TREND PLOTS
    print("\n  Creating seasonal trend plots...")
    
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    metrics = ['Duration', 'Flow', 'Count']
    
    for season in seasons:
        for metric in metrics:
            # Get seasonal trend data
            tau_col = f'{season}_{metric}_Tau'
            p_col = f'{season}_{metric}_P_Value'
            trend_col = f'{season}_{metric}_Trend'
            
            if all(col in results_df.columns for col in [tau_col, p_col, trend_col]):
                # Get significant increasing trends
                increasing_seasonal = results_df[
                    (results_df[trend_col] == 'increasing') & 
                    (results_df[p_col] < 0.10)
                ].nsmallest(6, p_col)
                
                # Get significant decreasing trends
                decreasing_seasonal = results_df[
                    (results_df[trend_col] == 'decreasing') & 
                    (results_df[p_col] < 0.10)
                ].nsmallest(6, p_col)
                
                # Create seasonal plots if we have data
                if len(increasing_seasonal) > 0 or len(decreasing_seasonal) > 0:
                    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
                    fig.suptitle(f'{season} {metric} Trends - Top 6 Increasing & Decreasing (90% Confidence)', fontsize=16)
                    
                    # Plot increasing trends
                    for i, (_, station_row) in enumerate(increasing_seasonal.iterrows()):
                        if i >= 6:
                            break
                        
                        station_id = station_row['Station_ID']
                        station_data = original_df[
                            (original_df['Station_ID'] == station_id) & 
                            (original_df['Season'] == season)
                        ]
                        
                        if len(station_data) > 0:
                            if metric == 'Duration':
                                annual_data = station_data.groupby('Year')['Duration_Days'].mean()
                                ylabel = 'Duration (days)'
                            elif metric == 'Flow':
                                annual_data = station_data.groupby('Year')['Mean_SFD_Flow_Percentile'].mean()
                                ylabel = 'Flow Percentile (%)'
                            else:  # Count
                                annual_data = station_data.groupby('Year').size()
                                ylabel = 'Event Count'
                            
                            axes[0, i].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='green')
                            axes[0, i].set_title(f'{station_id}\np={station_row[p_col]:.4f}', fontsize=10)
                            axes[0, i].set_xlabel('Year')
                            axes[0, i].set_ylabel(ylabel)
                            axes[0, i].grid(True, alpha=0.3)
                    
                    # Plot decreasing trends
                    for i, (_, station_row) in enumerate(decreasing_seasonal.iterrows()):
                        if i >= 6:
                            break
                        
                        station_id = station_row['Station_ID']
                        station_data = original_df[
                            (original_df['Station_ID'] == station_id) & 
                            (original_df['Season'] == season)
                        ]
                        
                        if len(station_data) > 0:
                            if metric == 'Duration':
                                annual_data = station_data.groupby('Year')['Duration_Days'].mean()
                                ylabel = 'Duration (days)'
                            elif metric == 'Flow':
                                annual_data = station_data.groupby('Year')['Mean_SFD_Flow_Percentile'].mean()
                                ylabel = 'Flow Percentile (%)'
                            else:  # Count
                                annual_data = station_data.groupby('Year').size()
                                ylabel = 'Event Count'
                            
                            axes[1, i].plot(annual_data.index, annual_data.values, 'o-', alpha=0.7, color='red')
                            axes[1, i].set_title(f'{station_id}\np={station_row[p_col]:.4f}', fontsize=10)
                            axes[1, i].set_xlabel('Year')
                            axes[1, i].set_ylabel(ylabel)
                            axes[1, i].grid(True, alpha=0.3)
                    
                    # Hide empty subplots
                    for i in range(len(increasing_seasonal), 6):
                        axes[0, i].set_visible(False)
                    for i in range(len(decreasing_seasonal), 6):
                        axes[1, i].set_visible(False)
                    
                    # Add row labels
                    axes[0, 0].text(-0.1, 0.5, 'Increasing Trends', transform=axes[0, 0].transAxes, 
                                   rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
                    axes[1, 0].text(-0.1, 0.5, 'Decreasing Trends', transform=axes[1, 0].transAxes, 
                                   rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(f'results/seasonal_{season.lower()}_{metric.lower()}_trends.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"    ✓ {season} {metric} trends plot saved")
    
    # Summary plot: Trend distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Duration trends
    duration_counts = results_df['Duration_Trend'].value_counts()
    axes[0].pie(duration_counts.values, labels=[x.replace('_', ' ').title() for x in duration_counts.index], 
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Duration Trends Distribution')
    
    # Flow percentile trends
    flow_counts = results_df['Flow_Percentile_Trend'].value_counts()
    axes[1].pie(flow_counts.values, labels=[x.replace('_', ' ').title() for x in flow_counts.index], 
                autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Flow Percentile Trends Distribution')
    
    # Event count trends
    count_counts = results_df['Event_Count_Trend'].value_counts()
    axes[2].pie(count_counts.values, labels=[x.replace('_', ' ').title() for x in count_counts.index], 
                autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Event Count Trends Distribution')
    
    plt.tight_layout()
    plt.savefig('results/trend_distribution_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Trend distribution summary plot saved")
    
    # Create comprehensive summary table
    summary_data = []
    
    # Duration summary
    for trend_type in ['increasing', 'decreasing', 'no_trend']:
        count = len(results_df[results_df['Duration_Trend'] == trend_type])
        significant_count = len(results_df[
            (results_df['Duration_Trend'] == trend_type) & 
            (results_df['Duration_P_Value'] < 0.10)
        ])
        summary_data.append({
            'Metric': 'Duration (days)',
            'Trend': trend_type.replace('_', ' ').title(),
            'Total_Stations': count,
            'Significant_Stations': significant_count,
            'Percentage': f"{(count/len(results_df)*100):.1f}%"
        })
    
    # Flow percentile summary
    for trend_type in ['increasing', 'decreasing', 'no_trend']:
        count = len(results_df[results_df['Flow_Percentile_Trend'] == trend_type])
        significant_count = len(results_df[
            (results_df['Flow_Percentile_Trend'] == trend_type) & 
            (results_df['Flow_Percentile_P_Value'] < 0.10)
        ])
        summary_data.append({
            'Metric': 'Flow Percentile (%)',
            'Trend': trend_type.replace('_', ' ').title(),
            'Total_Stations': count,
            'Significant_Stations': significant_count,
            'Percentage': f"{(count/len(results_df)*100):.1f}%"
        })
    
    # Event count summary
    for trend_type in ['increasing', 'decreasing', 'no_trend']:
        count = len(results_df[results_df['Event_Count_Trend'] == trend_type])
        significant_count = len(results_df[
            (results_df['Event_Count_Trend'] == trend_type) & 
            (results_df['Event_Count_P_Value'] < 0.10)
        ])
        summary_data.append({
            'Metric': 'Event Count (per year)',
            'Trend': trend_type.replace('_', ' ').title(),
            'Total_Stations': count,
            'Significant_Stations': significant_count,
            'Percentage': f"{(count/len(results_df)*100):.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('results/trend_summary_table.csv', index=False)
    print("  ✓ Comprehensive summary table saved")
    
    # Create seasonal summary table
    seasonal_summary_data = []
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        for metric in ['Duration', 'Flow', 'Count']:
            trend_col = f'{season}_{metric}_Trend'
            p_col = f'{season}_{metric}_P_Value'
            
            if trend_col in results_df.columns:
                for trend_type in ['increasing', 'decreasing', 'no_trend']:
                    count = len(results_df[results_df[trend_col] == trend_type])
                    significant_count = len(results_df[
                        (results_df[trend_col] == trend_type) & 
                        (results_df[p_col] < 0.10)
                    ])
                    
                    seasonal_summary_data.append({
                        'Season': season,
                        'Metric': f'{metric} {"(days)" if metric == "Duration" else "(%)" if metric == "Flow" else "(count)"}',
                        'Trend': trend_type.replace('_', ' ').title(),
                        'Total_Stations': count,
                        'Significant_Stations': significant_count,
                        'Percentage': f"{(count/len(results_df)*100):.1f}%" if count > 0 else "0.0%"
                    })
    
    seasonal_summary_df = pd.DataFrame(seasonal_summary_data)
    seasonal_summary_df.to_csv('results/seasonal_trend_summary_table.csv', index=False)
    print("  ✓ Seasonal summary table saved")
    
    print(f"\n4. COMPREHENSIVE SUMMARY TABLE:")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    
    print(f"\n5. DETAILED SEASONAL TRENDS SUMMARY:")
    print("-" * 50)
    
    # Print seasonal summary by season
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        print(f"\n{season.upper()} TRENDS:")
        season_data = seasonal_summary_df[seasonal_summary_df['Season'] == season]
        
        for metric in ['Duration (days)', 'Flow (%)', 'Count (count)']:
            metric_data = season_data[season_data['Metric'] == metric]
            if len(metric_data) > 0:
                print(f"  {metric}:")
                for _, row in metric_data.iterrows():
                    if row['Total_Stations'] > 0:
                        print(f"    {row['Trend']}: {row['Total_Stations']} stations ({row['Significant_Stations']} significant)")
    
    # Create seasonal comparison plot
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Seasonal Trend Distribution Comparison (90% Confidence)', fontsize=16)
    
    metrics_plot = ['Duration', 'Flow', 'Count']
    seasons_plot = ['Winter', 'Spring', 'Summer', 'Fall']
    
    for i, metric in enumerate(metrics_plot):
        for j, season in enumerate(seasons_plot):
            trend_col = f'{season}_{metric}_Trend'
            
            if trend_col in results_df.columns:
                # Get trend counts for this season/metric
                trend_counts = results_df[trend_col].value_counts()
                
                # Filter out insufficient_data for cleaner plots
                trend_counts = trend_counts[trend_counts.index != 'insufficient_data']
                
                if len(trend_counts) > 0:
                    colors = {'increasing': 'green', 'decreasing': 'red', 'no_trend': 'gray'}
                    plot_colors = [colors.get(trend, 'blue') for trend in trend_counts.index]
                    
                    axes[i, j].pie(trend_counts.values, 
                                  labels=[t.replace('_', ' ').title() for t in trend_counts.index],
                                  autopct='%1.1f%%', 
                                  colors=plot_colors,
                                  startangle=90)
                    
                    metric_name = 'Duration' if metric == 'Duration' else 'Flow %' if metric == 'Flow' else 'Event Count'
                    axes[i, j].set_title(f'{season} {metric_name}', fontsize=12)
                else:
                    axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i, j].transAxes)
                    axes[i, j].set_title(f'{season} {metric}', fontsize=12)
            else:
                axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'{season} {metric}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/seasonal_trend_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Seasonal trend comparison plot saved")
    
    # Calculate seasonal trend statistics
    print(f"\n6. SEASONAL TREND STATISTICS:")
    print("-" * 50)
    
    seasonal_stats = {}
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        seasonal_stats[season] = {}
        
        for metric in ['Duration', 'Flow', 'Count']:
            trend_col = f'{season}_{metric}_Trend'
            p_col = f'{season}_{metric}_P_Value'
            
            if trend_col in results_df.columns:
                total_with_data = len(results_df[results_df[trend_col] != 'insufficient_data'])
                increasing = len(results_df[results_df[trend_col] == 'increasing'])
                decreasing = len(results_df[results_df[trend_col] == 'decreasing'])
                no_trend = len(results_df[results_df[trend_col] == 'no_trend'])
                
                seasonal_stats[season][metric] = {
                    'total': total_with_data,
                    'increasing': increasing,
                    'decreasing': decreasing,
                    'no_trend': no_trend,
                    'significant_trends': increasing + decreasing
                }
        
        print(f"\n{season}:")
        if seasonal_stats[season]:
            for metric in ['Duration', 'Flow', 'Count']:
                if metric in seasonal_stats[season]:
                    stats = seasonal_stats[season][metric]
                    total = stats['total']
                    sig_trends = stats['significant_trends']
                    if total > 0:
                        pct_sig = (sig_trends / total) * 100
                        print(f"  {metric}: {sig_trends}/{total} stations with significant trends ({pct_sig:.1f}%)")
                        print(f"    Increasing: {stats['increasing']} ({(stats['increasing']/total)*100:.1f}%)")
                        print(f"    Decreasing: {stats['decreasing']} ({(stats['decreasing']/total)*100:.1f}%)")
    
    # Save seasonal statistics
    seasonal_stats_summary = []
    for season in seasonal_stats:
        for metric in seasonal_stats[season]:
            stats = seasonal_stats[season][metric]
            if stats['total'] > 0:
                seasonal_stats_summary.append({
                    'Season': season,
                    'Metric': metric,
                    'Total_Stations_With_Data': stats['total'],
                    'Increasing_Trends': stats['increasing'],
                    'Decreasing_Trends': stats['decreasing'],
                    'No_Trend': stats['no_trend'],
                    'Total_Significant_Trends': stats['significant_trends'],
                    'Percent_With_Significant_Trends': f"{(stats['significant_trends']/stats['total'])*100:.1f}%"
                })
    
    seasonal_stats_df = pd.DataFrame(seasonal_stats_summary)
    seasonal_stats_df.to_csv('results/seasonal_trend_statistics.csv', index=False)
    print(f"\n  ✓ Seasonal trend statistics saved to results/seasonal_trend_statistics.csv")
    
    print(f"\nAll plots and results saved to results/ directory")
    print(f"Total stations analyzed: {len(results_df)}")
    print(f"Data period: {original_df['Year'].min()} - {original_df['Year'].max()}")
    print(f"Total flash drought events: {len(original_df)}")
    
    print(f"\nFILES CREATED:")
    print("- mann_kendall_trend_results.csv (detailed results)")
    print("- trend_summary_table.csv (annual trends summary)")
    print("- seasonal_trend_summary_table.csv (seasonal trends summary)")
    print("- seasonal_trend_statistics.csv (seasonal statistics)")
    print("- Multiple trend plots (annual and seasonal)")
    print("- seasonal_trend_comparison.png (seasonal comparison)")

if __name__ == "__main__":
    results_df, original_df = analyze_trends()
    generate_summary_and_plots(results_df, original_df)
    print("\nAnalysis complete!")