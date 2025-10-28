#!/usr/bin/env python3
"""
Diagnostic script to compare annual vs seasonal trend calculations
"""

import pandas as pd
import numpy as np

def analyze_trend_differences():
    """Compare annual vs seasonal trends to identify discrepancies"""
    
    # Load results
    results_df = pd.read_csv('results/mann_kendall_trend_results.csv')
    
    print("DIAGNOSTIC: Annual vs Seasonal Trend Comparison")
    print("=" * 60)
    
    # Check a few example stations
    print("\nExample Station Analysis:")
    print("-" * 30)
    
    # Get first few stations with both annual and seasonal data
    example_stations = results_df.head(5)
    
    for _, station in example_stations.iterrows():
        station_id = station['Station_ID']
        print(f"\nStation {station_id}:")
        
        # Annual trends
        print(f"  Annual Duration Trend: {station['Duration_Trend']} (p={station['Duration_P_Value']:.4f})")
        print(f"  Annual Flow Trend: {station['Flow_Percentile_Trend']} (p={station['Flow_Percentile_P_Value']:.4f})")
        
        # Seasonal trends
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        for season in seasons:
            dur_trend = station[f'{season}_Duration_Trend']
            dur_p = station[f'{season}_Duration_P_Value']
            flow_trend = station[f'{season}_Flow_Trend']
            flow_p = station[f'{season}_Flow_P_Value']
            
            if pd.notna(dur_trend) and dur_trend != 'insufficient_data':
                print(f"    {season} Duration: {dur_trend} (p={dur_p:.4f})")
            if pd.notna(flow_trend) and flow_trend != 'insufficient_data':
                print(f"    {season} Flow: {flow_trend} (p={flow_p:.4f})")
    
    # Count mismatches
    print(f"\n\nTREND COUNT ANALYSIS:")
    print("-" * 30)
    
    # Annual counts
    annual_dur_inc = len(results_df[results_df['Duration_Trend'] == 'increasing'])
    annual_dur_dec = len(results_df[results_df['Duration_Trend'] == 'decreasing'])
    annual_flow_inc = len(results_df[results_df['Flow_Percentile_Trend'] == 'increasing'])
    annual_flow_dec = len(results_df[results_df['Flow_Percentile_Trend'] == 'decreasing'])
    
    print(f"Annual Duration Trends: {annual_dur_inc} increasing, {annual_dur_dec} decreasing")
    print(f"Annual Flow Trends: {annual_flow_inc} increasing, {annual_flow_dec} decreasing")
    
    # Seasonal counts (sum across all seasons)
    seasonal_dur_inc = 0
    seasonal_dur_dec = 0
    seasonal_flow_inc = 0
    seasonal_flow_dec = 0
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        dur_col = f'{season}_Duration_Trend'
        flow_col = f'{season}_Flow_Trend'
        
        seasonal_dur_inc += len(results_df[results_df[dur_col] == 'increasing'])
        seasonal_dur_dec += len(results_df[results_df[dur_col] == 'decreasing'])
        seasonal_flow_inc += len(results_df[results_df[flow_col] == 'increasing'])
        seasonal_flow_dec += len(results_df[results_df[flow_col] == 'decreasing'])
    
    print(f"Seasonal Duration Trends (total): {seasonal_dur_inc} increasing, {seasonal_dur_dec} decreasing")
    print(f"Seasonal Flow Trends (total): {seasonal_flow_inc} increasing, {seasonal_flow_dec} decreasing")
    
    print(f"\nWhy numbers don't match:")
    print("1. Annual trends use ALL events across all seasons for each year")
    print("2. Seasonal trends use ONLY events within each specific season")
    print("3. A station can have different trends in different seasons")
    print("4. Seasonal totals count each station up to 4 times (once per season)")
    
    # Show stations with conflicting trends
    print(f"\n\nSTATIONS WITH CONFLICTING TRENDS:")
    print("-" * 40)
    
    conflicts = 0
    for _, station in results_df.iterrows():
        annual_dur = station['Duration_Trend']
        seasonal_trends = []
        
        for season in ['Winter', 'Spring', 'Summer', 'Fall']:
            seasonal_trend = station[f'{season}_Duration_Trend']
            if pd.notna(seasonal_trend) and seasonal_trend != 'insufficient_data':
                seasonal_trends.append(seasonal_trend)
        
        # Check if any seasonal trend differs from annual
        if annual_dur == 'no_trend' and any(t != 'no_trend' for t in seasonal_trends):
            conflicts += 1
            if conflicts <= 5:  # Show first 5 examples
                print(f"Station {station['Station_ID']}: Annual={annual_dur}, Seasonal={seasonal_trends}")
    
    print(f"\nTotal stations with conflicting annual vs seasonal trends: {conflicts}")

if __name__ == "__main__":
    analyze_trend_differences()