import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the processed data
df = pd.read_csv('processed_streamflow_data.csv')

# Convert date columns to datetime
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])

print("="*80)
print("DETAILED STATION-SPECIFIC DURATION EVOLUTION ANALYSIS")
print("="*80)

# Get top 10 most active stations for detailed analysis
top_stations = df['Station_ID'].value_counts().head(10)
print(f"\nTop 10 Most Active Stations:")
for i, (station, count) in enumerate(top_stations.items(), 1):
    print(f"{i:2d}. Station {station}: {count} events")

# Create detailed analysis for each top station
def analyze_station_evolution(station_id, show_plots=True):
    station_data = df[df['Station_ID'] == station_id].copy()
    station_data = station_data.sort_values('Onset_Time')
    
    print(f"\n{'='*60}")
    print(f"STATION {station_id} - DURATION EVOLUTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Total Events: {len(station_data)}")
    print(f"Time Period: {station_data['Year'].min()} - {station_data['Year'].max()}")
    print(f"Duration Range: {station_data['Duration_Days'].min():.0f} - {station_data['Duration_Days'].max():.0f} days")
    print(f"Average Duration: {station_data['Duration_Days'].mean():.1f} days")
    
    # Monthly evolution
    monthly_evolution = station_data.groupby(['Year', 'Month_Name'])['Duration_Days'].agg([
        'count', 'mean', 'median', 'min', 'max'
    ]).round(2)
    
    print(f"\nMonthly Evolution (showing years with events):")
    print(monthly_evolution.head(10))
    
    # Yearly trends
    yearly_trends = station_data.groupby('Year')['Duration_Days'].agg([
        'count', 'mean', 'median', 'min', 'max'
    ]).round(2)
    
    print(f"\nYearly Trends:")
    print(yearly_trends)
    
    # Monthly patterns across all years
    monthly_patterns = station_data.groupby('Month_Name')['Duration_Days'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(2)
    
    print(f"\nMonthly Patterns (All Years Combined):")
    print(monthly_patterns)
    
    if show_plots and len(station_data) > 5:  # Only plot if enough data
        # Create visualization for this station
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Station {station_id} - Duration Evolution Analysis', fontsize=14, fontweight='bold')
        
        # Plot 1: Duration over time
        ax1 = axes[0, 0]
        ax1.scatter(station_data['Onset_Time'], station_data['Duration_Days'], alpha=0.7, s=50)
        ax1.set_title('Duration Over Time')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Duration (Days)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Monthly box plot
        ax2 = axes[0, 1]
        if len(station_data['Month_Name'].unique()) > 1:
            sns.boxplot(data=station_data, x='Month_Name', y='Duration_Days', ax=ax2)
            ax2.set_title('Duration by Month')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Duration by Month')
        
        # Plot 3: Yearly trend
        ax3 = axes[1, 0]
        if len(yearly_trends) > 1:
            ax3.plot(yearly_trends.index, yearly_trends['mean'], marker='o', linewidth=2, markersize=6)
            ax3.set_title('Average Duration Trend by Year')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Average Duration (Days)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data\nfor yearly trend', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Average Duration Trend by Year')
        
        # Plot 4: Duration distribution
        ax4 = axes[1, 1]
        ax4.hist(station_data['Duration_Days'], bins=min(20, len(station_data)//2), alpha=0.7, edgecolor='black')
        ax4.set_title('Duration Distribution')
        ax4.set_xlabel('Duration (Days)')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'station_{station_id}_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved as 'station_{station_id}_analysis.png'")
    
    return station_data

# Analyze top 5 stations in detail
print(f"\n{'='*80}")
print("DETAILED ANALYSIS OF TOP 5 STATIONS")
print(f"{'='*80}")

for station in top_stations.head(5).index:
    analyze_station_evolution(station, show_plots=True)
    print("\n" + "-"*80)# Create
 a comprehensive comparison across all stations
print(f"\n{'='*80}")
print("COMPREHENSIVE COMPARISON ACROSS ALL STATIONS")
print(f"{'='*80}")

# Evolution patterns by decade
df['Decade'] = (df['Year'] // 10) * 10
decade_analysis = df.groupby(['Station_ID', 'Decade'])['Duration_Days'].agg([
    'count', 'mean', 'median'
]).round(2)

print("\nDecade-wise Duration Evolution (Sample):")
print(decade_analysis.head(15))

# Monthly evolution trends across all stations
monthly_evolution_all = df.groupby(['Year', 'Month_Name'])['Duration_Days'].agg([
    'count', 'mean', 'median'
]).round(2)

print(f"\nMonthly Evolution Trends (All Stations Combined):")
print(monthly_evolution_all.head(15))

# Create summary statistics for each station
station_summary = df.groupby('Station_ID').agg({
    'Duration_Days': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'Year': ['min', 'max'],
    'Month': 'nunique'
}).round(2)

station_summary.columns = ['Event_Count', 'Avg_Duration', 'Median_Duration', 'Duration_Std', 
                          'Min_Duration', 'Max_Duration', 'First_Year', 'Last_Year', 'Months_Active']

print(f"\nStation Summary Statistics (Top 10):")
print(station_summary.sort_values('Event_Count', ascending=False).head(10))

# Export detailed station summaries
station_summary.to_csv('station_summary_statistics.csv')
decade_analysis.to_csv('decade_wise_duration_analysis.csv')
monthly_evolution_all.to_csv('monthly_evolution_all_stations.csv')

print(f"\nAdditional exported files:")
print("• station_summary_statistics.csv - Comprehensive station statistics")
print("• decade_wise_duration_analysis.csv - Duration evolution by decade")
print("• monthly_evolution_all_stations.csv - Monthly trends across all stations")

# Key insights
print(f"\n{'='*80}")
print("KEY INSIGHTS FROM DURATION EVOLUTION ANALYSIS")
print(f"{'='*80}")

print(f"1. TEMPORAL COVERAGE:")
print(f"   • Dataset spans {df['Year'].max() - df['Year'].min() + 1} years ({df['Year'].min()}-{df['Year'].max()})")
print(f"   • {df['Station_ID'].nunique()} unique monitoring stations")
print(f"   • {len(df)} total flash drought events")

print(f"\n2. DURATION CHARACTERISTICS:")
print(f"   • Average duration: {df['Duration_Days'].mean():.1f} days")
print(f"   • Median duration: {df['Duration_Days'].median():.1f} days")
print(f"   • Range: {df['Duration_Days'].min():.0f} - {df['Duration_Days'].max():.0f} days")

print(f"\n3. SEASONAL PATTERNS:")
seasonal_avg = df.groupby('Month_Name')['Duration_Days'].mean().sort_values(ascending=False)
print(f"   • Longest average durations: {list(seasonal_avg.head(3).index)}")
print(f"   • Shortest average durations: {list(seasonal_avg.tail(3).index)}")

print(f"\n4. STATION ACTIVITY:")
print(f"   • Most active station: {top_stations.index[0]} ({top_stations.iloc[0]} events)")
print(f"   • Average events per station: {len(df) / df['Station_ID'].nunique():.1f}")

print(f"\n5. TEMPORAL TRENDS:")
yearly_avg = df.groupby('Year')['Duration_Days'].mean()
recent_avg = yearly_avg.tail(10).mean()
early_avg = yearly_avg.head(10).mean()
print(f"   • Recent decade average (2015-2024): {recent_avg:.1f} days")
print(f"   • Early decade average (1981-1990): {early_avg:.1f} days")
print(f"   • Trend: {'Increasing' if recent_avg > early_avg else 'Decreasing'} duration over time")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE - All files exported successfully!")
print(f"{'='*80}")