import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read and process data
df = pd.read_csv('conus_daily_resampled_streamflow_flash_droughts.csv')
df['Mean_SFD_Flow_Percentile'] = df['Mean_SFD_Flow_Percentile'].round(2)
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
df['Termination_Time'] = pd.to_datetime(df['Termination_Time'])
df['Year'] = df['Onset_Time'].dt.year
df['Month'] = df['Onset_Time'].dt.month
df['Month_Name'] = df['Onset_Time'].dt.strftime('%B')
df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                3: 'Spring', 4: 'Spring', 5: 'Spring',
                                6: 'Summer', 7: 'Summer', 8: 'Summer',
                                9: 'Fall', 10: 'Fall', 11: 'Fall'})

# Select most active station
target_station = 3208500
station_data = df[df['Station_ID'] == target_station].copy().sort_values('Onset_Time')

print("="*80)
print(f"COMPREHENSIVE ANALYSIS FOR STATION {target_station}")
print("="*80)

# Basic Statistics
print(f"\n1. BASIC STATISTICS")
print("-" * 40)
print(f"Total Events: {len(station_data)}")
print(f"Time Period: {station_data['Year'].min()} - {station_data['Year'].max()}")
print(f"Duration - Mean: {station_data['Duration_Days'].mean():.1f} days")
print(f"Duration - Median: {station_data['Duration_Days'].median():.1f} days")
print(f"Duration - Range: {station_data['Duration_Days'].min():.0f} - {station_data['Duration_Days'].max():.0f} days")

# Yearly Evolution
print(f"\n2. YEARLY EVOLUTION")
print("-" * 40)
yearly_stats = station_data.groupby('Year')['Duration_Days'].agg(['count', 'mean', 'median']).round(2)
print(yearly_stats.head(10))

# Monthly Patterns
print(f"\n3. MONTHLY PATTERNS")
print("-" * 40)
monthly_stats = station_data.groupby('Month_Name')['Duration_Days'].agg(['count', 'mean', 'median']).round(2)
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_stats = monthly_stats.reindex([m for m in month_order if m in monthly_stats.index])
print(monthly_stats)

# Seasonal Analysis
print(f"\n4. SEASONAL ANALYSIS")
print("-" * 40)
seasonal_stats = station_data.groupby('Season')['Duration_Days'].agg(['count', 'mean', 'median']).round(2)
print(seasonal_stats)

# Trend Analysis
print(f"\n5. TREND ANALYSIS")
print("-" * 40)
if len(yearly_stats) > 2:
    slope, intercept, r_value, p_value, std_err = stats.linregress(yearly_stats.index, yearly_stats['mean'])
    print(f"Linear Trend: {slope:.3f} days per year")
    print(f"R-squared: {r_value**2:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Trend: {'Increasing' if slope > 0 else 'Decreasing'}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

# Extreme Events
print(f"\n6. EXTREME EVENTS")
print("-" * 40)
duration_90th = station_data['Duration_Days'].quantile(0.9)
duration_10th = station_data['Duration_Days'].quantile(0.1)
extreme_long = station_data[station_data['Duration_Days'] >= duration_90th]
extreme_short = station_data[station_data['Duration_Days'] <= duration_10th]

print(f"Long Events (≥{duration_90th:.0f} days): {len(extreme_long)} events")
print(f"Short Events (≤{duration_10th:.0f} days): {len(extreme_short)} events")

print(f"\n7. TOP 5 LONGEST EVENTS")
print("-" * 40)
longest = station_data.nlargest(5, 'Duration_Days')
for i, (_, event) in enumerate(longest.iterrows(), 1):
    print(f"{i}. {event['Onset_Time'].strftime('%Y-%m-%d')}: {event['Duration_Days']:.0f} days")

print(f"\nCreating visualizations...")

# Create comprehensive visualization
plt.style.use('default')
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle(f'Station {target_station} - Duration Evolution Analysis', fontsize=16, fontweight='bold')

# Plot 1: Timeline
ax = axes[0, 0]
ax.scatter(station_data['Onset_Time'], station_data['Duration_Days'], alpha=0.7, s=50)
ax.set_title('Duration Timeline')
ax.set_ylabel('Duration (Days)')
ax.tick_params(axis='x', rotation=45)

# Plot 2: Yearly trend
ax = axes[0, 1]
yearly_avg = station_data.groupby('Year')['Duration_Days'].mean()
ax.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2)
ax.set_title('Average Duration by Year')
ax.set_ylabel('Average Duration (Days)')

# Plot 3: Monthly boxplot
ax = axes[0, 2]
existing_months = [m for m in month_order if m in station_data['Month_Name'].values]
if len(existing_months) > 1:
    sns.boxplot(data=station_data, x='Month_Name', y='Duration_Days', ax=ax, order=existing_months)
    ax.tick_params(axis='x', rotation=45)
ax.set_title('Duration by Month')

# Plot 4: Seasonal bars
ax = axes[1, 0]
seasonal_avg = station_data.groupby('Season')['Duration_Days'].mean()
ax.bar(seasonal_avg.index, seasonal_avg.values, color=['lightblue', 'lightgreen', 'orange', 'lightcoral'])
ax.set_title('Average Duration by Season')
ax.set_ylabel('Average Duration (Days)')

# Plot 5: Duration histogram
ax = axes[1, 1]
ax.hist(station_data['Duration_Days'], bins=15, alpha=0.7, edgecolor='black')
ax.axvline(station_data['Duration_Days'].mean(), color='red', linestyle='--', label='Mean')
ax.axvline(station_data['Duration_Days'].median(), color='orange', linestyle='--', label='Median')
ax.set_title('Duration Distribution')
ax.set_xlabel('Duration (Days)')
ax.legend()

# Plot 6: Events per year
ax = axes[1, 2]
events_per_year = station_data.groupby('Year').size()
ax.bar(events_per_year.index, events_per_year.values, alpha=0.7)
ax.set_title('Events per Year')
ax.set_ylabel('Number of Events')
ax.tick_params(axis='x', rotation=45)

# Plot 7: Duration vs Onset Rate
ax = axes[2, 0]
ax.scatter(station_data['Onset_Rate_Days'], station_data['Duration_Days'], alpha=0.7)
ax.set_title('Duration vs Onset Rate')
ax.set_xlabel('Onset Rate (Days)')
ax.set_ylabel('Duration (Days)')

# Plot 8: Monthly frequency
ax = axes[2, 1]
monthly_counts = station_data['Month_Name'].value_counts().reindex(existing_months)
ax.bar(monthly_counts.index, monthly_counts.values, color='lightgreen')
ax.set_title('Event Frequency by Month')
ax.set_ylabel('Number of Events')
ax.tick_params(axis='x', rotation=45)

# Plot 9: Flow percentile vs Duration
ax = axes[2, 2]
ax.scatter(station_data['Mean_SFD_Flow_Percentile'], station_data['Duration_Days'], alpha=0.7)
ax.set_title('Flow Percentile vs Duration')
ax.set_xlabel('Flow Percentile (%)')
ax.set_ylabel('Duration (Days)')

plt.tight_layout()
plt.savefig(f'station_{target_station}_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization saved as 'station_{target_station}_complete_analysis.png'")

# Export data
station_data.to_csv(f'station_{target_station}_data.csv', index=False)
yearly_stats.to_csv(f'station_{target_station}_yearly_stats.csv')
monthly_stats.to_csv(f'station_{target_station}_monthly_stats.csv')
seasonal_stats.to_csv(f'station_{target_station}_seasonal_stats.csv')

print(f"\nExported files:")
print(f"• station_{target_station}_data.csv")
print(f"• station_{target_station}_yearly_stats.csv")
print(f"• station_{target_station}_monthly_stats.csv")
print(f"• station_{target_station}_seasonal_stats.csv")
print(f"• station_{target_station}_complete_analysis.png")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)