import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the CSV file
df = pd.read_csv('conus_daily_resampled_streamflow_flash_droughts.csv')

# Format Mean_SFD_Flow_Percentile to 2 decimal places
df['Mean_SFD_Flow_Percentile'] = df['Mean_SFD_Flow_Percentile'].round(2)

# Convert date columns to datetime
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
df['Termination_Time'] = pd.to_datetime(df['Termination_Time'])

# Extract year and month from Onset_Time
df['Year'] = df['Onset_Time'].dt.year
df['Month'] = df['Onset_Time'].dt.month
df['Month_Name'] = df['Onset_Time'].dt.strftime('%B')

print("Data Overview:")
print(f"Total records: {len(df)}")
print(f"Unique Station IDs: {df['Station_ID'].nunique()}")
print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Duration range: {df['Duration_Days'].min()} - {df['Duration_Days'].max()} days")

# Monthly Duration Statistics by Station and Year
monthly_stats = df.groupby(['Station_ID', 'Year', 'Month_Name'])['Duration_Days'].agg([
    'mean', 'median', 'count', 'min', 'max'
]).round(2)

print("\n" + "="*60)
print("MONTHLY DURATION STATISTICS BY STATION AND YEAR")
print("="*60)
print(monthly_stats.head(15))

# Overall monthly trends across all years
overall_monthly = df.groupby(['Month_Name'])['Duration_Days'].agg([
    'mean', 'median', 'count', 'std'
]).round(2)

print("\n" + "="*60)
print("OVERALL MONTHLY DURATION TRENDS (ALL STATIONS)")
print("="*60)
print(overall_monthly)

# Yearly Duration Statistics by Station
yearly_stats = df.groupby(['Station_ID', 'Year'])['Duration_Days'].agg([
    'mean', 'median', 'count', 'min', 'max'
]).round(2)

print("\n" + "="*60)
print("YEARLY DURATION STATISTICS BY STATION")
print("="*60)
print(yearly_stats.head(15))

# Create visualizations
plt.style.use('default')
sns.set_palette("husl")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Streamflow Flash Drought Duration Analysis', fontsize=16, fontweight='bold')

# Plot 1: Box plot of duration by month
ax1 = axes[0, 0]
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

existing_months = [month for month in month_order if month in df['Month_Name'].values]

sns.boxplot(data=df, x='Month_Name', y='Duration_Days', ax=ax1, order=existing_months)
ax1.set_title('Duration Distribution by Month (All Years)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Duration (Days)')
ax1.tick_params(axis='x', rotation=45)

# Plot 2: Average duration by year
ax2 = axes[0, 1]
yearly_avg = df.groupby('Year')['Duration_Days'].mean()
ax2.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
ax2.set_title('Average Duration Trend by Year')
ax2.set_xlabel('Year')
ax2.set_ylabel('Average Duration (Days)')
ax2.grid(True, alpha=0.3)

# Plot 3: Heatmap of average duration by month and year
ax3 = axes[1, 0]
pivot_data = df.pivot_table(values='Duration_Days', index='Month', columns='Year', aggfunc='mean')
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Duration (Days)'})
ax3.set_title('Average Duration Heatmap (Month vs Year)')
ax3.set_xlabel('Year')
ax3.set_ylabel('Month')

# Plot 4: Duration trends by station (top 5 stations with most events)
ax4 = axes[1, 1]
top_stations = df['Station_ID'].value_counts().head(5).index
for station in top_stations:
    station_data = df[df['Station_ID'] == station].groupby('Year')['Duration_Days'].mean()
    ax4.plot(station_data.index, station_data.values, marker='o', label=f'Station {station}', linewidth=2)

ax4.set_title('Duration Trends by Top 5 Active Stations')
ax4.set_xlabel('Year')
ax4.set_ylabel('Average Duration (Days)')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('streamflow_duration_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'streamflow_duration_analysis.png'")

# Export processed data
monthly_summary = df.groupby(['Station_ID', 'Month_Name'])['Duration_Days'].agg([
    'count', 'mean', 'median', 'min', 'max', 'std'
]).round(2)

yearly_summary = df.groupby(['Station_ID', 'Year'])['Duration_Days'].agg([
    'count', 'mean', 'median', 'min', 'max', 'std'
]).round(2)

monthly_summary.to_csv('monthly_duration_summary.csv')
yearly_summary.to_csv('yearly_duration_summary.csv')
df.to_csv('processed_streamflow_data.csv', index=False)

print("\nExported files:")
print("• monthly_duration_summary.csv - Monthly statistics by station")
print("• yearly_duration_summary.csv - Yearly statistics by station") 
print("• processed_streamflow_data.csv - Main dataset with added columns")

print("\nAnalysis complete!")