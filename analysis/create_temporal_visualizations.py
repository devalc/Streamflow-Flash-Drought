import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_parquet('data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
df['Year'] = df['Onset_Time'].dt.year
df['Month'] = df['Onset_Time'].dt.month
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

target_vars = ['Mean_SFD_Flow_Percentile', 'Duration_Days']

# Set style
plt.style.use('default')
sns.set_palette("husl")

# 1. Monthly patterns heatmap
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Monthly heatmap for Mean_SFD_Flow_Percentile
ax = axes[0, 0]
monthly_pivot = df.groupby(['AGGECOREGION', 'Month'])['Mean_SFD_Flow_Percentile'].mean().unstack()
sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
ax.set_title('Mean SFD Flow Percentile - Monthly Patterns')
ax.set_xlabel('Month')
ax.set_ylabel('Ecoregion')
ax.set_xticklabels([month_names[m] for m in range(1, 13)])

# Monthly heatmap for Duration_Days
ax = axes[0, 1]
monthly_pivot = df.groupby(['AGGECOREGION', 'Month'])['Duration_Days'].mean().unstack()
sns.heatmap(monthly_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
ax.set_title('Duration Days - Monthly Patterns')
ax.set_xlabel('Month')
ax.set_ylabel('Ecoregion')
ax.set_xticklabels([month_names[m] for m in range(1, 13)])

# Seasonal patterns for Mean_SFD_Flow_Percentile
ax = axes[1, 0]
seasonal_data = df.pivot_table(values='Mean_SFD_Flow_Percentile', index=['AGGECOREGION'], columns='Season', aggfunc='mean')
seasonal_data = seasonal_data[['Spring', 'Summer', 'Fall', 'Winter']]
sns.heatmap(seasonal_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
ax.set_title('Mean SFD Flow Percentile - Seasonal Patterns')
ax.set_xlabel('Season')
ax.set_ylabel('Ecoregion')

# Seasonal patterns for Duration_Days
ax = axes[1, 1]
seasonal_data = df.pivot_table(values='Duration_Days', index=['AGGECOREGION'], columns='Season', aggfunc='mean')
seasonal_data = seasonal_data[['Spring', 'Summer', 'Fall', 'Winter']]
sns.heatmap(seasonal_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
ax.set_title('Duration Days - Seasonal Patterns')
ax.set_xlabel('Season')
ax.set_ylabel('Ecoregion')

plt.tight_layout()
plt.savefig('results/monthly_seasonal_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Annual trends
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Annual trends for Mean_SFD_Flow_Percentile
ax = axes[0]
for ecoregion in df['AGGECOREGION'].unique():
    if pd.isna(ecoregion):
        continue
    eco_data = df[df['AGGECOREGION'] == ecoregion]
    annual_means = eco_data.groupby('Year')['Mean_SFD_Flow_Percentile'].mean()
    
    if len(annual_means) > 1:
        ax.plot(annual_means.index, annual_means.values, marker='o', label=ecoregion, alpha=0.7, linewidth=2)

ax.set_title('Mean SFD Flow Percentile - Annual Trends (1981-2024)')
ax.set_xlabel('Year')
ax.set_ylabel('Mean SFD Flow Percentile')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

# Annual trends for Duration_Days
ax = axes[1]
for ecoregion in df['AGGECOREGION'].unique():
    if pd.isna(ecoregion):
        continue
    eco_data = df[df['AGGECOREGION'] == ecoregion]
    annual_means = eco_data.groupby('Year')['Duration_Days'].mean()
    
    if len(annual_means) > 1:
        ax.plot(annual_means.index, annual_means.values, marker='o', label=ecoregion, alpha=0.7, linewidth=2)

ax.set_title('Duration Days - Annual Trends (1981-2024)')
ax.set_xlabel('Year')
ax.set_ylabel('Duration (Days)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/annual_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Ecoregion comparisons
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Boxplot for Mean_SFD_Flow_Percentile by ecoregion
ax = axes[0, 0]
sns.boxplot(data=df, x='AGGECOREGION', y='Mean_SFD_Flow_Percentile', ax=ax)
ax.set_title('Mean SFD Flow Percentile by Ecoregion')
ax.tick_params(axis='x', rotation=45)

# Boxplot for Duration_Days by ecoregion
ax = axes[0, 1]
sns.boxplot(data=df, x='AGGECOREGION', y='Duration_Days', ax=ax)
ax.set_title('Duration Days by Ecoregion')
ax.tick_params(axis='x', rotation=45)

# Monthly distribution violin plot
ax = axes[1, 0]
sns.violinplot(data=df, x='Month', y='Mean_SFD_Flow_Percentile', ax=ax)
ax.set_title('Mean SFD Flow Percentile - Monthly Distribution')
ax.set_xticklabels([month_names[m] for m in range(1, 13)])

# Seasonal distribution violin plot
ax = axes[1, 1]
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
sns.violinplot(data=df, x='Season', y='Duration_Days', order=season_order, ax=ax)
ax.set_title('Duration Days - Seasonal Distribution')

plt.tight_layout()
plt.savefig('results/ecoregion_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Event frequency analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Event frequency by month and ecoregion
ax = axes[0]
event_counts = df.groupby(['AGGECOREGION', 'Month']).size().unstack(fill_value=0)
sns.heatmap(event_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('SFD Event Frequency by Month')
ax.set_xlabel('Month')
ax.set_ylabel('Ecoregion')
ax.set_xticklabels([month_names[m] for m in range(1, 13)])

# Event frequency by season and ecoregion
ax = axes[1]
event_counts_seasonal = df.groupby(['AGGECOREGION', 'Season']).size().unstack(fill_value=0)
event_counts_seasonal = event_counts_seasonal[['Spring', 'Summer', 'Fall', 'Winter']]
sns.heatmap(event_counts_seasonal, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
ax.set_title('SFD Event Frequency by Season')
ax.set_xlabel('Season')
ax.set_ylabel('Ecoregion')

plt.tight_layout()
plt.savefig('results/event_frequency.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations created successfully!")
print("Files saved:")
print("  - results/monthly_seasonal_patterns.png")
print("  - results/annual_trends.png")
print("  - results/ecoregion_distributions.png")
print("  - results/event_frequency.png")