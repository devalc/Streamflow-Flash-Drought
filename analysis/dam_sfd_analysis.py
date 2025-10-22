import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Loading SFD events with static attributes for dam analysis...")
df = pd.read_parquet('../data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Onset_Time'].min()} to {df['Onset_Time'].max()}")

# Check available dam-related columns
dam_columns = ['RAW_DIS_NEAREST_DAM', 'RAW_AVG_DIS_ALLDAMS', 'RAW_DIS_NEAREST_MAJ_DAM', 
               'RAW_AVG_DIS_ALL_MAJ_DAMS', 'FRESHW_WITHDRAWAL']
available_dam_cols = [col for col in dam_columns if col in df.columns]
print(f"Available dam-related columns: {available_dam_cols}")

# Convert Onset_Time to datetime and extract temporal components
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
df['Year'] = df['Onset_Time'].dt.year
df['Month'] = df['Onset_Time'].dt.month
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Define summer monsoon season (June-August)
df['Is_Summer_Monsoon'] = df['Month'].isin([6, 7, 8])

# Check for missing values in dam columns
print("\nMissing values in dam-related columns:")
for col in available_dam_cols:
    missing_pct = (df[col].isna().sum() / len(df)) * 100
    print(f"{col}: {missing_pct:.1f}% missing values")

def classify_dam_presence(df, dam_distance_cols):
    """
    Classify catchments based on dam presence/absence using distance thresholds
    """
    print(f"\n{'='*60}")
    print("CLASSIFYING DAM PRESENCE/ABSENCE")
    print(f"{'='*60}")
    
    results = {}
    
    for col in dam_distance_cols:
        if col in df.columns:
            # Handle negative values and missing data (-999 likely means no dam data)
            # Replace -999 and other negative values with NaN for proper handling
            df[col] = df[col].replace(-999, np.nan)
            df[col] = df[col].where(df[col] >= 0, np.nan)
            
            # Remove missing values for analysis
            valid_data = df[col].dropna()
            
            if len(valid_data) == 0:
                print(f"\n{col}: No valid data available")
                continue
                
            print(f"\n{col} Statistics:")
            print(f"  Valid records: {len(valid_data):,} ({len(valid_data)/len(df)*100:.1f}%)")
            print(f"  Mean distance: {valid_data.mean():.1f} km")
            print(f"  Median distance: {valid_data.median():.1f} km")
            print(f"  Min distance: {valid_data.min():.1f} km")
            print(f"  Max distance: {valid_data.max():.1f} km")
            
            # Create dam influence categories for valid data only
            df[f'{col}_Category'] = pd.cut(df[col], 
                                         bins=[0, 10, 50, np.inf],
                                         labels=['High_Dam_Influence', 'Moderate_Dam_Influence', 'Low_Dam_Influence'],
                                         include_lowest=True)
            
            # Binary classification: presence vs absence
            # Use median as threshold for binary classification
            threshold = valid_data.median()
            df[f'{col}_Binary'] = df[col].apply(lambda x: 'Dam_Present' if pd.notna(x) and x <= threshold 
                                              else 'Dam_Absent' if pd.notna(x) else np.nan)
            
            results[col] = {
                'threshold': threshold,
                'categories': df[f'{col}_Category'].value_counts(),
                'binary': df[f'{col}_Binary'].value_counts()
            }
            
            print(f"  Threshold for presence/absence: {threshold:.1f} km")
            print(f"  Dam influence categories:")
            for cat, count in results[col]['categories'].items():
                print(f"    {cat}: {count} ({count/len(df)*100:.1f}%)")
    
    return results

def analyze_sfd_frequency_by_dam_presence(df, dam_cols):
    """
    Analyze SFD frequency in presence/absence of dams during summer monsoon
    """
    print(f"\n{'='*60}")
    print("SFD FREQUENCY ANALYSIS BY DAM PRESENCE")
    print(f"{'='*60}")
    
    # Focus on summer monsoon season
    summer_df = df[df['Is_Summer_Monsoon'] == True].copy()
    
    frequency_results = {}
    
    for col in dam_cols:
        if f'{col}_Binary' in df.columns:
            print(f"\nAnalyzing {col}:")
            print("-" * 40)
            
            # Calculate SFD frequency by catchment and dam presence
            catchment_freq = summer_df.groupby(['Station_ID', f'{col}_Binary']).size().reset_index(name='SFD_Count')
            
            # Get total years for frequency calculation
            total_years = summer_df['Year'].nunique()
            catchment_freq['SFD_Frequency'] = catchment_freq['SFD_Count'] / total_years
            
            # Statistical comparison
            dam_present = catchment_freq[catchment_freq[f'{col}_Binary'] == 'Dam_Present']['SFD_Frequency']
            dam_absent = catchment_freq[catchment_freq[f'{col}_Binary'] == 'Dam_Absent']['SFD_Frequency']
            
            # Remove any NaN values
            dam_present = dam_present.dropna()
            dam_absent = dam_absent.dropna()
            
            if len(dam_present) > 0 and len(dam_absent) > 0:
                # Statistical test
                stat, p_value = stats.mannwhitneyu(dam_present, dam_absent, alternative='two-sided')
                
                print(f"  Dam Present - Mean frequency: {dam_present.mean():.2f} ± {dam_present.std():.2f}")
                print(f"  Dam Absent - Mean frequency: {dam_absent.mean():.2f} ± {dam_absent.std():.2f}")
                print(f"  Statistical test (Mann-Whitney U): p = {p_value:.4f}")
                
                if p_value < 0.05:
                    if dam_present.mean() > dam_absent.mean():
                        print(f"  Result: Significantly HIGHER frequency with dams present")
                    else:
                        print(f"  Result: Significantly LOWER frequency with dams present")
                else:
                    print(f"  Result: No significant difference")
                
                frequency_results[col] = {
                    'dam_present_freq': dam_present,
                    'dam_absent_freq': dam_absent,
                    'p_value': p_value,
                    'effect_size': (dam_present.mean() - dam_absent.mean()) / np.sqrt((dam_present.var() + dam_absent.var()) / 2)
                }
    
    return frequency_results

def analyze_sfd_severity_by_dam_presence(df, dam_cols):
    """
    Analyze SFD severity (mean flow percentile and duration) by dam presence
    """
    print(f"\n{'='*60}")
    print("SFD SEVERITY ANALYSIS BY DAM PRESENCE")
    print(f"{'='*60}")
    
    # Focus on summer monsoon season
    summer_df = df[df['Is_Summer_Monsoon'] == True].copy()
    
    severity_results = {}
    severity_vars = ['Mean_SFD_Flow_Percentile', 'Duration_Days']
    
    for col in dam_cols:
        if f'{col}_Binary' in df.columns:
            print(f"\nAnalyzing {col}:")
            print("-" * 40)
            
            severity_results[col] = {}
            
            for var in severity_vars:
                if var in summer_df.columns:
                    # Compare severity between dam presence/absence
                    dam_present = summer_df[summer_df[f'{col}_Binary'] == 'Dam_Present'][var].dropna()
                    dam_absent = summer_df[summer_df[f'{col}_Binary'] == 'Dam_Absent'][var].dropna()
                    
                    if len(dam_present) > 0 and len(dam_absent) > 0:
                        # Statistical test
                        stat, p_value = stats.mannwhitneyu(dam_present, dam_absent, alternative='two-sided')
                        
                        print(f"\n  {var}:")
                        print(f"    Dam Present - Mean: {dam_present.mean():.2f} ± {dam_present.std():.2f}")
                        print(f"    Dam Absent - Mean: {dam_absent.mean():.2f} ± {dam_absent.std():.2f}")
                        print(f"    Statistical test: p = {p_value:.4f}")
                        
                        if p_value < 0.05:
                            if dam_present.mean() > dam_absent.mean():
                                print(f"    Result: Significantly HIGHER {var.lower()} with dams present")
                            else:
                                print(f"    Result: Significantly LOWER {var.lower()} with dams present")
                        else:
                            print(f"    Result: No significant difference")
                        
                        severity_results[col][var] = {
                            'dam_present': dam_present,
                            'dam_absent': dam_absent,
                            'p_value': p_value,
                            'effect_size': (dam_present.mean() - dam_absent.mean()) / np.sqrt((dam_present.var() + dam_absent.var()) / 2)
                        }
    
    return severity_results

def analyze_ecoregion_dam_interactions(df, dam_cols):
    """
    Analyze how dam effects vary across different ecoregions
    """
    print(f"\n{'='*60}")
    print("ECOREGION-DAM INTERACTION ANALYSIS")
    print(f"{'='*60}")
    
    summer_df = df[df['Is_Summer_Monsoon'] == True].copy()
    
    interaction_results = {}
    
    for col in dam_cols:
        if f'{col}_Binary' in df.columns:
            print(f"\nAnalyzing {col} across ecoregions:")
            print("-" * 50)
            
            interaction_results[col] = {}
            
            for ecoregion in summer_df['AGGECOREGION'].unique():
                if pd.isna(ecoregion):
                    continue
                
                eco_data = summer_df[summer_df['AGGECOREGION'] == ecoregion]
                
                if len(eco_data) > 10:  # Minimum sample size
                    # Calculate frequency by dam presence in this ecoregion
                    freq_by_dam = eco_data.groupby([f'{col}_Binary', 'Station_ID']).size().reset_index(name='Count')
                    freq_summary = freq_by_dam.groupby(f'{col}_Binary')['Count'].agg(['mean', 'std', 'count'])
                    
                    # Calculate severity by dam presence
                    severity_by_dam = eco_data.groupby(f'{col}_Binary')[['Mean_SFD_Flow_Percentile', 'Duration_Days']].agg(['mean', 'std'])
                    
                    interaction_results[col][ecoregion] = {
                        'frequency': freq_summary,
                        'severity': severity_by_dam,
                        'sample_size': len(eco_data)
                    }
                    
                    print(f"\n  {ecoregion} (n={len(eco_data)}):")
                    if 'Dam_Present' in freq_summary.index and 'Dam_Absent' in freq_summary.index:
                        dam_freq = freq_summary.loc['Dam_Present', 'mean']
                        no_dam_freq = freq_summary.loc['Dam_Absent', 'mean']
                        print(f"    Avg SFD frequency - Dam Present: {dam_freq:.2f}, Dam Absent: {no_dam_freq:.2f}")
                        
                        if 'Dam_Present' in severity_by_dam.index and 'Dam_Absent' in severity_by_dam.index:
                            dam_flow = severity_by_dam.loc['Dam_Present', ('Mean_SFD_Flow_Percentile', 'mean')]
                            no_dam_flow = severity_by_dam.loc['Dam_Absent', ('Mean_SFD_Flow_Percentile', 'mean')]
                            print(f"    Avg flow percentile - Dam Present: {dam_flow:.1f}, Dam Absent: {no_dam_flow:.1f}")
    
    return interaction_results

def create_dam_sfd_visualizations(df, dam_cols, frequency_results, severity_results):
    """
    Create comprehensive visualizations for dam-SFD relationships
    """
    print(f"\n{'='*60}")
    print("CREATING DAM-SFD VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create main figure
    fig = plt.figure(figsize=(20, 16))
    
    # Summer monsoon data
    summer_df = df[df['Is_Summer_Monsoon'] == True].copy()
    
    plot_idx = 1
    
    for col in dam_cols:
        if f'{col}_Binary' in df.columns and col in frequency_results:
            
            # 1. SFD Frequency Distribution by Dam Presence
            ax = plt.subplot(4, 4, plot_idx)
            
            freq_data = []
            labels = []
            
            if col in frequency_results:
                freq_data.extend([frequency_results[col]['dam_present_freq'], 
                                frequency_results[col]['dam_absent_freq']])
                labels.extend(['Dam Present', 'Dam Absent'])
                
                ax.boxplot(freq_data, labels=labels)
                ax.set_title(f'SFD Frequency Distribution\n{col.replace("RAW_", "").replace("_", " ")}')
                ax.set_ylabel('SFD Frequency (events/year)')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
            
            # 2. SFD Severity (Flow Percentile) by Dam Presence
            ax = plt.subplot(4, 4, plot_idx)
            
            if col in severity_results and 'Mean_SFD_Flow_Percentile' in severity_results[col]:
                sev_data = [severity_results[col]['Mean_SFD_Flow_Percentile']['dam_present'],
                           severity_results[col]['Mean_SFD_Flow_Percentile']['dam_absent']]
                
                ax.boxplot(sev_data, labels=['Dam Present', 'Dam Absent'])
                ax.set_title(f'SFD Flow Percentile\n{col.replace("RAW_", "").replace("_", " ")}')
                ax.set_ylabel('Mean SFD Flow Percentile')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
            
            # 3. SFD Duration by Dam Presence
            ax = plt.subplot(4, 4, plot_idx)
            
            if col in severity_results and 'Duration_Days' in severity_results[col]:
                dur_data = [severity_results[col]['Duration_Days']['dam_present'],
                           severity_results[col]['Duration_Days']['dam_absent']]
                
                ax.boxplot(dur_data, labels=['Dam Present', 'Dam Absent'])
                ax.set_title(f'SFD Duration\n{col.replace("RAW_", "").replace("_", " ")}')
                ax.set_ylabel('Duration (days)')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
            
            # 4. Spatial Distribution - Scatter plot of dam distance vs SFD characteristics
            ax = plt.subplot(4, 4, plot_idx)
            
            # Create scatter plot of dam distance vs mean flow percentile
            valid_data = summer_df[[col, 'Mean_SFD_Flow_Percentile']].dropna()
            
            if len(valid_data) > 0:
                scatter = ax.scatter(valid_data[col], valid_data['Mean_SFD_Flow_Percentile'], 
                                   alpha=0.6, s=30)
                
                # Add trend line
                if len(valid_data) > 10:
                    z = np.polyfit(valid_data[col], valid_data['Mean_SFD_Flow_Percentile'], 1)
                    p = np.poly1d(z)
                    ax.plot(valid_data[col].sort_values(), p(valid_data[col].sort_values()), 
                           "r--", alpha=0.8, linewidth=2)
                    
                    # Calculate correlation
                    corr, p_val = stats.pearsonr(valid_data[col], valid_data['Mean_SFD_Flow_Percentile'])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_xlabel(f'{col.replace("RAW_", "").replace("_", " ")} (km)')
                ax.set_ylabel('Mean SFD Flow Percentile')
                ax.set_title(f'Dam Distance vs SFD Severity')
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # 5. Ecoregion comparison
    if plot_idx <= 16:
        ax = plt.subplot(4, 4, plot_idx)
        
        # Create heatmap of SFD frequency by ecoregion and dam presence
        if len(dam_cols) > 0 and f'{dam_cols[0]}_Binary' in summer_df.columns:
            freq_pivot = summer_df.groupby(['AGGECOREGION', f'{dam_cols[0]}_Binary']).size().unstack(fill_value=0)
            
            if not freq_pivot.empty:
                sns.heatmap(freq_pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
                ax.set_title('SFD Count by Ecoregion\nand Dam Presence')
                ax.set_xlabel('Dam Presence')
                ax.set_ylabel('Ecoregion')
        
        plot_idx += 1
    
    # 6. Monthly pattern comparison
    if plot_idx <= 16:
        ax = plt.subplot(4, 4, plot_idx)
        
        if len(dam_cols) > 0 and f'{dam_cols[0]}_Binary' in df.columns:
            monthly_pattern = df.groupby(['Month', f'{dam_cols[0]}_Binary']).size().unstack(fill_value=0)
            
            if not monthly_pattern.empty:
                monthly_pattern.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title('Monthly SFD Pattern\nby Dam Presence')
                ax.set_xlabel('Month')
                ax.set_ylabel('SFD Count')
                ax.legend(title='Dam Status')
                ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('../results/dam_sfd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_spatial_distribution_maps(df, dam_cols):
    """
    Create spatial distribution analysis focusing on summer monsoon season
    """
    print(f"\n{'='*60}")
    print("SPATIAL DISTRIBUTION ANALYSIS - SUMMER MONSOON")
    print(f"{'='*60}")
    
    summer_df = df[df['Is_Summer_Monsoon'] == True].copy()
    
    spatial_results = {}
    
    for col in dam_cols:
        if f'{col}_Binary' in df.columns:
            print(f"\nSpatial analysis for {col}:")
            print("-" * 40)
            
            # Calculate catchment-level statistics
            catchment_stats = summer_df.groupby(['Station_ID', f'{col}_Binary']).agg({
                'Mean_SFD_Flow_Percentile': ['mean', 'count'],
                'Duration_Days': 'mean',
                'AGGECOREGION': 'first'
            }).round(2)
            
            catchment_stats.columns = ['_'.join(col).strip() for col in catchment_stats.columns]
            catchment_stats = catchment_stats.reset_index()
            
            # Rename columns for clarity
            catchment_stats.rename(columns={
                'Mean_SFD_Flow_Percentile_mean': 'Mean_Severity',
                'Mean_SFD_Flow_Percentile_count': 'SFD_Frequency',
                'Duration_Days_mean': 'Mean_Duration',
                'AGGECOREGION_first': 'Ecoregion'
            }, inplace=True)
            
            spatial_results[col] = catchment_stats
            
            # Summary statistics
            dam_present_stats = catchment_stats[catchment_stats[f'{col}_Binary'] == 'Dam_Present']
            dam_absent_stats = catchment_stats[catchment_stats[f'{col}_Binary'] == 'Dam_Absent']
            
            print(f"  Catchments with dams present: {len(dam_present_stats)}")
            print(f"  Catchments with dams absent: {len(dam_absent_stats)}")
            
            if len(dam_present_stats) > 0 and len(dam_absent_stats) > 0:
                print(f"  Mean SFD frequency - Dam Present: {dam_present_stats['SFD_Frequency'].mean():.2f}")
                print(f"  Mean SFD frequency - Dam Absent: {dam_absent_stats['SFD_Frequency'].mean():.2f}")
                print(f"  Mean severity - Dam Present: {dam_present_stats['Mean_Severity'].mean():.2f}")
                print(f"  Mean severity - Dam Absent: {dam_absent_stats['Mean_Severity'].mean():.2f}")
    
    return spatial_results

# Main execution
if __name__ == "__main__":
    
    print(f"\n{'='*80}")
    print("DAM-STREAMFLOW FLASH DROUGHT ANALYSIS")
    print(f"{'='*80}")
    
    # Filter to available dam columns
    available_dam_distance_cols = [col for col in ['RAW_DIS_NEAREST_DAM', 'RAW_DIS_NEAREST_MAJ_DAM'] 
                                  if col in df.columns]
    
    if len(available_dam_distance_cols) == 0:
        print("ERROR: No dam distance columns found in dataset!")
        print("Available columns:", df.columns.tolist())
        exit()
    
    # Step 1: Classify dam presence/absence
    dam_classification = classify_dam_presence(df, available_dam_distance_cols)
    
    # Step 2: Analyze SFD frequency by dam presence
    frequency_results = analyze_sfd_frequency_by_dam_presence(df, available_dam_distance_cols)
    
    # Step 3: Analyze SFD severity by dam presence  
    severity_results = analyze_sfd_severity_by_dam_presence(df, available_dam_distance_cols)
    
    # Step 4: Analyze ecoregion-dam interactions
    interaction_results = analyze_ecoregion_dam_interactions(df, available_dam_distance_cols)
    
    # Step 5: Spatial distribution analysis
    spatial_results = create_spatial_distribution_maps(df, available_dam_distance_cols)
    
    # Step 6: Create visualizations
    create_dam_sfd_visualizations(df, available_dam_distance_cols, frequency_results, severity_results)
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save dam classification results
    for col in available_dam_distance_cols:
        if f'{col}_Binary' in df.columns:
            dam_summary = df.groupby([f'{col}_Binary', 'AGGECOREGION']).agg({
                'Mean_SFD_Flow_Percentile': ['mean', 'std', 'count'],
                'Duration_Days': ['mean', 'std'],
                col: ['mean', 'median']
            }).round(2)
            
            dam_summary.to_csv(f'../results/dam_analysis_{col.lower()}.csv')
    
    # Save spatial results
    for col, spatial_data in spatial_results.items():
        spatial_data.to_csv(f'../results/spatial_dam_analysis_{col.lower()}.csv', index=False)
    
    # Create comprehensive summary report
    with open('../results/DAM_SFD_ANALYSIS_SUMMARY.md', 'w') as f:
        f.write("# Dam-Streamflow Flash Drought Analysis Summary\n\n")
        
        f.write("## Research Question\n")
        f.write("How do dams influence the frequency and severity of streamflow flash droughts (SFDs), ")
        f.write("particularly during the summer monsoon season?\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- **Total SFD Events**: {len(df):,}\n")
        f.write(f"- **Summer Monsoon Events**: {len(df[df['Is_Summer_Monsoon']==True]):,}\n")
        f.write(f"- **Date Range**: {df['Onset_Time'].min().strftime('%Y-%m-%d')} to {df['Onset_Time'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Unique Catchments**: {df['Station_ID'].nunique():,}\n")
        f.write(f"- **Ecoregions**: {df['AGGECOREGION'].nunique()}\n\n")
        
        f.write("## Dam Variables Analyzed\n")
        for col in available_dam_distance_cols:
            if col in dam_classification:
                threshold = dam_classification[col]['threshold']
                f.write(f"- **{col.replace('RAW_', '').replace('_', ' ')}**: Threshold = {threshold:.1f} km\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### SFD Frequency Analysis\n")
        for col in available_dam_distance_cols:
            if col in frequency_results:
                dam_freq = frequency_results[col]['dam_present_freq'].mean()
                no_dam_freq = frequency_results[col]['dam_absent_freq'].mean()
                p_val = frequency_results[col]['p_value']
                
                f.write(f"**{col.replace('RAW_', '').replace('_', ' ')}:**\n")
                f.write(f"- Dam Present: {dam_freq:.2f} events/year\n")
                f.write(f"- Dam Absent: {no_dam_freq:.2f} events/year\n")
                f.write(f"- Statistical significance: p = {p_val:.4f}\n")
                
                if p_val < 0.05:
                    if dam_freq > no_dam_freq:
                        f.write("- **Result**: Significantly HIGHER frequency with dams present\n\n")
                    else:
                        f.write("- **Result**: Significantly LOWER frequency with dams present\n\n")
                else:
                    f.write("- **Result**: No significant difference\n\n")
        
        f.write("### SFD Severity Analysis\n")
        for col in available_dam_distance_cols:
            if col in severity_results:
                f.write(f"**{col.replace('RAW_', '').replace('_', ' ')}:**\n")
                
                for var in ['Mean_SFD_Flow_Percentile', 'Duration_Days']:
                    if var in severity_results[col]:
                        dam_sev = severity_results[col][var]['dam_present'].mean()
                        no_dam_sev = severity_results[col][var]['dam_absent'].mean()
                        p_val = severity_results[col][var]['p_value']
                        
                        f.write(f"- {var.replace('_', ' ')}: Dam Present = {dam_sev:.2f}, Dam Absent = {no_dam_sev:.2f} (p = {p_val:.4f})\n")
                f.write("\n")
        
        f.write("## Files Generated\n")
        f.write("- `dam_sfd_analysis.png` - Comprehensive visualizations\n")
        f.write("- `dam_analysis_*.csv` - Dam classification and statistics by ecoregion\n")
        f.write("- `spatial_dam_analysis_*.csv` - Spatial analysis results\n")
        f.write("- `DAM_SFD_ANALYSIS_SUMMARY.md` - This summary report\n\n")
        
        f.write("## Interpretation\n")
        f.write("This analysis provides insights into how dam infrastructure affects streamflow flash drought ")
        f.write("patterns across different ecoregions. The results can inform water management strategies ")
        f.write("and drought mitigation planning, particularly during critical summer monsoon periods.\n")
    
    print("\nDam-SFD analysis complete!")
    print("Files saved:")
    print("  - results/dam_sfd_analysis.png")
    print("  - results/dam_analysis_*.csv")
    print("  - results/spatial_dam_analysis_*.csv") 
    print("  - results/DAM_SFD_ANALYSIS_SUMMARY.md")