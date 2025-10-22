import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("Loading SFD events with static attributes for ENSO analysis...")
df = pd.read_parquet('../data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Onset_Time'].min()} to {df['Onset_Time'].max()}")

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

# Handle dam distance data (remove -999 values)
dam_columns = ['RAW_DIS_NEAREST_DAM', 'RAW_DIS_NEAREST_MAJ_DAM']
for col in dam_columns:
    if col in df.columns:
        df[col] = df[col].replace(-999, np.nan)
        df[col] = df[col].where(df[col] >= 0, np.nan)

def define_enso_years():
    """
    Define El Niño and La Niña years based on historical records
    Using standard NOAA ONI (Oceanic Niño Index) classifications
    """
    
    # El Niño years (ONI >= +0.5°C for 5+ consecutive months)
    # Based on NOAA Climate Prediction Center data
    el_nino_years = [
        1982, 1983, 1986, 1987, 1991, 1992, 1994, 1995, 1997, 1998, 
        2002, 2003, 2004, 2005, 2006, 2007, 2009, 2010, 2014, 2015, 2016, 2018, 2019, 2023
    ]
    
    # La Niña years (ONI <= -0.5°C for 5+ consecutive months)
    la_nina_years = [
        1984, 1985, 1988, 1989, 1995, 1996, 1998, 1999, 2000, 2001, 
        2007, 2008, 2010, 2011, 2016, 2017, 2020, 2021, 2022, 2023, 2024
    ]
    
    # Neutral years (neither El Niño nor La Niña)
    all_years = set(range(1981, 2025))
    enso_years = set(el_nino_years + la_nina_years)
    neutral_years = sorted(list(all_years - enso_years))
    
    print(f"ENSO Year Classification:")
    print(f"  El Niño years ({len(el_nino_years)}): {sorted(el_nino_years)}")
    print(f"  La Niña years ({len(la_nina_years)}): {sorted(la_nina_years)}")
    print(f"  Neutral years ({len(neutral_years)}): {neutral_years}")
    
    return el_nino_years, la_nina_years, neutral_years

def classify_enso_phase(df):
    """
    Classify each SFD event by ENSO phase
    """
    print(f"\n{'='*60}")
    print("CLASSIFYING ENSO PHASES")
    print(f"{'='*60}")
    
    el_nino_years, la_nina_years, neutral_years = define_enso_years()
    
    # Create ENSO classification
    def get_enso_phase(year):
        if year in el_nino_years:
            return 'El_Nino'
        elif year in la_nina_years:
            return 'La_Nina'
        else:
            return 'Neutral'
    
    df['ENSO_Phase'] = df['Year'].apply(get_enso_phase)
    
    # Print summary
    enso_counts = df['ENSO_Phase'].value_counts()
    print(f"ENSO Phase Distribution:")
    for phase, count in enso_counts.items():
        print(f"  {phase}: {count:,} events ({count/len(df)*100:.1f}%)")
    
    # Years covered
    print(f"\nYears by ENSO Phase:")
    print(f"  El Niño years: {sorted(el_nino_years)}")
    print(f"  La Niña years: {sorted(la_nina_years)}")
    print(f"  Neutral years: {len(neutral_years)} years")
    
    return df

def classify_dam_presence_for_enso(df):
    """
    Classify dam presence for ENSO analysis
    """
    print(f"\n{'='*60}")
    print("CLASSIFYING DAM PRESENCE FOR ENSO ANALYSIS")
    print(f"{'='*60}")
    
    for col in dam_columns:
        if col in df.columns:
            # Use median as threshold for binary classification
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                threshold = valid_data.median()
                df[f'{col}_Binary'] = df[col].apply(
                    lambda x: 'Dam_Present' if pd.notna(x) and x <= threshold 
                    else 'Dam_Absent' if pd.notna(x) else np.nan
                )
                
                print(f"{col}:")
                print(f"  Threshold: {threshold:.1f} km")
                print(f"  Dam Present: {(df[f'{col}_Binary'] == 'Dam_Present').sum():,}")
                print(f"  Dam Absent: {(df[f'{col}_Binary'] == 'Dam_Absent').sum():,}")
    
    return df

def analyze_sfd_frequency_by_enso_season(df):
    """
    Analyze SFD frequency by ENSO phase, season, and dam presence across ecoregions
    """
    print(f"\n{'='*60}")
    print("SFD FREQUENCY ANALYSIS BY ENSO PHASE AND SEASON")
    print(f"{'='*60}")
    
    results = {}
    
    for dam_col in dam_columns:
        if f'{dam_col}_Binary' in df.columns:
            print(f"\nAnalyzing {dam_col}:")
            print("-" * 50)
            
            # Calculate frequency by catchment, ENSO phase, season, and dam presence
            freq_analysis = df.groupby([
                'Station_ID', 'AGGECOREGION', 'ENSO_Phase', 'Season', f'{dam_col}_Binary'
            ]).size().reset_index(name='SFD_Count')
            
            # Calculate total years for each ENSO phase for frequency calculation
            enso_year_counts = df.groupby('ENSO_Phase')['Year'].nunique()
            
            # Add frequency calculation
            freq_analysis['Years_in_Phase'] = freq_analysis['ENSO_Phase'].map(enso_year_counts)
            freq_analysis['SFD_Frequency'] = freq_analysis['SFD_Count'] / freq_analysis['Years_in_Phase']
            
            # Summary statistics by ENSO phase, season, and dam presence
            summary_stats = freq_analysis.groupby([
                'ENSO_Phase', 'Season', f'{dam_col}_Binary'
            ])['SFD_Frequency'].agg(['mean', 'std', 'count', 'median']).round(3)
            
            results[dam_col] = {
                'detailed': freq_analysis,
                'summary': summary_stats
            }
            
            print(f"\nSummary Statistics for {dam_col}:")
            print(summary_stats)
            
            # Statistical comparisons
            print(f"\nStatistical Comparisons:")
            for enso_phase in ['El_Nino', 'La_Nina', 'Neutral']:
                for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                    phase_season_data = freq_analysis[
                        (freq_analysis['ENSO_Phase'] == enso_phase) & 
                        (freq_analysis['Season'] == season)
                    ]
                    
                    if len(phase_season_data) > 0:
                        dam_present = phase_season_data[
                            phase_season_data[f'{dam_col}_Binary'] == 'Dam_Present'
                        ]['SFD_Frequency']
                        dam_absent = phase_season_data[
                            phase_season_data[f'{dam_col}_Binary'] == 'Dam_Absent'
                        ]['SFD_Frequency']
                        
                        if len(dam_present) > 5 and len(dam_absent) > 5:
                            stat, p_value = stats.mannwhitneyu(
                                dam_present, dam_absent, alternative='two-sided'
                            )
                            
                            print(f"  {enso_phase} {season}:")
                            print(f"    Dam Present: {dam_present.mean():.3f} ± {dam_present.std():.3f}")
                            print(f"    Dam Absent: {dam_absent.mean():.3f} ± {dam_absent.std():.3f}")
                            print(f"    p-value: {p_value:.4f}")
    
    return results

def analyze_ecoregion_enso_patterns(df):
    """
    Analyze how different ecoregions respond to ENSO phases
    """
    print(f"\n{'='*60}")
    print("ECOREGION-ENSO PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    ecoregion_results = {}
    
    for dam_col in dam_columns:
        if f'{dam_col}_Binary' in df.columns:
            print(f"\nAnalyzing {dam_col} across ecoregions:")
            print("-" * 50)
            
            # Calculate mean frequency by ecoregion, ENSO phase, and dam presence
            eco_enso_summary = df.groupby([
                'AGGECOREGION', 'ENSO_Phase', f'{dam_col}_Binary'
            ]).agg({
                'Station_ID': 'nunique',  # Number of unique catchments
                'Duration_Days': 'mean',
                'Mean_SFD_Flow_Percentile': 'mean'
            }).round(2)
            
            eco_enso_summary.columns = ['Num_Catchments', 'Mean_Duration', 'Mean_Flow_Percentile']
            
            ecoregion_results[dam_col] = eco_enso_summary
            
            print(f"\nEcoregion Summary for {dam_col}:")
            print(eco_enso_summary)
    
    return ecoregion_results

def create_enso_sfd_visualizations(df, frequency_results, ecoregion_results):
    """
    Create comprehensive visualizations for ENSO-SFD analysis
    """
    print(f"\n{'='*60}")
    print("CREATING ENSO-SFD VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # Create main figure with multiple subplots
    fig = plt.figure(figsize=(24, 20))
    
    # Color scheme for ENSO phases
    enso_colors = {'El_Nino': '#d62728', 'La_Nina': '#1f77b4', 'Neutral': '#2ca02c'}
    
    plot_idx = 1
    
    # 1. SFD Frequency by ENSO Phase and Season (Dam Present vs Absent)
    for i, dam_col in enumerate(dam_columns):
        if dam_col in frequency_results:
            ax = plt.subplot(4, 4, plot_idx)
            
            # Prepare data for plotting
            plot_data = frequency_results[dam_col]['detailed']
            
            # Create grouped bar plot
            pivot_data = plot_data.groupby([
                'ENSO_Phase', 'Season', f'{dam_col}_Binary'
            ])['SFD_Frequency'].mean().unstack(f'{dam_col}_Binary')
            
            if not pivot_data.empty:
                pivot_data.plot(kind='bar', ax=ax, width=0.8, 
                               color=['lightcoral', 'lightblue'])
                ax.set_title(f'SFD Frequency by ENSO & Season\n{dam_col.replace("RAW_", "").replace("_", " ")}')
                ax.set_xlabel('ENSO Phase - Season')
                ax.set_ylabel('Mean SFD Frequency')
                ax.legend(title='Dam Status', bbox_to_anchor=(1.05, 1))
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # 2. Seasonal patterns by ENSO phase
    ax = plt.subplot(4, 4, plot_idx)
    
    seasonal_enso = df.groupby(['ENSO_Phase', 'Season']).size().unstack('Season')
    seasonal_enso = seasonal_enso[['Spring', 'Summer', 'Fall', 'Winter']]  # Order seasons
    
    seasonal_enso.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('SFD Count by ENSO Phase and Season')
    ax.set_xlabel('ENSO Phase')
    ax.set_ylabel('SFD Count')
    ax.legend(title='Season', bbox_to_anchor=(1.05, 1))
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3)
    
    plot_idx += 1
    
    # 3. Ecoregion response to ENSO phases
    ax = plt.subplot(4, 4, plot_idx)
    
    eco_enso_counts = df.groupby(['AGGECOREGION', 'ENSO_Phase']).size().unstack('ENSO_Phase')
    
    if not eco_enso_counts.empty:
        sns.heatmap(eco_enso_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
        ax.set_title('SFD Count by Ecoregion and ENSO Phase')
        ax.set_xlabel('ENSO Phase')
        ax.set_ylabel('Ecoregion')
    
    plot_idx += 1
    
    # 4. Dam presence effect across ENSO phases
    if len(dam_columns) > 0 and f'{dam_columns[0]}_Binary' in df.columns:
        ax = plt.subplot(4, 4, plot_idx)
        
        enso_dam_freq = df.groupby([
            'ENSO_Phase', f'{dam_columns[0]}_Binary'
        ]).size().unstack(f'{dam_columns[0]}_Binary')
        
        if not enso_dam_freq.empty:
            enso_dam_freq.plot(kind='bar', ax=ax, width=0.8,
                              color=['lightcoral', 'lightblue'])
            ax.set_title('SFD Count by ENSO Phase and Dam Presence')
            ax.set_xlabel('ENSO Phase')
            ax.set_ylabel('SFD Count')
            ax.legend(title='Dam Status')
            ax.tick_params(axis='x', rotation=0)
            ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 5-8. Individual ecoregion analysis
    ecoregions = df['AGGECOREGION'].unique()[:4]  # Show top 4 ecoregions
    
    for eco in ecoregions:
        if plot_idx > 16:
            break
            
        ax = plt.subplot(4, 4, plot_idx)
        
        eco_data = df[df['AGGECOREGION'] == eco]
        if len(eco_data) > 0:
            eco_seasonal = eco_data.groupby(['ENSO_Phase', 'Season']).size().unstack('Season')
            eco_seasonal = eco_seasonal.reindex(['Spring', 'Summer', 'Fall', 'Winter'], axis=1)
            
            if not eco_seasonal.empty:
                eco_seasonal.plot(kind='bar', ax=ax, width=0.8, stacked=True)
                ax.set_title(f'{eco} - Seasonal SFD Patterns')
                ax.set_xlabel('ENSO Phase')
                ax.set_ylabel('SFD Count')
                ax.legend(title='Season', bbox_to_anchor=(1.05, 1))
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 9. Box plot of SFD severity by ENSO phase
    if plot_idx <= 16:
        ax = plt.subplot(4, 4, plot_idx)
        
        sns.boxplot(data=df, x='ENSO_Phase', y='Mean_SFD_Flow_Percentile', ax=ax)
        ax.set_title('SFD Severity by ENSO Phase')
        ax.set_xlabel('ENSO Phase')
        ax.set_ylabel('Mean SFD Flow Percentile')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # 10. Duration comparison by ENSO phase
    if plot_idx <= 16:
        ax = plt.subplot(4, 4, plot_idx)
        
        sns.boxplot(data=df, x='ENSO_Phase', y='Duration_Days', ax=ax)
        ax.set_title('SFD Duration by ENSO Phase')
        ax.set_xlabel('ENSO Phase')
        ax.set_ylabel('Duration (days)')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig('../results/enso_sfd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    
    print(f"\n{'='*80}")
    print("ENSO-STREAMFLOW FLASH DROUGHT ANALYSIS")
    print(f"{'='*80}")
    
    # Step 1: Classify ENSO phases
    df = classify_enso_phase(df)
    
    # Step 2: Classify dam presence
    df = classify_dam_presence_for_enso(df)
    
    # Step 3: Analyze SFD frequency by ENSO phase and season
    frequency_results = analyze_sfd_frequency_by_enso_season(df)
    
    # Step 4: Analyze ecoregion-ENSO patterns
    ecoregion_results = analyze_ecoregion_enso_patterns(df)
    
    # Step 5: Create visualizations
    create_enso_sfd_visualizations(df, frequency_results, ecoregion_results)
    
    # Save results
    print(f"\n{'='*60}")
    print("SAVING ENSO-SFD ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    # Save detailed frequency results
    for dam_col, results in frequency_results.items():
        results['detailed'].to_csv(f'../results/enso_sfd_frequency_{dam_col.lower()}.csv', index=False)
        results['summary'].to_csv(f'../results/enso_sfd_summary_{dam_col.lower()}.csv')
    
    # Save ecoregion results
    for dam_col, eco_results in ecoregion_results.items():
        eco_results.to_csv(f'../results/ecoregion_enso_patterns_{dam_col.lower()}.csv')
    
    # Create comprehensive summary report
    with open('../results/ENSO_SFD_ANALYSIS_SUMMARY.md', 'w') as f:
        f.write("# ENSO-Streamflow Flash Drought Analysis Summary\n\n")
        
        f.write("## Research Question\n")
        f.write("How do streamflow flash droughts (SFDs) vary during different seasons ")
        f.write("in El Niño and La Niña years, and what is the frequency of SFDs in the ")
        f.write("absence/presence of dams across all ecoregions?\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- **Total SFD Events**: {len(df):,}\n")
        f.write(f"- **Date Range**: {df['Onset_Time'].min().strftime('%Y-%m-%d')} to {df['Onset_Time'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Unique Catchments**: {df['Station_ID'].nunique():,}\n")
        f.write(f"- **Ecoregions**: {df['AGGECOREGION'].nunique()}\n\n")
        
        f.write("## ENSO Phase Distribution\n")
        enso_counts = df['ENSO_Phase'].value_counts()
        for phase, count in enso_counts.items():
            f.write(f"- **{phase}**: {count:,} events ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### Seasonal Patterns by ENSO Phase\n")
        seasonal_summary = df.groupby(['ENSO_Phase', 'Season']).size().unstack('Season')
        for enso_phase in seasonal_summary.index:
            f.write(f"**{enso_phase}:**\n")
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                if season in seasonal_summary.columns:
                    count = seasonal_summary.loc[enso_phase, season]
                    f.write(f"- {season}: {count:,} events\n")
            f.write("\n")
        
        f.write("### Dam Influence Across ENSO Phases\n")
        if len(dam_columns) > 0 and f'{dam_columns[0]}_Binary' in df.columns:
            dam_enso_summary = df.groupby(['ENSO_Phase', f'{dam_columns[0]}_Binary']).size().unstack(f'{dam_columns[0]}_Binary')
            for enso_phase in dam_enso_summary.index:
                f.write(f"**{enso_phase}:**\n")
                if 'Dam_Present' in dam_enso_summary.columns:
                    dam_present = dam_enso_summary.loc[enso_phase, 'Dam_Present']
                    f.write(f"- Dam Present: {dam_present:,} events\n")
                if 'Dam_Absent' in dam_enso_summary.columns:
                    dam_absent = dam_enso_summary.loc[enso_phase, 'Dam_Absent']
                    f.write(f"- Dam Absent: {dam_absent:,} events\n")
                f.write("\n")
        
        f.write("## Files Generated\n")
        f.write("- `enso_sfd_analysis.png` - Comprehensive visualizations\n")
        f.write("- `enso_sfd_frequency_*.csv` - Detailed frequency analysis by dam type\n")
        f.write("- `enso_sfd_summary_*.csv` - Summary statistics by ENSO phase and season\n")
        f.write("- `ecoregion_enso_patterns_*.csv` - Ecoregion-specific ENSO responses\n")
        f.write("- `ENSO_SFD_ANALYSIS_SUMMARY.md` - This summary report\n\n")
        
        f.write("## Interpretation\n")
        f.write("This analysis reveals how climate oscillations (ENSO) interact with dam infrastructure ")
        f.write("to influence streamflow flash drought patterns across different seasons and ecoregions. ")
        f.write("The results provide insights for seasonal drought forecasting and water management ")
        f.write("strategies during El Niño and La Niña events.\n")
    
    print("\nENSO-SFD analysis complete!")
    print("Files saved:")
    print("  - results/enso_sfd_analysis.png")
    print("  - results/enso_sfd_frequency_*.csv")
    print("  - results/enso_sfd_summary_*.csv")
    print("  - results/ecoregion_enso_patterns_*.csv")
    print("  - results/ENSO_SFD_ANALYSIS_SUMMARY.md")