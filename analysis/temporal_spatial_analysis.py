import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

print("Loading SFD events with static attributes...")
df = pd.read_parquet('data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['Onset_Time'].min()} to {df['Onset_Time'].max()}")
print(f"Unique ecoregions: {df['AGGECOREGION'].nunique()}")

# Convert Onset_Time to datetime if it's not already
df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])

# Extract temporal components
df['Year'] = df['Onset_Time'].dt.year
df['Month'] = df['Onset_Time'].dt.month
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Define month names for better visualization
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['Month_Name'] = df['Month'].map(month_names)

print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
print(f"Total years analyzed: {df['Year'].nunique()}")

# Target variables for analysis
target_vars = ['Mean_SFD_Flow_Percentile', 'Duration_Days']

print(f"\nAnalyzing variables: {target_vars}")
print(f"Ecoregions: {sorted(df['AGGECOREGION'].unique())}")

def monthly_seasonal_analysis(df, target_vars):
    """Analyze monthly and seasonal patterns for each ecoregion"""
    
    print(f"\n{'='*60}")
    print("MONTHLY AND SEASONAL ANALYSIS")
    print(f"{'='*60}")
    
    results = {}
    
    # Monthly analysis
    monthly_stats = df.groupby(['AGGECOREGION', 'Month'])[target_vars].agg(['mean', 'std', 'count']).round(2)
    monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns]
    
    # Seasonal analysis
    seasonal_stats = df.groupby(['AGGECOREGION', 'Season'])[target_vars].agg(['mean', 'std', 'count']).round(2)
    seasonal_stats.columns = ['_'.join(col).strip() for col in seasonal_stats.columns]
    
    # Annual trends
    annual_stats = df.groupby(['AGGECOREGION', 'Year'])[target_vars].agg(['mean', 'std', 'count']).round(2)
    annual_stats.columns = ['_'.join(col).strip() for col in annual_stats.columns]
    
    results['monthly'] = monthly_stats
    results['seasonal'] = seasonal_stats
    results['annual'] = annual_stats
    
    # Print summary statistics
    print("\nMONTHLY PATTERNS SUMMARY:")
    print("-" * 40)
    
    for var in target_vars:
        print(f"\n{var}:")
        monthly_means = df.groupby(['AGGECOREGION', 'Month'])[var].mean().unstack()
        
        # Find peak and low months for each ecoregion
        for ecoregion in monthly_means.index:
            peak_month = monthly_means.loc[ecoregion].idxmax()
            low_month = monthly_means.loc[ecoregion].idxmin()
            peak_val = monthly_means.loc[ecoregion, peak_month]
            low_val = monthly_means.loc[ecoregion, low_month]
            
            print(f"  {ecoregion}: Peak={month_names[peak_month]}({peak_val:.1f}), Low={month_names[low_month]}({low_val:.1f})")
    
    print("\nSEASONAL PATTERNS SUMMARY:")
    print("-" * 40)
    
    for var in target_vars:
        print(f"\n{var}:")
        seasonal_means = df.groupby(['AGGECOREGION', 'Season'])[var].mean().unstack()
        
        # Find peak and low seasons for each ecoregion
        for ecoregion in seasonal_means.index:
            peak_season = seasonal_means.loc[ecoregion].idxmax()
            low_season = seasonal_means.loc[ecoregion].idxmin()
            peak_val = seasonal_means.loc[ecoregion, peak_season]
            low_val = seasonal_means.loc[ecoregion, low_season]
            
            print(f"  {ecoregion}: Peak={peak_season}({peak_val:.1f}), Low={low_season}({low_val:.1f})")
    
    return results

def temporal_trend_analysis(df, target_vars):
    """Analyze temporal trends over years"""
    
    print(f"\n{'='*60}")
    print("TEMPORAL TREND ANALYSIS")
    print(f"{'='*60}")
    
    trend_results = {}
    
    for var in target_vars:
        print(f"\nAnalyzing trends for {var}:")
        print("-" * 30)
        
        var_trends = {}
        
        for ecoregion in df['AGGECOREGION'].unique():
            if pd.isna(ecoregion):
                continue
                
            eco_data = df[df['AGGECOREGION'] == ecoregion]
            annual_means = eco_data.groupby('Year')[var].mean()
            
            if len(annual_means) > 3:  # Need at least 4 years for trend analysis
                # Linear regression for trend
                years = annual_means.index.values
                values = annual_means.values
                
                # Remove any NaN values
                mask = ~np.isnan(values)
                if mask.sum() > 3:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(years[mask], values[mask])
                    
                    # Determine trend direction and significance
                    if p_value < 0.05:
                        if slope > 0:
                            trend = "Increasing*"
                        else:
                            trend = "Decreasing*"
                    else:
                        if slope > 0:
                            trend = "Increasing"
                        else:
                            trend = "Decreasing"
                    
                    var_trends[ecoregion] = {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend': trend,
                        'years': len(annual_means)
                    }
                    
                    print(f"  {ecoregion}: {trend} (slope={slope:.3f}, R²={r_value**2:.3f}, p={p_value:.3f})")
        
        trend_results[var] = var_trends
    
    return trend_results

def ecoregion_similarity_analysis(df, target_vars):
    """Analyze similarity between ecoregions using clustering"""
    
    print(f"\n{'='*60}")
    print("ECOREGION SIMILARITY ANALYSIS")
    print(f"{'='*60}")
    
    # Create feature matrix for each ecoregion
    ecoregion_features = []
    ecoregion_names = []
    
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.isna(ecoregion):
            continue
            
        eco_data = df[df['AGGECOREGION'] == ecoregion]
        
        features = []
        
        # Monthly means for each variable
        for var in target_vars:
            monthly_means = eco_data.groupby('Month')[var].mean()
            # Ensure all 12 months are represented
            for month in range(1, 13):
                features.append(monthly_means.get(month, np.nan))
        
        # Seasonal means for each variable
        for var in target_vars:
            seasonal_means = eco_data.groupby('Season')[var].mean()
            for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                features.append(seasonal_means.get(season, np.nan))
        
        # Overall statistics
        for var in target_vars:
            features.extend([
                eco_data[var].mean(),
                eco_data[var].std(),
                eco_data[var].median()
            ])
        
        # Handle any NaN values
        features = np.array(features)
        features = np.nan_to_num(features, nan=np.nanmean(features))
        
        ecoregion_features.append(features)
        ecoregion_names.append(ecoregion)
    
    # Convert to array and standardize
    feature_matrix = np.array(ecoregion_features)
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Hierarchical clustering
    linkage_matrix = linkage(feature_matrix_scaled, method='ward')
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(feature_matrix_scaled)
    
    # Create similarity results
    similarity_results = {
        'feature_matrix': feature_matrix_scaled,
        'ecoregion_names': ecoregion_names,
        'linkage_matrix': linkage_matrix,
        'cluster_labels': cluster_labels,
        'pca_features': pca_features,
        'pca_explained_variance': pca.explained_variance_ratio_
    }
    
    # Print clustering results
    print("\nHIERARCHICAL CLUSTERING GROUPS:")
    print("-" * 35)
    
    # Get clusters at different levels
    for n_clusters in [2, 3, 4]:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        print(f"\n{n_clusters} Clusters:")
        for i in range(1, n_clusters + 1):
            cluster_members = [ecoregion_names[j] for j in range(len(ecoregion_names)) if clusters[j] == i]
            print(f"  Cluster {i}: {', '.join(cluster_members)}")
    
    print(f"\nK-MEANS CLUSTERING (3 clusters):")
    print("-" * 30)
    for i in range(3):
        cluster_members = [ecoregion_names[j] for j in range(len(ecoregion_names)) if cluster_labels[j] == i]
        print(f"  Cluster {i}: {', '.join(cluster_members)}")
    
    print(f"\nPCA ANALYSIS:")
    print("-" * 15)
    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.1%}")
    
    return similarity_results

def spatial_pattern_analysis(df, target_vars):
    """Analyze spatial patterns and correlations"""
    
    print(f"\n{'='*60}")
    print("SPATIAL PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    spatial_results = {}
    
    # Calculate ecoregion summary statistics
    ecoregion_summary = df.groupby('AGGECOREGION')[target_vars].agg([
        'mean', 'median', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    # Calculate coefficient of variation (CV) for each ecoregion
    for var in target_vars:
        cv_col = f'{var}_CV'
        ecoregion_summary[(var, 'cv')] = (ecoregion_summary[(var, 'std')] / 
                                         ecoregion_summary[(var, 'mean')] * 100)
    
    spatial_results['summary'] = ecoregion_summary
    
    print("\nECOREGION SUMMARY STATISTICS:")
    print("-" * 35)
    
    for var in target_vars:
        print(f"\n{var}:")
        var_summary = ecoregion_summary[var].sort_values('mean', ascending=False)
        
        print("  Ranking by mean value:")
        for i, (ecoregion, stats) in enumerate(var_summary.iterrows(), 1):
            print(f"    {i}. {ecoregion}: {stats['mean']:.1f} ± {stats['std']:.1f} "
                  f"(CV: {stats['std']/stats['mean']*100:.1f}%)")
    
    # Correlation analysis between ecoregions
    correlation_results = {}
    
    for var in target_vars:
        # Create pivot table with ecoregions as columns and time as rows
        pivot_data = df.pivot_table(
            values=var, 
            index=['Year', 'Month'], 
            columns='AGGECOREGION', 
            aggfunc='mean'
        )
        
        # Calculate correlation matrix
        corr_matrix = pivot_data.corr()
        correlation_results[var] = corr_matrix
        
        print(f"\n{var} - HIGHEST CORRELATIONS BETWEEN ECOREGIONS:")
        print("-" * 50)
        
        # Find highest correlations (excluding diagonal)
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                eco1 = corr_matrix.columns[i]
                eco2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_values.append((eco1, eco2, corr_val))
        
        # Sort by correlation strength
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Print top 5 correlations
        for eco1, eco2, corr_val in corr_values[:5]:
            print(f"  {eco1} - {eco2}: {corr_val:.3f}")
    
    spatial_results['correlations'] = correlation_results
    
    return spatial_results

def create_comprehensive_visualizations(df, target_vars, monthly_results, similarity_results, spatial_results, trend_results):
    """Create comprehensive visualizations for temporal and spatial analysis"""
    
    print(f"\n{'='*60}")
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Monthly patterns heatmap
    for i, var in enumerate(target_vars):
        ax = plt.subplot(4, 4, i*2 + 1)
        
        # Create monthly heatmap
        monthly_pivot = df.groupby(['AGGECOREGION', 'Month'])[var].mean().unstack()
        
        sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
        ax.set_title(f'{var} - Monthly Patterns')
        ax.set_xlabel('Month')
        ax.set_ylabel('Ecoregion')
        
        # Set month labels
        ax.set_xticklabels([month_names[m] for m in range(1, 13)])
    
    # 2. Seasonal patterns
    for i, var in enumerate(target_vars):
        ax = plt.subplot(4, 4, i*2 + 2)
        
        # Create seasonal boxplot
        seasonal_data = df.pivot_table(values=var, index=['AGGECOREGION'], columns='Season', aggfunc='mean')
        seasonal_data = seasonal_data[['Spring', 'Summer', 'Fall', 'Winter']]  # Order seasons
        
        sns.heatmap(seasonal_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax)
        ax.set_title(f'{var} - Seasonal Patterns')
        ax.set_xlabel('Season')
        ax.set_ylabel('Ecoregion')
    
    # 3. Annual trends
    ax = plt.subplot(4, 4, 5)
    
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.isna(ecoregion):
            continue
        eco_data = df[df['AGGECOREGION'] == ecoregion]
        annual_means = eco_data.groupby('Year')[target_vars[0]].mean()
        
        if len(annual_means) > 1:
            ax.plot(annual_means.index, annual_means.values, marker='o', label=ecoregion, alpha=0.7)
    
    ax.set_title(f'{target_vars[0]} - Annual Trends')
    ax.set_xlabel('Year')
    ax.set_ylabel(target_vars[0])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Duration annual trends
    ax = plt.subplot(4, 4, 6)
    
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.isna(ecoregion):
            continue
        eco_data = df[df['AGGECOREGION'] == ecoregion]
        annual_means = eco_data.groupby('Year')[target_vars[1]].mean()
        
        if len(annual_means) > 1:
            ax.plot(annual_means.index, annual_means.values, marker='o', label=ecoregion, alpha=0.7)
    
    ax.set_title(f'{target_vars[1]} - Annual Trends')
    ax.set_xlabel('Year')
    ax.set_ylabel(target_vars[1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 5. Ecoregion similarity dendrogram
    ax = plt.subplot(4, 4, 7)
    
    dendrogram(similarity_results['linkage_matrix'], 
               labels=similarity_results['ecoregion_names'],
               ax=ax, orientation='top')
    ax.set_title('Ecoregion Similarity (Hierarchical Clustering)')
    ax.tick_params(axis='x', rotation=45)
    
    # 6. PCA visualization
    ax = plt.subplot(4, 4, 8)
    
    scatter = ax.scatter(similarity_results['pca_features'][:, 0], 
                        similarity_results['pca_features'][:, 1],
                        c=similarity_results['cluster_labels'], 
                        cmap='viridis', s=100, alpha=0.7)
    
    # Add ecoregion labels
    for i, name in enumerate(similarity_results['ecoregion_names']):
        ax.annotate(name, (similarity_results['pca_features'][i, 0], 
                          similarity_results['pca_features'][i, 1]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel(f'PC1 ({similarity_results["pca_explained_variance"][0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({similarity_results["pca_explained_variance"][1]:.1%} variance)')
    ax.set_title('Ecoregion Similarity (PCA)')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # 7. Correlation heatmap for Mean_SFD_Flow_Percentile
    ax = plt.subplot(4, 4, 9)
    
    corr_matrix = spatial_results['correlations'][target_vars[0]]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, ax=ax, square=True)
    ax.set_title(f'{target_vars[0]} - Ecoregion Correlations')
    
    # 8. Correlation heatmap for Duration_Days
    ax = plt.subplot(4, 4, 10)
    
    corr_matrix = spatial_results['correlations'][target_vars[1]]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, ax=ax, square=True)
    ax.set_title(f'{target_vars[1]} - Ecoregion Correlations')
    
    # 9. Monthly distribution violin plots
    ax = plt.subplot(4, 4, 11)
    
    sns.violinplot(data=df, x='Month', y=target_vars[0], ax=ax)
    ax.set_title(f'{target_vars[0]} - Monthly Distribution')
    ax.set_xticklabels([month_names[m] for m in range(1, 13)])
    
    # 10. Seasonal distribution violin plots
    ax = plt.subplot(4, 4, 12)
    
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    sns.violinplot(data=df, x='Season', y=target_vars[1], order=season_order, ax=ax)
    ax.set_title(f'{target_vars[1]} - Seasonal Distribution')
    
    # 11. Ecoregion comparison boxplot
    ax = plt.subplot(4, 4, 13)
    
    sns.boxplot(data=df, x='AGGECOREGION', y=target_vars[0], ax=ax)
    ax.set_title(f'{target_vars[0]} by Ecoregion')
    ax.tick_params(axis='x', rotation=45)
    
    # 12. Duration comparison boxplot
    ax = plt.subplot(4, 4, 14)
    
    sns.boxplot(data=df, x='AGGECOREGION', y=target_vars[1], ax=ax)
    ax.set_title(f'{target_vars[1]} by Ecoregion')
    ax.tick_params(axis='x', rotation=45)
    
    # 13. Trend slopes visualization
    ax = plt.subplot(4, 4, 15)
    
    # Extract trend slopes for visualization
    ecoregions = []
    flow_slopes = []
    duration_slopes = []
    
    for eco in similarity_results['ecoregion_names']:
        if eco in trend_results[target_vars[0]]:
            ecoregions.append(eco)
            flow_slopes.append(trend_results[target_vars[0]][eco]['slope'])
            duration_slopes.append(trend_results[target_vars[1]].get(eco, {}).get('slope', 0))
    
    if ecoregions:
        x_pos = np.arange(len(ecoregions))
        width = 0.35
        
        ax.bar(x_pos - width/2, flow_slopes, width, label=target_vars[0], alpha=0.7)
        ax.bar(x_pos + width/2, duration_slopes, width, label=target_vars[1], alpha=0.7)
        
        ax.set_xlabel('Ecoregion')
        ax.set_ylabel('Trend Slope (per year)')
        ax.set_title('Temporal Trend Slopes by Ecoregion')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ecoregions, rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 14. Event frequency by month and ecoregion
    ax = plt.subplot(4, 4, 16)
    
    event_counts = df.groupby(['AGGECOREGION', 'Month']).size().unstack(fill_value=0)
    
    sns.heatmap(event_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title('SFD Event Frequency by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Ecoregion')
    ax.set_xticklabels([month_names[m] for m in range(1, 13)])
    
    plt.tight_layout()
    plt.savefig('../results/temporal_spatial_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    
    # Run all analyses
    monthly_results = monthly_seasonal_analysis(df, target_vars)
    trend_results = temporal_trend_analysis(df, target_vars)
    similarity_results = ecoregion_similarity_analysis(df, target_vars)
    spatial_results = spatial_pattern_analysis(df, target_vars)
    
    # Create visualizations
    create_comprehensive_visualizations(df, target_vars, monthly_results, 
                                      similarity_results, spatial_results, trend_results)
    
    # Save detailed results
    print(f"\n{'='*60}")
    print("SAVING DETAILED RESULTS")
    print(f"{'='*60}")
    
    # Save monthly patterns
    monthly_results['monthly'].to_csv('../results/monthly_patterns.csv')
    monthly_results['seasonal'].to_csv('../results/seasonal_patterns.csv')
    monthly_results['annual'].to_csv('../results/annual_patterns.csv')
    
    # Save similarity analysis
    similarity_df = pd.DataFrame(
        similarity_results['feature_matrix'],
        index=similarity_results['ecoregion_names']
    )
    similarity_df.to_csv('../results/ecoregion_similarity_features.csv')
    
    # Save spatial correlations
    for var in target_vars:
        spatial_results['correlations'][var].to_csv(f'../results/{var}_ecoregion_correlations.csv')
    
    # Save trend analysis
    trend_summary = []
    for var in target_vars:
        for eco, trend_data in trend_results[var].items():
            trend_summary.append({
                'Variable': var,
                'Ecoregion': eco,
                'Slope': trend_data['slope'],
                'R_squared': trend_data['r_squared'],
                'P_value': trend_data['p_value'],
                'Trend': trend_data['trend'],
                'Years': trend_data['years']
            })
    
    trend_df = pd.DataFrame(trend_summary)
    trend_df.to_csv('../results/temporal_trends.csv', index=False)
    
    # Create summary report
    with open('../results/TEMPORAL_SPATIAL_SUMMARY.md', 'w') as f:
        f.write("# Temporal and Spatial Analysis Summary\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- **Total Events**: {len(df):,}\n")
        f.write(f"- **Date Range**: {df['Onset_Time'].min().strftime('%Y-%m-%d')} to {df['Onset_Time'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"- **Years Analyzed**: {df['Year'].nunique()}\n")
        f.write(f"- **Ecoregions**: {df['AGGECOREGION'].nunique()}\n\n")
        
        f.write("## Key Findings\n\n")
        
        f.write("### Temporal Trends\n")
        for var in target_vars:
            f.write(f"\n**{var}:**\n")
            for eco, trend_data in trend_results[var].items():
                significance = "*" if trend_data['p_value'] < 0.05 else ""
                f.write(f"- {eco}: {trend_data['trend']} (slope={trend_data['slope']:.3f}, R²={trend_data['r_squared']:.3f}){significance}\n")
        
        f.write("\n### Ecoregion Clusters\n")
        f.write("Based on temporal and seasonal patterns:\n")
        for i in range(3):
            cluster_members = [similarity_results['ecoregion_names'][j] 
                             for j in range(len(similarity_results['ecoregion_names'])) 
                             if similarity_results['cluster_labels'][j] == i]
            f.write(f"- **Cluster {i+1}**: {', '.join(cluster_members)}\n")
        
        f.write(f"\n### PCA Analysis\n")
        f.write(f"- PC1 explains {similarity_results['pca_explained_variance'][0]:.1%} of variance\n")
        f.write(f"- PC2 explains {similarity_results['pca_explained_variance'][1]:.1%} of variance\n")
        f.write(f"- Total variance explained: {sum(similarity_results['pca_explained_variance']):.1%}\n")
        
        f.write("\n## Files Generated\n")
        f.write("- `temporal_spatial_analysis.png` - Comprehensive visualizations\n")
        f.write("- `monthly_patterns.csv` - Monthly statistics by ecoregion\n")
        f.write("- `seasonal_patterns.csv` - Seasonal statistics by ecoregion\n")
        f.write("- `annual_patterns.csv` - Annual statistics by ecoregion\n")
        f.write("- `temporal_trends.csv` - Trend analysis results\n")
        f.write("- `ecoregion_similarity_features.csv` - Similarity analysis features\n")
        f.write("- `*_ecoregion_correlations.csv` - Correlation matrices\n")
    
    print("\nAnalysis complete!")
    print("Files saved:")
    print("  - results/temporal_spatial_analysis.png")
    print("  - results/monthly_patterns.csv")
    print("  - results/seasonal_patterns.csv") 
    print("  - results/annual_patterns.csv")
    print("  - results/temporal_trends.csv")
    print("  - results/ecoregion_similarity_features.csv")
    print("  - results/TEMPORAL_SPATIAL_SUMMARY.md")
    print("  - results/*_ecoregion_correlations.csv")