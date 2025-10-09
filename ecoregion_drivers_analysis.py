import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading SFD events with static attributes...")
df = pd.read_parquet('data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique ecoregions: {df['AGGECOREGION'].nunique()}")

# Display basic info about target variables
target_vars = ['Duration_Days', 'Onset_Rate_Days', 'Mean_SFD_Flow_Percentile']
print("\nTarget Variables Summary:")
for var in target_vars:
    if var in df.columns:
        print(f"{var}: mean={df[var].mean():.2f}, std={df[var].std():.2f}, range=[{df[var].min():.2f}, {df[var].max():.2f}]")

# Exclude specified columns
exclude_cols = ['station_id', 'onset_time'] + [col for col in df.columns if 'station' in col.lower() or 'onset_time' in col.lower() or 'id' in col.lower()]
exclude_cols = [col for col in exclude_cols if col in df.columns]
print(f"\nExcluding columns: {exclude_cols}")

# Get potential driver columns (numeric columns excluding targets and excluded columns)
driver_cols = [col for col in df.columns if col not in target_vars + exclude_cols + ['AGGECOREGION']]
numeric_cols = df[driver_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\nPotential driver columns ({len(numeric_cols)}): {numeric_cols[:10]}...")  # Show first 10

def analyze_drivers_by_ecoregion(df, target_var, driver_cols, top_n=10):
    """Analyze drivers for a target variable across ecoregions using Random Forest"""
    
    results = {}
    
    # Overall analysis (all ecoregions combined)
    print(f"\n{'='*60}")
    print(f"ANALYZING DRIVERS FOR: {target_var}")
    print(f"{'='*60}")
    
    # Prepare data for overall analysis
    X = df[driver_cols].fillna(df[driver_cols].median())
    y = df[target_var].fillna(df[target_var].median())
    
    # Remove any remaining NaN or infinite values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(X_clean) > 0:
        # Train Random Forest for overall analysis
        rf_overall = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_overall.fit(X_clean, y_clean)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': driver_cols,
            'importance': rf_overall.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nOVERALL TOP {top_n} DRIVERS (All Ecoregions):")
        print("-" * 50)
        for i, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<30}: {row['importance']:.4f}")
        
        results['overall'] = importance_df.head(top_n)
        
        # Calculate R² score
        y_pred = rf_overall.predict(X_clean)
        r2 = r2_score(y_clean, y_pred)
        print(f"\nOverall Model R² Score: {r2:.4f}")
    
    # Analysis by ecoregion
    ecoregion_results = {}
    
    print(f"\nDRIVERS BY ECOREGION:")
    print("-" * 50)
    
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.isna(ecoregion):
            continue
            
        eco_df = df[df['AGGECOREGION'] == ecoregion]
        
        if len(eco_df) < 10:  # Skip if too few samples
            continue
            
        # Prepare data for this ecoregion
        X_eco = eco_df[driver_cols].fillna(eco_df[driver_cols].median())
        y_eco = eco_df[target_var].fillna(eco_df[target_var].median())
        
        # Remove any remaining NaN or infinite values
        mask_eco = np.isfinite(X_eco).all(axis=1) & np.isfinite(y_eco)
        X_eco_clean = X_eco[mask_eco]
        y_eco_clean = y_eco[mask_eco]
        
        if len(X_eco_clean) > 5:  # Need minimum samples
            try:
                rf_eco = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                rf_eco.fit(X_eco_clean, y_eco_clean)
                
                importance_eco = pd.DataFrame({
                    'feature': driver_cols,
                    'importance': rf_eco.feature_importances_
                }).sort_values('importance', ascending=False)
                
                ecoregion_results[ecoregion] = importance_eco.head(5)
                
                print(f"\nEcoregion {ecoregion} (n={len(eco_df)}):")
                print(f"  Mean {target_var}: {y_eco_clean.mean():.2f}")
                print("  Top 5 drivers:")
                for i, row in importance_eco.head(5).iterrows():
                    print(f"    {row['feature']:<25}: {row['importance']:.4f}")
                    
            except Exception as e:
                print(f"  Error analyzing ecoregion {ecoregion}: {str(e)}")
    
    results['by_ecoregion'] = ecoregion_results
    return results

def create_correlation_analysis(df, target_vars, driver_cols):
    """Create correlation analysis between drivers and targets"""
    
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    # Calculate correlations
    corr_data = df[target_vars + driver_cols].corr()
    
    # Extract correlations with target variables
    target_correlations = {}
    
    for target in target_vars:
        if target in corr_data.columns:
            target_corr = corr_data[target].drop(target_vars).abs().sort_values(ascending=False)
            target_correlations[target] = target_corr.head(10)
            
            print(f"\nTop 10 correlations with {target}:")
            print("-" * 40)
            for feature, corr_val in target_corr.head(10).items():
                direction = "+" if corr_data[target][feature] > 0 else "-"
                print(f"{feature:<30}: {direction}{corr_val:.4f}")
    
    return target_correlations

# Run the analysis
if __name__ == "__main__":
    
    # Check if we have the required target variables
    available_targets = [var for var in target_vars if var in df.columns]
    
    if not available_targets:
        print("Error: None of the target variables found in the dataset!")
        print(f"Available columns: {list(df.columns)}")
    else:
        print(f"Found target variables: {available_targets}")
        
        # Run correlation analysis first
        correlations = create_correlation_analysis(df, available_targets, numeric_cols)
        
        # Run Random Forest analysis for each target variable
        all_results = {}
        
        for target in available_targets:
            results = analyze_drivers_by_ecoregion(df, target, numeric_cols, top_n=15)
            all_results[target] = results 
       # Create summary visualization
        print(f"\n{'='*60}")
        print("CREATING SUMMARY VISUALIZATIONS")
        print(f"{'='*60}")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Target variable distributions by ecoregion
        ax1 = plt.subplot(3, 3, 1)
        for i, target in enumerate(available_targets):
            df.boxplot(column=target, by='AGGECOREGION', ax=ax1 if i == 0 else plt.subplot(3, 3, i+1))
            plt.title(f'{target} by Ecoregion')
            plt.xticks(rotation=45)
            plt.suptitle('')  # Remove default title
        
        # Plot 2: Feature importance comparison
        if len(available_targets) > 1:
            ax_imp = plt.subplot(3, 2, 3)
            
            # Get top features for each target
            top_features_by_target = {}
            for target in available_targets:
                if target in all_results and 'overall' in all_results[target]:
                    top_features_by_target[target] = all_results[target]['overall']['feature'].head(10).tolist()
            
            # Create comparison plot if we have results
            if top_features_by_target:
                all_top_features = list(set().union(*top_features_by_target.values()))[:15]
                
                importance_matrix = []
                for target in available_targets:
                    if target in all_results and 'overall' in all_results[target]:
                        target_importance = all_results[target]['overall'].set_index('feature')['importance']
                        row = [target_importance.get(feat, 0) for feat in all_top_features]
                        importance_matrix.append(row)
                
                if importance_matrix:
                    sns.heatmap(importance_matrix, 
                              xticklabels=[f[:20] for f in all_top_features], 
                              yticklabels=available_targets,
                              annot=True, fmt='.3f', cmap='viridis', ax=ax_imp)
                    plt.title('Feature Importance Comparison')
                    plt.xticks(rotation=45, ha='right')
        
        # Plot 3: Correlation heatmap for top drivers
        ax_corr = plt.subplot(3, 2, 4)
        
        # Get top drivers across all targets
        all_important_features = set()
        for target in available_targets:
            if target in correlations:
                all_important_features.update(correlations[target].head(8).index)
        
        if all_important_features:
            corr_subset = df[list(all_important_features) + available_targets].corr()
            mask = np.triu(np.ones_like(corr_subset, dtype=bool))
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, ax=ax_corr)
            plt.title('Correlation Matrix: Top Drivers & Targets')
        
        # Plot 4: Ecoregion summary statistics
        ax_eco = plt.subplot(3, 1, 3)
        
        eco_summary = df.groupby('AGGECOREGION')[available_targets].agg(['mean', 'std', 'count'])
        eco_summary.columns = ['_'.join(col).strip() for col in eco_summary.columns]
        
        # Plot mean values
        eco_means = eco_summary[[col for col in eco_summary.columns if 'mean' in col]]
        eco_means.plot(kind='bar', ax=ax_eco)
        plt.title('Mean Target Values by Ecoregion')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig('ecoregion_drivers_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        print(f"Dataset contains {len(df)} SFD events across {df['AGGECOREGION'].nunique()} ecoregions")
        print(f"Analyzed {len(numeric_cols)} potential driver variables")
        
        for target in available_targets:
            if target in all_results and 'overall' in all_results[target]:
                top_driver = all_results[target]['overall'].iloc[0]
                print(f"\nFor {target}:")
                print(f"  Most important driver: {top_driver['feature']} (importance: {top_driver['importance']:.4f})")
                print(f"  Analyzed across {len(all_results[target]['by_ecoregion'])} ecoregions")
        
        print(f"\nVisualization saved as: ecoregion_drivers_analysis.png")
        print("Analysis complete!")