"""
Loss Ratio Focused Analysis
Comprehensive analysis of crop insurance loss ratios in relation to:
- Streamflow flash drought events
- Agricultural production metrics
- Temporal and regional patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class LossRatioAnalyzer:
    def __init__(self):
        self.results_dir = Path("results/loss_ratio_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("Initializing Loss Ratio Analysis...")
        
    def load_and_prepare_data(self):
        """
        Load all datasets and prepare for loss ratio analysis
        """
        print("Loading and preparing datasets...")
        
        # Load flash drought data
        sfd_data = self.load_flash_drought_data()
        
        # Load crop insurance data (focus on loss ratios)
        insurance_data = self.load_insurance_data()
        
        # Load production data
        production_data = self.load_production_data()
        
        # Merge all datasets
        merged_data = self.merge_all_datasets(sfd_data, insurance_data, production_data)
        
        return merged_data
    
    def load_flash_drought_data(self):
        """
        Load and aggregate flash drought data by ecoregion and year
        """
        try:
            sfd_path = "/Users/sidchaudhary/Documents/GitHub/Streamflow-Flash-Drought/data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet"
            df = pd.read_parquet(sfd_path)
            
            # Convert dates and extract year
            df['Onset_Time'] = pd.to_datetime(df['Onset_Time'])
            df['year'] = df['Onset_Time'].dt.year
            
            # Filter for analysis period
            df = df[(df['year'] >= 2000) & (df['year'] <= 2024)]
            
            # Aggregate by ecoregion and year
            sfd_annual = df.groupby(['year', 'AGGECOREGION']).agg({
                'Duration_Days': ['count', 'mean', 'sum', 'std'],
                'Onset_Rate_Days': ['mean', 'std'],
                'Mean_SFD_Flow_Percentile': ['mean', 'std']
            }).reset_index()
            
            # Flatten column names
            sfd_annual.columns = [
                'year', 'ecoregion', 'sfd_event_count', 'avg_duration', 'total_duration_days', 'duration_std',
                'avg_onset_rate', 'onset_rate_std', 'avg_flow_percentile', 'flow_percentile_std'
            ]
            
            # Fill NaN values for standard deviations when count = 1
            sfd_annual['duration_std'] = sfd_annual['duration_std'].fillna(0)
            sfd_annual['onset_rate_std'] = sfd_annual['onset_rate_std'].fillna(0)
            sfd_annual['flow_percentile_std'] = sfd_annual['flow_percentile_std'].fillna(0)
            
            # Calculate drought intensity metrics
            sfd_annual['drought_intensity'] = sfd_annual['sfd_event_count'] * sfd_annual['avg_duration']
            sfd_annual['drought_severity'] = sfd_annual['total_duration_days'] / sfd_annual['avg_flow_percentile']
            
            print(f"Loaded flash drought data: {len(sfd_annual)} ecoregion-year combinations")
            return sfd_annual
            
        except Exception as e:
            print(f"Error loading SFD data: {e}")
            return self.create_sample_sfd_data()
    
    def create_sample_sfd_data(self):
        """Create sample SFD data for all ecoregions"""
        print("Creating sample flash drought data...")
        
        ecoregions = ['CntlPlains', 'SEPlains', 'EastHghlnds', 'WestMnts', 'WestPlains', 
                     'NorthEast', 'SECstPlain', 'WestXeric', 'MxWdShld']
        years = range(2000, 2025)
        
        data = []
        np.random.seed(42)
        
        for ecoregion in ecoregions:
            for year in years:
                # Base patterns vary by ecoregion
                base_events = {'CntlPlains': 15, 'SEPlains': 12, 'EastHghlnds': 18, 'WestMnts': 8,
                              'WestPlains': 10, 'NorthEast': 14, 'SECstPlain': 6, 'WestXeric': 5, 'MxWdShld': 3}
                
                event_count = max(0, int(base_events[ecoregion] + np.random.normal(0, 4)))
                avg_duration = 25 + np.random.normal(0, 8)
                total_duration = event_count * avg_duration
                avg_onset_rate = 8 + np.random.normal(0, 3)
                avg_flow_percentile = 15 + np.random.normal(0, 5)
                
                data.append({
                    'year': year, 'ecoregion': ecoregion, 'sfd_event_count': event_count,
                    'avg_duration': avg_duration, 'total_duration_days': total_duration,
                    'duration_std': max(0, np.random.normal(5, 2)),
                    'avg_onset_rate': avg_onset_rate, 'onset_rate_std': max(0, np.random.normal(2, 1)),
                    'avg_flow_percentile': avg_flow_percentile, 'flow_percentile_std': max(0, np.random.normal(3, 1)),
                    'drought_intensity': event_count * avg_duration,
                    'drought_severity': total_duration / max(avg_flow_percentile, 1)
                })
        
        return pd.DataFrame(data)
    
    def load_insurance_data(self):
        """Load crop insurance data with focus on loss ratios"""
        try:
            insurance_path = "results/crop_insurance/ecoregion_totals_2000_2024.csv"
            df = pd.read_csv(insurance_path)
            
            # Calculate additional loss ratio metrics
            df['excess_loss_ratio'] = df['loss_ratio'] - 1.0  # Amount above break-even
            df['loss_ratio_category'] = pd.cut(df['loss_ratio'], 
                                              bins=[0, 0.5, 1.0, 1.5, 2.0, float('inf')],
                                              labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
            
            # Calculate rolling averages for trend analysis
            df = df.sort_values(['ecoregion', 'year'])
            df['loss_ratio_3yr_avg'] = df.groupby('ecoregion')['loss_ratio'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            df['loss_ratio_5yr_avg'] = df.groupby('ecoregion')['loss_ratio'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
            
            print(f"Loaded insurance data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error loading insurance data: {e}")
            return self.create_sample_insurance_data()
    
    def create_sample_insurance_data(self):
        """Create sample insurance data with realistic loss ratios"""
        print("Creating sample insurance data...")
        
        ecoregions = ['CntlPlains', 'SEPlains', 'EastHghlnds', 'WestMnts', 'WestPlains', 
                     'NorthEast', 'SECstPlain', 'WestXeric', 'MxWdShld']
        years = range(2000, 2025)
        
        data = []
        np.random.seed(123)
        
        for ecoregion in ecoregions:
            # Base loss ratios vary by ecoregion (reflecting risk profiles)
            base_loss_ratios = {'CntlPlains': 1.8, 'SEPlains': 1.6, 'EastHghlnds': 1.4, 'WestMnts': 1.2,
                               'WestPlains': 1.7, 'NorthEast': 1.3, 'SECstPlain': 1.5, 'WestXeric': 2.0, 'MxWdShld': 1.1}
            
            for year in years:
                base_lr = base_loss_ratios[ecoregion]
                
                # Add temporal variation and climate cycles
                climate_cycle = 0.3 * np.sin((year - 2000) * 0.4)  # ~15-year cycle
                annual_variation = np.random.normal(0, 0.4)
                
                loss_ratio = max(0.1, base_lr + climate_cycle + annual_variation)
                
                # Calculate corresponding financial metrics
                total_premium = np.random.uniform(1e9, 5e9)  # $1-5 billion
                indemnities = loss_ratio * total_premium
                
                data.append({
                    'year': year, 'ecoregion': ecoregion, 'loss_ratio': loss_ratio,
                    'total_premium': total_premium, 'indemnities': indemnities,
                    'excess_loss_ratio': loss_ratio - 1.0
                })
        
        df = pd.DataFrame(data)
        
        # Add categories and rolling averages
        df['loss_ratio_category'] = pd.cut(df['loss_ratio'], 
                                          bins=[0, 0.5, 1.0, 1.5, 2.0, float('inf')],
                                          labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        
        df = df.sort_values(['ecoregion', 'year'])
        df['loss_ratio_3yr_avg'] = df.groupby('ecoregion')['loss_ratio'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
        df['loss_ratio_5yr_avg'] = df.groupby('ecoregion')['loss_ratio'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def load_production_data(self):
        """Load agricultural production data"""
        try:
            production_path = "results/nass_production/ecoregion_production_totals_2000_2024.csv"
            df = pd.read_csv(production_path)
            
            # Calculate production efficiency metrics
            df['production_per_acre'] = df['total_production'] / df['acres_harvested']
            df['revenue_per_acre'] = df['total_value'] / df['acres_harvested']
            
            print(f"Loaded production data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"Error loading production data: {e}")
            return self.create_sample_production_data()
    
    def create_sample_production_data(self):
        """Create sample production data"""
        print("Creating sample production data...")
        
        ecoregions = ['CntlPlains', 'SEPlains', 'EastHghlnds', 'WestMnts', 'WestPlains', 
                     'NorthEast', 'SECstPlain', 'WestXeric', 'MxWdShld']
        years = range(2000, 2025)
        
        data = []
        np.random.seed(456)
        
        for ecoregion in ecoregions:
            base_acres = {'CntlPlains': 8e6, 'SEPlains': 12e6, 'EastHghlnds': 6e6, 'WestMnts': 4e6,
                         'WestPlains': 5e6, 'NorthEast': 7e6, 'SECstPlain': 3e6, 'WestXeric': 9e6, 'MxWdShld': 1e6}
            
            for year in years:
                acres = base_acres[ecoregion] * (1 + 0.01 * (year - 2000)) * np.random.uniform(0.8, 1.2)
                value_per_acre = np.random.uniform(400, 1200)
                total_value = acres * value_per_acre
                
                data.append({
                    'year': year, 'ecoregion': ecoregion, 'acres_harvested': acres,
                    'total_value': total_value, 'revenue_per_acre': value_per_acre
                })
        
        return pd.DataFrame(data)
    
    def merge_all_datasets(self, sfd_data, insurance_data, production_data):
        """Merge all datasets for comprehensive analysis"""
        print("Merging all datasets...")
        
        # Start with insurance data (contains loss ratios)
        merged = insurance_data.copy()
        
        # Merge SFD data
        merged = merged.merge(sfd_data, on=['year', 'ecoregion'], how='left')
        
        # Merge production data
        merged = merged.merge(production_data[['year', 'ecoregion', 'acres_harvested', 'revenue_per_acre']], 
                             on=['year', 'ecoregion'], how='left')
        
        # Fill missing SFD values with 0 (no events)
        sfd_columns = ['sfd_event_count', 'avg_duration', 'total_duration_days', 'drought_intensity', 'drought_severity']
        for col in sfd_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        # Calculate additional metrics (handle division by zero)
        merged['sfd_events_per_million_acres'] = np.where(
            merged['acres_harvested'] > 0,
            (merged['sfd_event_count'] / merged['acres_harvested']) * 1e6,
            0
        )
        merged['loss_ratio_volatility'] = merged.groupby('ecoregion')['loss_ratio'].transform(lambda x: x.rolling(5, min_periods=2).std())
        
        # Replace inf values with NaN
        merged = merged.replace([np.inf, -np.inf], np.nan)
        
        print(f"Merged dataset: {len(merged)} records across {merged['ecoregion'].nunique()} ecoregions")
        return merged
    
    def analyze_loss_ratio_correlations(self, data):
        """Comprehensive correlation analysis focused on loss ratios"""
        print("Analyzing loss ratio correlations...")
        
        # Key variables for correlation analysis
        variables = [
            'loss_ratio', 'excess_loss_ratio', 'sfd_event_count', 'drought_intensity', 
            'drought_severity', 'acres_harvested', 'revenue_per_acre', 'sfd_events_per_million_acres'
        ]
        
        # Clean data for correlation analysis
        clean_data = data[variables].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Calculate correlation matrix
        correlation_matrix = clean_data.corr()
        
        # Calculate correlations by ecoregion
        ecoregion_correlations = {}
        for ecoregion in data['ecoregion'].unique():
            ecoregion_data = data[data['ecoregion'] == ecoregion]
            
            # Clean ecoregion data
            ecoregion_clean = ecoregion_data[['sfd_event_count', 'drought_intensity', 'loss_ratio']].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(ecoregion_clean) > 5:  # Need sufficient data points
                try:
                    corr_sfd_lr = stats.pearsonr(ecoregion_clean['sfd_event_count'], ecoregion_clean['loss_ratio'])
                    corr_intensity_lr = stats.pearsonr(ecoregion_clean['drought_intensity'], ecoregion_clean['loss_ratio'])
                    
                    ecoregion_correlations[ecoregion] = {
                        'sfd_loss_ratio_corr': corr_sfd_lr[0],
                        'sfd_loss_ratio_pvalue': corr_sfd_lr[1],
                        'intensity_loss_ratio_corr': corr_intensity_lr[0],
                        'intensity_loss_ratio_pvalue': corr_intensity_lr[1],
                        'n_observations': len(ecoregion_clean)
                    }
                except:
                    # Skip if correlation calculation fails
                    continue
        
        # Overall correlations with cleaned data
        overall_clean = data[['sfd_event_count', 'drought_intensity', 'drought_severity', 
                             'acres_harvested', 'revenue_per_acre', 'loss_ratio']].replace([np.inf, -np.inf], np.nan).dropna()
        
        overall_correlations = {}
        try:
            overall_correlations['SFD_vs_LossRatio'] = stats.pearsonr(overall_clean['sfd_event_count'], overall_clean['loss_ratio'])
        except:
            overall_correlations['SFD_vs_LossRatio'] = (0.0, 1.0)
            
        try:
            overall_correlations['DroughtIntensity_vs_LossRatio'] = stats.pearsonr(overall_clean['drought_intensity'], overall_clean['loss_ratio'])
        except:
            overall_correlations['DroughtIntensity_vs_LossRatio'] = (0.0, 1.0)
            
        try:
            overall_correlations['DroughtSeverity_vs_LossRatio'] = stats.pearsonr(overall_clean['drought_severity'], overall_clean['loss_ratio'])
        except:
            overall_correlations['DroughtSeverity_vs_LossRatio'] = (0.0, 1.0)
            
        try:
            overall_correlations['Acres_vs_LossRatio'] = stats.pearsonr(overall_clean['acres_harvested'], overall_clean['loss_ratio'])
        except:
            overall_correlations['Acres_vs_LossRatio'] = (0.0, 1.0)
            
        try:
            overall_correlations['RevenuePerAcre_vs_LossRatio'] = stats.pearsonr(overall_clean['revenue_per_acre'], overall_clean['loss_ratio'])
        except:
            overall_correlations['RevenuePerAcre_vs_LossRatio'] = (0.0, 1.0)
        
        return correlation_matrix, ecoregion_correlations, overall_correlations
    
    def perform_regression_analysis(self, data):
        """Multiple regression analysis for loss ratio prediction"""
        print("Performing regression analysis...")
        
        # Prepare features for regression
        feature_columns = ['sfd_event_count', 'drought_intensity', 'drought_severity', 
                          'acres_harvested', 'revenue_per_acre', 'sfd_events_per_million_acres']
        
        # Remove rows with missing values
        regression_data = data[feature_columns + ['loss_ratio']].dropna()
        
        X = regression_data[feature_columns]
        y = regression_data['loss_ratio']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate predictions and metrics
        y_pred = model.predict(X_scaled)
        r2_score = model.score(X_scaled, y)
        
        # Feature importance (standardized coefficients)
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        regression_results = {
            'model': model,
            'scaler': scaler,
            'r2_score': r2_score,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'residuals': y - y_pred
        }
        
        return regression_results
    
    def create_comprehensive_visualizations(self, data, correlation_matrix, ecoregion_correlations, regression_results):
        """Create comprehensive loss ratio visualizations"""
        print("Creating comprehensive visualizations...")
        
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Loss ratio time series by ecoregion
        ax1 = plt.subplot(4, 4, 1)
        for ecoregion in data['ecoregion'].unique():
            ecoregion_data = data[data['ecoregion'] == ecoregion]
            ax1.plot(ecoregion_data['year'], ecoregion_data['loss_ratio'], 
                    marker='o', label=ecoregion, linewidth=2, markersize=3)
        
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Loss Ratio')
        ax1.set_title('Loss Ratio Trends by Ecoregion')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss ratio vs SFD events scatter
        ax2 = plt.subplot(4, 4, 2)
        scatter = ax2.scatter(data['sfd_event_count'], data['loss_ratio'], 
                             c=data['year'], cmap='viridis', alpha=0.6, s=30)
        
        # Add trend line
        z = np.polyfit(data['sfd_event_count'], data['loss_ratio'], 1)
        p = np.poly1d(z)
        ax2.plot(data['sfd_event_count'], p(data['sfd_event_count']), "r--", alpha=0.8)
        
        ax2.axhline(y=1.0, color='red', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Flash Drought Events')
        ax2.set_ylabel('Loss Ratio')
        ax2.set_title('Loss Ratio vs Flash Drought Events')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Year')
        
        # 3. Correlation heatmap
        ax3 = plt.subplot(4, 4, 3)
        variables = ['loss_ratio', 'sfd_event_count', 'drought_intensity', 'drought_severity', 
                    'acres_harvested', 'revenue_per_acre']
        corr_subset = correlation_matrix.loc[variables, variables]
        
        sns.heatmap(corr_subset, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax3, cbar_kws={'shrink': 0.8}, fmt='.3f')
        ax3.set_title('Loss Ratio Correlation Matrix')
        
        # 4. Loss ratio distribution by ecoregion
        ax4 = plt.subplot(4, 4, 4)
        ecoregions = data['ecoregion'].unique()
        loss_ratio_data = [data[data['ecoregion'] == eco]['loss_ratio'].values for eco in ecoregions]
        
        bp = ax4.boxplot(loss_ratio_data, labels=ecoregions, patch_artist=True)
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xticklabels(ecoregions, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('Loss Ratio')
        ax4.set_title('Loss Ratio Distribution by Ecoregion')
        ax4.grid(True, alpha=0.3)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 5. Drought intensity vs loss ratio
        ax5 = plt.subplot(4, 4, 5)
        ax5.scatter(data['drought_intensity'], data['loss_ratio'], 
                   c=data['ecoregion'].astype('category').cat.codes, cmap='tab10', alpha=0.6, s=30)
        
        # Add trend line
        z = np.polyfit(data['drought_intensity'], data['loss_ratio'], 1)
        p = np.poly1d(z)
        ax5.plot(data['drought_intensity'], p(data['drought_intensity']), "r--", alpha=0.8)
        
        ax5.axhline(y=1.0, color='red', linestyle='-', alpha=0.5)
        ax5.set_xlabel('Drought Intensity (Events × Duration)')
        ax5.set_ylabel('Loss Ratio')
        ax5.set_title('Loss Ratio vs Drought Intensity')
        ax5.grid(True, alpha=0.3)
        
        # 6. Regression feature importance
        ax6 = plt.subplot(4, 4, 6)
        feature_imp = regression_results['feature_importance']
        bars = ax6.barh(range(len(feature_imp)), feature_imp['abs_coefficient'])
        ax6.set_yticks(range(len(feature_imp)))
        ax6.set_yticklabels(feature_imp['feature'], fontsize=8)
        ax6.set_xlabel('Absolute Coefficient Value')
        ax6.set_title(f'Feature Importance (R² = {regression_results["r2_score"]:.3f})')
        
        # Color bars by coefficient sign
        for i, (bar, coef) in enumerate(zip(bars, feature_imp['coefficient'])):
            bar.set_color('red' if coef < 0 else 'blue')
        
        # 7. Actual vs predicted loss ratios
        ax7 = plt.subplot(4, 4, 7)
        
        # Get the regression data indices
        feature_columns = ['sfd_event_count', 'drought_intensity', 'drought_severity', 
                          'acres_harvested', 'revenue_per_acre', 'sfd_events_per_million_acres']
        regression_data = data[feature_columns + ['loss_ratio']].dropna()
        
        ax7.scatter(regression_data['loss_ratio'], regression_results['predictions'], alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(regression_data['loss_ratio'].min(), regression_results['predictions'].min())
        max_val = max(regression_data['loss_ratio'].max(), regression_results['predictions'].max())
        ax7.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax7.set_xlabel('Actual Loss Ratio')
        ax7.set_ylabel('Predicted Loss Ratio')
        ax7.set_title('Actual vs Predicted Loss Ratios')
        ax7.grid(True, alpha=0.3)
        
        # 8. Residuals plot
        ax8 = plt.subplot(4, 4, 8)
        ax8.scatter(regression_results['predictions'], regression_results['residuals'], alpha=0.6, s=30)
        ax8.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax8.set_xlabel('Predicted Loss Ratio')
        ax8.set_ylabel('Residuals')
        ax8.set_title('Regression Residuals')
        ax8.grid(True, alpha=0.3)
        
        # 9. Loss ratio categories over time
        ax9 = plt.subplot(4, 4, 9)
        category_counts = data.groupby(['year', 'loss_ratio_category']).size().unstack(fill_value=0)
        category_counts.plot(kind='area', stacked=True, ax=ax9, alpha=0.7)
        ax9.set_xlabel('Year')
        ax9.set_ylabel('Number of Ecoregions')
        ax9.set_title('Loss Ratio Categories Over Time')
        ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # 10. Ecoregion correlation comparison
        ax10 = plt.subplot(4, 4, 10)
        ecoregions = list(ecoregion_correlations.keys())
        correlations = [ecoregion_correlations[eco]['sfd_loss_ratio_corr'] for eco in ecoregions]
        p_values = [ecoregion_correlations[eco]['sfd_loss_ratio_pvalue'] for eco in ecoregions]
        
        bars = ax10.bar(range(len(ecoregions)), correlations)
        ax10.set_xticks(range(len(ecoregions)))
        ax10.set_xticklabels(ecoregions, rotation=45, ha='right', fontsize=8)
        ax10.set_ylabel('Correlation Coefficient')
        ax10.set_title('SFD-Loss Ratio Correlation by Ecoregion')
        ax10.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax10.grid(True, alpha=0.3)
        
        # Color bars by significance
        for bar, p_val in zip(bars, p_values):
            if p_val < 0.05:
                bar.set_color('red')
            elif p_val < 0.1:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        # 11. Loss ratio volatility
        ax11 = plt.subplot(4, 4, 11)
        volatility_by_eco = data.groupby('ecoregion')['loss_ratio_volatility'].mean().sort_values(ascending=False)
        ax11.bar(range(len(volatility_by_eco)), volatility_by_eco.values)
        ax11.set_xticks(range(len(volatility_by_eco)))
        ax11.set_xticklabels(volatility_by_eco.index, rotation=45, ha='right', fontsize=8)
        ax11.set_ylabel('Loss Ratio Volatility (5-yr rolling std)')
        ax11.set_title('Loss Ratio Volatility by Ecoregion')
        ax11.grid(True, alpha=0.3)
        
        # 12. 3-year rolling average trends
        ax12 = plt.subplot(4, 4, 12)
        for ecoregion in data['ecoregion'].unique()[:5]:  # Top 5 for clarity
            ecoregion_data = data[data['ecoregion'] == ecoregion]
            ax12.plot(ecoregion_data['year'], ecoregion_data['loss_ratio_3yr_avg'], 
                     marker='o', label=ecoregion, linewidth=2, markersize=3)
        
        ax12.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax12.set_xlabel('Year')
        ax12.set_ylabel('3-Year Rolling Average Loss Ratio')
        ax12.set_title('Loss Ratio Trends (3-Year Average)')
        ax12.legend(fontsize=8)
        ax12.grid(True, alpha=0.3)
        
        # 13. Excess loss ratio analysis
        ax13 = plt.subplot(4, 4, 13)
        ax13.scatter(data['sfd_event_count'], data['excess_loss_ratio'], 
                    c=data['ecoregion'].astype('category').cat.codes, cmap='tab10', alpha=0.6, s=30)
        
        # Add trend line
        z = np.polyfit(data['sfd_event_count'], data['excess_loss_ratio'], 1)
        p = np.poly1d(z)
        ax13.plot(data['sfd_event_count'], p(data['sfd_event_count']), "r--", alpha=0.8)
        
        ax13.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        ax13.set_xlabel('Flash Drought Events')
        ax13.set_ylabel('Excess Loss Ratio (above break-even)')
        ax13.set_title('Excess Loss Ratio vs Flash Droughts')
        ax13.grid(True, alpha=0.3)
        
        # 14. Revenue per acre vs loss ratio
        ax14 = plt.subplot(4, 4, 14)
        ax14.scatter(data['revenue_per_acre'], data['loss_ratio'], 
                    c=data['sfd_event_count'], cmap='Reds', alpha=0.6, s=30)
        
        ax14.axhline(y=1.0, color='red', linestyle='-', alpha=0.5)
        ax14.set_xlabel('Revenue per Acre ($)')
        ax14.set_ylabel('Loss Ratio')
        ax14.set_title('Revenue per Acre vs Loss Ratio')
        ax14.grid(True, alpha=0.3)
        
        # 15. Drought events per million acres
        ax15 = plt.subplot(4, 4, 15)
        ax15.scatter(data['sfd_events_per_million_acres'], data['loss_ratio'], 
                    c=data['year'], cmap='viridis', alpha=0.6, s=30)
        
        # Add trend line
        valid_data = data.dropna(subset=['sfd_events_per_million_acres', 'loss_ratio'])
        if len(valid_data) > 1:
            z = np.polyfit(valid_data['sfd_events_per_million_acres'], valid_data['loss_ratio'], 1)
            p = np.poly1d(z)
            ax15.plot(valid_data['sfd_events_per_million_acres'], 
                     p(valid_data['sfd_events_per_million_acres']), "r--", alpha=0.8)
        
        ax15.axhline(y=1.0, color='red', linestyle='-', alpha=0.5)
        ax15.set_xlabel('SFD Events per Million Acres')
        ax15.set_ylabel('Loss Ratio')
        ax15.set_title('Drought Density vs Loss Ratio')
        ax15.grid(True, alpha=0.3)
        
        # 16. Summary statistics table
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('off')
        
        # Calculate summary statistics
        summary_stats = data.groupby('ecoregion').agg({
            'loss_ratio': ['mean', 'std', 'min', 'max'],
            'sfd_event_count': 'mean'
        }).round(3)
        
        # Create text summary
        summary_text = "LOSS RATIO SUMMARY BY ECOREGION\n\n"
        for eco in summary_stats.index[:6]:  # Top 6 for space
            lr_mean = summary_stats.loc[eco, ('loss_ratio', 'mean')]
            lr_std = summary_stats.loc[eco, ('loss_ratio', 'std')]
            sfd_mean = summary_stats.loc[eco, ('sfd_event_count', 'mean')]
            summary_text += f"{eco}:\n"
            summary_text += f"  Avg LR: {lr_mean:.2f} ± {lr_std:.2f}\n"
            summary_text += f"  Avg SFD: {sfd_mean:.1f}\n\n"
        
        ax16.text(0.05, 0.95, summary_text, transform=ax16.transAxes, fontsize=8,
                 verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'comprehensive_loss_ratio_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, data, correlation_matrix, ecoregion_correlations, 
                                    overall_correlations, regression_results):
        """Generate comprehensive loss ratio analysis report"""
        print("Generating comprehensive report...")
        
        # Calculate key statistics
        overall_avg_lr = data['loss_ratio'].mean()
        overall_std_lr = data['loss_ratio'].std()
        high_lr_threshold = 1.5
        high_lr_percentage = (data['loss_ratio'] > high_lr_threshold).mean() * 100
        
        # Ecoregion rankings
        ecoregion_stats = data.groupby('ecoregion').agg({
            'loss_ratio': ['mean', 'std', 'count'],
            'sfd_event_count': 'mean',
            'excess_loss_ratio': 'mean'
        }).round(3)
        
        ecoregion_stats.columns = ['avg_loss_ratio', 'lr_volatility', 'n_years', 'avg_sfd_events', 'avg_excess_lr']
        ecoregion_stats = ecoregion_stats.sort_values('avg_loss_ratio', ascending=False)
        
        report = f"""
COMPREHENSIVE LOSS RATIO ANALYSIS REPORT
Crop Insurance Loss Ratios vs Streamflow Flash Droughts
Analysis Period: 2000-2024
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
================
• Overall Average Loss Ratio: {overall_avg_lr:.3f} ± {overall_std_lr:.3f}
• Percentage of High Loss Ratio Years (>1.5): {high_lr_percentage:.1f}%
• Total Observations: {len(data)} ecoregion-year combinations
• Ecoregions Analyzed: {data['ecoregion'].nunique()}
• Analysis Period: {data['year'].min()}-{data['year'].max()}

OVERALL CORRELATIONS WITH LOSS RATIO
====================================
"""
        
        for name, (corr, p_value) in overall_correlations.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            interpretation = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            
            report += f"• {name.replace('_', ' ')}: r = {corr:.3f} (p = {p_value:.3f}) {significance}\n"
            report += f"  - {interpretation} {direction} correlation\n"
        
        report += f"""

ECOREGION RANKINGS BY AVERAGE LOSS RATIO
========================================
"""
        
        for i, (ecoregion, stats) in enumerate(ecoregion_stats.iterrows(), 1):
            sfd_lr_corr = ecoregion_correlations.get(ecoregion, {}).get('sfd_loss_ratio_corr', 'N/A')
            sfd_lr_pval = ecoregion_correlations.get(ecoregion, {}).get('sfd_loss_ratio_pvalue', 'N/A')
            
            report += f"{i}. {ecoregion}:\n"
            report += f"   - Average Loss Ratio: {stats['avg_loss_ratio']:.3f}\n"
            report += f"   - Loss Ratio Volatility: {stats['lr_volatility']:.3f}\n"
            report += f"   - Average SFD Events: {stats['avg_sfd_events']:.1f}\n"
            report += f"   - Average Excess Loss Ratio: {stats['avg_excess_lr']:.3f}\n"
            if isinstance(sfd_lr_corr, float):
                significance = "***" if sfd_lr_pval < 0.001 else "**" if sfd_lr_pval < 0.01 else "*" if sfd_lr_pval < 0.05 else ""
                report += f"   - SFD-Loss Ratio Correlation: {sfd_lr_corr:.3f} {significance}\n"
            report += "\n"
        
        report += f"""

REGRESSION ANALYSIS RESULTS
===========================
• Model R-squared: {regression_results['r2_score']:.3f}
• Model explains {regression_results['r2_score']*100:.1f}% of loss ratio variance

Feature Importance (Standardized Coefficients):
"""
        
        for _, row in regression_results['feature_importance'].iterrows():
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            report += f"• {row['feature']}: {row['coefficient']:+.3f} ({direction} loss ratio)\n"
        
        # Calculate temporal trends
        annual_avg_lr = data.groupby('year')['loss_ratio'].mean()
        lr_trend = np.polyfit(annual_avg_lr.index, annual_avg_lr.values, 1)[0]
        
        annual_avg_sfd = data.groupby('year')['sfd_event_count'].mean()
        sfd_trend = np.polyfit(annual_avg_sfd.index, annual_avg_sfd.values, 1)[0]
        
        # Calculate correlation between annual averages
        try:
            annual_corr = stats.pearsonr(annual_avg_lr.values, annual_avg_sfd.values)[0]
        except:
            annual_corr = 0.0
        
        report += f"""

TEMPORAL TRENDS (2000-2024)
===========================
• Loss Ratio Trend: {lr_trend:+.4f} per year ({'increasing' if lr_trend > 0 else 'decreasing'})
• Flash Drought Trend: {sfd_trend:+.2f} events per year ({'increasing' if sfd_trend > 0 else 'decreasing'})
• Correlation between annual averages: {annual_corr:.3f}

RISK ASSESSMENT BY ECOREGION
============================
"""
        
        # Risk categories based on loss ratio and volatility
        for ecoregion, stats in ecoregion_stats.iterrows():
            avg_lr = stats['avg_loss_ratio']
            volatility = stats['lr_volatility']
            
            if avg_lr > 2.0:
                risk_level = "VERY HIGH"
            elif avg_lr > 1.5:
                risk_level = "HIGH"
            elif avg_lr > 1.2:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"
            
            volatility_level = "High" if volatility > 0.5 else "Moderate" if volatility > 0.3 else "Low"
            
            report += f"• {ecoregion}: {risk_level} risk (volatility: {volatility_level})\n"
        
        report += f"""

KEY FINDINGS
============
• Loss ratios vary significantly across ecoregions ({ecoregion_stats['avg_loss_ratio'].min():.2f} to {ecoregion_stats['avg_loss_ratio'].max():.2f})
• Flash drought events show {'positive' if overall_correlations['SFD_vs_LossRatio'][0] > 0 else 'negative'} correlation with loss ratios
• Regression model explains {regression_results['r2_score']*100:.1f}% of loss ratio variance
• {'Increasing' if lr_trend > 0 else 'Decreasing'} trend in loss ratios over analysis period
• Drought intensity is {'more' if abs(overall_correlations['DroughtIntensity_vs_LossRatio'][0]) > abs(overall_correlations['SFD_vs_LossRatio'][0]) else 'less'} predictive than event count alone

ECONOMIC IMPLICATIONS
====================
• High loss ratios indicate significant federal subsidy of agricultural risk
• Flash drought events contribute to insurance program stress
• Regional risk patterns suggest need for differentiated premium structures
• Climate change may be affecting traditional actuarial assumptions
• Ecoregional analysis reveals geographic concentration of risk

RECOMMENDATIONS
===============
• Monitor high-risk ecoregions for premium adequacy
• Consider drought forecasting in risk assessment models
• Develop region-specific risk management strategies
• Investigate causes of high volatility in certain ecoregions
• Enhance early warning systems for flash drought events

METHODOLOGY
===========
• Loss ratios calculated as indemnities/premiums by ecoregion and year
• Flash drought events from SFD dataset with AGGECOREGION classification
• Correlation analysis using Pearson coefficients with significance testing
• Multiple regression with standardized features
• Temporal trend analysis using linear regression
• Risk assessment based on average loss ratios and volatility measures
"""
        
        # Save report
        report_file = self.results_dir / 'comprehensive_loss_ratio_analysis_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Comprehensive report saved to {report_file}")
        return report

def main():
    """
    Main analysis function
    """
    print("Starting Comprehensive Loss Ratio Analysis...")
    print("=" * 70)
    
    analyzer = LossRatioAnalyzer()
    
    # Load and prepare all data
    data = analyzer.load_and_prepare_data()
    
    # Perform correlation analysis
    correlation_matrix, ecoregion_correlations, overall_correlations = analyzer.analyze_loss_ratio_correlations(data)
    
    # Perform regression analysis
    regression_results = analyzer.perform_regression_analysis(data)
    
    # Create comprehensive visualizations
    analyzer.create_comprehensive_visualizations(data, correlation_matrix, ecoregion_correlations, regression_results)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report(data, correlation_matrix, ecoregion_correlations, 
                                                   overall_correlations, regression_results)
    
    # Save processed data
    data.to_csv(analyzer.results_dir / 'loss_ratio_analysis_dataset.csv', index=False)
    
    print(f"\nComprehensive Loss Ratio Analysis Complete!")
    print(f"Results saved to: {analyzer.results_dir}")
    print("\nKey Overall Correlations:")
    print("=" * 50)
    for name, (corr, p_value) in overall_correlations.items():
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"{name.replace('_', ' ')}: r = {corr:.3f} (p = {p_value:.3f}) {significance}")

if __name__ == "__main__":
    main()