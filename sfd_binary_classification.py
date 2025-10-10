import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading SFD events with static attributes...")
df = pd.read_parquet('data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Unique ecoregions: {df['AGGECOREGION'].nunique()}")

# Check target variable
target_var = 'Mean_SFD_Flow_Percentile'
if target_var not in df.columns:
    print(f"Error: {target_var} not found in dataset!")
    print(f"Available columns: {list(df.columns)}")
    exit()

print(f"\n{target_var} statistics:")
print(f"Mean: {df[target_var].mean():.2f}")
print(f"Std: {df[target_var].std():.2f}")
print(f"Min: {df[target_var].min():.2f}")
print(f"Max: {df[target_var].max():.2f}")
print(f"25th percentile: {df[target_var].quantile(0.25):.2f}")
print(f"50th percentile: {df[target_var].quantile(0.50):.2f}")
print(f"75th percentile: {df[target_var].quantile(0.75):.2f}")

# Categorize Mean_SFD_Flow_Percentile into 2 classes based on median
def categorize_flow_percentile_binary(value, median_val):
    if value <= median_val:
        return 'Low'
    else:
        return 'High'

# Calculate median for binary split
median_val = df[target_var].median()

print(f"\nBinary class boundary:")
print(f"Low: <= {median_val:.2f}")
print(f"High: > {median_val:.2f}")

# Create binary categorical target
df['Flow_Category'] = df[target_var].apply(lambda x: categorize_flow_percentile_binary(x, median_val))

# Check class distribution
class_counts = df['Flow_Category'].value_counts()
print(f"\nClass distribution:")
for category, count in class_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{category}: {count} ({percentage:.1f}%)")

# Exclude specified columns
exclude_cols = ['Station_ID', 'Onset_Time', 'Duration_Days', 'Onset_Rate_Days', target_var, 'Flow_Category']
exclude_cols += [col for col in df.columns if any(term in col.lower() for term in ['station', 'onset_time', 'id'])]
exclude_cols = [col for col in exclude_cols if col in df.columns]

print(f"\nExcluding columns: {exclude_cols}")

# Get potential driver columns (numeric columns excluding targets and excluded columns)
driver_cols = [col for col in df.columns if col not in exclude_cols + ['AGGECOREGION']]
numeric_cols = df[driver_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\nPotential driver columns ({len(numeric_cols)}): {numeric_cols[:10]}...")

def build_binary_classification_model(df, target_col, feature_cols, test_size=0.2):
    """Build and evaluate a binary Random Forest classification model"""
    
    print(f"\n{'='*60}")
    print("BUILDING BINARY CLASSIFICATION MODEL")
    print(f"{'='*60}")
    
    # Prepare data
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove any remaining NaN or infinite values
    mask = np.isfinite(X).all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"Clean dataset size: {len(X_clean)} samples")
    print(f"Features: {len(feature_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Build Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    print("\nTraining Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 20 Most Important Features:")
    print("-" * 50)
    for i, row in feature_importance.head(20).iterrows():
        print(f"{row['feature']:<30}: {row['importance']:.4f}")
    
    return rf_model, feature_importance, X_test, y_test, y_pred, y_pred_proba

def analyze_by_ecoregion_binary(df, target_col, feature_cols):
    """Analyze feature importance by ecoregion for binary classification"""
    
    print(f"\n{'='*60}")
    print("ECOREGION-SPECIFIC BINARY ANALYSIS")
    print(f"{'='*60}")
    
    ecoregion_results = {}
    
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.isna(ecoregion):
            continue
            
        eco_df = df[df['AGGECOREGION'] == ecoregion].copy()
        
        if len(eco_df) < 50:  # Skip if too few samples
            print(f"\nSkipping {ecoregion}: insufficient samples ({len(eco_df)})")
            continue
        
        print(f"\n{ecoregion} (n={len(eco_df)}):")
        
        # Check class distribution in this ecoregion
        class_dist = eco_df[target_col].value_counts()
        print(f"  Class distribution: {dict(class_dist)}")
        
        # Skip if any class has too few samples
        if class_dist.min() < 10:
            print(f"  Skipping: some classes have < 10 samples")
            continue
        
        # Prepare data
        X_eco = eco_df[feature_cols].fillna(eco_df[feature_cols].median())
        y_eco = eco_df[target_col]
        
        # Remove any remaining NaN or infinite values
        mask_eco = np.isfinite(X_eco).all(axis=1) & y_eco.notna()
        X_eco_clean = X_eco[mask_eco]
        y_eco_clean = y_eco[mask_eco]
        
        if len(X_eco_clean) < 30:
            print(f"  Skipping: insufficient clean samples ({len(X_eco_clean)})")
            continue
        
        try:
            # Build model for this ecoregion
            rf_eco = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_eco.fit(X_eco_clean, y_eco_clean)
            
            # Get feature importance
            importance_eco = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_eco.feature_importances_
            }).sort_values('importance', ascending=False)
            
            ecoregion_results[ecoregion] = importance_eco.head(10)
            
            # Calculate metrics
            y_pred_eco = rf_eco.predict(X_eco_clean)
            accuracy_eco = accuracy_score(y_eco_clean, y_pred_eco)
            
            # Calculate AUC if possible
            try:
                y_pred_proba_eco = rf_eco.predict_proba(X_eco_clean)
                auc_eco = roc_auc_score(y_eco_clean, y_pred_proba_eco[:, 1])
                print(f"  Accuracy: {accuracy_eco:.4f}, AUC: {auc_eco:.4f}")
            except:
                print(f"  Accuracy: {accuracy_eco:.4f}")
            
            print("  Top 5 drivers:")
            for i, row in importance_eco.head(5).iterrows():
                print(f"    {row['feature']:<25}: {row['importance']:.4f}")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return ecoregion_results

def create_binary_visualizations(df, target_col, feature_importance, ecoregion_results, y_test, y_pred_proba):
    """Create comprehensive visualizations for binary classification"""
    
    print(f"\n{'='*60}")
    print("CREATING BINARY VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Class distribution overall
    ax1 = plt.subplot(3, 3, 1)
    class_counts = df[target_col].value_counts()
    colors = ['#ff7f0e', '#2ca02c']
    ax1.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', colors=colors)
    ax1.set_title('Overall Class Distribution')
    
    # Plot 2: Class distribution by ecoregion
    ax2 = plt.subplot(3, 3, 2)
    class_by_eco = pd.crosstab(df['AGGECOREGION'], df[target_col], normalize='index') * 100
    class_by_eco.plot(kind='bar', stacked=True, ax=ax2, color=colors)
    ax2.set_title('Class Distribution by Ecoregion (%)')
    ax2.set_xlabel('Ecoregion')
    ax2.set_ylabel('Percentage')
    ax2.legend(title='Flow Category')
    plt.xticks(rotation=45)
    
    # Plot 3: Top 15 feature importance
    ax3 = plt.subplot(3, 3, 3)
    top_features = feature_importance.head(15)
    ax3.barh(range(len(top_features)), top_features['importance'])
    ax3.set_yticks(range(len(top_features)))
    ax3.set_yticklabels([f[:20] for f in top_features['feature']])
    ax3.set_xlabel('Feature Importance')
    ax3.set_title('Top 15 Most Important Features')
    ax3.invert_yaxis()
    
    # Plot 4: Target variable distribution with binary split
    ax4 = plt.subplot(3, 3, 4)
    df['Mean_SFD_Flow_Percentile'].hist(bins=50, alpha=0.7, ax=ax4)
    median_val = df['Mean_SFD_Flow_Percentile'].median()
    ax4.axvline(median_val, color='red', linestyle='--', label=f'Median: {median_val:.2f}')
    ax4.set_xlabel('Mean SFD Flow Percentile')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution with Binary Split')
    ax4.legend()
    
    # Plot 5: ROC Curve
    ax5 = plt.subplot(3, 3, 5)
    from sklearn.metrics import roc_curve, auc
    
    # Convert string labels to binary for ROC curve
    y_test_binary = (y_test == 'High').astype(int)
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    ax5.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax5.set_xlim([0.0, 1.0])
    ax5.set_ylim([0.0, 1.05])
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve')
    ax5.legend(loc="lower right")
    
    # Plot 6: Boxplot by ecoregion
    ax6 = plt.subplot(3, 3, 6)
    df.boxplot(column='Mean_SFD_Flow_Percentile', by='AGGECOREGION', ax=ax6)
    ax6.axhline(median_val, color='red', linestyle='--', alpha=0.7)
    ax6.set_xlabel('Ecoregion')
    ax6.set_ylabel('Mean SFD Flow Percentile')
    ax6.set_title('Flow Percentile by Ecoregion')
    plt.xticks(rotation=45)
    plt.suptitle('')  # Remove default title
    
    # Plot 7: Feature importance heatmap by ecoregion
    if ecoregion_results:
        ax7 = plt.subplot(3, 2, 5)
        
        # Get top features across all ecoregions
        all_features = set()
        for eco_importance in ecoregion_results.values():
            all_features.update(eco_importance.head(8)['feature'])
        
        top_features_list = list(all_features)[:15]
        
        # Create importance matrix
        importance_matrix = []
        ecoregion_names = []
        
        for eco_name, eco_importance in ecoregion_results.items():
            eco_dict = eco_importance.set_index('feature')['importance'].to_dict()
            row = [eco_dict.get(feat, 0) for feat in top_features_list]
            importance_matrix.append(row)
            ecoregion_names.append(eco_name)
        
        if importance_matrix:
            sns.heatmap(importance_matrix, 
                       xticklabels=[f[:15] for f in top_features_list], 
                       yticklabels=ecoregion_names,
                       annot=True, fmt='.3f', cmap='viridis', ax=ax7)
            ax7.set_title('Feature Importance by Ecoregion')
            plt.xticks(rotation=45, ha='right')
    
    # Plot 8: Class means for top features
    ax8 = plt.subplot(3, 2, 6)
    top_5_features = feature_importance.head(5)['feature'].tolist()
    
    class_means = df.groupby(target_col)[top_5_features].mean()
    class_means.T.plot(kind='bar', ax=ax8, color=colors)
    ax8.set_title('Feature Means by Class')
    ax8.set_xlabel('Features')
    ax8.set_ylabel('Mean Value')
    ax8.legend(title='Flow Category')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('sfd_binary_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the binary analysis
if __name__ == "__main__":
    
    # Build binary classification model
    model, feature_importance, X_test, y_test, y_pred, y_pred_proba = build_binary_classification_model(
        df, 'Flow_Category', numeric_cols
    )
    
    # Analyze by ecoregion
    ecoregion_results = analyze_by_ecoregion_binary(df, 'Flow_Category', numeric_cols)
    
    # Create visualizations
    create_binary_visualizations(df, 'Flow_Category', feature_importance, ecoregion_results, y_test, y_pred_proba)
    
    # Print confusion matrix
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print(f"{'='*60}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['High', 'Low'], 
                yticklabels=['High', 'Low'])
    plt.title('Binary Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Final summary
    print(f"\n{'='*60}")
    print("BINARY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    median_val = df['Mean_SFD_Flow_Percentile'].median()
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score((y_test == 'High').astype(int), y_pred_proba[:, 1])
    
    print(f"Dataset: {len(df)} SFD events across {df['AGGECOREGION'].nunique()} ecoregions")
    print(f"Target variable: {target_var}")
    print(f"Binary split at median: {median_val:.2f}")
    print(f"Low: ≤ {median_val:.2f}, High: > {median_val:.2f}")
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Features analyzed: {len(numeric_cols)}")
    
    print(f"\nTop 5 Most Important Drivers:")
    for i, row in feature_importance.head(5).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print(f"\nEcoregions analyzed: {len(ecoregion_results)}")
    
    # Class distribution by ecoregion
    print(f"\nClass Distribution by Ecoregion:")
    for ecoregion in df['AGGECOREGION'].unique():
        if pd.notna(ecoregion):
            eco_df = df[df['AGGECOREGION'] == ecoregion]
            class_dist = eco_df['Flow_Category'].value_counts()
            total = len(eco_df)
            low_pct = (class_dist.get('Low', 0) / total) * 100
            high_pct = (class_dist.get('High', 0) / total) * 100
            print(f"  {ecoregion}: Low {low_pct:.1f}%, High {high_pct:.1f}%")
    
    # Save results
    print(f"\nSaving results...")
    feature_importance.to_csv('binary_feature_importance.csv', index=False)
    
    # Save ecoregion results
    with open('binary_ecoregion_results.txt', 'w') as f:
        f.write("BINARY CLASSIFICATION - ECOREGION-SPECIFIC FEATURE IMPORTANCE\n")
        f.write("="*60 + "\n\n")
        
        for eco_name, eco_importance in ecoregion_results.items():
            f.write(f"{eco_name}:\n")
            for i, row in eco_importance.head(10).iterrows():
                f.write(f"  {row['feature']:<30}: {row['importance']:.4f}\n")
            f.write("\n")
    
    print("Binary analysis complete!")
    print("Files saved:")
    print("  - sfd_binary_classification_analysis.png")
    print("  - binary_confusion_matrix.png") 
    print("  - binary_feature_importance.csv")
    print("  - binary_ecoregion_results.txt")