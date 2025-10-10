# Streamflow Flash Drought Classification Analysis

## Overview

This analysis implements machine learning classification to predict Mean SFD Flow Percentile categories (Low vs High) using environmental and hydrological drivers across 9 ecoregions in the United States.

## Project Structure

```
├── analysis/
│   └── improved_sfd_classification.py    # Main analysis script
├── results/
│   ├── improved_confusion_matrix.png     # Model performance visualization
│   ├── improved_feature_importance.csv   # Feature rankings
│   └── model_comparison.csv              # Algorithm comparison
├── docs/
│   └── SFD_ANALYSIS_README.md            # This documentation
├── data/
│   └── SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet  # Input dataset
└── src/
    └── (existing project files)
```

## Key Results

### Model Performance
- **Accuracy**: 57.6% (binary classification)
- **AUC Score**: 59.8%
- **Cross-validation**: 56.9% (±0.6%)
- **Approach**: Ensemble model (Random Forest + Gradient Boosting + Logistic Regression)

### Top Predictive Features
1. **PRE_MM_S11** - November precipitation
2. **SWC_PC_S04** - April soil water content
3. **SWC_PC_S02_SQUARED** - February soil water content (squared)
4. **PRE_MM_SYR** - Annual precipitation sum
5. **SWC_PC_S03** - March soil water content

### Key Insights
- **Soil water content** (especially winter/spring) is the strongest predictor category
- **Precipitation patterns** (annual and seasonal) are critical
- **Water body characteristics** provide significant predictive power
- **Feature engineering** improved model performance
- **Regional differences** exist across the 9 ecoregions

## Classification Setup

### Target Variable
- **Mean_SFD_Flow_Percentile**: Average flow percentile during P20-to-P40 recovery period
- **Binary Split**: Median threshold at 15.14th percentile
  - **Low**: ≤ 15.14th percentile (50% of events)
  - **High**: > 15.14th percentile (50% of events)

### Input Features
- **250+ environmental drivers** including:
  - Climate variables (precipitation, temperature, evapotranspiration)
  - Soil properties (water content, organic carbon, texture)
  - Topographic features (elevation, slope, aspect)
  - Land cover and land use
  - Water body characteristics
  - Human impact indicators

## Methodology

### 1. Feature Engineering
- **Seasonal ratios**: November precipitation / Annual precipitation
- **Seasonal differences**: Winter-Summer soil water content
- **Interaction terms**: Precipitation × Elevation
- **Polynomial features**: Squared terms for key variables
- **Binned categories**: Elevation bins with dummy encoding

### 2. Feature Selection
- **Statistical selection**: F-test for feature relevance
- **Random Forest importance**: Tree-based feature ranking
- **Recursive Feature Elimination**: Iterative feature removal
- **Combined approach**: Intersection of multiple methods
- **Final selection**: 80 most important features

### 3. Model Development
- **Ensemble approach**: Voting classifier combining:
  - Random Forest (300 trees, optimized parameters)
  - Gradient Boosting (200 estimators)
  - Logistic Regression (L2 regularization)
- **Cross-validation**: 5-fold stratified validation
- **Preprocessing**: Robust scaling for linear models

### 4. Evaluation
- **Accuracy**: Overall classification performance
- **AUC Score**: Area under ROC curve
- **Precision/Recall**: Class-specific performance
- **Confusion Matrix**: Detailed error analysis

## Usage

### Running the Analysis
```bash
cd analysis/
python improved_sfd_classification.py
```

### Requirements
- Python 3.7+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- Required data file: `data/SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet`

### Outputs
- **Confusion matrix**: Visual model performance
- **Feature importance**: CSV with ranked features
- **Model comparison**: Performance across algorithms
- **Console output**: Detailed results and metrics

## Applications

### Drought Monitoring
- **Early warning systems**: Monitor key predictive features
- **Risk assessment**: Identify high-risk periods and regions
- **Resource allocation**: Target monitoring efforts

### Water Management
- **Reservoir operations**: Anticipate drought severity
- **Agricultural planning**: Seasonal drought risk assessment
- **Emergency preparedness**: Advance warning for severe events

### Research Applications
- **Climate studies**: Understanding drought drivers
- **Hydrological modeling**: Feature importance insights
- **Regional analysis**: Ecoregion-specific patterns

## Limitations

1. **Moderate accuracy** (57.6%) indicates remaining unexplained variance
2. **Static features** don't capture temporal dynamics
3. **Linear threshold** may not reflect natural boundaries
4. **Regional variability** suggests need for specialized models

## Future Improvements

1. **Temporal features**: Include antecedent conditions
2. **Deep learning**: Neural networks for complex patterns
3. **Regional models**: Ecoregion-specific classifiers
4. **Dynamic thresholds**: Adaptive classification boundaries
5. **Ensemble expansion**: Additional algorithm integration

## Contact & Support

For questions about this analysis or to request modifications, please refer to the project repository or contact the development team.

## Version History

- **v1.0**: Initial binary classification implementation
- **v1.1**: Feature engineering and ensemble modeling
- **v1.2**: Organized project structure and documentation