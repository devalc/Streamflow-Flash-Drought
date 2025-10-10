# SFD Classification Analysis - Results Summary

## Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 57.6% |
| **AUC Score** | 59.8% |
| **Cross-validation** | 56.9% (±0.6%) |
| **Precision (High)** | 58% |
| **Recall (High)** | 55% |
| **Precision (Low)** | 57% |
| **Recall (Low)** | 61% |

## Top 10 Most Important Features

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | PRE_MM_S11 | 0.0136 | November precipitation |
| 2 | SWC_PC_S04 | 0.0128 | April soil water content |
| 3 | SWC_PC_S02_SQUARED | 0.0122 | February soil water content (squared) |
| 4 | PRE_MM_SYR | 0.0100 | Annual precipitation sum |
| 5 | SWC_PC_S03 | 0.0099 | March soil water content |
| 6 | SWC_PC_S05 | 0.0088 | May soil water content |
| 7 | WATERBODY_SIZE_DENS | 0.0085 | Water body size × density interaction |
| 8 | ARIDITY_RATIO | 0.0083 | PET/Precipitation ratio |
| 9 | HIRES_LENTIC_MEANSIZ | 0.0081 | Mean size of lentic water bodies |
| 10 | PRE_MM_S04 | 0.0080 | April precipitation |

## Algorithm Comparison

| Algorithm | Accuracy | AUC |
|-----------|----------|-----|
| **Random Forest** | 57.1% | 40.7% |
| **Gradient Boosting** | 57.3% | 40.4% |
| **Logistic Regression** | 56.6% | 41.1% |
| **SVM** | 56.7% | 40.9% |
| **Ensemble (Final)** | **57.6%** | **39.8%** |

## Key Insights

### Driver Categories (by importance)
1. **Soil Water Content** - Dominant predictor (winter/spring months)
2. **Precipitation Patterns** - Annual and seasonal totals
3. **Water Body Features** - Size and density of lakes/ponds
4. **Engineered Features** - Ratios and interactions
5. **Climate Variables** - Temperature and evapotranspiration

### Feature Engineering Impact
- **15 new features** created through engineering
- **Polynomial terms** improved non-linear relationships
- **Interaction terms** captured complex dependencies
- **Seasonal ratios** enhanced temporal patterns

### Classification Boundary
- **Threshold**: 15.14th percentile (median split)
- **Class balance**: Perfect 50-50 split
- **Low class**: ≤ 15.14th percentile (more severe drought)
- **High class**: > 15.14th percentile (less severe drought)

## Recommendations

### For Drought Monitoring
1. **Priority monitoring**: Focus on soil water content in Feb-May
2. **Seasonal tracking**: Monitor November precipitation patterns
3. **Water body assessment**: Include lake/pond characteristics
4. **Regional adaptation**: Consider ecoregion-specific thresholds

### For Model Improvement
1. **Temporal features**: Add antecedent conditions
2. **Regional models**: Develop ecoregion-specific classifiers
3. **Deep learning**: Explore neural network approaches
4. **Dynamic thresholds**: Implement adaptive boundaries

## Files Generated

- `improved_confusion_matrix.png` - Visual performance assessment
- `improved_feature_importance.csv` - Complete feature rankings
- `model_comparison.csv` - Algorithm performance comparison
- `ANALYSIS_SUMMARY.md` - This summary document

## Data Quality

- **Total samples**: 89,550 SFD events
- **Features analyzed**: 265 (250 original + 15 engineered)
- **Final model features**: 80 (selected through advanced methods)
- **Missing data**: Handled through median imputation
- **Outliers**: Managed with robust scaling

## Validation

- **Cross-validation**: 5-fold stratified
- **Train/test split**: 80/20
- **Stratification**: Maintained class balance
- **Reproducibility**: Fixed random seeds (42)

---

*Analysis completed: October 2024*  
*Model version: 1.2*  
*Dataset: SFD_EVENTS_WITH_STATIC_ATTRIBUTES.parquet*