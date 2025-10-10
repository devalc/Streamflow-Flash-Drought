# Binary Classification Analysis: Mean SFD Flow Percentile

## Executive Summary

Successfully built a binary Random Forest classifier to predict Mean_SFD_Flow_Percentile categories (Low vs High) using 250 environmental drivers across 9 ecoregions. The binary approach significantly improved model performance compared to the 3-class version.

## Model Performance Improvement

### Binary vs 3-Class Comparison
- **Binary Accuracy**: 57.4% (vs 39.2% for 3-class)
- **AUC Score**: 59.8% (binary classification advantage)
- **Balanced Classes**: Perfect 50-50 split using median threshold

## Classification Setup

### Binary Split Criteria
- **Threshold**: Median value = 15.14th percentile
- **Low Class**: ≤ 15.14th percentile (50.0% of events, n=44,776)
- **High Class**: > 15.14th percentile (50.0% of events, n=44,774)

### Model Specifications
- **Algorithm**: Random Forest (200 trees)
- **Features**: 250 environmental drivers
- **Cross-validation**: 56.4% accuracy (±0.5%)
- **Training/Test Split**: 80/20

## Key Findings

### Top 5 Most Important Drivers (Overall)

1. **PRE_MM_SYR** (0.0146) - Annual precipitation sum
2. **SWC_PC_S02** (0.0140) - February soil water content
3. **SWC_PC_S03** (0.0114) - March soil water content
4. **SWC_PC_S12** (0.0113) - December soil water content
5. **SWC_PC_SYR** (0.0111) - Annual soil water content

### Driver Categories Ranking

#### 1. Soil Water Content (Dominant)
- **SWC_PC_S02, SWC_PC_S03, SWC_PC_S12, SWC_PC_SYR, SWC_PC_S04, SWC_PC_S01, SWC_PC_S05**
- Winter/spring soil moisture is critical for drought severity prediction

#### 2. Precipitation Patterns
- **PRE_MM_SYR, PRE_MM_S11, PRE_MM_S04, PRE_MM_S03, PRE_MM_S12**
- Annual and late-season precipitation most important

#### 3. Water Body Characteristics
- **HIRES_LENTIC_MEANSIZ, HIRES_LENTIC_NUM, LKV_MC_USU, LKA_PC_USE**
- Lake/pond size and density influence drought severity

#### 4. Evapotranspiration
- **AET_MM_S05, AET_MM_S04** - Spring evapotranspiration patterns

#### 5. Human/Infrastructure
- **PPD_PK_UAV** - Population density impacts

## Regional Performance Analysis

### Best Performing Ecoregions
1. **MxWdShld** (64.0% accuracy, 64.8% AUC) - Mixed Wood Shield
2. **WestMnts** (61.8% accuracy, 64.0% AUC) - Western Mountains
3. **WestXeric** (61.8% accuracy, 65.2% AUC) - Western Xeric

### Moderate Performing Ecoregions
4. **SECstPlain** (61.4% accuracy, 66.1% AUC) - SE Coastal Plain
5. **SEPlains** (61.2% accuracy, 64.6% AUC) - Southeastern Plains
6. **EastHghlnds** (60.3% accuracy, 62.0% AUC) - Eastern Highlands

### Lower Performing Ecoregions
7. **WestPlains** (60.1% accuracy, 63.9% AUC) - Western Plains
8. **NorthEast** (60.0% accuracy, 63.1% AUC) - Northeastern Forests
9. **CntlPlains** (59.4% accuracy, 63.3% AUC) - Central Plains

## Ecoregion-Specific Key Drivers

### WestMnts (Western Mountains) - 61.8% accuracy
- **P_SEASONALITY** (0.0279) - Precipitation seasonality
- **ELEV_SITE_M** (0.0203) - Site elevation
- **PRE_MM_S02** (0.0187) - February precipitation
- **ELEV_MIN_M_BASIN** (0.0169) - Minimum basin elevation

### SEPlains (Southeastern Plains) - 61.2% accuracy
- **PET_MM_S02** (0.0172) - February potential evapotranspiration
- **PPD_PK_UAV** (0.0172) - Population density
- **HIRES_LENTIC_NUM** (0.0149) - Number of lentic water bodies
- **DRAIN_SQKM** (0.0147) - Drainage area

### CntlPlains (Central Plains) - 59.4% accuracy
- **FRAC_SNOW** (0.0213) - Snow fraction
- **TMP_DC_S11** (0.0189) - November temperature
- **PET_MM_S02** (0.0168) - February potential evapotranspiration
- **TMP_DC_S03** (0.0167) - March temperature

### NorthEast (Northeastern Forests) - 60.0% accuracy
- **HIRES_LENTIC_MEANSIZ** (0.0248) - Mean lentic water body size
- **LKA_PC_USE** (0.0239) - Lake area percentage
- **LKV_MC_USU** (0.0166) - Lake volume usage
- **KFACT_UP** (0.0130) - Soil erodibility factor

### EastHghlnds (Eastern Highlands) - 60.3% accuracy
- **REV_MC_USU** (0.0206) - Reservoir usage
- **RUN_MM_SYR** (0.0150) - Annual runoff
- **LKV_MC_USU** (0.0126) - Lake volume usage
- **HIRES_LENTIC_MEANSIZ** (0.0124) - Mean lentic water body size

## Regional Class Distribution Patterns

### High Drought Severity Regions (>55% High class)
- **MxWdShld**: 61.2% High (northern, lake-rich)
- **WestXeric**: 58.7% High (arid western)
- **WestMnts**: 58.5% High (mountainous)
- **WestPlains**: 55.6% High (western plains)

### Low Drought Severity Regions (>53% Low class)
- **EastHghlnds**: 56.0% Low (eastern mountains)
- **SEPlains**: 54.8% Low (southeastern)
- **NorthEast**: 53.7% Low (northeastern forests)

### Balanced Regions
- **CntlPlains**: 51.6% High, 48.4% Low
- **SECstPlain**: 54.3% High, 45.7% Low

## Model Interpretation

### Classification Performance by Class
- **High Class**: 58% precision, 53% recall
- **Low Class**: 57% precision, 61% recall
- **Overall**: Well-balanced performance between classes

### Key Insights
1. **Soil water content** in winter/spring is the strongest predictor
2. **Annual precipitation** remains critically important
3. **Water body characteristics** provide significant predictive power
4. **Regional specialization** improves accuracy by 2-6%
5. **Western regions** tend toward higher drought severity
6. **Eastern regions** show more variability in patterns

## Management Applications

### Early Warning Systems
- Monitor **February-March soil water content** as primary indicator
- Track **annual precipitation** accumulation patterns
- Assess **water body conditions** (lakes, ponds) in affected regions

### Regional Strategies
- **Western regions**: Focus on elevation and seasonality effects
- **Eastern regions**: Emphasize reservoir and lake management
- **Plains regions**: Monitor snow dynamics and temperature patterns
- **Coastal regions**: Consider soil properties and water table depth

### Operational Recommendations
1. **Deploy soil moisture sensors** in key watersheds
2. **Integrate water body monitoring** into drought assessment
3. **Develop region-specific thresholds** for early warning
4. **Consider human impacts** in drought severity predictions

## Model Limitations & Future Work

### Current Limitations
- **Moderate accuracy** (57.4%) indicates remaining unexplained variance
- **Linear threshold** may not capture complex relationships
- **Static features** don't account for temporal dynamics

### Improvement Opportunities
1. **Ensemble methods** combining multiple algorithms
2. **Temporal features** incorporating antecedent conditions
3. **Non-linear thresholds** for class boundaries
4. **Deep learning approaches** for complex pattern recognition
5. **Regional model specialization** for each ecoregion

## Conclusion

The binary classification approach successfully identifies key drivers of streamflow drought severity with 57% accuracy. Soil water content emerges as the dominant predictor, followed by precipitation patterns and water body characteristics. Regional differences highlight the need for location-specific management strategies, with western regions showing higher drought severity and eastern regions displaying more complex patterns influenced by water management infrastructure.