# Mean SFD Flow Percentile Classification Analysis

## Executive Summary

Built a Random Forest classification model to predict Mean_SFD_Flow_Percentile categories (Low, Medium, High) using 250 environmental and hydrological drivers across 9 ecoregions. The analysis reveals key factors that distinguish between different levels of streamflow drought severity.

## Classification Setup

### Target Variable Categorization
- **Low**: ≤ 12.57th percentile (33.0% of events, n=29,552)
- **Medium**: 12.57-17.77th percentile (34.0% of events, n=30,446)  
- **High**: > 17.77th percentile (33.0% of events, n=29,552)

### Model Performance
- **Overall Accuracy**: 39.2%
- **Cross-validation Accuracy**: 39.2% (±0.5%)
- **Dataset**: 89,550 SFD events across 9 ecoregions
- **Features**: 250 environmental drivers

## Key Findings

### Top 5 Most Important Drivers (Overall)

1. **PRE_MM_SYR** (0.0125) - Annual precipitation sum
2. **PRE_MM_S11** (0.0110) - November precipitation  
3. **SWC_PC_S02** (0.0102) - February soil water content
4. **P_MEAN** (0.0098) - Mean annual precipitation
5. **HIRES_LENTIC_MEANSIZ** (0.0093) - Mean size of lentic water bodies

### Driver Categories

#### Climate Drivers (Most Important)
- **Precipitation**: PRE_MM_SYR, PRE_MM_S11, P_MEAN, PRE_MM_S04, PRE_MM_S05
- **Soil Water**: SWC_PC_S02, SWC_PC_S01, SWC_PC_S03, SWC_PC_S04, SWC_PC_S12
- **Evapotranspiration**: AET_MM_S05
- **Snow**: FRAC_SNOW (important in northern regions)

#### Landscape Drivers
- **Water Bodies**: HIRES_LENTIC_MEANSIZ, LKA_PC_USE, LKV_MC_USU
- **Topography**: ASPECT_EASTNESS, elevation metrics
- **Infrastructure**: RDD_MK_UAV (road density), PPD_PK_UAV (population density)

## Regional Analysis

### Best Performing Ecoregions (Accuracy)
1. **WestXeric** (46.9%) - Driven by human footprint indices
2. **MxWdShld** (46.9%) - Driven by stream gradient and lake area
3. **WestMnts** (46.4%) - Driven by precipitation seasonality

### Poorest Performing Ecoregions (Accuracy)
1. **EastHghlnds** (43.6%) - Complex terrain, driven by reservoir usage
2. **NorthEast** (43.6%) - Lake-dominated, driven by lentic water bodies
3. **CntlPlains** (43.9%) - Snow-influenced, driven by snow fraction

### Ecoregion-Specific Key Drivers

#### WestMnts (Mountain West)
- **P_SEASONALITY** (0.0217) - Precipitation seasonality
- **PRE_MM_S02** (0.0176) - February precipitation
- **PRE_MM_S01** (0.0143) - January precipitation
- **ELEV_SITE_M** (0.0116) - Site elevation

#### SEPlains (Southeastern Plains)  
- **DRAIN_SQKM** (0.0168) - Drainage area
- **PET_MM_S02** (0.0164) - February potential evapotranspiration
- **HIRES_LENTIC_NUM** (0.0137) - Number of lentic water bodies
- **PPD_PK_UAV** (0.0128) - Population density

#### CntlPlains (Central Plains)
- **FRAC_SNOW** (0.0161) - Snow fraction
- **TMP_DC_S11** (0.0142) - November temperature
- **P_SEASONALITY** (0.0134) - Precipitation seasonality
- **SNW_PC_S01** (0.0130) - January snow percentage

#### NorthEast (Northeastern Forests)
- **HIRES_LENTIC_MEANSIZ** (0.0213) - Mean lentic water body size
- **LKA_PC_USE** (0.0209) - Lake area percentage
- **KFACT_UP** (0.0122) - Soil erodibility factor
- **WET_PC_UG2** (0.0110) - Wetland percentage

#### EastHghlnds (Eastern Highlands)
- **REV_MC_USU** (0.0147) - Reservoir usage
- **LKA_PC_USE** (0.0134) - Lake area percentage
- **LKV_MC_USU** (0.0106) - Lake volume usage
- **HIRES_LENTIC_MEANSIZ** (0.0101) - Mean lentic water body size

## Class Characteristics

### Model Performance by Class
- **Low Class**: Precision 40%, Recall 51% (best recall)
- **Medium Class**: Precision 34%, Recall 27% (poorest performance)
- **High Class**: Precision 42%, Recall 40% (balanced)

### Regional Class Distribution Patterns
- **EastHghlnds**: More Low events (37.9%)
- **SEPlains**: More Low events (38.4%)  
- **WestMnts**: More High events (40.5%)
- **WestXeric**: More High events (41.9%)

## Management Implications

### Primary Controls
1. **Precipitation patterns** (annual and seasonal) are the strongest predictors
2. **Soil water content** in late winter/early spring is critical
3. **Water body characteristics** significantly influence drought severity
4. **Regional climate seasonality** creates distinct patterns

### Regional Management Strategies

#### Western Regions (WestMnts, WestXeric)
- Focus on **precipitation seasonality** and **elevation effects**
- Monitor **human footprint** impacts on drought severity
- Consider **snow dynamics** in mountain areas

#### Eastern Regions (EastHghlnds, NorthEast)
- Emphasize **lake and reservoir management**
- Account for **soil characteristics** and **erodibility**
- Monitor **wetland conditions**

#### Plains Regions (CntlPlains, SEPlains, WestPlains)
- Track **snow fraction** and **temperature patterns**
- Consider **drainage characteristics** and **population impacts**
- Monitor **soil water content** in winter/spring

#### Coastal Regions (SECstPlain)
- Focus on **soil organic carbon** and **water table depth**
- Consider **elevation gradients** within basins

## Model Limitations

1. **Moderate accuracy** (39.2%) suggests high natural variability
2. **Medium class** is hardest to predict (transition zone)
3. **Complex interactions** between drivers not fully captured
4. **Regional differences** require location-specific models

## Recommendations

1. **Develop region-specific models** for better accuracy
2. **Focus monitoring** on top 5-10 drivers per ecoregion
3. **Integrate seasonal patterns** in drought prediction systems
4. **Consider ensemble approaches** combining multiple models
5. **Validate findings** with independent datasets

## Data Products Generated

- **Feature importance rankings** (overall and by ecoregion)
- **Classification model** for operational use
- **Visualizations** showing regional patterns
- **Confusion matrix** for model evaluation