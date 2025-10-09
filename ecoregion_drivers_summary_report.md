# Streamflow Flash Drought Drivers Analysis by Ecoregion

## Executive Summary

Analysis of 89,550 SFD events across 9 ecoregions identified key drivers for three critical drought characteristics:
- **Duration_Days**: How long droughts persist
- **Onset_Rate_Days**: How quickly droughts develop  
- **Mean_SFD_Flow_Percentile**: Severity of flow reduction during recovery

## Key Findings

### Overall Top Drivers (All Ecoregions Combined)

#### Duration_Days (R² = 0.135)
1. **P_MEAN** (0.125) - Mean annual precipitation is the strongest predictor
2. **PRE_MM_SYR** (0.046) - Annual precipitation sum
3. **PRE_MM_S11** (0.025) - November precipitation
4. **NO10AVE** (0.025) - Soil nitrogen content
5. **ELEV_STD_M_BASIN** (0.020) - Elevation variability in basin

#### Onset_Rate_Days (R² = 0.086)  
1. **HGB** (0.063) - Hydrologic group B soils
2. **PET_MEAN** (0.029) - Mean potential evapotranspiration
3. **SOC_TH_UAV** (0.027) - Soil organic carbon thickness
4. **HIRES_LENTIC_MEANSIZ** (0.018) - Mean size of lentic water bodies
5. **WTDEPAVE** (0.016) - Average water table depth

#### Mean_SFD_Flow_Percentile (R² = 0.101)
1. **SWC_PC_S03** (0.097) - March soil water content
2. **PRE_MM_S11** (0.039) - November precipitation  
3. **FRAC_SNOW** (0.037) - Fraction of precipitation as snow
4. **LKA_PC_USE** (0.021) - Lake area percentage
5. **HIRES_LENTIC_MEANSIZ** (0.020) - Mean size of lentic water bodies

## Regional Variations

### Duration_Days by Ecoregion

**Longest Droughts:**
- MxWdShld (69.5 days) - Driven by human footprint index (HFT_IX_U93)
- WestXeric (62.5 days) - Driven by land cover (GLC_PC_U09) 
- WestMnts (61.7 days) - Driven by climate moisture index (CMI_IX_S11)

**Shortest Droughts:**
- NorthEast (32.7 days) - Driven by soil erodibility (KFACT_UP)
- SEPlains (34.2 days) - Driven by basin fragmentation (FRAGUN_BASIN)
- EastHghlnds (34.7 days) - Driven by bulk density (BDAVE)

### Onset_Rate_Days by Ecoregion

**Fastest Onset:**
- WestPlains (10.0 days) - Driven by February snow (SNW_PC_S02)
- SEPlains (10.2 days) - Driven by stream gradient (SGR_DK_SAV)
- WestXeric (10.5 days) - Driven by silt content (SLT_PC_UAV)

**Slowest Onset:**
- MxWdShld (12.5 days) - Driven by silt content (SLT_PC_UAV)
- NorthEast (11.3 days) - Driven by May evapotranspiration (AET_MM_S05)
- EastHghlnds (11.1 days) - Driven by hydrologic group B soils (HGB)

### Mean_SFD_Flow_Percentile by Ecoregion

**Most Severe (Lowest Percentiles):**
- EastHghlnds (14.5%) - Driven by reservoir usage (REV_MC_USU)
- SEPlains (14.5%) - Driven by basin fragmentation (FRAGUN_BASIN)
- NorthEast (14.8%) - Driven by lake area (LKA_PC_USE)

**Least Severe (Higher Percentiles):**
- MxWdShld (16.6%) - Driven by lentic water body density (HIRES_LENTIC_DENS)
- WestXeric (16.3%) - Driven by maximum elevation (ELE_MT_SMX)
- WestMnts (16.2%) - Driven by annual runoff (RUN_MM_SYR)

## Driver Categories

### Climate Drivers
- **Precipitation**: P_MEAN, PRE_MM_SYR, PRE_MM_S11 (strongest overall)
- **Temperature**: TMP_DC_S10, TMP_DC_S12 (important in CntlPlains)
- **Snow**: FRAC_SNOW, SNW_PC_S02 (critical in northern/mountain regions)
- **Evapotranspiration**: PET_MEAN, AET_MM_S05 (moderate influence)

### Landscape Drivers  
- **Topography**: ELEV_STD_M_BASIN, SLOPE_PCT, ASPECT_DEGREES
- **Soil Properties**: HGB, SOC_TH_UAV, WTDEPAVE, KFACT_UP
- **Land Cover**: GLC_PC_U09, FOR_PC_USE, CRP_PC_USE
- **Water Bodies**: HIRES_LENTIC_MEANSIZ, LKA_PC_USE

### Human Impact Drivers
- **Infrastructure**: FRAGUN_BASIN, RAW_DIS_NEAREST_DAM
- **Land Use**: URB_PC_USE, IRE_PC_USE, PST_PC_USE  
- **Population**: PPD_PK_UAV, POP_CT_USU
- **Development**: HFT_IX_U93, GDP_UD_USU

## Management Implications

1. **Precipitation** is the dominant control on drought duration across all regions
2. **Soil properties** (especially hydrologic groups) strongly influence onset rates
3. **Water body characteristics** affect both onset speed and severity
4. **Human modifications** (fragmentation, dams) significantly impact drought patterns
5. **Regional differences** require tailored management approaches for each ecoregion

## Recommendations

1. **Monitor precipitation patterns** as primary early warning indicator
2. **Consider soil characteristics** when assessing drought vulnerability  
3. **Evaluate water body management** for drought mitigation potential
4. **Account for human impacts** in drought prediction models
5. **Develop region-specific strategies** based on dominant local drivers