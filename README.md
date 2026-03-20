# BOQ Embodied Carbon Estimation using ML

Estimating embodied carbon and Carbon Cost Intensity (CCI) for Indian construction
projects using Bill of Quantities data and ICE DB V4.1 emission factors.

## Projects
- Bot, Eco, Mall, Zen: BOQ data (no rates)
- PA: BOQ with unit rates, WBS codes, payment tracking

## Structure
- src/         Python pipeline scripts (one per part)
- data/raw/    Original Excel files (not committed to git)
- data/processed/  Intermediate CSV outputs
- outputs/     Final figures, tables, trained models
- notebooks/   Exploratory analysis

## Parts
1. Data ingestion and cleaning
2. ICE DB emission factor mapping
3. Carbon Cost Intensity dataset
4. ML model training and comparison
5. GGBS scenario analysis
6. Visualisations and paper figures
