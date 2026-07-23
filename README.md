# Open-Pit Drill Fleet Performance Analytics

An end-to-end Python analytics pipeline for transforming drill-by-hole operational records into engineering insights, statistical evidence, anomaly alerts, machine-learning predictions, visualizations, dashboards, and exportable reports.

## Project Overview

Mining operations generate large volumes of drilling data, but much of it remains underused in operational decision-making. This project analyzes open-pit drilling records at hole, rig, operator, and date level to evaluate fleet performance and identify potential improvement opportunities.

The pipeline retains suspicious records and adds transparent quality flags rather than silently deleting them.

## Main Capabilities

- Excel data ingestion and schema validation
- Data cleaning and engineering feature creation
- Data-quality and outlier flagging
- KPI analysis by:
  - rig
  - operator
  - hole type
  - date
  - rig–operator combination
- Statistical testing, including:
  - D’Agostino normality test
  - Spearman correlation
  - Mann–Whitney U test
  - Kruskal–Wallis test
  - Welch ANOVA
  - chi-square testing
  - robust OLS regression
  - median quantile regression
  - Tukey HSD comparisons
- Static PNG charts
- Interactive Plotly dashboards
- Anomaly detection using:
  - engineering/domain rules
  - robust z-scores
  - Isolation Forest
  - Local Outlier Factor
  - One-Class SVM
- Penetration-rate regression benchmarking
- Low-performance risk classification
- Temporal holdout validation
- Out-of-fold classification-threshold selection
- Prediction intervals
- Permutation importance
- Partial dependence plots
- Optional SHAP explainability
- Optional Optuna tuning for XGBoost
- Exportable reports, models, predictions, tables, and charts

## Repository Structure

```text
.
├── open_pit_drill_fleet_analytics.py
├── requirements_open_pit_drill_analytics.txt
├── README.md
├── LICENSE
└── drill_analytics_outputs/
    ├── data/
    ├── tables/
    ├── statistics/
    ├── figures/
    ├── dashboard/
    ├── models/
    ├── predictions/
    ├── explainability/
    └── reports/
```

The output directory is created automatically when the pipeline runs.

## Input Data

The pipeline expects an Excel workbook containing drill-by-hole records.

Required columns:

| Source column | Meaning |
|---|---|
| `Hole` | Hole identifier |
| `Type` | Hole type |
| `Length (m)` | Hole length |
| `Equipment` | Drill rig or equipment identifier |
| `Average Penetration Rate (m/hr)` | Average drilling penetration rate |
| `Operator` | Operator identifier |
| `Date` | Drilling date |
| `Redrill of` | Original hole identifier for redrilled holes |
| `Wet` | Wet/dry hole indicator |

Optional columns supported by the pipeline include:

- `Material`
- `Design Diameter (mm)`
- `Best Available Diameter (mm)`
- `Comments`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DataMiningLegend1/Open-Pit-Drill-Fleet-Performance-Analytics.git
cd Open-Pit-Drill-Fleet-Performance-Analytics
```

### 2. Create and activate a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements_open_pit_drill_analytics.txt
```

Some advanced features depend on optional packages such as XGBoost, LightGBM, CatBoost, Optuna, Plotly, Statsmodels, and SHAP. The script detects whether these packages are installed and skips unavailable optional features.

## Usage

### Standard run

```bash
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_analytics_outputs
```

### Quick validation run

Use fast mode to test the pipeline with a smaller model set:

```bash
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_outputs_fast \
  --fast
```

### Run with XGBoost tuning

```bash
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_analytics_outputs \
  --tune-xgboost
```

### Skip SHAP analysis

```bash
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_analytics_outputs \
  --skip-shap
```

### View all command-line options

```bash
python open_pit_drill_fleet_analytics.py --help
```

## Key Outputs

After execution, the pipeline generates:

- cleaned and feature-engineered data
- overall and grouped KPI tables
- statistical-test results
- anomaly-detection results
- model-comparison tables
- holdout predictions
- trained machine-learning models
- regression and classification metadata
- static figures
- interactive HTML dashboards
- an executive Markdown report
- pipeline logs and run metadata

Important files include:

```text
drill_analytics_outputs/data/drilling_cleaned_feature_engineered.csv
drill_analytics_outputs/tables/regression_model_comparison.csv
drill_analytics_outputs/tables/classification_model_comparison.csv
drill_analytics_outputs/tables/anomaly_detection_results.csv
drill_analytics_outputs/dashboard/interactive_dashboard.html
drill_analytics_outputs/reports/executive_report.md
drill_analytics_outputs/models/best_penetration_rate_regressor.joblib
drill_analytics_outputs/models/best_low_performance_classifier.joblib
```

## Modelling Design

### Penetration-rate regression

The regression benchmark compares multiple algorithms, including:

- Dummy median baseline
- Ridge regression
- Elastic Net
- Huber regression
- K-nearest neighbors
- support vector regression
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- Histogram Gradient Boosting
- XGBoost, LightGBM, and CatBoost when installed

### Low-performance classification

Low performance is defined using the lower penetration-rate quantile within each hole type. Penetration rate is excluded from the classification predictors to prevent target leakage.

### Validation

The pipeline prefers date-aware train/test splitting. Recent dates are reserved as a temporal holdout whenever the dataset supports it.

## Data-Quality Philosophy

The project follows a transparent quality-control approach:

- suspicious values are retained
- quality flags are added to each record
- non-positive rates are flagged
- implausibly high rates are flagged
- robust group-level statistical outliers are flagged
- consensus anomaly scores combine several detection methods

Flagged observations should be reviewed by engineers before correction or removal.

## Engineering Limitations

The dataset may not contain all operational drivers of drilling performance. Important missing variables may include:

- shift
- start and finish times
- delays
- maintenance state
- bit condition
- geology and rock hardness
- planned depth
- drilling cost
- dispatch conditions

The results identify patterns and associations. They should not be interpreted as causal proof or used as an automated personnel-evaluation system.

## Author

**Davie Mdumuka**  
Mining Engineer and Data Analyst  
Python | SQL | Power BI | Machine Learning

## License

This project is distributed under the license included in the repository.
