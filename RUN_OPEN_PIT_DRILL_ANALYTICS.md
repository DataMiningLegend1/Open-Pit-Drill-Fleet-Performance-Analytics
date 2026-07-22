# Run the Open-Pit Drill Fleet Analytics Pipeline

## 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements_open_pit_drill_analytics.txt
```

## 3. Quick validation run

For Bash or zsh:

```bash
python open_pit_drill_fleet_analytics.py --input "Drilling by Hole_3167_1136_086_001_2026-01-29T07_58_47.818Z(2).xlsx" --output-dir drill_outputs_fast --fast
```

For PowerShell, use either a single line or backticks for line continuation:

```powershell
python .\open_pit_drill_fleet_analytics.py --input "Drilling by Hole_3167_1136_086_001_2026-01-29T07_58_47.818Z(2).xlsx" --output-dir .\drill_outputs_fast --fast
```

```powershell
python .\open_pit_drill_fleet_analytics.py `
  --input "Drilling by Hole_3167_1136_086_001_2026-01-29T07_58_47.818Z(2).xlsx" `
  --output-dir .\drill_outputs_fast `
  --fast
```

## 4. Comprehensive portfolio run

For PowerShell:

```powershell
python .\open_pit_drill_fleet_analytics.py --input "Drilling by Hole_3167_1136_086_001_2026-01-29T07_58_47.818Z(2).xlsx" --output-dir .\drill_outputs --skip-shap
```

Remove `--skip-shap` when SHAP is installed and you want global explainability charts.

## 5. Optional XGBoost tuning

```bash
python open_pit_drill_fleet_analytics.py --input "Drilling by Hole_3167_1136_086_001_2026-01-29T07_58_47.818Z(2).xlsx" --output-dir drill_outputs_tuned --tune-xgboost --optuna-trials 40 --skip-shap
```

## Main output folders

- `data/`: cleaned, feature-engineered records
- `tables/`: KPIs, model comparisons, anomaly results
- `statistics/`: statistical tests and inferential summaries
- `figures/`: portfolio-ready PNG charts
- `dashboard/`: interactive Plotly HTML dashboards
- `models/`: serialized best models and metadata
- `predictions/`: untouched temporal-holdout predictions
- `explainability/`: feature-importance tables
- `reports/`: executive report
