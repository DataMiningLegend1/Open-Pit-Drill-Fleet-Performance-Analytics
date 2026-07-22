#!/usr/bin/env python3
"""
Open-Pit Drill Fleet Performance Analytics
===========================================

End-to-end portfolio pipeline for drill-by-hole operational data.

Capabilities
------------
• Excel ingestion and schema validation
• transparent data-quality rules and engineering feature creation
• KPI tables by rig, operator, hole type, date, and rig–operator combination
• statistical inference: normality, Spearman, Mann–Whitney, Kruskal–Wallis,
  Welch ANOVA, chi-square, robust OLS, median quantile regression, Tukey HSD
• static PNG visualizations plus interactive Plotly dashboards
• anomaly detection: robust/domain rules, Isolation Forest, LOF, One-Class SVM
• date-aware regression benchmark for penetration-rate prediction
• date-aware classification benchmark for low-performance risk
• modern model suite: linear/robust, KNN, SVM, Random Forest, Extra Trees,
  AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM,
  and CatBoost when installed
• out-of-fold threshold selection, temporal holdout testing, prediction intervals
• permutation importance, partial dependence, and optional SHAP
• optional Optuna tuning of XGBoost regression
• exportable reports, models, predictions, tables, and charts

The pipeline keeps suspicious rows and flags them. It does not silently delete them.
The low-performance target is defined within hole type and penetration rate is
excluded from classification predictors, preventing target leakage.

Example
-------
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_outputs

Quick validation
----------------
python open_pit_drill_fleet_analytics.py \
  --input "Drilling by Hole.xlsx" \
  --output-dir drill_outputs_fast \
  --fast
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import OneClassSVM, SVC, SVR

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.oneway import anova_oneway
    STATSMODELS = True
except Exception:
    STATSMODELS = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST = True
except Exception:
    XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM = True
except Exception:
    LIGHTGBM = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST = True
except Exception:
    CATBOOST = False

try:
    import optuna
    OPTUNA = True
except Exception:
    OPTUNA = False

try:
    import shap
    SHAP = True
except Exception:
    SHAP = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
LOG = logging.getLogger("drill_analytics")


@dataclass(frozen=True)
class Config:
    random_state: int = 42
    max_plausible_rate: float = 250.0
    robust_z_threshold: float = 4.5
    anomaly_contamination: float = 0.04
    low_performance_quantile: float = 0.25
    test_date_fraction: float = 0.25
    cv_folds: int = 5
    precision_floor: float = 0.35
    confidence_level: float = 0.90
    optuna_trials: int = 30
    fast: bool = False
    skip_shap: bool = False


ALIASES = {
    "Hole": "hole",
    "Type": "hole_type",
    "Material": "material",
    "Design Diameter (mm)": "design_diameter_mm",
    "Best Available Diameter (mm)": "best_available_diameter_mm",
    "Length (m)": "length_m",
    "Equipment": "equipment",
    "Average Penetration Rate (m/hr)": "penetration_rate_m_per_hr",
    "Operator": "operator",
    "Date": "date",
    "Redrill of": "redrill_of",
    "Wet": "wet",
    "Comments": "comments",
}
REQUIRED = {
    "hole", "hole_type", "length_m", "equipment", "penetration_rate_m_per_hr",
    "operator", "date", "redrill_of", "wet",
}
MISSING = {"", "--", "-", "na", "n/a", "none", "null", "nan"}

PREDICTORS = [
    "hole_type", "material", "design_diameter_mm", "best_available_diameter_mm",
    "diameter_deviation_mm", "length_m", "equipment", "operator", "wet",
    "redrill_flag", "hole_row_index", "hole_number", "day_of_week",
    "day_of_month", "week_of_year", "is_weekend",
]
NUMERIC = [
    "design_diameter_mm", "best_available_diameter_mm", "diameter_deviation_mm",
    "length_m", "redrill_flag", "hole_row_index", "hole_number", "day_of_week",
    "day_of_month", "week_of_year", "is_weekend",
]
CATEGORICAL = ["hole_type", "material", "equipment", "operator", "wet"]


def configure_logging(output: Path, verbose: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(output / "pipeline.log", mode="w", encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def output_tree(base: Path) -> dict[str, Path]:
    names = ["data", "tables", "statistics", "figures", "dashboard", "models", "predictions", "explainability", "reports"]
    paths = {"base": base}
    for name in names:
        paths[name] = base / name
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


def snake(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_").lower()


def save_json(obj: Any, path: Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, (np.integer,)): return int(value)
        if isinstance(value, (np.floating,)): return None if not np.isfinite(value) else float(value)
        if isinstance(value, (pd.Timestamp, datetime)): return value.isoformat()
        if isinstance(value, Path): return str(value)
        if isinstance(value, np.ndarray): return value.tolist()
        return str(value) if not isinstance(value, (str, int, float, bool, list, dict, type(None))) else value
    path.write_text(json.dumps(obj, indent=2, default=convert), encoding="utf-8")


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def letters_index(value: Any) -> float:
    if pd.isna(value): return np.nan
    text = str(value).upper().strip()
    if not text.isalpha(): return np.nan
    result = 0
    for char in text: result = result * 26 + ord(char) - 64
    return float(result)


def robust_z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0: return pd.Series(np.nan, index=x.index)
    med = x.median(); mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0: return pd.Series(0.0, index=x.index)
    return 0.67448975 * (x - med) / mad


def rmse(y: Iterable[float], p: Iterable[float]) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def load_clean_engineer(path: Path, sheet: str | int, cfg: Config) -> pd.DataFrame:
    LOG.info("Reading %s", path)
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if not isinstance(df, pd.DataFrame): raise ValueError("Select a single worksheet.")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={c: ALIASES.get(c, snake(c)) for c in df.columns})
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    df.insert(0, "record_id", np.arange(1, len(df) + 1))

    for col in df.select_dtypes(include="object"):
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].map(lambda v: np.nan if isinstance(v, str) and v.lower() in MISSING else v)
    for col in ["design_diameter_mm", "best_available_diameter_mm", "length_m", "penetration_rate_m_per_hr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_date = pd.to_numeric(df["date"], errors="coerce")
    regular_date = pd.to_datetime(df["date"], errors="coerce")
    excel_date = pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric_date, unit="D")
    df["date"] = regular_date.where(~numeric_date.between(1, 100000), excel_date)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    bool_map = {True: True, False: False, 1: True, 0: False, "true": True, "false": False, "yes": True, "no": False, "wet": True, "dry": False}
    df["wet"] = df["wet"].map(lambda v: bool_map.get(v, bool_map.get(str(v).strip().lower(), np.nan)) if not pd.isna(v) else np.nan).astype("boolean")
    df["redrill_flag"] = df["redrill_of"].notna().astype(int)
    df["diameter_deviation_mm"] = df["best_available_diameter_mm"] - df["design_diameter_mm"]

    parts = df["hole"].astype("string").str.extract(r"^([A-Za-z]+)\s*(\d+)?$")
    df["hole_row_letters"] = parts[0].str.upper()
    df["hole_row_index"] = df["hole_row_letters"].map(letters_index)
    df["hole_number"] = pd.to_numeric(parts[1], errors="coerce")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["day_name"] = df["date"].dt.day_name().str[:3]
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("Float64")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    rate = df["penetration_rate_m_per_hr"]
    df["estimated_drilling_hours"] = df["length_m"] / rate.where(rate > 0)
    df["estimated_drilling_minutes"] = df["estimated_drilling_hours"] * 60
    df["rate_nonpositive_flag"] = rate.notna() & (rate <= 0)
    df["rate_domain_outlier_flag"] = rate.notna() & (rate > cfg.max_plausible_rate)
    df["rate_robust_z_global"] = robust_z(rate)
    df["rate_robust_z_group"] = df.groupby(["hole_type", "equipment"], dropna=False)["penetration_rate_m_per_hr"].transform(robust_z)
    df["rate_statistical_outlier_flag"] = rate.notna() & (df["rate_robust_z_group"].abs() > cfg.robust_z_threshold)
    df["rate_any_quality_flag"] = df[["rate_nonpositive_flag", "rate_domain_outlier_flag", "rate_statistical_outlier_flag"]].any(axis=1)
    df["modeling_valid_rate"] = rate.notna() & df["length_m"].notna() & ~df["rate_any_quality_flag"]

    valid = df["modeling_valid_rate"]
    low_map = df.loc[valid].groupby("hole_type")["penetration_rate_m_per_hr"].quantile(cfg.low_performance_quantile)
    low_global = df.loc[valid, "penetration_rate_m_per_hr"].quantile(cfg.low_performance_quantile)
    df["low_performance_benchmark"] = df["hole_type"].map(low_map).fillna(low_global)
    df["low_performance_flag"] = np.where(valid, (rate <= df["low_performance_benchmark"]).astype(int), np.nan)
    df["equipment_operator"] = df["equipment"].astype("string").fillna("Unknown") + " | " + df["operator"].astype("string").fillna("Unknown")
    LOG.info("Loaded %d rows; %d valid for modelling", len(df), int(valid.sum()))
    return df


def quality_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "column": c, "dtype": str(df[c].dtype), "rows": len(df),
        "missing_count": int(df[c].isna().sum()), "missing_pct": df[c].isna().mean() * 100,
        "unique_count": int(df[c].nunique(dropna=True)),
        "examples": " | ".join(df[c].dropna().astype(str).unique()[:5]),
    } for c in df.columns]).sort_values(["missing_pct", "column"], ascending=[False, True])


def group_kpis(df: pd.DataFrame, group: str) -> pd.DataFrame:
    x = df.copy(); x[group] = x[group].astype("string").fillna("Unknown")
    out = x.groupby(group, dropna=False).agg(
        holes=("hole", "count"), total_length_m=("length_m", "sum"),
        mean_rate_m_per_hr=("penetration_rate_m_per_hr", "mean"),
        median_rate_m_per_hr=("penetration_rate_m_per_hr", "median"),
        estimated_drilling_hours=("estimated_drilling_hours", "sum"),
        redrill_rate_pct=("redrill_flag", lambda s: s.mean() * 100),
        wet_rate_pct=("wet", lambda s: s.astype("Float64").mean() * 100),
        quality_flag_rate_pct=("rate_any_quality_flag", lambda s: s.mean() * 100),
        valid_model_rows=("modeling_valid_rate", "sum"),
    ).reset_index()
    return out.sort_values("total_length_m", ascending=False)


def export_tables(df: pd.DataFrame, paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    valid = df.loc[df["modeling_valid_rate"], "penetration_rate_m_per_hr"]
    overall = pd.DataFrame([
        ("Total records", len(df), "count"), ("Unique holes", df["hole"].nunique(), "count"),
        ("Unique equipment", df["equipment"].nunique(), "count"), ("Unique operators", df["operator"].nunique(), "count"),
        ("Total drilled length", df["length_m"].sum(), "m"), ("Median penetration rate", valid.median(), "m/hr"),
        ("Mean penetration rate", valid.mean(), "m/hr"), ("P10 penetration rate", valid.quantile(.1), "m/hr"),
        ("P90 penetration rate", valid.quantile(.9), "m/hr"), ("Redrill rate", df["redrill_flag"].mean()*100, "%"),
        ("Wet-hole rate", df["wet"].astype("Float64").mean()*100, "%"),
        ("Missing penetration rates", df["penetration_rate_m_per_hr"].isna().sum(), "count"),
        ("Domain outliers", df["rate_domain_outlier_flag"].sum(), "count"),
        ("Statistical outliers", df["rate_statistical_outlier_flag"].sum(), "count"),
        ("Valid modelling rows", df["modeling_valid_rate"].sum(), "count"),
    ], columns=["metric", "value", "unit"])
    tables = {"overall_kpis": overall, "data_quality": quality_table(df)}
    for group in ["equipment", "operator", "hole_type", "date", "equipment_operator"]:
        tables[f"kpis_by_{group}"] = group_kpis(df, group)
    matrix = pd.pivot_table(df.loc[df["modeling_valid_rate"]], index="operator", columns="equipment", values="penetration_rate_m_per_hr", aggfunc="median")
    tables["operator_equipment_median_rate"] = matrix.reset_index()
    df.to_csv(paths["data"] / "drilling_cleaned_feature_engineered.csv", index=False)
    for name, table in tables.items(): table.to_csv(paths["tables"] / f"{name}.csv", index=False)
    return tables


def statistics_analysis(df: pd.DataFrame, paths: dict[str, Path], fast: bool) -> pd.DataFrame:
    x = df.loc[df["modeling_valid_rate"]].copy(); ycol = "penetration_rate_m_per_hr"; rows = []
    def add(name: str, stat: float, p: float, effect: float | None, effect_name: str, n: int, note: str):
        rows.append({"test": name, "statistic": stat, "p_value": p, "effect_size": effect, "effect_name": effect_name, "n": n, "significant_0_05": p < .05, "interpretation": note})
    rate = x[ycol].dropna()
    if len(rate) >= 20:
        s, p = stats.normaltest(rate); add("D'Agostino normality", float(s), float(p), float(stats.skew(rate)), "skewness", len(rate), "Small p indicates non-normality.")
    pair = x[["length_m", ycol]].dropna()
    if len(pair) >= 10:
        s, p = stats.spearmanr(pair["length_m"], pair[ycol]); add("Spearman length vs rate", float(s), float(p), float(s), "rho", len(pair), "Association is not causation.")
    wet = x.loc[x["wet"] == True, ycol].dropna(); dry = x.loc[x["wet"] == False, ycol].dropna()  # noqa
    if len(wet) >= 5 and len(dry) >= 5:
        s, p = stats.mannwhitneyu(wet, dry, alternative="two-sided"); add("Mann-Whitney wet vs dry", float(s), float(p), None, "", len(wet)+len(dry), "Compares distributions robustly.")
    for group in ["equipment", "operator", "hole_type"]:
        groups = [g[ycol].dropna().to_numpy() for _, g in x.groupby(group) if g[ycol].notna().sum() >= 5]
        if len(groups) >= 2:
            s, p = stats.kruskal(*groups); eps = max(0, (s-len(groups)+1)/(sum(map(len,groups))-len(groups)))
            add(f"Kruskal-Wallis by {group}", float(s), float(p), float(eps), "epsilon squared", sum(map(len,groups)), "At least one distribution differs if significant.")
            if STATSMODELS:
                try:
                    w = anova_oneway(groups, use_var="unequal"); add(f"Welch ANOVA by {group}", float(w.statistic), float(w.pvalue), None, "", sum(map(len,groups)), "Robust to unequal variances.")
                except Exception: pass
    ct = pd.crosstab(x["wet"].astype("string"), x["redrill_flag"])
    if ct.shape == (2,2):
        s,p,_,_ = stats.chi2_contingency(ct); n=ct.to_numpy().sum(); add("Chi-square wet vs redrill", float(s), float(p), float(math.sqrt(s/n)), "Cramer's V", int(n), "Association does not prove causation.")
    result = pd.DataFrame(rows); result.to_csv(paths["statistics"] / "statistical_tests.csv", index=False)
    if STATSMODELS and len(x) >= 50 and not fast:
        sx=x.copy(); sx["wet_int"]=sx["wet"].astype("Float64").astype(float)
        formula="np.log1p(penetration_rate_m_per_hr) ~ length_m + wet_int + redrill_flag + C(hole_type) + C(equipment) + C(operator)"
        try: (paths["statistics"] / "robust_ols_summary.txt").write_text(smf.ols(formula, sx).fit(cov_type="HC3").summary().as_text())
        except Exception as e: LOG.warning("Robust OLS skipped: %s", e)
        try: (paths["statistics"] / "median_quantile_regression.txt").write_text(smf.quantreg(formula, sx).fit(q=.5, max_iter=5000).summary().as_text())
        except Exception as e: LOG.warning("Quantile regression skipped: %s", e)
        for group in ["equipment", "operator"]:
            sub=x[[ycol,group]].dropna(); keep=sub[group].value_counts(); sub=sub[sub[group].isin(keep[keep>=5].index)]
            if sub[group].nunique()>=2:
                try: (paths["statistics"] / f"tukey_{group}.txt").write_text(str(pairwise_tukeyhsd(sub[ycol],sub[group])))
                except Exception: pass
    return result


def visualizations(df: pd.DataFrame, paths: dict[str, Path], fast: bool = False) -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    x=df.loc[df["modeling_valid_rate"]].copy(); y="penetration_rate_m_per_hr"
    fig,ax=plt.subplots(figsize=(10,6)); ax.hist(x[y].dropna(), bins=35, alpha=.8); ax.axvline(x[y].median(),ls="--",label="Median"); ax.legend(); ax.set(title="Penetration Rate Distribution",xlabel="m/hr",ylabel="Holes"); save_fig(fig,paths["figures"]/"01_rate_distribution.png")
    order=x.groupby("equipment")[y].median().sort_values(ascending=False).index
    fig,ax=plt.subplots(figsize=(11,6)); sns.boxplot(data=x,x="equipment",y=y,order=order,showfliers=False,ax=ax); ax.set(title="Penetration Rate by Rig",xlabel="Equipment",ylabel="m/hr"); save_fig(fig,paths["figures"]/"02_rate_by_equipment.png")
    order=x.groupby("operator")[y].median().sort_values(ascending=False).index
    fig,ax=plt.subplots(figsize=(12,7)); sns.boxplot(data=x,y="operator",x=y,order=order,showfliers=False,ax=ax); ax.set(title="Penetration Rate by Operator",xlabel="m/hr",ylabel="Operator"); save_fig(fig,paths["figures"]/"03_rate_by_operator.png")
    fig,ax=plt.subplots(figsize=(10,7)); sns.scatterplot(data=x,x="length_m",y=y,hue="equipment",style="wet",alpha=.75,ax=ax); ax.set(title="Hole Length vs Penetration Rate",xlabel="Length (m)",ylabel="m/hr"); save_fig(fig,paths["figures"]/"04_length_vs_rate.png")
    if fast:
        return
    daily=x.groupby("date",as_index=False).agg(total_length_m=("length_m","sum"),median_rate=(y,"median"))
    fig,ax=plt.subplots(figsize=(12,6)); ax.plot(daily["date"],daily["median_rate"],marker="o"); ax.tick_params(axis="x",rotation=45); ax.set(title="Daily Median Penetration Rate",xlabel="Date",ylabel="m/hr"); save_fig(fig,paths["figures"]/"05_daily_rate.png")
    fig,ax=plt.subplots(figsize=(12,6)); ax.bar(daily["date"].dt.strftime("%Y-%m-%d"),daily["total_length_m"]); ax.tick_params(axis="x",rotation=45); ax.set(title="Daily Drilled Length",xlabel="Date",ylabel="m"); save_fig(fig,paths["figures"]/"06_daily_length.png")
    matrix=pd.pivot_table(x,index="operator",columns="equipment",values=y,aggfunc="median")
    fig,ax=plt.subplots(figsize=(11,7)); sns.heatmap(matrix,annot=True,fmt=".1f",cmap="viridis",ax=ax); ax.set_title("Median Rate: Operator × Equipment"); save_fig(fig,paths["figures"]/"07_operator_equipment_heatmap.png")
    q=pd.Series({"Missing rate":df[y].isna().sum(),"Domain outlier":df["rate_domain_outlier_flag"].sum(),"Statistical outlier":df["rate_statistical_outlier_flag"].sum(),"Valid modelling rows":df["modeling_valid_rate"].sum()}).sort_values()
    fig,ax=plt.subplots(figsize=(10,5)); ax.barh(q.index,q.values); ax.set(title="Data Quality Summary",xlabel="Records"); save_fig(fig,paths["figures"]/"08_quality_summary.png")

    if PLOTLY:
        fig=make_subplots(rows=2,cols=2,subplot_titles=("Daily length","Daily median rate","Rate by equipment","Length vs rate"))
        fig.add_trace(go.Bar(x=daily.date,y=daily.total_length_m,name="Length"),row=1,col=1)
        fig.add_trace(go.Scatter(x=daily.date,y=daily.median_rate,mode="lines+markers",name="Median rate"),row=1,col=2)
        eq=group_kpis(x,"equipment"); fig.add_trace(go.Bar(x=eq.equipment,y=eq.median_rate_m_per_hr,name="Equipment"),row=2,col=1)
        for rig,g in x.groupby("equipment"): fig.add_trace(go.Scatter(x=g.length_m,y=g[y],mode="markers",name=str(rig),text=g.hole),row=2,col=2)
        fig.update_layout(title="Open-Pit Drill Fleet Performance Analytics",height=900,template="plotly_white")
        fig.write_html(paths["dashboard"]/"interactive_dashboard.html",include_plotlyjs="cdn")
        px.scatter(df,x="length_m",y=y,color="rate_any_quality_flag",symbol="equipment",hover_data=["hole","operator","date","redrill_of","wet","comments"],title="Quality and Outlier Explorer").write_html(paths["dashboard"]/"quality_explorer.html",include_plotlyjs="cdn")


def preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",RobustScaler())]),num),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=2))]),cat),
    ],verbose_feature_names_out=False)


def anomaly_detection(df: pd.DataFrame, cfg: Config, paths: dict[str, Path]) -> pd.DataFrame:
    s=df.loc[df["penetration_rate_m_per_hr"].notna() & df["length_m"].notna()].copy()
    num=["length_m","design_diameter_mm","best_available_diameter_mm","penetration_rate_m_per_hr","estimated_drilling_hours","hole_row_index","hole_number"]
    cat=["hole_type","equipment","operator","wet","redrill_flag"]
    X=preprocessor(num,cat).fit_transform(s[num+cat])
    iso=IsolationForest(n_estimators=50 if cfg.fast else 500,contamination=cfg.anomaly_contamination,random_state=cfg.random_state,n_jobs=-1)
    s["isolation_forest_anomaly"]=(iso.fit_predict(X)==-1).astype(int); s["isolation_score"]=-iso.score_samples(X)
    lof=LocalOutlierFactor(n_neighbors=min(30,max(5,len(s)//20)),contamination=cfg.anomaly_contamination)
    s["lof_anomaly"]=(lof.fit_predict(X)==-1).astype(int); s["lof_score"]=-lof.negative_outlier_factor_
    oc=OneClassSVM(nu=cfg.anomaly_contamination,gamma="scale")
    s["one_class_svm_anomaly"]=(oc.fit_predict(X)==-1).astype(int)
    s["domain_robust_anomaly"]=(s["rate_domain_outlier_flag"]|s["rate_statistical_outlier_flag"]).astype(int)
    s["anomaly_votes"]=s[["isolation_forest_anomaly","lof_anomaly","one_class_svm_anomaly","domain_robust_anomaly"]].sum(axis=1)
    s["consensus_anomaly_flag"]=(s["anomaly_votes"]>=2).astype(int)
    cols=["record_id","hole","date","equipment","operator","hole_type","length_m","penetration_rate_m_per_hr","rate_robust_z_group","rate_domain_outlier_flag","rate_statistical_outlier_flag","isolation_forest_anomaly","lof_anomaly","one_class_svm_anomaly","anomaly_votes","consensus_anomaly_flag","isolation_score","lof_score"]
    s[cols].sort_values(["anomaly_votes","isolation_score"],ascending=[False,False]).to_csv(paths["tables"]/"anomaly_detection_results.csv",index=False)
    for c in ["isolation_forest_anomaly","lof_anomaly","one_class_svm_anomaly","anomaly_votes","consensus_anomaly_flag","isolation_score","lof_score"]:
        df[c]=np.nan; df.loc[s.index,c]=s[c]
    fig,ax=plt.subplots(figsize=(11,7)); sns.scatterplot(data=s,x="length_m",y="penetration_rate_m_per_hr",hue="anomaly_votes",size="anomaly_votes",sizes=(30,160),palette="viridis",ax=ax); ax.set_title("Consensus Anomaly Detection"); save_fig(fig,paths["figures"]/"09_consensus_anomalies.png")
    return df


def model_data(df: pd.DataFrame):
    cols=[c for c in PREDICTORS if c in df]; num=[c for c in NUMERIC if c in cols]; cat=[c for c in CATEGORICAL if c in cols]
    X=df[cols].copy()
    for c in cat: X[c]=X[c].astype("string")
    return X,num,cat


def split_indices(df: pd.DataFrame,cfg:Config):
    dates=sorted(pd.to_datetime(df.date.dropna().unique()))
    if len(dates)>=4:
        n=max(2,math.ceil(len(dates)*cfg.test_date_fraction)); n=min(n,len(dates)-2); test_dates=dates[-n:]
        test=df.index[df.date.isin(test_dates)]; train=df.index[~df.date.isin(test_dates)]
        if len(test)>=20 and len(train)>=100:return train,test,test_dates
    rng=np.random.default_rng(cfg.random_state); idx=np.array(df.index); rng.shuffle(idx); cut=int(len(idx)*(1-cfg.test_date_fraction)); return pd.Index(idx[:cut]),pd.Index(idx[cut:]),[]


def cv_object(y:pd.Series,groups:pd.Series,classification:bool,cfg:Config):
    ng=groups.dropna().nunique()
    if ng>=3:return GroupKFold(min(cfg.cv_folds,ng)),{"groups":groups}
    if classification:
        n=max(2,min(cfg.cv_folds,int(y.value_counts().min())));return StratifiedKFold(n,shuffle=True,random_state=cfg.random_state),{}
    return KFold(min(cfg.cv_folds,3),shuffle=True,random_state=cfg.random_state),{}


def regression_registry(cfg:Config)->dict[str,BaseEstimator]:
    rs=cfg.random_state
    if cfg.fast:
        return {
            "DummyMedian":DummyRegressor(strategy="median"),
            "Ridge":Ridge(alpha=1),
            "RandomForest":RandomForestRegressor(n_estimators=25,min_samples_leaf=3,max_features="sqrt",random_state=rs,n_jobs=1),
        }
    m={
        "DummyMedian":DummyRegressor(strategy="median"),"Ridge":Ridge(alpha=1),
        "ElasticNet":ElasticNet(alpha=.02,l1_ratio=.4,max_iter=20000,random_state=rs),
        "Huber":HuberRegressor(max_iter=2000),"KNN":KNeighborsRegressor(12,weights="distance"),
        "SVR_RBF":SVR(C=10),
        "RandomForest":RandomForestRegressor(n_estimators=500,min_samples_leaf=3,max_features="sqrt",random_state=rs,n_jobs=-1),
        "ExtraTrees":ExtraTreesRegressor(n_estimators=600,min_samples_leaf=2,max_features=.8,random_state=rs,n_jobs=-1),
        "AdaBoost":AdaBoostRegressor(n_estimators=250,learning_rate=.03,random_state=rs),
        "GradientBoosting":GradientBoostingRegressor(n_estimators=300,learning_rate=.03,max_depth=3,loss="huber",random_state=rs),
        "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=350,learning_rate=.05,l2_regularization=1,random_state=rs),
    }
    if XGBOOST:m["XGBoost"]=XGBRegressor(n_estimators=700,max_depth=4,learning_rate=.03,subsample=.85,colsample_bytree=.85,min_child_weight=3,reg_alpha=.05,reg_lambda=1.5,objective="reg:squarederror",random_state=rs,n_jobs=-1)
    if LIGHTGBM:m["LightGBM"]=LGBMRegressor(n_estimators=700,learning_rate=.03,num_leaves=24,min_child_samples=20,subsample=.85,colsample_bytree=.85,reg_lambda=1,random_state=rs,n_jobs=-1,verbose=-1)
    if CATBOOST:m["CatBoost"]=CatBoostRegressor(iterations=700,depth=6,learning_rate=.03,loss_function="RMSE",random_seed=rs,verbose=False,allow_writing_files=False)
    return m


def classification_registry(cfg:Config)->dict[str,BaseEstimator]:
    rs=cfg.random_state
    if cfg.fast:
        return {
            "DummyPrior":DummyClassifier(strategy="prior"),
            "LogisticRegression":LogisticRegression(class_weight="balanced",max_iter=3000,random_state=rs),
            "RandomForest":RandomForestClassifier(n_estimators=25,min_samples_leaf=3,max_features="sqrt",class_weight="balanced_subsample",random_state=rs,n_jobs=1),
        }
    m={
        "DummyPrior":DummyClassifier(strategy="prior"),
        "LogisticRegression":LogisticRegression(class_weight="balanced",max_iter=5000,random_state=rs),
        "KNN":KNeighborsClassifier(15,weights="distance"),"SVC_RBF":SVC(C=5,class_weight="balanced",probability=True,random_state=rs),
        "RandomForest":RandomForestClassifier(n_estimators=500,min_samples_leaf=3,class_weight="balanced_subsample",random_state=rs,n_jobs=-1),
        "ExtraTrees":ExtraTreesClassifier(n_estimators=600,min_samples_leaf=2,class_weight="balanced",random_state=rs,n_jobs=-1),
        "AdaBoost":AdaBoostClassifier(n_estimators=250,learning_rate=.03,random_state=rs),
        "GradientBoosting":GradientBoostingClassifier(n_estimators=300,learning_rate=.03,max_depth=3,random_state=rs),
        "HistGradientBoosting":HistGradientBoostingClassifier(max_iter=350,learning_rate=.05,l2_regularization=1,random_state=rs),
    }
    if XGBOOST:m["XGBoost"]=XGBClassifier(n_estimators=700,max_depth=4,learning_rate=.03,subsample=.85,colsample_bytree=.85,min_child_weight=3,reg_alpha=.05,reg_lambda=1.5,eval_metric="logloss",random_state=rs,n_jobs=-1)
    if LIGHTGBM:m["LightGBM"]=LGBMClassifier(n_estimators=700,learning_rate=.03,num_leaves=24,min_child_samples=20,class_weight="balanced",random_state=rs,n_jobs=-1,verbose=-1)
    if CATBOOST:m["CatBoost"]=CatBoostClassifier(iterations=700,depth=6,learning_rate=.03,auto_class_weights="Balanced",random_seed=rs,verbose=False,allow_writing_files=False)
    return m


def flatten(name:str,res:dict[str,np.ndarray])->dict[str,Any]:
    row={"model":name}
    for k,v in res.items():
        if not k.startswith("test_"):continue
        metric=k[5:]; a=np.asarray(v,float)
        if metric.startswith("neg_"):metric=metric[4:];a=-a
        row[f"cv_{metric}_mean"]=np.nanmean(a);row[f"cv_{metric}_std"]=np.nanstd(a)
    row["fit_time_seconds"]=np.sum(res.get("fit_time",[np.nan]));return row


def choose_threshold(y:pd.Series,p:np.ndarray,floor:float)->float:
    precision,recall,t=precision_recall_curve(y,p)
    if len(t)==0:return .5
    valid=precision[:-1]>=floor
    if valid.any():
        ids=np.flatnonzero(valid);return float(t[ids[np.argmax(recall[:-1][valid])]])
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12);return float(t[np.nanargmax(f)])


def explain(model:Pipeline,X:pd.DataFrame,y:pd.Series,task:str,name:str,cfg:Config,paths:dict[str,Path])->None:
    if cfg.fast:
        return
    try:
        imp=permutation_importance(model,X,y,scoring="neg_mean_absolute_error" if task=="regression" else "average_precision",n_repeats=4 if cfg.fast else 20,random_state=cfg.random_state,n_jobs=-1)
        tab=pd.DataFrame({"feature":X.columns,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False)
        tab.to_csv(paths["explainability"]/f"{task}_permutation_importance.csv",index=False)
        top=tab.head(15).sort_values("importance_mean");fig,ax=plt.subplots(figsize=(10,7));ax.barh(top.feature,top.importance_mean,xerr=top.importance_std);ax.set_title(f"{task.title()} Permutation Importance — {name}");save_fig(fig,paths["figures"]/f"16_{task}_importance.png")
    except Exception as e:LOG.warning("Permutation importance skipped: %s",e)
    for feature in [f for f in ["length_m","hole_row_index"] if f in X][:2]:
        try:
            fig,ax=plt.subplots(figsize=(8,6));PartialDependenceDisplay.from_estimator(model,X,[feature],ax=ax);ax.set_title(f"{task.title()} Partial Dependence — {feature}");save_fig(fig,paths["figures"]/f"17_{task}_pdp_{feature}.png")
        except Exception:pass
    if cfg.skip_shap or not SHAP:return
    try:
        sample=X.sample(min(200,len(X)),random_state=cfg.random_state)
        if sample.empty:return
        prep=model.named_steps.get("preprocess")
        if prep is None:
            Xt=sample.to_numpy()
            names=list(sample.columns)
        else:
            Xt=prep.transform(sample)
            names=list(prep.get_feature_names_out())
        est=model.named_steps.get("model",model)
        if isinstance(est,TransformedTargetRegressor):est=est.regressor_
        if not callable(est):
            return
        explainer_kwargs={"check_additivity":False}
        try:
            explainer=shap.Explainer(est,Xt,feature_names=names,**explainer_kwargs)
        except TypeError:
            explainer=shap.Explainer(est,Xt,feature_names=names)
        except Exception:
            return
        sv=explainer(Xt)
        if getattr(sv,"values",None) is None:
            return
        plt.figure(figsize=(10,7));shap.plots.bar(sv,max_display=min(15,len(names)),show=False);plt.tight_layout();plt.savefig(paths["figures"]/f"18_{task}_shap.png",dpi=180,bbox_inches="tight");plt.close()
    except Exception as e:LOG.info("SHAP skipped: %s",e)


def regression_benchmark(df:pd.DataFrame,cfg:Config,paths:dict[str,Path])->dict[str,Any]:
    d=df.loc[df.modeling_valid_rate].copy();X,num,cat=model_data(d);y=d.penetration_rate_m_per_hr.astype(float);train,test,test_dates=split_indices(d,cfg);Xt,Xh=X.loc[train],X.loc[test];yt,yh=y.loc[train],y.loc[test];groups=d.loc[train,"date"].astype("string");cv,kw=cv_object(yt,groups,False,cfg);prep=preprocessor(num,cat)
    rows=[];fitted={}
    for name,est in regression_registry(cfg).items():
        LOG.info("Regression: %s",name);pipe=Pipeline([("preprocess",clone(prep)),("model",TransformedTargetRegressor(regressor=est,func=np.log1p,inverse_func=np.expm1,check_inverse=False))])
        try:
            res=cross_validate(pipe,Xt,yt,cv=cv,scoring={"neg_mae":"neg_mean_absolute_error","neg_rmse":"neg_root_mean_squared_error","r2":"r2"},n_jobs=1,error_score="raise",**kw);row=flatten(name,res);pipe.fit(Xt,yt);pred=np.maximum(pipe.predict(Xh),0);row.update({"holdout_mae":mean_absolute_error(yh,pred),"holdout_rmse":rmse(yh,pred),"holdout_r2":r2_score(yh,pred),"status":"success"});fitted[name]=pipe
        except Exception as e:row={"model":name,"status":f"failed: {e}"};LOG.warning("%s failed: %s",name,e)
        rows.append(row)
    comp=pd.DataFrame(rows);ok=comp[comp.status=="success"].sort_values(["cv_rmse_mean","holdout_mae"]);comp=pd.concat([ok,comp[comp.status!="success"]]);comp.to_csv(paths["tables"]/"regression_model_comparison.csv",index=False);best=str(ok.iloc[0].model);model=fitted[best];pred=np.maximum(model.predict(Xh),0)
    oof=cross_val_predict(clone(model),Xt,yt,cv=cv,method="predict",n_jobs=1,**kw);resid=np.abs(yt.to_numpy()-oof);alpha=1-cfg.confidence_level;q=min(1,math.ceil((len(resid)+1)*(1-alpha))/len(resid));radius=float(np.quantile(resid,q,method="higher"))
    out=d.loc[test,["record_id","hole","date","equipment","operator","penetration_rate_m_per_hr"]].copy();out["predicted_rate_m_per_hr"]=pred;out["error"]=out.penetration_rate_m_per_hr-pred;out["lower_interval"]=np.maximum(pred-radius,0);out["upper_interval"]=pred+radius;out.to_csv(paths["predictions"]/"regression_holdout_predictions.csv",index=False);joblib.dump(model,paths["models"]/"best_penetration_rate_regressor.joblib")
    save_json({"best_model":best,"test_dates":test_dates,"holdout":{"mae":mean_absolute_error(yh,pred),"rmse":rmse(yh,pred),"r2":r2_score(yh,pred)},"interval_radius":radius},paths["models"]/"regression_metadata.json")
    fig,ax=plt.subplots(figsize=(8,8));ax.scatter(yh,pred,alpha=.75);lo=min(yh.min(),pred.min());hi=max(yh.max(),pred.max());ax.plot([lo,hi],[lo,hi],ls="--");ax.set(title=f"Actual vs Predicted — {best}",xlabel="Actual m/hr",ylabel="Predicted m/hr");save_fig(fig,paths["figures"]/"10_regression_actual_predicted.png");explain(model,Xh,yh,"regression",best,cfg,paths)
    return {"best_name":best,"comparison":comp,"holdout":{"mae":mean_absolute_error(yh,pred),"rmse":rmse(yh,pred),"r2":r2_score(yh,pred)}}


def classification_benchmark(df:pd.DataFrame,cfg:Config,paths:dict[str,Path])->dict[str,Any]:
    d=df.loc[df.modeling_valid_rate & df.low_performance_flag.notna()].copy();X,num,cat=model_data(d);y=d.low_performance_flag.astype(int);train,test,test_dates=split_indices(d,cfg);Xt,Xh=X.loc[train],X.loc[test];yt,yh=y.loc[train],y.loc[test]
    if yt.nunique()<2 or yh.nunique()<2:return {"skipped":True}
    groups=d.loc[train,"date"].astype("string");cv,kw=cv_object(yt,groups,True,cfg);prep=preprocessor(num,cat);rows=[];fitted={}
    for name,est in classification_registry(cfg).items():
        LOG.info("Classification: %s",name);pipe=Pipeline([("preprocess",clone(prep)),("model",est)])
        try:
            res=cross_validate(pipe,Xt,yt,cv=cv,scoring={"roc_auc":"roc_auc","average_precision":"average_precision","balanced_accuracy":"balanced_accuracy","f1":"f1"},n_jobs=1,error_score="raise",**kw);row=flatten(name,res);pipe.fit(Xt,yt);p=pipe.predict_proba(Xh)[:,1];row.update({"holdout_roc_auc":roc_auc_score(yh,p),"holdout_pr_auc":average_precision_score(yh,p),"status":"success"});fitted[name]=pipe
        except Exception as e:row={"model":name,"status":f"failed: {e}"};LOG.warning("%s failed: %s",name,e)
        rows.append(row)
    comp=pd.DataFrame(rows);ok=comp[comp.status=="success"].sort_values(["cv_average_precision_mean","cv_roc_auc_mean"],ascending=False);comp=pd.concat([ok,comp[comp.status!="success"]]);comp.to_csv(paths["tables"]/"classification_model_comparison.csv",index=False);best=str(ok.iloc[0].model);model=fitted[best]
    oof=cross_val_predict(clone(model),Xt,yt,cv=cv,method="predict_proba",n_jobs=1,**kw)[:,1];threshold=choose_threshold(yt,oof,cfg.precision_floor);p=model.predict_proba(Xh)[:,1];pred=(p>=threshold).astype(int)
    metrics={"threshold":threshold,"roc_auc":roc_auc_score(yh,p),"pr_auc":average_precision_score(yh,p),"precision":precision_score(yh,pred,zero_division=0),"recall":recall_score(yh,pred,zero_division=0),"f1":f1_score(yh,pred,zero_division=0),"balanced_accuracy":balanced_accuracy_score(yh,pred),"brier":brier_score_loss(yh,p)}
    out=d.loc[test,["record_id","hole","date","equipment","operator","penetration_rate_m_per_hr","low_performance_benchmark","low_performance_flag"]].copy();out["risk_probability"]=p;out["predicted_low_performance"]=pred;out["threshold"]=threshold;out.to_csv(paths["predictions"]/"classification_holdout_predictions.csv",index=False);joblib.dump(model,paths["models"]/"best_low_performance_classifier.joblib");save_json({"best_model":best,"test_dates":test_dates,"metrics":metrics,"classification_report":classification_report(yh,pred,output_dict=True,zero_division=0),"confusion_matrix":confusion_matrix(yh,pred).tolist()},paths["models"]/"classification_metadata.json")
    fig,ax=plt.subplots(figsize=(7,6));ConfusionMatrixDisplay.from_predictions(yh,pred,display_labels=["Normal/high","Low"],cmap="Blues",colorbar=False,ax=ax);ax.set_title(f"Holdout Confusion Matrix — {best}");save_fig(fig,paths["figures"]/"12_confusion_matrix.png")
    fig,ax=plt.subplots(figsize=(8,6));RocCurveDisplay.from_predictions(yh,p,name=best,ax=ax);save_fig(fig,paths["figures"]/"13_roc_curve.png")
    fig,ax=plt.subplots(figsize=(8,6));PrecisionRecallDisplay.from_predictions(yh,p,name=best,ax=ax);save_fig(fig,paths["figures"]/"14_precision_recall.png")
    if len(yh)>=30:
        a,b=calibration_curve(yh,p,n_bins=6,strategy="quantile");fig,ax=plt.subplots(figsize=(7,6));ax.plot(b,a,marker="o");ax.plot([0,1],[0,1],ls="--");ax.set(title="Risk Calibration",xlabel="Predicted probability",ylabel="Observed rate");save_fig(fig,paths["figures"]/"15_calibration.png")
    explain(model,Xh,yh,"classification",best,cfg,paths);return {"best_name":best,"comparison":comp,"metrics":metrics}


def tune_xgb(df:pd.DataFrame,cfg:Config,paths:dict[str,Path])->dict[str,Any]|None:
    if not (OPTUNA and XGBOOST):LOG.warning("Optuna/XGBoost unavailable");return None
    d=df.loc[df.modeling_valid_rate].copy();X,num,cat=model_data(d);y=d.penetration_rate_m_per_hr.astype(float);train,test,_=split_indices(d,cfg);Xt,Xh=X.loc[train],X.loc[test];yt,yh=y.loc[train],y.loc[test];groups=d.loc[train,"date"].astype("string");cv,kw=cv_object(yt,groups,False,cfg);prep=preprocessor(num,cat)
    def objective(trial):
        est=XGBRegressor(n_estimators=trial.suggest_int("n_estimators",250,1000,50),max_depth=trial.suggest_int("max_depth",2,8),learning_rate=trial.suggest_float("learning_rate",.01,.15,log=True),subsample=trial.suggest_float("subsample",.6,1),colsample_bytree=trial.suggest_float("colsample_bytree",.6,1),reg_alpha=trial.suggest_float("reg_alpha",1e-5,2,log=True),reg_lambda=trial.suggest_float("reg_lambda",1e-3,5,log=True),objective="reg:squarederror",random_state=cfg.random_state,n_jobs=-1)
        pipe=Pipeline([("preprocess",clone(prep)),("model",TransformedTargetRegressor(regressor=est,func=np.log1p,inverse_func=np.expm1,check_inverse=False))]);r=cross_validate(pipe,Xt,yt,cv=cv,scoring="neg_root_mean_squared_error",n_jobs=1,**kw);return -r["test_score"].mean()
    study=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=cfg.random_state));study.optimize(objective,n_trials=cfg.optuna_trials,show_progress_bar=False);study.trials_dataframe().to_csv(paths["tables"]/"optuna_trials.csv",index=False);save_json({"best_params":study.best_params,"best_cv_rmse":study.best_value},paths["models"]/"optuna_best.json");return {"best_params":study.best_params,"best_cv_rmse":study.best_value}


def report(df:pd.DataFrame,tables:dict[str,pd.DataFrame],tests:pd.DataFrame,reg:dict[str,Any],clf:dict[str,Any],paths:dict[str,Path],cfg:Config)->None:
    valid=df.loc[df.modeling_valid_rate];eq=tables["kpis_by_equipment"].sort_values("median_rate_m_per_hr",ascending=False);op=tables["kpis_by_operator"].sort_values("median_rate_m_per_hr",ascending=False);sig=tests.loc[tests.significant_0_05] if not tests.empty else pd.DataFrame();lines="\n".join(f"- {r.test}: p={r.p_value:.4g}" for r in sig.itertuples()) or "- No test reached p < 0.05."
    text=f"""# Open-Pit Drill Fleet Performance Analytics\n\n## Executive summary\n\nThe source contains **{len(df):,} holes**, **{df.equipment.nunique()} rigs**, **{df.operator.nunique()} operators**, and **{df.date.nunique()} dates**. Total recorded length is **{df.length_m.sum():,.1f} m**.\n\nAfter transparent screening, **{int(df.modeling_valid_rate.sum()):,} records** were eligible for modelling. There are **{int(df.penetration_rate_m_per_hr.isna().sum())} missing rates**, **{int(df.rate_domain_outlier_flag.sum())} domain-limit outliers above {cfg.max_plausible_rate:g} m/hr**, and **{int(df.rate_statistical_outlier_flag.sum())} robust statistical outliers**. All remain in the cleaned export with flags.\n\nQuality-filtered median penetration rate is **{valid.penetration_rate_m_per_hr.median():.2f} m/hr**. Redrill rate is **{df.redrill_flag.mean()*100:.2f}%** and wet-hole rate is **{df.wet.astype('Float64').mean()*100:.2f}%**.\n\n## Operational highlights\n\n- Highest median-rate rig: **{eq.iloc[0].equipment}** at **{eq.iloc[0].median_rate_m_per_hr:.2f} m/hr**.\n- Highest median-rate operator: **{op.iloc[0].operator}** at **{op.iloc[0].median_rate_m_per_hr:.2f} m/hr**.\n- Consensus anomalies requiring engineering review: **{int((df.consensus_anomaly_flag==1).sum())}**.\n\n## Statistical evidence\n\n{lines}\n\n## Predictive modelling\n\nBest regression model: **{reg.get('best_name','N/A')}**; temporal holdout MAE **{reg.get('holdout',{}).get('mae',float('nan')):.2f} m/hr**, RMSE **{reg.get('holdout',{}).get('rmse',float('nan')):.2f} m/hr**.\n\nBest low-performance classifier: **{clf.get('best_name','N/A')}**. Metrics are saved in `models/classification_metadata.json`. Threshold selection uses out-of-fold training probabilities only; the temporal holdout remains untouched until final evaluation.\n\n## Engineering recommendations\n\n1. Review consensus anomalies before correcting or deleting any record.\n2. Compare operators within rig and hole-type context, not using raw averages alone.\n3. Treat model risk as a prioritization signal, not an automated personnel judgement.\n4. Add shift, start/finish time, delays, maintenance state, bit condition, geology, planned depth, and cost fields.\n5. Monitor drift by date and validate findings with supervisors and operators.\n\n## Limitation\n\nThis dataset lacks explicit plans, delay codes, maintenance, cost, and direct rock-hardness measurements. Results demonstrate analytics competence and association, not causal proof or an autonomous dispatch recommendation.\n"""
    (paths["reports"]/"executive_report.md").write_text(text,encoding="utf-8")


def run(args:argparse.Namespace)->None:
    cfg=Config(random_state=args.random_state,max_plausible_rate=args.max_plausible_rate,robust_z_threshold=args.robust_z_threshold,anomaly_contamination=args.anomaly_contamination,low_performance_quantile=args.low_performance_quantile,test_date_fraction=args.test_date_fraction,cv_folds=2 if args.fast else args.cv_folds,precision_floor=args.precision_floor,confidence_level=args.confidence_level,optuna_trials=args.optuna_trials,fast=args.fast,skip_shap=args.skip_shap)
    base=Path(args.output_dir).expanduser().resolve();configure_logging(base,args.verbose);paths=output_tree(base);LOG.info("Starting pipeline")
    sheet=int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df=load_clean_engineer(Path(args.input).expanduser().resolve(),sheet,cfg);df=anomaly_detection(df,cfg,paths);tables=export_tables(df,paths);tests=statistics_analysis(df,paths,cfg.fast);visualizations(df,paths,cfg.fast);reg=regression_benchmark(df,cfg,paths);clf=classification_benchmark(df,cfg,paths);tuned=tune_xgb(df,cfg,paths) if args.tune_xgboost else None;report(df,tables,tests,reg,clf,paths,cfg)
    save_json({"generated_at":datetime.now(),"input":args.input,"configuration":asdict(cfg),"rows":len(df),"valid_model_rows":int(df.modeling_valid_rate.sum()),"regression_best":reg.get("best_name"),"classification_best":clf.get("best_name"),"tuning":tuned,"dependencies":{"statsmodels":STATSMODELS,"plotly":PLOTLY,"xgboost":XGBOOST,"lightgbm":LIGHTGBM,"catboost":CATBOOST,"optuna":OPTUNA,"shap":SHAP}},paths["base"]/"run_metadata.json");LOG.info("Complete: %s",base)


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Advanced open-pit drill fleet analytics",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",required=True);p.add_argument("--output-dir",default="drill_analytics_outputs");p.add_argument("--sheet",default="0")
    p.add_argument("--random-state",type=int,default=42);p.add_argument("--max-plausible-rate",type=float,default=250);p.add_argument("--robust-z-threshold",type=float,default=4.5);p.add_argument("--anomaly-contamination",type=float,default=.04);p.add_argument("--low-performance-quantile",type=float,default=.25);p.add_argument("--test-date-fraction",type=float,default=.25);p.add_argument("--cv-folds",type=int,default=5);p.add_argument("--precision-floor",type=float,default=.35);p.add_argument("--confidence-level",type=float,default=.90);p.add_argument("--optuna-trials",type=int,default=30)
    p.add_argument("--fast",action="store_true",help="Quick three-model validation mode");p.add_argument("--tune-xgboost",action="store_true");p.add_argument("--skip-shap",action="store_true");p.add_argument("--verbose",action="store_true");return p


if __name__=="__main__":
    try:run(parser().parse_args())
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as exc:LOG.exception("Pipeline failed: %s",exc);raise SystemExit(1) from exc
