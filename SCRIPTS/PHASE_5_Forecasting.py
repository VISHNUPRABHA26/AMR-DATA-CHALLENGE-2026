# ============================================================
# AMR DATA CHALLENGE 2026 — PHASE 5: FORECASTING
# Nosocomial Neuroinvasive Bacterial AMR Study
# ============================================================
# OBJECTIVE:
#   Forecast MDR rates per organism × country from 2025–2030
#   using ARIMA and Prophet models on 2004–2024 time series.
#   Flag series predicted to cross MDR > 50% threshold.
#
# METHODS:
#   - Minimum 8 valid yearly data points to fit model
#     (Hyndman & Athanasopoulos 2021, FPP3; Box & Jenkins 1976)
#   - WHO GLASS 2023 minimum isolate threshold: >= 10 per cell
#   - Dangerous threshold: MDR rate > 50%
#     (WHO GLASS 2023; ECDC EARS-Net 2023)
#   - Models: ARIMA (auto-order selection via AIC minimization)
#             Prophet (trend-only, no sub-annual seasonality)
#   - Forecast horizon: 2025–2030 (6 years)
#
# INPUT:
#   ~/AMR_DATA_CHALLENGE/PHASE1_outputs/atlas_filtered_mdr_flagged.csv
#
# OUTPUT (saved to ~/AMR_DATA_CHALLENGE/PHASE8_outputs/):
#   P5_timeseries_input.csv         — aggregated MDR rates per org×country×year
#   P5_skipped_series.csv           — series excluded with reason
#   P5_arima_forecasts.csv          — ARIMA forecasts 2025-2030
#   P5_prophet_forecasts.csv        — Prophet forecasts 2025-2030
#   P5_combined_forecasts.csv       — merged ARIMA + Prophet with agreement flag
#   P5_threshold_alerts.csv         — series crossing MDR>50% within 2025-2030
#   P5_model_diagnostics.csv        — AIC, RMSE, model order per series
#   plots/                          — one plot per organism (all countries overlaid)
#
# REFERENCES:
#   Magiorakos et al. 2012. CMI 18(3):268-281
#   Box & Jenkins. 1976. Time Series Analysis. Holden-Day.
#   Hyndman & Athanasopoulos. 2021. Forecasting: Principles and Practice (3rd ed.)
#   Taylor & Letham. 2018. The American Statistician. 72(1):37-45 [Prophet]
#   WHO GLASS. 2023. Global Antimicrobial Resistance and Use Surveillance Report.
#   ECDC EARS-Net. 2023. Antimicrobial Resistance Surveillance in Europe.
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
import logging
warnings.filterwarnings("ignore")

# Suppress all prophet/stan/statsmodels noise
for logger_name in ["prophet", "cmdstanpy", "cmdstan", "stan", "pystan",
                    "statsmodels", "statsmodels.tsa", "statsmodels.base"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

# Suppress statsmodels ConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter("ignore", ConvergenceWarning)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import itertools

from prophet import Prophet

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================
# PATHS
# ============================================================
ATLAS_PATH = os.path.expanduser(
    "~/AMR_DATA_CHALLENGE/phase1_outputs/atlas_filtered_mdr_flagged.csv"
)
OUT_DIR      = os.path.expanduser("~/AMR_DATA_CHALLENGE/phase5_outputs")
PLOT_DIR     = os.path.join(OUT_DIR, "plots")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

def save(df, name):
    path = os.path.join(OUT_DIR, name + ".csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {name}.csv  ({len(df):,} rows)")

# ============================================================
# STUDY PARAMETERS
# ============================================================
MIN_ISOLATES      = 10   # WHO GLASS 2023 minimum per cell
MIN_VALID_YEARS   = 8    # Hyndman & Athanasopoulos 2021 minimum for ARIMA
MDR_THRESHOLD     = 50.0 # WHO GLASS / ECDC EARS-Net danger threshold (%)
FORECAST_YEARS    = list(range(2025, 2031))  # 2025–2030 inclusive

TARGET_ORGANISMS = [
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Escherichia coli",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Enterobacter cloacae",
    "Enterococcus faecium",
    "Staphylococcus epidermidis",
    "Serratia marcescens",
    "Citrobacter koseri",
    "Streptococcus pneumoniae",
    "Haemophilus influenzae",
    "Salmonella spp"
]

# ============================================================
# MDR CLASS MAP — Magiorakos et al. 2012 (ECDC/CDC standard)
# Identical to Phase 2 definition to ensure MDR_v2 consistency
# Non-susceptible = Resistant OR Intermediate
# MDR = non-susceptible in >= 3 antibiotic classes
# ============================================================
MDR_CLASSES = {
    "Penicillins": [
        "Ampicillin_I", "Amoxycillin clavulanate_I",
        "Ampicillin sulbactam_I", "Piperacillin tazobactam_I", "Penicillin_I"
    ],
    "Cephalosporins": [
        "Cefepime_I", "Ceftazidime_I", "Ceftriaxone_I", "Ceftaroline_I",
        "Ceftazidime avibactam_I", "Ceftolozane tazobactam_I"
    ],
    "Monobactams":      ["Aztreonam_I"],
    "Carbapenems":      ["Meropenem_I", "Imipenem_I", "Doripenem_I"],
    "Fluoroquinolones": ["Levofloxacin_I", "Ciprofloxacin_I"],
    "Aminoglycosides":  ["Amikacin_I", "Gentamicin_I"],
    "Glycopeptides":    ["Vancomycin_I", "Teicoplanin_I"],
    "Polymyxins":       ["Colistin_I"],
    "Tetracyclines":    ["Minocycline_I", "Tigecycline_I"],
    "Trimethoprim_Sulfonamides": ["Trimethoprim sulfa_I"],
    "Oxazolidinones":   ["Linezolid_I"],
    "Lipopeptides":     ["Daptomycin_I"],
    "Penicillinase_Resistant_Penicillins": ["Oxacillin_I"],
    "Lincosamides":     ["Clindamycin_I"],
    "Macrolides":       ["Erythromycin_I"]
}
TOTAL_CLASSES = len(MDR_CLASSES)  # 15

# ============================================================
# STEP 1 — LOAD ATLAS PHASE 1 OUTPUT
# ============================================================
print("=" * 65)
print("STEP 1 — LOADING ATLAS PHASE 1 OUTPUT")
print("=" * 65)

atlas = pd.read_csv(ATLAS_PATH, low_memory=False)
print(f"  Loaded: {len(atlas):,} rows x {atlas.shape[1]} columns")

# Resolve column names (defensive — handles case variants)
def resolve(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

SP_COL   = resolve(atlas, ["Species"])
CTRY_COL = resolve(atlas, ["Country"])
YR_COL   = resolve(atlas, ["Year"])
AGE_COL  = resolve(atlas, ["Age Group", "Age.Group"])

print(f"  Species col : {SP_COL}")
print(f"  Country col : {CTRY_COL}")
print(f"  Year col    : {YR_COL}")
print(f"  Age col     : {AGE_COL}")

# Standardize organism labels
def get_organism_label(species_val):
    for org in TARGET_ORGANISMS:
        if org.lower() in str(species_val).lower():
            return org
    return None  # exclude non-target organisms

atlas["Organism_Label"] = atlas[SP_COL].apply(get_organism_label)
atlas = atlas[atlas["Organism_Label"].notna()].copy()
print(f"  After organism filter: {len(atlas):,} rows")

# Standardize year
atlas[YR_COL] = pd.to_numeric(atlas[YR_COL], errors="coerce")
atlas = atlas[atlas[YR_COL].notna()].copy()
atlas[YR_COL] = atlas[YR_COL].astype(int)

# ============================================================
# STEP 2 — BUILD MDR_v2 (Magiorakos 2012, R+I = non-susceptible)
# This replicates Phase 2 compute_mxp() exactly.
# Non-susceptible in >= 3 antibiotic classes = MDR
# ============================================================
print("\n" + "=" * 65)
print("STEP 2 — REBUILDING MDR_v2 (Magiorakos 2012 correct definition)")
print("  Non-susceptible = Resistant OR Intermediate")
print("  MDR = non-susceptible in >= 3 antibiotic classes")
print("  Processing... (this will take several minutes on 40MB file)")
print("=" * 65)

atlas_cols = set(atlas.columns)

def compute_mdr_v2(row):
    """
    Replicates Phase 2 compute_mxp() MDR flag.
    Returns 1 if non-susceptible in >= 3 classes, else 0.
    Reference: Magiorakos et al. 2012, CMI 18(3):268-281
    """
    ns_count = 0
    for cls, drugs in MDR_CLASSES.items():
        available = [d for d in drugs if d in atlas_cols]
        if not available:
            continue
        vals = [row[d] for d in available if pd.notna(row[d])]
        if not vals:
            continue
        if any(v in ["Resistant", "Intermediate"] for v in vals):
            ns_count += 1
    return 1 if ns_count >= 3 else 0

atlas["MDR_v2"] = atlas.apply(compute_mdr_v2, axis=1)

total_mdr = atlas["MDR_v2"].sum()
print(f"  MDR_v2 isolates : {total_mdr:,} ({total_mdr/len(atlas)*100:.1f}%)")
print(f"  Non-MDR         : {len(atlas)-total_mdr:,}")

# ============================================================
# STEP 3 — AGGREGATE MDR RATE PER ORGANISM × COUNTRY × YEAR
# Formula: MDR_Rate = MDR_v2_count / Total_isolates * 100
# Minimum isolates per cell: >= 10 (WHO GLASS 2023)
# ============================================================
print("\n" + "=" * 65)
print("STEP 3 — AGGREGATING MDR RATE PER ORGANISM × COUNTRY × YEAR")
print(f"  Minimum isolates per cell: >= {MIN_ISOLATES} (WHO GLASS 2023)")
print("=" * 65)

grouped = atlas.groupby(["Organism_Label", CTRY_COL, YR_COL]).agg(
    Total_N   = ("MDR_v2", "count"),
    MDR_Count = ("MDR_v2", "sum")
).reset_index()
grouped.columns = ["Organism", "Country", "Year", "Total_N", "MDR_Count"]

# Apply WHO GLASS minimum threshold
grouped["MDR_Rate_%"] = np.where(
    grouped["Total_N"] >= MIN_ISOLATES,
    (grouped["MDR_Count"] / grouped["Total_N"] * 100).round(2),
    np.nan  # below threshold -> NaN, not used in modelling
)
grouped["Sufficient"] = (grouped["Total_N"] >= MIN_ISOLATES)

print(f"  Total organism×country×year cells : {len(grouped):,}")
print(f"  Cells with sufficient data (>=10) : {grouped['Sufficient'].sum():,}")
print(f"  Cells below threshold             : {(~grouped['Sufficient']).sum():,}")

save(grouped, "P5_timeseries_input")

# ============================================================
# STEP 4 — BUILD TIME SERIES PER ORGANISM × COUNTRY
# Filter: >= 8 valid (sufficient) yearly data points
# ============================================================
print("\n" + "=" * 65)
print("STEP 4 — BUILDING TIME SERIES PER ORGANISM × COUNTRY")
print(f"  Minimum valid yearly points: >= {MIN_VALID_YEARS}")
print("=" * 65)

series_list   = []  # accepted series
skipped_list  = []  # rejected series with reason

for org in TARGET_ORGANISMS:
    org_data = grouped[grouped["Organism"] == org]
    countries = org_data["Country"].unique()

    for country in countries:
        ts_raw = org_data[org_data["Country"] == country].copy()
        ts_raw = ts_raw.sort_values("Year")

        # Valid points = cells with sufficient isolates
        valid = ts_raw[ts_raw["Sufficient"] == True]
        n_valid = len(valid)

        if n_valid < MIN_VALID_YEARS:
            skipped_list.append({
                "Organism":      org,
                "Country":       country,
                "Total_Years":   len(ts_raw),
                "Valid_Years":   n_valid,
                "Reason":        f"Fewer than {MIN_VALID_YEARS} valid data points"
            })
            continue

        series_list.append({
            "Organism": org,
            "Country":  country,
            "ts":       valid[["Year", "MDR_Rate_%"]].reset_index(drop=True)
        })

print(f"  Series accepted for modelling : {len(series_list)}")
print(f"  Series skipped                : {len(skipped_list)}")

df_skipped = pd.DataFrame(skipped_list)
save(df_skipped, "P5_skipped_series")

# ============================================================
# HELPER: ARIMA ORDER SELECTION BY AIC
# Grid search over p in [0,1,2], d in [0,1], q in [0,1,2]
# d determined by ADF test for stationarity
# Reference: Box & Jenkins 1976; Hyndman & Athanasopoulos 2021
# ============================================================

def select_arima_order(ts_values):
    """
    Select best ARIMA(p,d,q) order by minimizing AIC.
    d is set by ADF test: if series is non-stationary, d=1; else d=0.

    Fallback hierarchy (scientifically defensible):
      1. Best AIC across p in {0,1,2}, q in {0,1,2}
      2. ARIMA(1,d,0) — simple autoregressive model
      3. ARIMA(0,1,0) — random walk (guaranteed to converge on any series)
         Justification: random walk is the minimum assumption for a
         non-stationary epidemiological time series with unknown structure.
         Reference: Hyndman & Athanasopoulos 2021, FPP3, Ch. 9.1

    Returns (p, d, q, best_aic)
    """
    # ADF test for stationarity (p-value < 0.05 = stationary)
    try:
        adf_result = adfuller(ts_values, autolag="AIC")
        adf_pval   = adf_result[1]
        d = 0 if adf_pval < 0.05 else 1
    except Exception:
        d = 1  # default to differencing if ADF test fails

    best_aic   = np.inf
    best_order = None

    # Grid search — restrict to low orders for short series (n=8-21)
    # High-order models (p=2,q=2) are over-parameterised for n<20
    for p, q in itertools.product(range(3), range(3)):
        try:
            model = SARIMAX(
                ts_values,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=200, method="lbfgs")
            if result.aic < best_aic:
                best_aic   = result.aic
                best_order = (p, d, q)
        except Exception:
            continue

    # Fallback 1: ARIMA(1,d,0) — simple AR model
    if best_order is None:
        try:
            model = SARIMAX(
                ts_values,
                order=(1, d, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False, maxiter=500, method="nm")
            best_order = (1, d, 0)
            best_aic   = result.aic
        except Exception:
            pass

    # Fallback 2: ARIMA(0,1,0) — random walk (always converges)
    if best_order is None:
        try:
            model = SARIMAX(
                ts_values,
                order=(0, 1, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False)
            best_order = (0, 1, 0)
            best_aic   = result.aic
        except Exception:
            # If even this fails, return a flag
            return 0, 1, 0, np.nan

    return best_order[0], best_order[1], best_order[2], best_aic


def fit_arima_forecast(ts_df, forecast_years):
    """
    Fit best ARIMA model and forecast for given years.
    Returns forecast df, diagnostics dict.
    ts_df: DataFrame with columns [Year, MDR_Rate_%]
    """
    ts_values = ts_df["MDR_Rate_%"].values.astype(float)
    last_year = int(ts_df["Year"].max())

    p, d, q, aic = select_arima_order(ts_values)

    try:
        model  = SARIMAX(
            ts_values,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        # Try lbfgs first (more stable for short series), fall back to nm
        try:
            result = model.fit(disp=False, maxiter=500, method="lbfgs")
        except Exception:
            result = model.fit(disp=False, maxiter=1000, method="nm")

        # In-sample RMSE
        fitted   = result.fittedvalues
        rmse     = np.sqrt(np.mean((ts_values[d:] - fitted[d:]) ** 2))

        # Forecast steps = number of years from last observed to 2030
        n_steps  = max(forecast_years) - last_year
        forecast = result.get_forecast(steps=n_steps)
        # Normalise to numpy arrays — conf_int() returns DataFrame or ndarray
        # depending on statsmodels version. Use numpy indexing for safety.
        fc_mean_raw = forecast.predicted_mean
        fc_mean = np.asarray(fc_mean_raw).flatten()
        fc_ci_raw = forecast.conf_int(alpha=0.05)
        if hasattr(fc_ci_raw, 'values'):
            fc_ci = fc_ci_raw.values
        else:
            fc_ci = np.asarray(fc_ci_raw)
        if fc_ci.ndim == 1:
            fc_ci = fc_ci.reshape(-1, 2)

        # Extract only the years we need (2025-2030)
        future_years = list(range(last_year + 1, max(forecast_years) + 1))
        out_rows = []
        for i, yr in enumerate(future_years):
            if yr in forecast_years:
                lo = float(np.clip(fc_ci[i, 0], 0, 100))
                hi = float(np.clip(fc_ci[i, 1], 0, 100))
                mn = float(np.clip(fc_mean[i],  0, 100))
                out_rows.append({
                    "Year":               yr,
                    "ARIMA_Forecast_%":   round(mn, 2),
                    "ARIMA_CI_Lower_%":   round(lo, 2),
                    "ARIMA_CI_Upper_%":   round(hi, 2)
                })

        diagnostics = {
            "ARIMA_p": p, "ARIMA_d": d, "ARIMA_q": q,
            "ARIMA_AIC": round(aic, 3),
            "ARIMA_RMSE": round(rmse, 3),
            "ARIMA_Status": "OK"
        }
        return pd.DataFrame(out_rows), diagnostics

    except Exception as e:
        diagnostics = {
            "ARIMA_p": None, "ARIMA_d": None, "ARIMA_q": None,
            "ARIMA_AIC": None, "ARIMA_RMSE": None,
            "ARIMA_Status": f"FAILED: {str(e)}"
        }
        return None, diagnostics


def fit_prophet_forecast(ts_df, forecast_years):
    """
    Fit Prophet model and forecast for given years.
    Annual data — no sub-annual seasonality.
    Uncertainty via Monte Carlo (uncertainty_samples=500).
    Reference: Taylor & Letham 2018, The American Statistician 72(1):37-45
    Returns forecast df, diagnostics dict.
    """
    # Prophet requires columns: ds (datetime), y (value)
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(ts_df["Year"].astype(str) + "-01-01"),
        "y":  ts_df["MDR_Rate_%"].astype(float)
    })

    try:
        m = Prophet(
            yearly_seasonality=False,   # annual data, no sub-annual seasonality
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=500,    # Monte Carlo uncertainty
            changepoint_prior_scale=0.05  # conservative — avoids overfitting short series
        )
        m.fit(prophet_df)

        # Future dataframe: all years from data start to 2030
        last_year = int(ts_df["Year"].max())
        n_future  = max(forecast_years) - last_year
        future    = m.make_future_dataframe(periods=n_future, freq="YS")
        forecast  = m.predict(future)

        # In-sample RMSE
        insample = forecast[forecast["ds"].dt.year <= last_year]
        actuals  = ts_df["MDR_Rate_%"].values
        pred_in  = insample["yhat"].values[-len(actuals):]
        rmse     = np.sqrt(np.mean((actuals - pred_in) ** 2))

        out_rows = []
        for yr in forecast_years:
            row_f = forecast[forecast["ds"].dt.year == yr]
            if len(row_f) == 0:
                continue
            mn = float(np.clip(row_f["yhat"].values[0],       0, 100))
            lo = float(np.clip(row_f["yhat_lower"].values[0], 0, 100))
            hi = float(np.clip(row_f["yhat_upper"].values[0], 0, 100))
            out_rows.append({
                "Year":                 yr,
                "Prophet_Forecast_%":   round(mn, 2),
                "Prophet_CI_Lower_%":   round(lo, 2),
                "Prophet_CI_Upper_%":   round(hi, 2)
            })

        diagnostics = {
            "Prophet_RMSE":   round(rmse, 3),
            "Prophet_Status": "OK"
        }
        return pd.DataFrame(out_rows), diagnostics

    except Exception as e:
        diagnostics = {
            "Prophet_RMSE":   None,
            "Prophet_Status": f"FAILED: {str(e)}"
        }
        return None, diagnostics

# ============================================================
# STEP 5 — FIT ARIMA AND PROPHET FOR ALL ACCEPTED SERIES
# ============================================================
print("\n" + "=" * 65)
print("STEP 5 — FITTING ARIMA AND PROPHET MODELS")
print(f"  Total series to model: {len(series_list)}")
print(f"  Forecast horizon: 2025–2030")
print("=" * 65)

arima_rows      = []
prophet_rows    = []
diagnostic_rows = []

for idx, entry in enumerate(series_list):
    org     = entry["Organism"]
    country = entry["Country"]
    ts_df   = entry["ts"]

    if (idx + 1) % 50 == 0 or idx == 0:
        print(f"  Processing [{idx+1}/{len(series_list)}]: {org} — {country}")

    # --- ARIMA ---
    arima_fc, arima_diag = fit_arima_forecast(ts_df, FORECAST_YEARS)
    if arima_fc is not None:
        for _, row in arima_fc.iterrows():
            arima_rows.append({
                "Organism": org,
                "Country":  country,
                **row.to_dict()
            })

    # --- Prophet ---
    prophet_fc, prophet_diag = fit_prophet_forecast(ts_df, FORECAST_YEARS)
    if prophet_fc is not None:
        for _, row in prophet_fc.iterrows():
            prophet_rows.append({
                "Organism": org,
                "Country":  country,
                **row.to_dict()
            })

    # --- Diagnostics ---
    diag_row = {
        "Organism":        org,
        "Country":         country,
        "N_Valid_Years":   len(ts_df),
        "Year_Min":        int(ts_df["Year"].min()),
        "Year_Max":        int(ts_df["Year"].max()),
        "MDR_Rate_Mean_%": round(ts_df["MDR_Rate_%"].mean(), 2),
        "MDR_Rate_Last_%": round(ts_df["MDR_Rate_%"].iloc[-1], 2),
    }
    diag_row.update(arima_diag)
    diag_row.update(prophet_diag)
    diagnostic_rows.append(diag_row)

print(f"\n  ARIMA forecast rows   : {len(arima_rows)}")
print(f"  Prophet forecast rows : {len(prophet_rows)}")

df_arima   = pd.DataFrame(arima_rows)
df_prophet = pd.DataFrame(prophet_rows)
df_diag    = pd.DataFrame(diagnostic_rows)

save(df_arima,   "P5_arima_forecasts")
save(df_prophet, "P5_prophet_forecasts")
save(df_diag,    "P5_model_diagnostics")

# ============================================================
# STEP 6 — MERGE ARIMA AND PROPHET FORECASTS
# Agreement flag: both models within 10 percentage points
# ============================================================
print("\n" + "=" * 65)
print("STEP 6 — MERGING ARIMA AND PROPHET FORECASTS")
print("=" * 65)

if len(df_prophet) > 0:
    if len(df_arima) > 0:
        df_combined = pd.merge(
            df_arima, df_prophet,
            on=["Organism", "Country", "Year"],
            how="outer"
        )
    else:
        # ARIMA fully failed — use Prophet only, add empty ARIMA columns
        print("  NOTE: ARIMA produced 0 rows. Combined table will use Prophet only.")
        print("        ARIMA columns will be NaN. Alerts will fire on Prophet forecasts.")
        df_combined = df_prophet.copy()
        df_combined["ARIMA_Forecast_%"] = np.nan
        df_combined["ARIMA_CI_Lower_%"] = np.nan
        df_combined["ARIMA_CI_Upper_%"] = np.nan

    # Agreement: both forecasts within 10 percentage points
    df_combined["Model_Agreement"] = df_combined.apply(
        lambda r: "AGREE" if (
            pd.notna(r.get("ARIMA_Forecast_%")) and
            pd.notna(r.get("Prophet_Forecast_%")) and
            abs(r["ARIMA_Forecast_%"] - r["Prophet_Forecast_%"]) <= 10.0
        ) else "DISAGREE",
        axis=1
    )

    # Ensemble mean (average of both when both available)
    df_combined["Ensemble_Mean_%"] = df_combined.apply(
        lambda r: round(
            np.nanmean([r.get("ARIMA_Forecast_%"), r.get("Prophet_Forecast_%")]), 2
        ) if pd.notna(r.get("ARIMA_Forecast_%")) or pd.notna(r.get("Prophet_Forecast_%"))
        else np.nan,
        axis=1
    )

    save(df_combined, "P5_combined_forecasts")
    print(f"  Combined rows: {len(df_combined):,}")
    agree_pct = (df_combined["Model_Agreement"] == "AGREE").mean() * 100
    print(f"  Model agreement rate: {agree_pct:.1f}%")
else:
    df_combined = pd.DataFrame()
    print("  WARNING: One or both model outputs are empty — check diagnostics.")

# ============================================================
# STEP 7 — THRESHOLD ALERTS (MDR > 50%)
# Flag: any year in 2025-2030 where Ensemble_Mean_% > 50
# Reference: WHO GLASS 2023; ECDC EARS-Net 2023
# ============================================================
print("\n" + "=" * 65)
print(f"STEP 7 — THRESHOLD ALERTS (MDR Rate > {MDR_THRESHOLD}%)")
print("  Reference: WHO GLASS 2023 / ECDC EARS-Net 2023")
print("=" * 65)

alert_rows = []

if len(df_combined) > 0:
    for (org, country), grp in df_combined.groupby(["Organism", "Country"]):
        grp = grp.sort_values("Year")

        # Current MDR rate (last observed year)
        diag_row = df_diag[
            (df_diag["Organism"] == org) &
            (df_diag["Country"]  == country)
        ]
        current_mdr = float(diag_row["MDR_Rate_Last_%"].values[0]) \
            if len(diag_row) > 0 else np.nan

        # Already above threshold?
        already_above = current_mdr > MDR_THRESHOLD if pd.notna(current_mdr) else False

        # Check each forecast year
        for _, frow in grp.iterrows():
            arima_val   = frow.get("ARIMA_Forecast_%")
            prophet_val = frow.get("Prophet_Forecast_%")
            ensemble_val = frow.get("Ensemble_Mean_%")

            arima_alert    = pd.notna(arima_val)   and float(arima_val)   > MDR_THRESHOLD
            prophet_alert  = pd.notna(prophet_val) and float(prophet_val) > MDR_THRESHOLD
            ensemble_alert = pd.notna(ensemble_val) and float(ensemble_val) > MDR_THRESHOLD

            # Fire alert if ANY available model crosses threshold
            # (not requiring both — single model alert is valid and reported separately)
            if arima_alert or prophet_alert or ensemble_alert:
                # Determine which models agree on alert
                models_alerting = []
                if arima_alert:   models_alerting.append("ARIMA")
                if prophet_alert: models_alerting.append("Prophet")

                alert_rows.append({
                    "Organism":             org,
                    "Country":              country,
                    "Alert_Year":           int(frow["Year"]),
                    "Current_MDR_Rate_%":   round(current_mdr, 2) if pd.notna(current_mdr) else None,
                    "Already_Above_50":     already_above,
                    "ARIMA_Forecast_%":     frow.get("ARIMA_Forecast_%"),
                    "Prophet_Forecast_%":   frow.get("Prophet_Forecast_%"),
                    "Ensemble_Mean_%":      frow.get("Ensemble_Mean_%"),
                    "ARIMA_Alert":          arima_alert,
                    "Prophet_Alert":        prophet_alert,
                    "Both_Models_Alert":    arima_alert and prophet_alert,
                    "Models_Alerting":      "+".join(models_alerting)
                })

df_alerts = pd.DataFrame(alert_rows)

if len(df_alerts) > 0:
    # Keep earliest alert year per organism×country
    first_alert = df_alerts.sort_values("Alert_Year").groupby(
        ["Organism", "Country"]
    ).first().reset_index()

    save(df_alerts,  "P5_threshold_alerts")
    save(first_alert, "P5_first_threshold_crossing")

    print(f"\n  Total alert records         : {len(df_alerts):,}")
    print(f"  Unique organism×country pairs alerting: "
          f"{df_alerts[['Organism','Country']].drop_duplicates().shape[0]}")
    print(f"  Both-model alerts           : {df_alerts['Both_Models_Alert'].sum():,}")

    print("\n  TOP ALERTS — Both models agree, crossing 50% threshold:")
    top = first_alert[first_alert["Both_Models_Alert"] == True].sort_values(
        "Alert_Year"
    ).head(20)
    if len(top) > 0:
        for _, r in top.iterrows():
            print(f"    {r['Organism']:<35} {r['Country']:<25} "
                  f"→ {r['Alert_Year']}  "
                  f"[ARIMA:{r['ARIMA_Forecast_%']:.1f}% "
                  f"Prophet:{r['Prophet_Forecast_%']:.1f}%]")
    else:
        print("    No both-model agreement alerts found.")
else:
    print("  No threshold alerts triggered.")

# ============================================================
# STEP 8 — PLOTS: ONE PER ORGANISM
# Each plot: all qualifying countries overlaid
# Historical (solid) + ARIMA forecast (dashed) + Prophet (dotted)
# Red horizontal line at MDR=50% threshold
# ============================================================
print("\n" + "=" * 65)
print("STEP 8 — GENERATING FORECAST PLOTS")
print("=" * 65)

for org in TARGET_ORGANISMS:
    org_series = [s for s in series_list if s["Organism"] == org]
    if len(org_series) == 0:
        continue

    # Filter combined forecasts for this organism
    if len(df_combined) > 0:
        org_fc = df_combined[df_combined["Organism"] == org]
    else:
        org_fc = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(14, 7))

    colors = cm.tab20(np.linspace(0, 1, len(org_series)))

    for i, entry in enumerate(org_series):
        country = entry["Country"]
        ts_df   = entry["ts"]
        color   = colors[i]

        # Historical line
        ax.plot(
            ts_df["Year"], ts_df["MDR_Rate_%"],
            color=color, linewidth=1.5,
            label=country if len(org_series) <= 20 else "_nolegend_"
        )

        if len(org_fc) > 0:
            fc_c = org_fc[org_fc["Country"] == country].sort_values("Year")
            if len(fc_c) > 0:
                # Connect historical end to forecast start
                last_obs_yr  = ts_df["Year"].max()
                last_obs_val = ts_df[ts_df["Year"] == last_obs_yr]["MDR_Rate_%"].values[0]

                # ARIMA forecast
                if "ARIMA_Forecast_%" in fc_c.columns:
                    arima_yrs  = [last_obs_yr] + fc_c["Year"].tolist()
                    arima_vals = [last_obs_val] + fc_c["ARIMA_Forecast_%"].tolist()
                    ax.plot(
                        arima_yrs, arima_vals,
                        color=color, linewidth=1.2,
                        linestyle="--", alpha=0.8
                    )

                # Prophet forecast
                if "Prophet_Forecast_%" in fc_c.columns:
                    prop_yrs  = [last_obs_yr] + fc_c["Year"].tolist()
                    prop_vals = [last_obs_val] + fc_c["Prophet_Forecast_%"].tolist()
                    ax.plot(
                        prop_yrs, prop_vals,
                        color=color, linewidth=1.0,
                        linestyle=":", alpha=0.6
                    )

    # Threshold line
    ax.axhline(
        y=MDR_THRESHOLD, color="red", linewidth=1.5,
        linestyle="-.", label=f"MDR={MDR_THRESHOLD}% threshold (WHO GLASS 2023)"
    )

    # Forecast region shading
    ax.axvspan(2024.5, 2030.5, alpha=0.05, color="gray", label="Forecast region")

    ax.set_xlim(2004, 2030)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("MDR Rate (%)", fontsize=11)
    ax.set_title(
        f"{org}\nMDR Rate 2004–2024 (observed) + 2025–2030 forecast\n"
        f"-- ARIMA   :  Prophet   |  Red line = 50% danger threshold",
        fontsize=10
    )
    ax.set_xticks(list(range(2004, 2031, 2)))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    if len(org_series) <= 20:
        ax.legend(
            fontsize=7, loc="upper left",
            ncol=2, framealpha=0.7
        )

    plt.tight_layout()
    safe_name = org.replace(" ", "_").replace(".", "")
    plot_path = os.path.join(PLOT_DIR, f"P5_{safe_name}_forecast.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: P5_{safe_name}_forecast.png")

# ============================================================
# STEP 9 — FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("PHASE 5 COMPLETE — SUMMARY")
print("=" * 65)

print(f"""
  Input isolates (after MDR_v2 rebuild) : {len(atlas):,}
  MDR_v2 rate                           : {atlas['MDR_v2'].mean()*100:.1f}%

  Time series built                     : {len(series_list)}
  Series skipped (< {MIN_VALID_YEARS} valid years)    : {len(skipped_list)}

  ARIMA forecast rows                   : {len(arima_rows)}
  Prophet forecast rows                 : {len(prophet_rows)}
  Threshold alerts (MDR > {MDR_THRESHOLD}%)       : {len(alert_rows)}

  Output files in: ~/AMR_DATA_CHALLENGE/phase5_outputs/
    P5_timeseries_input.csv
    P5_skipped_series.csv
    P5_arima_forecasts.csv
    P5_prophet_forecasts.csv
    P5_combined_forecasts.csv
    P5_threshold_alerts.csv
    P5_first_threshold_crossing.csv
    P5_model_diagnostics.csv
    plots/  (one PNG per organism)

  References:
    Magiorakos et al. 2012. CMI 18(3):268-281
    Box & Jenkins. 1976. Time Series Analysis.
    Hyndman & Athanasopoulos. 2021. FPP3 (3rd ed.)
    Taylor & Letham. 2018. Am Stat. 72(1):37-45
    WHO GLASS. 2023. AMR Surveillance Report.
    ECDC EARS-Net. 2023. AMR Surveillance Europe.
""")

# ============================================================
# AMR DATA CHALLENGE 2026 — PHASE 5: INTERACTIVE FORECAST VIZ
# ============================================================
# OBJECTIVE:
#   Build a single interactive HTML dashboard from Phase 5
#   forecast output CSVs. No external dependencies beyond
#   pandas, numpy, json, os.
#
# INPUT FILES (~/AMR_DATA_CHALLENGE/PHASE5_outputs/):
#   P5_combined_forecasts.csv
#   P5_timeseries_input.csv
#   P5_model_diagnostics.csv
#   P5_first_threshold_crossing.csv
#   P5_threshold_alerts.csv
#
# OUTPUT:
#   ~/AMR_DATA_CHALLENGE/PHASE5_outputs/P5_interactive_forecast.html
#
# LAYOUT :
#   - Title bar (dark blue gradient)
#   - Organism button bar (full names, with country count)
#   - Main row: Plotly chart + control panel
#   - Control panel: view selector, country dropdown,
#                    metric toggles, organism stats, references
#   - Colour bar + threshold note
#   - Alert summary table + organism summary grid
#   - Description box
#
# REFERENCES:
#   Magiorakos et al. 2012. CMI 18(3):268-281
#   Box & Jenkins. 1976. Time Series Analysis.
#   Hyndman & Athanasopoulos. 2021. FPP3 (3rd ed.)
#   Taylor & Letham. 2018. Am Stat. 72(1):37-45
#   WHO GLASS. 2023. AMR Surveillance Report.
#   ECDC EARS-Net. 2023. AMR Surveillance Europe.
# ============================================================

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
BASE_DIR  = os.path.expanduser("~/AMR_DATA_CHALLENGE/phase5_outputs")
COMB_CSV  = os.path.join(BASE_DIR, "P5_combined_forecasts.csv")
TS_CSV    = os.path.join(BASE_DIR, "P5_timeseries_input.csv")
DIAG_CSV  = os.path.join(BASE_DIR, "P5_model_diagnostics.csv")
FIRST_CSV = os.path.join(BASE_DIR, "P5_first_threshold_crossing.csv")
ALERT_CSV = os.path.join(BASE_DIR, "P5_threshold_alerts.csv")
OUT_HTML  = os.path.join(BASE_DIR, "P5_interactive_forecast.html")

# ============================================================
# STEP 1 — LOAD CSVs
# ============================================================
print("=" * 65)
print("STEP 1 — LOADING PHASE 5 OUTPUT FILES")
print("=" * 65)

comb  = pd.read_csv(COMB_CSV)
ts    = pd.read_csv(TS_CSV)
diag  = pd.read_csv(DIAG_CSV)
first = pd.read_csv(FIRST_CSV)

comb["Year"]  = comb["Year"].astype(int)
ts["Year"]    = ts["Year"].astype(int)

print(f"  Combined forecasts : {len(comb):,} rows")
print(f"  Timeseries input   : {len(ts):,} rows")
print(f"  Model diagnostics  : {len(diag):,} rows")
print(f"  First crossings    : {len(first):,} rows")

ORGANISMS = sorted(comb["Organism"].unique())
print(f"\n  Organisms found: {len(ORGANISMS)}")
for org in ORGANISMS:
    n = comb[comb["Organism"] == org]["Country"].nunique()
    print(f"    {org:<40} {n} countries")

# ============================================================
# STEP 2 — BUILD JS DATA STRUCTURE
# ============================================================
print("\n" + "=" * 65)
print("STEP 2 — BUILDING DATA TABLE FOR JS")
print("=" * 65)

# DATA[orgIdx] = {
#   organism, countries,
#   hist    : {country: [{year, rate}]},
#   arima   : {country: [{year, val, lo, hi}]},
#   prophet : {country: [{year, val, lo, hi}]},
#   ensemble: {country: [{year, val}]},
#   agree   : {country: [{year, agree}]},
#   diag    : {country: {rmse_a, rmse_p, order, n_years, last_mdr}},
#   alerts  : {country: {alert_year, already_above, both_models, models}}
# }

def safe_float(val, decimals=2):
    try:
        if pd.isna(val): return None
        return round(float(val), decimals)
    except Exception:
        return None

data_table = []
summary    = []

for org in ORGANISMS:
    ts_org   = ts[(ts["Organism"] == org) & (ts["Sufficient"] == True)].copy()
    fc_org   = comb[comb["Organism"] == org].copy()
    dg_org   = diag[diag["Organism"] == org].copy()
    al_org   = first[first["Organism"] == org].copy()

    countries = sorted(fc_org["Country"].unique())

    hist_d = {}; arima_d = {}; prophet_d = {}
    ens_d  = {}; agree_d = {}; diag_d   = {}; alert_d = {}

    for country in countries:
        # Historical observed
        h = ts_org[ts_org["Country"] == country].sort_values("Year")
        hist_d[country] = [
            {"year": int(r["Year"]), "rate": safe_float(r["MDR_Rate_%"])}
            for _, r in h.iterrows()
            if pd.notna(r["MDR_Rate_%"])
        ]

        # Forecasts
        fc = fc_org[fc_org["Country"] == country].sort_values("Year")
        arima_d[country]   = []
        prophet_d[country] = []
        ens_d[country]     = []
        agree_d[country]   = []

        for _, r in fc.iterrows():
            yr = int(r["Year"])
            arima_d[country].append({
                "year": yr,
                "val": safe_float(r.get("ARIMA_Forecast_%")),
                "lo":  safe_float(r.get("ARIMA_CI_Lower_%")),
                "hi":  safe_float(r.get("ARIMA_CI_Upper_%"))
            })
            prophet_d[country].append({
                "year": yr,
                "val": safe_float(r.get("Prophet_Forecast_%")),
                "lo":  safe_float(r.get("Prophet_CI_Lower_%")),
                "hi":  safe_float(r.get("Prophet_CI_Upper_%"))
            })
            ens_d[country].append({
                "year": yr,
                "val": safe_float(r.get("Ensemble_Mean_%"))
            })
            agree_d[country].append({
                "year": yr,
                "agree": str(r.get("Model_Agreement", "DISAGREE"))
            })

        # Diagnostics
        dg = dg_org[dg_org["Country"] == country]
        if len(dg) > 0:
            r = dg.iloc[0]
            try:
                order = f"({int(r['ARIMA_p'])},{int(r['ARIMA_d'])},{int(r['ARIMA_q'])})"
            except Exception:
                order = "N/A"
            diag_d[country] = {
                "rmse_a":  safe_float(r.get("ARIMA_RMSE")),
                "rmse_p":  safe_float(r.get("Prophet_RMSE")),
                "order":   order,
                "n_years": int(r["N_Valid_Years"]) if pd.notna(r.get("N_Valid_Years")) else None,
                "last_mdr": safe_float(r.get("MDR_Rate_Last_%"))
            }

        # Alerts
        al = al_org[al_org["Country"] == country]
        if len(al) > 0:
            r = al.iloc[0]
            alert_d[country] = {
                "alert_year":    int(r["Alert_Year"]),
                "already_above": bool(r["Already_Above_50"]),
                "both_models":   bool(r["Both_Models_Alert"]),
                "models":        str(r.get("Models_Alerting", ""))
            }

    data_table.append({
        "organism":  org,
        "countries": countries,
        "hist":      hist_d,
        "arima":     arima_d,
        "prophet":   prophet_d,
        "ensemble":  ens_d,
        "agree":     agree_d,
        "diag":      diag_d,
        "alerts":    alert_d
    })

    # Summary
    both_al  = al_org[al_org["Both_Models_Alert"] == True]
    emerging = al_org[(al_org["Already_Above_50"] == False) & (al_org["Both_Models_Alert"] == True)]
    already  = al_org[al_org["Already_Above_50"] == True]
    summary.append({
        "organism":          org,
        "n_countries":       len(countries),
        "n_alerts_both":     len(both_al),
        "n_emerging":        len(emerging),
        "n_already":         len(already),
        "mean_arima_rmse":   safe_float(dg_org["ARIMA_RMSE"].mean()) if len(dg_org) > 0 else None,
        "mean_prophet_rmse": safe_float(dg_org["Prophet_RMSE"].mean()) if len(dg_org) > 0 else None
    })

    print(f"  {org:<40} countries={len(countries):>3}  "
          f"both_alerts={len(both_al):>3}  emerging={len(emerging)}")

# ============================================================
# STEP 3 — SERIALISE TO JS
# ============================================================
data_js    = json.dumps(data_table,  separators=(",", ":"))
orgs_js    = json.dumps(ORGANISMS,   separators=(",", ":"))
summary_js = json.dumps(summary,     separators=(",", ":"))

# ============================================================
# STEP 4 — BUILD ORGANISM BUTTONS (full names + country count)
# ============================================================
org_buttons_html = ""
for i, (org, s) in enumerate(zip(ORGANISMS, summary)):
    active = " active" if i == 0 else ""
    org_buttons_html += (
        f'<button class="org-btn{active}" '
        f'onclick="selectOrganism({i})" '
        f'title="{org}">'
        f'{org} <span class="org-count">({s["n_countries"]})</span>'
        f'</button>\n'
    )

# ============================================================
# STEP 5 — CSS
# ============================================================
CSS = """
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f4f8;min-height:100vh;}
.page-wrap{max-width:1420px;margin:0 auto;padding:16px 16px 28px;
           display:flex;flex-direction:column;gap:0;}

/* TITLE */
.title-bar{background:linear-gradient(135deg,#1a3a5c 0%,#2471a3 100%);
           color:#fff;text-align:center;padding:14px 24px;
           border-radius:10px 10px 0 0;}
.title-bar h1{font-size:1.30rem;font-weight:800;letter-spacing:0.4px;}
.title-bar p{font-size:0.80rem;opacity:0.82;margin-top:3px;}

/* ORGANISM BUTTON BAR */
.org-bar{background:#1a3a5c;padding:8px 10px;display:flex;
         flex-wrap:wrap;gap:5px;border-bottom:2px solid #e6a817;}
.org-btn{background:#2c5282;color:#cbd5e0;border:1px solid #4a6fa5;
         border-radius:5px;padding:5px 10px;font-size:0.74rem;
         font-weight:600;cursor:pointer;transition:all 0.15s;
         white-space:nowrap;display:flex;align-items:center;gap:4px;}
.org-btn:hover{background:#3a6ea8;color:#fff;}
.org-btn.active{background:#e6a817;color:#1a1a00;border-color:#c8920f;}
.org-count{font-size:0.68rem;font-weight:400;opacity:0.75;}
.org-btn.active .org-count{opacity:0.65;}

/* MAIN ROW */
.main-row{display:flex;background:#fff;
          border-left:1px solid #d5dde5;
          border-right:1px solid #d5dde5;min-height:530px;}
.chart-area{flex:1 1 0%;min-width:0;padding:0;position:relative;}
#main-chart{width:100%;height:530px;}

/* CONTROL PANEL */
.ctrl-panel{width:245px;flex-shrink:0;background:#f7f9fc;
            border-left:2px solid #d5dde5;display:flex;
            flex-direction:column;gap:0;overflow-y:auto;max-height:530px;}
.ctrl-section{padding:11px 13px;border-bottom:1px solid #e0e6ed;}
.ctrl-section:last-child{border-bottom:none;}
.section-label{font-size:0.67rem;font-weight:700;color:#5d6d7e;
               text-transform:uppercase;letter-spacing:1px;margin-bottom:7px;}

/* VIEW BUTTONS */
.view-btns{display:flex;flex-direction:column;gap:4px;}
.view-btn{width:100%;padding:7px 10px;border:2px solid #c8d6e5;
          border-radius:6px;background:#fff;font-size:0.79rem;
          font-weight:600;color:#2c3e50;cursor:pointer;
          text-align:left;transition:all 0.15s;}
.view-btn:hover{background:#ebf5fb;border-color:#2980b9;}
.view-btn.active{background:#2471a3;color:#fff;border-color:#1a5276;}

/* METRIC TOGGLES */
.toggle-row{display:flex;flex-direction:column;gap:4px;}
.toggle-item{display:flex;align-items:center;gap:7px;
             font-size:0.77rem;color:#2c3e50;cursor:pointer;padding:2px 0;}
.toggle-item input{width:14px;height:14px;cursor:pointer;accent-color:#2471a3;}
.swatch{width:24px;height:4px;border-radius:2px;flex-shrink:0;}

/* COUNTRY SELECT */
.ctrl-select{width:100%;padding:7px 10px;border:2px solid #2980b9;
             border-radius:6px;background:#d6eaf8;font-size:0.82rem;
             font-weight:600;color:#1a252f;cursor:pointer;outline:none;
             appearance:none;-webkit-appearance:none;}
.country-count-note{font-size:0.70rem;color:#5d6d7e;margin-top:4px;}

/* STATS BOX */
.stats-box{background:#f0f4f8;border-radius:6px;padding:9px;
           font-size:0.73rem;line-height:1.9;color:#2c3e50;}
.stat-row{display:flex;justify-content:space-between;
          border-bottom:1px solid #dce4ec;padding:1px 0;}
.stat-row:last-child{border-bottom:none;}
.stat-val{font-weight:700;color:#2471a3;}
.low-cov-warn{margin-top:6px;background:#fadbd8;border-radius:4px;
              padding:4px 7px;font-size:0.70rem;color:#922b21;font-weight:600;}

/* COLORBAR */
.colorbar-row{background:#fff;border-left:1px solid #d5dde5;
              border-right:1px solid #d5dde5;border-top:1px solid #e8edf2;
              padding:8px 16px;display:flex;align-items:center;gap:14px;}
.cb-label{font-size:0.78rem;color:#333;font-weight:600;white-space:nowrap;}
.cb-gradient{display:block;width:100%;min-width:180px;height:18px;
             border-radius:4px;border:1px solid #ccc;
             background:linear-gradient(to right,
               #ffffcc 0%,#fecc5c 25%,#fd8d3c 50%,#f03b20 75%,#bd0026 100%);}
.cb-ticks{display:flex;justify-content:space-between;width:100%;
          font-size:0.70rem;color:#555;margin-top:2px;}
.cb-wrap{flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:2px;}
.threshold-note{font-size:0.72rem;color:#c0392b;font-weight:700;white-space:nowrap;}

/* ALERT / SUMMARY SECTION */
.section-outer{background:#fff;border-left:1px solid #d5dde5;
               border-right:1px solid #d5dde5;padding:10px 14px 12px;}
.section-outer h3{font-size:0.88rem;font-weight:700;color:#1a3a5c;margin-bottom:8px;}

/* ALERT TABLE */
.alert-table{width:100%;font-size:0.73rem;border-collapse:collapse;}
.alert-table th{background:#2471a3;color:#fff;padding:6px 7px;
                text-align:left;font-weight:600;}
.alert-table td{padding:5px 7px;border-bottom:1px solid #e8edf2;vertical-align:middle;}
.alert-table tr:hover td{background:#ebf5fb;}
.row-already{background:#fdf2f8!important;}
.row-emerging{background:#f0fff4!important;}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;
       font-size:0.68rem;font-weight:700;}
.badge-red{background:#fadbd8;color:#c0392b;}
.badge-green{background:#d5f5e3;color:#1e8449;}
.badge-orange{background:#fdebd0;color:#d35400;}
.badge-grey{background:#eaecee;color:#5d6d7e;}

/* SUMMARY GRID */
.summary-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
              gap:7px;margin-top:8px;}
.summary-card{background:#f7f9fc;border:1px solid #d5dde5;border-radius:6px;
              padding:9px 10px;font-size:0.71rem;cursor:pointer;
              transition:all 0.15s;}
.summary-card:hover{border-color:#2471a3;background:#ebf5fb;}
.summary-card.has-emerging{border-color:#27ae60;background:#f8fff8;}
.summary-card.has-alert{border-color:#e74c3c;background:#fff8f8;}
.sc-org{font-weight:700;color:#1a3a5c;font-size:0.74rem;
        margin-bottom:5px;line-height:1.3;}
.sc-row{display:flex;justify-content:space-between;
        color:#5d6d7e;padding:1px 0;border-bottom:1px solid #eef0f2;}
.sc-row:last-child{border-bottom:none;}
.sc-val{font-weight:700;color:#2471a3;}

/* DESC BOX */
.desc-box{background:#f8f9fa;border-left:7px solid #2980b9;
          border-radius:0 0 10px 10px;padding:14px 20px;
          font-size:0.83rem;line-height:1.75;color:#1a252f;
          box-shadow:0 4px 12px rgba(0,0,0,0.07);}
.desc-box strong{color:#1a3a5c;}
"""

# ============================================================
# STEP 6 — JAVASCRIPT
# ============================================================
JS = """
// ── EMBEDDED DATA ───────────────────────────────────────────
const DATA    = __DATA__;
const ORGS    = __ORGS__;
const SUMMARY = __SUMMARY__;

// Colour palette — 34 distinct colours for countries
const PAL = [
  '#2471a3','#c0392b','#27ae60','#e67e22','#8e44ad','#16a085',
  '#d35400','#1a5276','#922b21','#117864','#b7950b','#1f618d',
  '#6c3483','#0e6655','#784212','#2c3e50','#7b241c','#196f3d',
  '#4a235a','#154360','#5b2333','#0b5345','#6e2f1a','#1b2631',
  '#7e5109','#117a65','#5d6d7e','#85929e','#2e86c1','#a93226',
  '#1e8449','#d68910','#6d3461','#148f77'
];
function col(idx){ return PAL[idx % PAL.length]; }

// ── STATE ────────────────────────────────────────────────────
let _org  = 0;
let _view = 'forecast';
let _cty  = 'ALL';

// ── INIT ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildOrgButtons();
  populateCountrySelect();
  updateStats();
  redraw();
});

// ── ORG BUTTONS ──────────────────────────────────────────────
function buildOrgButtons(){
  // Already rendered server-side; just wire clicks
}

function selectOrganism(idx){
  _org = idx;
  _cty = 'ALL';
  document.querySelectorAll('.org-btn').forEach((b,i)=>{
    b.classList.toggle('active', i===idx);
  });
  populateCountrySelect();
  updateStats();
  if(_view === 'alerts'){
    buildAlertView();
  } else {
    redraw();
  }
}

// ── COUNTRY SELECT ───────────────────────────────────────────
function populateCountrySelect(){
  const sel = document.getElementById('cty-sel');
  const countries = DATA[_org].countries;
  sel.innerHTML = '<option value="ALL">All Countries</option>';
  countries.forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.textContent = c;
    sel.appendChild(o);
  });
  sel.value = _cty;
  const note = document.getElementById('cty-count-note');
  if(note) note.textContent = countries.length + ' countries for this organism';
}

function filterCountry(val){
  _cty = val;
  redraw();
}

// ── VIEW ─────────────────────────────────────────────────────
function setView(v){
  _view = v;
  ['forecast','ensemble'].forEach(id => {
    document.getElementById('btn-'+id).classList.toggle('active', id===v);
  });
  redraw();
}


// ── STATS ────────────────────────────────────────────────────
function updateStats(){
  const s = SUMMARY[_org];
  const org = DATA[_org];
  const anyAlert = Object.keys(org.alerts).length;
  const isLow = s.n_countries < 3;
  let html =
    '<div class="stat-row"><span>Countries modelled</span>'+
    '<span class="stat-val">'+s.n_countries+'</span></div>'+
    '<div class="stat-row"><span>Countries with alerts</span>'+
    '<span class="stat-val">'+anyAlert+'</span></div>'+
    '<div class="stat-row"><span>Both-model alerts</span>'+
    '<span class="stat-val" style="color:'+(s.n_alerts_both>0?'#c0392b':'inherit')+'">'+
    s.n_alerts_both+'</span></div>'+
    '<div class="stat-row"><span>Already &gt;50% (2024)</span>'+
    '<span class="stat-val" style="color:#c0392b;">'+s.n_already+'</span></div>'+
    '<div class="stat-row"><span>Emerging crisis</span>'+
    '<span class="stat-val" style="color:#27ae60;">'+s.n_emerging+'</span></div>'+
    '<div class="stat-row"><span>Mean ARIMA RMSE</span>'+
    '<span class="stat-val">'+(s.mean_arima_rmse!==null ? s.mean_arima_rmse+'%pp':'—')+'</span></div>'+
    '<div class="stat-row"><span>Mean Prophet RMSE</span>'+
    '<span class="stat-val">'+(s.mean_prophet_rmse!==null ? s.mean_prophet_rmse+'%pp':'—')+'</span></div>';
  if(isLow){
    html += '<div class="low-cov-warn">&#9888; LOW COVERAGE: &lt;3 series — results not generalisable across countries</div>';
  }
  document.getElementById('stats-box').innerHTML = html;
}

// ── MAIN CHART ───────────────────────────────────────────────
function redraw(){
  if(_view === 'alerts') return;
  const org      = DATA[_org];
  const ctryList = (_cty === 'ALL') ? org.countries : [_cty];

  const showHist    = document.getElementById('tog-hist').checked;
  const showArima   = document.getElementById('tog-arima').checked;
  const showProphet = document.getElementById('tog-prophet').checked;
  const showEns     = document.getElementById('tog-ensemble').checked;
  const showCI      = document.getElementById('tog-ci').checked;
  const showThresh  = document.getElementById('tog-threshold').checked;

  const traces = [];

  ctryList.forEach((country, ci) => {
    const palIdx = org.countries.indexOf(country);
    const c      = col(palIdx);
    const r      = parseInt(c.slice(1,3),16);
    const g      = parseInt(c.slice(3,5),16);
    const b      = parseInt(c.slice(5,7),16);
    const rgba   = (a) => 'rgba('+r+','+g+','+b+','+a+')';

    const hist    = org.hist[country]    || [];
    const arima   = org.arima[country]   || [];
    const prophet = org.prophet[country] || [];
    const ens     = org.ensemble[country]|| [];

    const histYr  = hist.map(d=>d.year);
    const histVal = hist.map(d=>d.rate);
    const lastYr  = histYr.length ? histYr[histYr.length-1] : null;
    const lastVal = histVal.length ? histVal[histVal.length-1] : null;

    const showLeg = (ci === 0) || (_cty !== 'ALL');

    // Historical
    if(showHist && hist.length){
      traces.push({
        x:histYr, y:histVal, type:'scatter', mode:'lines+markers',
        name:country, legendgroup:country,
        line:{color:c, width:2.2},
        marker:{size:4.5, color:c},
        hovertemplate:'<b>'+country+'</b><br>'+
          'Year: %{x}<br>MDR Rate: %{y:.1f}%<extra></extra>',
        showlegend:showLeg
      });
    }

    // ── FORECAST VIEW ─────────────────────────────────────
    if(_view === 'forecast'){

      // ARIMA line
      if(showArima){
        const aFilt = arima.filter(d=>d.val!==null);
        if(lastYr && aFilt.length){
          traces.push({
            x:[lastYr,...aFilt.map(d=>d.year)],
            y:[lastVal,...aFilt.map(d=>d.val)],
            type:'scatter', mode:'lines',
            name:country+' ARIMA', legendgroup:country,
            line:{color:c, width:2, dash:'dash'},
            hovertemplate:'<b>'+country+' ARIMA</b><br>'+
              'Year: %{x}<br>Forecast: %{y:.1f}%<extra></extra>',
            showlegend:false
          });
        }
        // ARIMA CI band
        if(showCI){
          const aCI = arima.filter(d=>d.lo!==null && d.hi!==null);
          if(aCI.length){
            const yrs = aCI.map(d=>d.year);
            traces.push({
              x:[...yrs,...yrs.slice().reverse()],
              y:[...aCI.map(d=>d.hi),...aCI.map(d=>d.lo).reverse()],
              type:'scatter', mode:'lines', fill:'toself',
              fillcolor:rgba(0.07), line:{color:'transparent'},
              hoverinfo:'skip', showlegend:false, legendgroup:country
            });
          }
        }
      }

      // Prophet line
      if(showProphet){
        const pFilt = prophet.filter(d=>d.val!==null);
        if(lastYr && pFilt.length){
          traces.push({
            x:[lastYr,...pFilt.map(d=>d.year)],
            y:[lastVal,...pFilt.map(d=>d.val)],
            type:'scatter', mode:'lines',
            name:country+' Prophet', legendgroup:country,
            line:{color:c, width:1.8, dash:'dot'},
            hovertemplate:'<b>'+country+' Prophet</b><br>'+
              'Year: %{x}<br>Forecast: %{y:.1f}%<extra></extra>',
            showlegend:false
          });
        }
        // Prophet CI band
        if(showCI){
          const pCI = prophet.filter(d=>d.lo!==null && d.hi!==null);
          if(pCI.length){
            const yrs = pCI.map(d=>d.year);
            traces.push({
              x:[...yrs,...yrs.slice().reverse()],
              y:[...pCI.map(d=>d.hi),...pCI.map(d=>d.lo).reverse()],
              type:'scatter', mode:'lines', fill:'toself',
              fillcolor:rgba(0.05), line:{color:'transparent'},
              hoverinfo:'skip', showlegend:false, legendgroup:country
            });
          }
        }
      }
    }

    // ── ENSEMBLE VIEW ─────────────────────────────────────
    if(_view === 'ensemble'){
      const eFilt = ens.filter(d=>d.val!==null);
      if(showEns && lastYr && eFilt.length){
        traces.push({
          x:[lastYr,...eFilt.map(d=>d.year)],
          y:[lastVal,...eFilt.map(d=>d.val)],
          type:'scatter', mode:'lines+markers',
          name:country+' Ensemble', legendgroup:country,
          line:{color:c, width:2.5},
          marker:{size:6, color:c, symbol:'diamond'},
          hovertemplate:'<b>'+country+' Ensemble</b><br>'+
            'Year: %{x}<br>MDR: %{y:.1f}%<extra></extra>',
          showlegend:showLeg
        });
      }
      // Outer CI envelope (max of ARIMA and Prophet CI)
      if(showCI){
        const aCI = arima.filter(d=>d.lo!==null && d.hi!==null);
        const pCI = prophet.filter(d=>d.lo!==null && d.hi!==null);
        const n   = Math.min(aCI.length, pCI.length);
        if(n > 0){
          const yrs  = aCI.slice(0,n).map(d=>d.year);
          const hiVals = yrs.map((_,i)=>Math.max(aCI[i].hi, pCI[i].hi));
          const loVals = yrs.map((_,i)=>Math.min(aCI[i].lo, pCI[i].lo));
          traces.push({
            x:[...yrs,...yrs.slice().reverse()],
            y:[...hiVals,...loVals.reverse()],
            type:'scatter', mode:'lines', fill:'toself',
            fillcolor:rgba(0.10),
            line:{color:rgba(0.25), width:1},
            hoverinfo:'skip', showlegend:false, legendgroup:country
          });
        }
      }
    }
  });

  // 50% threshold
  if(showThresh){
    traces.push({
      x:[2004,2030], y:[50,50], type:'scatter', mode:'lines',
      name:'50% Threshold (WHO GLASS 2023)',
      line:{color:'#c0392b', width:2.2, dash:'dashdot'},
      hoverinfo:'skip', showlegend:true
    });
  }
  // Forecast zone shading
  traces.push({
    x:[2024.5,2030,2030,2024.5], y:[0,0,105,105],
    type:'scatter', mode:'lines', fill:'toself',
    fillcolor:'rgba(0,0,0,0.025)', line:{color:'transparent'},
    hoverinfo:'skip', showlegend:false
  });

  const nCty = ctryList.length;
  const layout = {
    margin:{t:55,l:60,r:nCty>15?20:140,b:60},
    paper_bgcolor:'#ffffff', plot_bgcolor:'#f8fafc',
    xaxis:{
      title:{text:'Year', font:{size:13, color:'#1a3a5c', family:'Segoe UI, Arial, sans-serif'}},
      range:[2004,2030.5],
      tickvals:[2004,2006,2008,2010,2012,2014,2016,2018,2020,2022,2024,2026,2028,2030],
      tickfont:{size:11, color:'#1a3a5c', family:'Segoe UI, Arial, sans-serif'},
      gridcolor:'#e8edf2', showline:true, linecolor:'#bdc3c7',
      zeroline:false
    },
    yaxis:{
      title:{text:'MDR Rate (%)', font:{size:13, color:'#1a3a5c', family:'Segoe UI, Arial, sans-serif'}},
      range:[0,105],
      tickvals:[0,25,50,75,100],
      tickfont:{size:11, color:'#1a3a5c', family:'Segoe UI, Arial, sans-serif'},
      gridcolor:'#e8edf2', showline:true, linecolor:'#bdc3c7',
      zeroline:false
    },
    title:{
      text: ORGS[_org] +
        (_cty!=='ALL' ? ' — ' + _cty : ' — All ' + DATA[_org].countries.length + ' countries') +
        '<br><span style="font-size:11px;color:#7f8c8d;">' +
        (_view==='forecast' ? 'Solid = observed | Dashed = ARIMA | Dotted = Prophet' :
         _view==='ensemble' ? 'Solid = observed | Diamond = Ensemble mean | Shaded = CI envelope' : '') +
        ' &nbsp;|&nbsp; Red line = 50% WHO GLASS 2023 danger threshold</span>',
      font:{size:13, color:'#1a3a5c'}, x:0.5, xanchor:'center', y:0.98
    },
    showlegend: false,
    annotations:[],
    shapes:[{
      type:'line', x0:2024.5, x1:2024.5, y0:0, y1:105,
      line:{color:'#95a5a6', width:1.5, dash:'dot'}
    }]
  };

  Plotly.react('main-chart', traces, layout,
    {responsive:true, displayModeBar:true,
     modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d'],
     displaylogo:false});
}
"""

JS = (JS
      .replace("__DATA__",    data_js)
      .replace("__ORGS__",    orgs_js)
      .replace("__SUMMARY__", summary_js))

# ============================================================
# STEP 7 — BUILD HTML
# ============================================================
print("\n" + "=" * 65)
print("STEP 7 — BUILDING HTML")
print("=" * 65)

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Phase 5  MDR Forecasting 2025 to 2030 | AMR Data Challenge 2026</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{CSS}
</style>
</head>
<body>
<div class="page-wrap">

<!-- TITLE BAR -->
<div class="title-bar">
  <h1>Phase 5 &mdash; MDR Rate Forecasting 2025&ndash;2030</h1>
  <p>AMR Data Challenge 2026 &nbsp;&middot;&nbsp;
     ARIMA + Prophet Ensemble &nbsp;&middot;&nbsp;
     11 Organisms &nbsp;&middot;&nbsp;
     149 Country Time Series &nbsp;&middot;&nbsp;
     WHO GLASS Danger Threshold: 50%</p>
</div>

<!-- ORGANISM BUTTON BAR -->
<div class="org-bar">
{org_buttons_html}
</div>

<!-- MAIN ROW -->
<div class="main-row">

  <!-- CHART -->
  <div class="chart-area">
    <div id="main-chart"></div>
  </div>

  <!-- CONTROL PANEL -->
  <div class="ctrl-panel">

    <!-- VIEW SELECTOR -->
    <div class="ctrl-section">
      <div class="section-label">View</div>
      <div class="view-btns">
        <button class="view-btn active" id="btn-forecast"
                onclick="setView('forecast')">
          &#128200; Forecast Lines
        </button>
        <button class="view-btn" id="btn-ensemble"
                onclick="setView('ensemble')">
          &#9868; Ensemble + CI Bands
        </button>
      </div>
    </div>

    <!-- COUNTRY FILTER -->
    <div class="ctrl-section">
      <div class="section-label">Country Filter</div>
      <select id="cty-sel" class="ctrl-select"
              onchange="filterCountry(this.value)">
        <option value="ALL">All Countries</option>
      </select>
      <div class="country-count-note" id="cty-count-note"></div>
    </div>

    <!-- METRIC TOGGLES -->
    <div class="ctrl-section" id="metric-section">
      <div class="section-label">Show / Hide</div>
      <div class="toggle-row">
        <label class="toggle-item">
          <input type="checkbox" id="tog-hist" checked onchange="redraw()">
          <span class="swatch" style="background:#2471a3;height:3px;"></span>
          Historical MDR
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="tog-arima" checked onchange="redraw()">
          <span class="swatch"
            style="border-top:3px dashed #e74c3c;height:0;background:none;"></span>
          ARIMA Forecast
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="tog-prophet" checked onchange="redraw()">
          <span class="swatch"
            style="border-top:3px dotted #27ae60;height:0;background:none;"></span>
          Prophet Forecast
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="tog-ensemble" checked onchange="redraw()">
          <span class="swatch" style="background:#8e44ad;height:3px;"></span>
          Ensemble Mean
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="tog-ci" checked onchange="redraw()">
          <span class="swatch" style="background:#aed6f1;height:8px;border-radius:2px;"></span>
          95% CI Bands
        </label>
        <label class="toggle-item">
          <input type="checkbox" id="tog-threshold" checked onchange="redraw()">
          <span class="swatch" style="background:#c0392b;height:3px;"></span>
          50% Threshold
        </label>
      </div>
    </div>

    <!-- ORGANISM STATS -->
    <div class="ctrl-section">
      <div class="section-label">Organism Stats</div>
      <div class="stats-box" id="stats-box"></div>
    </div>

  </div><!-- /ctrl-panel -->
</div><!-- /main-row -->

<!-- COLOUR BAR -->
<div class="colorbar-row">
  <span class="cb-label">MDR Rate (%):</span>
  <div class="cb-wrap">
    <div class="cb-gradient"></div>
    <div class="cb-ticks">
      <span>0%</span><span>25%</span>
      <span>50%</span><span>75%</span><span>100%</span>
    </div>
  </div>
  <span class="threshold-note">
    &#9888; 50% = WHO GLASS 2023 / ECDC EARS-Net 2023 danger threshold
  </span>
</div>


<!-- DESCRIPTION BOX -->
<div class="desc-box">
  <strong>About this dashboard.</strong>
  This interactive viewer presents MDR rate forecasts (2025–2030)
  for 11 nosocomial neuroinvasive bacterial pathogens across
  149 organism-country time series built from Pfizer ATLAS data (2004–2024).
  MDR is defined per Magiorakos <em>et al.</em> (CMI 2012) as
  non-susceptible (R or I) to &#8805;3 antibiotic classes.
  Forecasting uses ARIMA (order selected by AIC minimisation
  with ADF stationarity test; Box &amp; Jenkins 1976;
  Hyndman &amp; Athanasopoulos 2021) and Prophet
  (piecewise linear trend, no seasonality; Taylor &amp; Letham 2018).
  Only series with &#8805;8 valid yearly data points
  (&#8805;10 isolates per cell, WHO GLASS 2023) are included.
  The <strong>50% MDR threshold</strong> is the WHO GLASS 2023 /
  ECDC EARS-Net 2023 high-burden alert level.
  <br><br>
  <strong>Views:</strong>
  <em>Forecast Lines</em> — historical (solid) + ARIMA (dashed) +
  Prophet (dotted) per country with optional 95% CI shading.
  <em>Ensemble + CI Bands</em> — ARIMA+Prophet ensemble mean
  (diamond markers) with outer uncertainty envelope.
  <em>Alert Summary</em> — organism overview grid +
  per-organism threshold alert table distinguishing
  <em>already in crisis</em> (&#62;50% MDR in 2024) from
  <em>emerging crisis</em> (projected to cross 50% by 2030).
  <br><br>
  <strong>Use organism buttons</strong> (full names, with country count in brackets)
  to switch pathogens.
  <strong>Country Filter</strong> isolates a single country trajectory.
  <strong>Show/Hide toggles</strong> control which model lines are displayed.
  <br><br>
  <strong>&#9888; Excluded organisms:</strong>
  <em>Citrobacter koseri</em> (474 isolates spread across 48 countries —
  no country reached &#8805;8 valid yearly data points) and
  <em>Salmonella</em> spp (only 8 total isolates in the filtered dataset)
  had insufficient temporal data for ARIMA/Prophet modelling.
  <strong>&#9888; Low coverage:</strong>
  <em>Haemophilus influenzae</em> has only 1 qualifying series —
  results are not generalisable across countries.
  <br><br>
  <strong>Data source:</strong> Pfizer ATLAS Surveillance Programme 2004–2024.
  <strong>MDR definition:</strong> Magiorakos <em>et al.</em> CMI 2012.
  <br><br>
  <strong>References:</strong><br>
  <strong>MDR definition:</strong> Magiorakos et al. 2012. CMI 18(3):268&#8211;281 &nbsp;&middot;&nbsp;
  <strong>ARIMA:</strong> Box &amp; Jenkins. 1976. Time Series Analysis. Holden-Day &nbsp;&middot;&nbsp;
  <strong>Forecast minimum:</strong> Hyndman &amp; Athanasopoulos. 2021. Forecasting: Principles and Practice (3rd ed.) &nbsp;&middot;&nbsp;
  <strong>Prophet:</strong> Taylor &amp; Letham. 2018. The American Statistician. 72(1):37&#8211;45 &nbsp;&middot;&nbsp;
  <strong>Threshold &amp; isolate minimum:</strong> WHO GLASS. 2023. Global AMR Surveillance Report &nbsp;&middot;&nbsp;
  <strong>Threshold:</strong> ECDC EARS-Net. 2023. AMR Surveillance in Europe.
</div>

</div><!-- /page-wrap -->

<!-- COPYRIGHT -->
<div style="text-align:center;padding:10px 0 6px;font-size:0.74rem;
     color:#7f8c8d;background:#f0f4f8;margin-top:0;
     border-top:1px solid #d5dde5;">
  &copy; 2026 PSG Institute of Medical Sciences &amp; Research
</div>

<script>
{JS}
</script>
</body>
</html>"""

# ============================================================
# STEP 8 — WRITE HTML
# ============================================================
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\n  Saved: {OUT_HTML}")
print(f"  File size: {os.path.getsize(OUT_HTML)/1024:.1f} KB")

print("\n" + "=" * 65)
print("PHASE 5 INTERACTIVE VISUALISATION — COMPLETE")
print("=" * 65)
print(f"""
  Output : {OUT_HTML}
  Open in any browser — no internet required for data.
  Plotly loaded from CDN (requires internet on first open).

  Dashboard features:
    - Full organism names on buttons with country count
    - 3 views: Forecast Lines / Ensemble+CI / Alert Summary
    - Country filter dropdown per organism
    - Show/Hide toggles for all model components
    - Organism stats panel
    - Alert table: existing crisis vs emerging crisis split
    - All-organism summary grid with click-to-navigate
""")

# ============================================================
# STEP 9 — BUILD ALERT SUMMARY HTML (standalone file)
# Same data, same styling, no Plotly required
# ============================================================
print("\\n" + "=" * 65)
print("STEP 9 — BUILDING ALERT SUMMARY HTML")
print("=" * 65)

# Organism buttons for alert page
_alert_org_btns = ""
for _i, (_org_name, _s) in enumerate(zip(ORGANISMS, summary)):
    _active = " active" if _i == 0 else ""
    _alert_org_btns += (
   	 f'<button class="org-btn{_active}" '
   	 f'onclick="switchOrg({_i})" '
   	 f'title="{_org_name}">'
   	 f'{_org_name} <span class="org-count">({_s["n_countries"]})</span>'
   	 f'</button>\n'
    )
    
    
_ALERT_CSS = (
    "*{box-sizing:border-box;margin:0;padding:0;}"
    "body{font-family:\'Segoe UI\',Arial,sans-serif;background:#f0f4f8;min-height:100vh;}"
    ".page-wrap{max-width:1420px;margin:0 auto;padding:16px 16px 28px;"
    "display:flex;flex-direction:column;gap:0;}"
    ".title-bar{background:linear-gradient(135deg,#1a3a5c 0%,#2471a3 100%);"
    "color:#fff;text-align:center;padding:14px 24px;border-radius:10px 10px 0 0;}"
    ".title-bar h1{font-size:1.30rem;font-weight:800;letter-spacing:0.4px;}"
    ".title-bar p{font-size:0.80rem;opacity:0.82;margin-top:3px;}"
    ".org-bar{background:#1a3a5c;padding:8px 10px;display:flex;"
    "flex-wrap:wrap;gap:5px;border-bottom:2px solid #e6a817;}"
    ".org-btn{background:#2c5282;color:#cbd5e0;border:1px solid #4a6fa5;"
    "border-radius:5px;padding:5px 10px;font-size:0.74rem;"
    "font-weight:600;cursor:pointer;transition:all 0.15s;"
    "white-space:nowrap;display:flex;align-items:center;gap:4px;}"
    ".org-btn:hover{background:#3a6ea8;color:#fff;}"
    ".org-btn.active{background:#e6a817;color:#1a1a00;border-color:#c8920f;}"
    ".org-count{font-size:0.68rem;font-weight:400;opacity:0.75;}"
    ".org-btn.active .org-count{opacity:0.65;}"
    ".content-area{background:#fff;border-left:1px solid #d5dde5;"
    "border-right:1px solid #d5dde5;padding:18px 20px 22px;}"
    ".section-title{font-size:1.0rem;font-weight:700;color:#1a3a5c;"
    "margin-bottom:10px;padding-bottom:6px;border-bottom:2px solid #e8edf2;}"
    ".summary-grid{display:grid;"
    "grid-template-columns:repeat(auto-fill,minmax(185px,1fr));"
    "gap:7px;margin-bottom:22px;}"
    ".summary-card{background:#f7f9fc;border:1px solid #d5dde5;border-radius:6px;"
    "padding:9px 10px;font-size:0.71rem;cursor:pointer;transition:all 0.15s;}"
    ".summary-card:hover{border-color:#2471a3;background:#ebf5fb;"
    "transform:translateY(-1px);}"
    ".summary-card.has-emerging{border-color:#27ae60;background:#f8fff8;}"
    ".summary-card.has-alert{border-color:#e74c3c;background:#fff8f8;}"
    ".summary-card.active-card{outline:3px solid #e6a817;outline-offset:1px;}"
    ".sc-org{font-weight:700;color:#1a3a5c;font-size:0.74rem;"
    "margin-bottom:5px;line-height:1.3;}"
    ".sc-row{display:flex;justify-content:space-between;"
    "color:#5d6d7e;padding:1px 0;border-bottom:1px solid #eef0f2;}"
    ".sc-row:last-child{border-bottom:none;}"
    ".sc-val{font-weight:700;color:#2471a3;}"
    ".alert-section{margin-top:4px;}"
    ".alert-header{display:flex;align-items:baseline;gap:10px;margin-bottom:6px;}"
    ".alert-header h3{font-size:0.92rem;font-weight:700;color:#1a3a5c;}"
    ".org-name{color:#c0392b;font-weight:700;}"
    ".legend-row{font-size:0.73rem;color:#5d6d7e;margin-bottom:8px;"
    "display:flex;align-items:center;gap:14px;flex-wrap:wrap;}"
    ".legend-swatch{display:inline-block;width:13px;height:13px;"
    "border-radius:2px;border:1px solid #d5dde5;"
    "vertical-align:middle;margin-right:4px;}"
    "#alert-table-wrap{max-height:420px;overflow-y:auto;border:1px solid #d5dde5;border-radius:4px;}""#alert-table-wrap thead th{position:sticky;top:0;z-index:2;}"".alert-table{width:100%;font-size:0.74rem;border-collapse:collapse;}"
    ".alert-table th{background:#2471a3;color:#fff;padding:7px 8px;"
    "text-align:left;font-weight:600;}"
    ".alert-table td{padding:6px 8px;border-bottom:1px solid #e8edf2;vertical-align:middle;}"
    ".alert-table tr:hover td{background:#ebf5fb;}"
    ".row-already{background:#fdf2f8!important;}"
    ".row-emerging{background:#f0fff4!important;}"
    ".badge{display:inline-block;padding:2px 8px;border-radius:10px;"
    "font-size:0.68rem;font-weight:700;}"
    ".badge-red{background:#fadbd8;color:#c0392b;}"
    ".badge-green{background:#d5f5e3;color:#1e8449;}"
    ".badge-orange{background:#fdebd0;color:#d35400;}"
    ".badge-grey{background:#eaecee;color:#5d6d7e;}"
    ".row-no-alert{background:#f8f9fa!important;color:#7f8c8d;}"".no-alert-box{text-align:center;color:#7f8c8d;padding:28px;"
    "background:#f8f9fa;border-radius:6px;font-size:0.83rem;"
    "border:1px dashed #d5dde5;}"
    ".desc-box{background:#f8f9fa;border-left:7px solid #2980b9;"
    "border-radius:0 0 10px 10px;padding:14px 20px;"
    "font-size:0.83rem;line-height:1.75;color:#1a252f;"
    "box-shadow:0 4px 12px rgba(0,0,0,0.07);}"
    ".desc-box strong{color:#1a3a5c;}"
    ".link-btn{display:inline-block;margin-top:10px;padding:7px 18px;"
    "background:#2471a3;color:#fff;border-radius:6px;"
    "text-decoration:none;font-size:0.80rem;font-weight:600;}"
    ".link-btn:hover{background:#1a5276;}"
)

_ALERT_JS = """
const DATA    = """ + data_js + """;
const ORGS    = """ + orgs_js + """;
const SUMMARY = """ + summary_js + """;

let _org = 0;

document.addEventListener('DOMContentLoaded', () => { renderAll(0); });

function switchOrg(idx){
  _org = idx;
  document.querySelectorAll('.org-btn').forEach((b,i)=>{
    b.classList.toggle('active', i===idx);
  });
  renderAll(idx);
}

function renderAll(idx){
  renderSummaryGrid(idx);
  renderAlertTable(idx);
}

function renderSummaryGrid(activeIdx){
  const grid = document.getElementById('summary-grid');
  grid.innerHTML = '';
  SUMMARY.forEach((s,i) => {
    const hasEm = s.n_emerging > 0;
    const hasAl = s.n_alerts_both > 0;
    let cls = 'summary-card';
    if(hasEm) cls += ' has-emerging';
    else if(hasAl) cls += ' has-alert';
    if(i === activeIdx) cls += ' active-card';
    const card = document.createElement('div');
    card.className = cls;
    card.style.cursor = 'pointer';
    card.title = 'Click to view ' + s.organism;
    card.onclick = () => switchOrg(i);
    card.innerHTML =
      '<div class="sc-org">'+s.organism+'</div>'+
      '<div class="sc-row"><span>Countries</span><span class="sc-val">'+s.n_countries+'</span></div>'+
      '<div class="sc-row"><span>Both-model alerts</span>'+
        '<span class="sc-val" style="color:'+(hasAl?'#c0392b':'#5d6d7e')+'">'+s.n_alerts_both+'</span></div>'+
      '<div class="sc-row"><span>Emerging crisis</span>'+
        '<span class="sc-val" style="color:'+(hasEm?'#27ae60':'#5d6d7e')+'">'+s.n_emerging+'</span></div>'+
      '<div class="sc-row"><span>Already &gt;50%</span>'+
        '<span class="sc-val" style="color:'+(s.n_already>0?'#c0392b':'#5d6d7e')+'">'+s.n_already+'</span></div>'+
      '<div class="sc-row"><span>ARIMA RMSE</span>'+
        '<span class="sc-val">'+(s.mean_arima_rmse!==null?s.mean_arima_rmse+'%pp':'—')+'</span></div>'+
      '<div class="sc-row"><span>Prophet RMSE</span>'+
        '<span class="sc-val">'+(s.mean_prophet_rmse!==null?s.mean_prophet_rmse+'%pp':'—')+'</span></div>';
    grid.appendChild(card);
  });
}

function renderAlertTable(idx){
  const org = DATA[idx];
  const alerts = org.alerts;
  document.getElementById('alert-org-name').textContent = ORGS[idx];
  const wrap = document.getElementById('alert-table-wrap');
  const tbody = document.getElementById('alert-tbody');
  tbody.innerHTML = '';
  wrap.style.display = '';
  document.getElementById('no-alert-msg').style.display = 'none';

  // Show ALL modelled countries — alerting ones first, then non-alerting
  const alertCountries = Object.keys(alerts).sort((a,b)=>
    alerts[a].alert_year - alerts[b].alert_year);
  const allCountries = org.countries;
  const nonAlertCountries = allCountries.filter(c => !alerts[c]);
  const orderedCountries = [...alertCountries, ...nonAlertCountries.sort()];

  orderedCountries.forEach(country => {
    const a = alerts[country] || null;
    const d = org.diag[country] || {};
    const tr = document.createElement('tr');

    if(a){
      // Country has an alert
      const alertYr = a.alert_year;
      const aF = (org.arima[country]||[]).find(x=>x.year===alertYr);
      const pF = (org.prophet[country]||[]).find(x=>x.year===alertYr);
      const eF = (org.ensemble[country]||[]).find(x=>x.year===alertYr);
      tr.className = a.already_above ? 'row-already' : 'row-emerging';
      const statusBadge = a.already_above
        ? '<span class="badge badge-red">Already &gt;50%</span>'
        : '<span class="badge badge-green">Emerging</span>';
      const confBadge = a.both_models
        ? '<span class="badge badge-orange">Both models &#10003;</span>'
        : '<span class="badge badge-grey">Single model</span>';
      tr.innerHTML =
        '<td><strong>'+country+'</strong></td>'+
        '<td style="text-align:center;">'+alertYr+'</td>'+
        '<td style="text-align:center;">'+(d.last_mdr!==undefined?d.last_mdr+'%':'—')+'</td>'+
        '<td style="text-align:center;">'+(aF&&aF.val!==null?aF.val.toFixed(1)+'%':'—')+'</td>'+
        '<td style="text-align:center;">'+(pF&&pF.val!==null?pF.val.toFixed(1)+'%':'—')+'</td>'+
        '<td style="text-align:center;">'+(eF&&eF.val!==null?eF.val.toFixed(1)+'%':'—')+'</td>'+
        '<td>'+statusBadge+'</td>'+
        '<td>'+confBadge+'</td>';
    } else {
      // Country modelled but no threshold alert — below 50% through 2030
      const lastEns = (org.ensemble[country]||[]).slice(-1)[0];
      tr.className = 'row-no-alert';
      tr.innerHTML =
        '<td><strong>'+country+'</strong></td>'+
        '<td style="text-align:center;color:#7f8c8d;">—</td>'+
        '<td style="text-align:center;">'+(d.last_mdr!==undefined?d.last_mdr+'%':'—')+'</td>'+
        '<td style="text-align:center;color:#7f8c8d;">—</td>'+
        '<td style="text-align:center;color:#7f8c8d;">—</td>'+
        '<td style="text-align:center;">'+(lastEns&&lastEns.val!==null?lastEns.val.toFixed(1)+'% (2030)':'—')+'</td>'+
        '<td><span class="badge badge-grey">Below 50%</span></td>'+
        '<td><span class="badge badge-grey">No alert</span></td>';
    }
    tbody.appendChild(tr);
  });
}
"""

_ALERT_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Phase 5 — Alert Summary | AMR Data Challenge 2026</title>

<style>
{_ALERT_CSS}
</style>

</head>
<body>

<div class="page-wrap">

<div class="title-bar">
  <h1>Phase 5 &mdash; MDR Threshold Alert Summary</h1>
  <p>
    AMR Data Challenge 2026 &nbsp;&middot;&nbsp;
    Countries projected to exceed 50% MDR rate by 2030
    &nbsp;&middot;&nbsp;
    WHO GLASS 2023 Danger Threshold: 50%
  </p>
</div>

<div class="org-bar">
{_alert_org_btns}
</div>

<div class="content-area">

  <div class="section-title">
    All Organisms &mdash; Alert Overview
    <span style="font-size:0.75rem;font-weight:400;color:#7f8c8d;">
      (click any card to switch organism)
    </span>
  </div>

  <div class="summary-grid" id="summary-grid"></div>

  <div class="alert-section">

    <div class="alert-header">
      <h3>
        Threshold Alerts &mdash;
        <span class="org-name" id="alert-org-name"></span>
      </h3>
    </div>

    <div class="legend-row">
      <span>
        <span class="legend-swatch" style="background:#fdf2f8;"></span>
        Pink = already &gt;50% MDR in 2024
      </span>

      <span>
        <span class="legend-swatch" style="background:#f0fff4;"></span>
        Green = emerging (projected to cross 50% in 2025–2030)
      </span>
    </div>

    <div id="no-alert-msg" style="display:none;"></div>

    <div id="alert-table-wrap">
      <table class="alert-table">
        <thead>
          <tr>
            <th>Country</th>
            <th style="text-align:center;">Alert Year</th>
            <th style="text-align:center;">Baseline MDR%</th>
            <th style="text-align:center;">ARIMA%</th>
            <th style="text-align:center;">Prophet%</th>
            <th style="text-align:center;">Ensemble%</th>
            <th>Status</th>
            <th>Confidence</th>
          </tr>
        </thead>

        <tbody id="alert-tbody"></tbody>

      </table>
    </div>

  </div>

</div>

<div class="desc-box">
  <strong>About this page.</strong>
  Standalone alert summary for all organism-country pairs projected
  to exceed the 50% MDR threshold between 2025–2030.
  Countries with no threshold alert are listed below alerting countries
  with their last observed MDR% and 2030 ensemble forecast for reference.
</div>

<div style="text-align:center;padding:10px 0 6px;font-size:0.74rem;
     color:#7f8c8d;background:#f0f4f8;margin-top:0;
     border-top:1px solid #d5dde5;">
  &copy; 2026 PSG Institute of Medical Sciences &amp; Research
</div>

</div>

<script>
{_ALERT_JS}
</script>

</body>
</html>
"""
OUT_ALERT_HTML = "P5_alert_summary.html"

with open(OUT_ALERT_HTML, "w", encoding="utf-8") as f:
    f.write(_ALERT_HTML)
print(f"\\n  Saved: {OUT_ALERT_HTML}")
print(f"  File size: {os.path.getsize(OUT_ALERT_HTML)/1024:.1f} KB")
print("\\n" + "=" * 65)
print("BOTH HTML FILES WRITTEN SUCCESSFULLY")
print("=" * 65)
print("  1. P5_interactive_forecast.html  — Forecast plots (unchanged)")
print("  2. P5_alert_summary.html         — Alert overview + threshold table")
