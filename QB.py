# %% [markdown]
# # QB Fantasy Projection Model
# Per-game prediction of 6 QB outcomes:
# passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds, fumbles_lost_total
#
# Pipeline:
#   Phase 2  — Exploratory Data Analysis
#   Phase 3  — Feature prep & pruning
#   Phase 4  — Baselines
#   Phase 5  — LightGBM (per-target, per-target Optuna search)
#   Phase 6  — Evaluation & diagnostics
#   Phase 7  — Feature importance / SHAP
#   Phase 8  — XGBoost challenger
#   Phase 9  — Ensemble / stacking

# %%
import io
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe in both script and notebook
import matplotlib.pyplot as plt

def _show(fig_path):
    """Display a saved figure inline when running in Jupyter; silent no-op in script mode."""
    try:
        from IPython.display import display, Image as _IPImage
        display(_IPImage(str(fig_path)))
    except Exception:
        pass
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    buf = getattr(sys.stdout, "buffer", None)
    if buf is not None:
        sys.stdout = io.TextIOWrapper(buf, encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    DATA_DIR = Path(__file__).resolve().parent
except NameError:
    DATA_DIR = Path.cwd()

MASTER_PATH = DATA_DIR / "data" / "master" / "nfl_master_dataset.parquet"
FIG_DIR     = DATA_DIR / "figures" / "qb_eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "passing_yards",
    "passing_tds",
    "interceptions",
    "rushing_yards",
    "rushing_tds",
    "fumbles_lost_total",
]

TRAIN_START = 2006
TRAIN_END   = 2023   # train through 2023; 2024 = val only; 2025 = test
VAL_START   = 2024
VAL_END     = 2024
TEST_YEAR   = 2025

# %%
# --- Load and scope to QBs ---
print("Loading master dataset...")
master = pd.read_parquet(MASTER_PATH)
print(f"  Master shape: {master.shape}")

qb = master[
    (master["position"] == "QB") &
    (master["season"] >= TRAIN_START)
].copy()

print(f"  QB rows (2006-2025): {len(qb):,}")
print(f"  Seasons: {sorted(qb['season'].unique())}")
print(f"  Unique players: {qb['player_id'].nunique():,}")

# %% [markdown]
# ## Phase 2 — Exploratory Data Analysis

# %%
# =============================================================================
# 2.1  TARGET DISTRIBUTIONS
# =============================================================================

print("\n" + "="*60)
print("2.1  TARGET DISTRIBUTIONS")
print("="*60)

# Split sets for distribution analysis
train = qb[qb["season"] <= TRAIN_END].copy()
val   = qb[qb["season"].between(VAL_START, VAL_END)].copy()   # 2024
test  = qb[qb["season"] == TEST_YEAR].copy()                  # 2025

print(f"\nSet sizes:  train={len(train):,}  val={len(val):,}  test={len(test):,}")
print()

# --- Summary statistics per target ---
stat_rows = []
for t in TARGETS:
    s = train[t].dropna()
    stat_rows.append({
        "target":   t,
        "n":        len(s),
        "mean":     s.mean(),
        "median":   s.median(),
        "std":      s.std(),
        "min":      s.min(),
        "max":      s.max(),
        "skewness": s.skew(),
        "kurtosis": s.kurtosis(),
        "pct_zero": (s == 0).mean() * 100,
        "pct_null": train[t].isna().mean() * 100,
    })

stats_df = pd.DataFrame(stat_rows).set_index("target")
print("Target summary statistics (train set 2006-2022):")
print(stats_df.to_string(float_format="{:.3f}".format))

# %%
# --- Distribution plots: histogram + KDE + Q-Q per target ---
fig, axes = plt.subplots(len(TARGETS), 3, figsize=(18, 4 * len(TARGETS)))
fig.suptitle("QB Target Distributions (Train 2006-2022)", fontsize=14, fontweight="bold")

for i, t in enumerate(TARGETS):
    s = train[t].dropna()

    # Histogram + KDE
    ax1 = axes[i, 0]
    ax1.hist(s, bins=50, color="steelblue", alpha=0.7, edgecolor="none", density=True)
    s.plot.kde(ax=ax1, color="darkblue", linewidth=2)
    ax1.set_title(f"{t}\nmean={s.mean():.2f}  skew={s.skew():.2f}  zeros={((s==0).mean()*100):.0f}%")
    ax1.set_xlabel(t)
    ax1.set_ylabel("Density")

    # Log-scale histogram (reveals tail behaviour)
    ax2 = axes[i, 1]
    nonzero = s[s > 0]
    if len(nonzero) > 0:
        ax2.hist(np.log1p(nonzero), bins=50, color="coral", alpha=0.7, edgecolor="none", density=True)
        ax2.set_title(f"{t} — log1p(nonzero)")
        ax2.set_xlabel("log1p(value)")
        ax2.set_ylabel("Density")

    # Q-Q plot vs normal
    ax3 = axes[i, 2]
    stats.probplot(s, dist="norm", plot=ax3)
    ax3.set_title(f"{t} — Q-Q vs Normal")

plt.tight_layout()
out = FIG_DIR / "target_distributions.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# %%
# --- Zero-inflation detail ---
print("\nZero-inflation detail (train):")
for t in TARGETS:
    s = train[t].dropna()
    n_zero  = (s == 0).sum()
    n_one   = (s == 1).sum()
    n_two   = (s == 2).sum()
    n_three = (s >= 3).sum()
    print(f"  {t:<25}  0:{n_zero:5d} ({n_zero/len(s)*100:4.1f}%)  "
          f"1:{n_one:5d}  2:{n_two:4d}  3+:{n_three:4d}")

# %%
# --- Distribution stability across eras ---
print("\nTarget means by era (train only):")
era_bins = [(2006,2012,"2006-12 no_snaps"), (2013,2015,"2013-15 no_ngs"), (2016,2022,"2016-22 full")]
header = f"  {'Target':<25}" + "".join(f"  {label:>20}" for _,_,label in era_bins)
print(header)
for t in TARGETS:
    row = f"  {t:<25}"
    for y1, y2, label in era_bins:
        sub = train[(train["season"] >= y1) & (train["season"] <= y2)][t].dropna()
        row += f"  {sub.mean():>8.3f} (n={len(sub):5d})"
    print(row)

# %%
# --- Season trend: has the game changed? ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle("QB Target Means by Season (train set)", fontsize=13, fontweight="bold")

for i, t in enumerate(TARGETS):
    by_yr = (
        train.groupby("season")[t]
        .agg(mean="mean", sem=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
    )
    ax = axes[i]
    ax.plot(by_yr["season"], by_yr["mean"], marker="o", color="steelblue")
    ax.fill_between(
        by_yr["season"],
        by_yr["mean"] - by_yr["sem"],
        by_yr["mean"] + by_yr["sem"],
        alpha=0.2, color="steelblue"
    )
    ax.axvline(2013, color="orange", linestyle="--", linewidth=1, label="snaps avail")
    ax.axvline(2016, color="green",  linestyle="--", linewidth=1, label="NGS avail")
    ax.set_title(t)
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean per game")
    ax.legend(fontsize=7)

plt.tight_layout()
out = FIG_DIR / "target_trend_by_season.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# %%
# =============================================================================
# 2.2  FEATURE DISTRIBUTIONS  (QB-relevant features only)
# =============================================================================

print("\n" + "="*60)
print("2.2  FEATURE NULL RATES (QB rows, train set)")
print("="*60)

# Define QB candidate features for passing_yards model
#
# Base = my full recommendation from feature audit.
# User overrides applied on top:
#   passing_yards      : all 4 windows (override: was L5+L20 only)
#   yards_per_attempt  : L5/L10/L20   (override: was L5/L20)
#   epa_per_dropback   : L5/L10/L20   (override: was L5/L20)
#   air_yards          : L5/L10/L20   (override: was L5/L20)
#   qb_pressure_rate   : L3/L10/L20   (override: was L5 only)
#   epa_per_opportunity: L5/L10/L20   (override: was L20 only)
#   NGS stats          : all 4 windows (override: was single windows each)
#   off_rb_receiving   : L5/L10/L20   (override: was drop all)
#   game_precip_mm     : keep          (override: was drop)
#   rest_days_opponent : keep          (override: was drop)
QB_FEATURES = [
    # --- Identity / context ---
    "depth_chart_rank",
    "games_played_current_season",
    "week",
    "season",

    # --- Fantasy points history (elite QB prior) ---
    "fantasy_pts_ewm10",
    "fantasy_pts_ewm20",
    "fantasy_pts_per_game_career",
    "passing_yards_per_game_career",
    "passing_yards_career_vs_recent",

    # --- Rolling passing volume ---
    # removed: passing_yards_L3 (too noisy)
    *[f"passing_yards_L{w}"   for w in [5, 10, 20]],
    *[f"passing_yards_ewm{w}" for w in [5, 10, 20]],

    # --- QB rate stats ---
    # removed: completion_pct_ewm20 (low SHAP; completion rate ignores yds/attempt, redundant with EPA)
    # removed: qb_cpoe_L5, qb_cpoe_ewm5/20 (low SHAP; redundant with epa_per_dropback/epa_per_opportunity)
    # removed: qb_pressure_rate_ewm5/10/20 (low SHAP; predicts sacks/turnovers not yards)
    # removed: qb_air_yards_per_attempt_ewm5 (low SHAP; ewm10/20 sufficient)
    # removed: epa_per_dropback_ewm5 (low SHAP; ewm10/20 sufficient)
    *[f"yards_per_attempt_ewm{w}"        for w in [5, 10, 20]],
    *[f"epa_per_dropback_ewm{w}"         for w in [10, 20]],
    *[f"qb_air_yards_per_attempt_ewm{w}" for w in [10, 20]],
    *[f"epa_per_opportunity_ewm{w}"      for w in [5, 10, 20]],

    # --- Rushing ---
    "rushing_yards_L20",
    "rushing_yards_ewm20",

    # --- NGS ---
    # removed: ngs_avg_time_to_throw_L5/L10 (low SHAP; L20 captures stable style fingerprint)
    # removed: ngs_avg_intended_air_yards_L5 (low SHAP; L10/L20 sufficient)
    # removed: ngs_aggressiveness_L5/L10/L20 (low SHAP across all windows; redundant with air yards)
    # removed: ngs_completion_pct_above_exp_L5/L10/L20 (low SHAP; redundant with EPA features)
    "ngs_avg_time_to_throw_L20",
    *[f"ngs_avg_intended_air_yards_L{w}" for w in [10, 20]],

    # --- Own offense quality ---
    "off_epa_per_play_L5", "off_epa_per_play_L20",
    "off_pass_rate_L5",
    *[f"off_rb_receiving_yards_L{w}" for w in [5, 10, 20]],
    "off_wr_wopr_L5",
    "off_wr_adot_L5",
    "off_te_epa_per_target_L5",
    "off_te_yprr_L5",
    "off_te_route_run_rate_L5",

    # --- Opponent defense quality ---
    "opp_def_qb_qb_epa_per_attempt_L5", "opp_def_qb_qb_epa_per_attempt_L20",
    "opp_def_qb_qb_cpoe_L5",
    "opp_def_team_team_epa_per_pass_L5",

    # --- Game environment ---
    # removed: age (collinear with season + games_played), rest_days_opponent (lowest SHAP)
    "game_location",
    "is_dome",
    "game_temp",
    "game_wind",
    "game_precip_mm",
    "rest_days",
]

# NOTE: do NOT filter QB_FEATURES against qb.columns here.
# EWM features (passing_yards_ewm*, epa_per_dropback_ewm*, etc.) are not yet
# computed at this point — they're added to qb in Phase 3.1b-EWMA.
# Filtering here would silently drop all EWM features before they exist.
# The authoritative missing-column check happens in Phase 3.1d, after EWM computation.
print(f"\nCandidate features (QB_FEATURES): {len(QB_FEATURES)}")

# Null rates — only check features that already exist in qb.columns (non-EWM)
print(f"\n{'Feature':<50} {'all%':>6} {'2006-12%':>9} {'2013-15%':>9} {'2016-22%':>9}")
print("-" * 88)
era1 = train[train["season"] <= 2012]
era2 = train[(train["season"] >= 2013) & (train["season"] <= 2015)]
era3 = train[train["season"] >= 2016]

problem_features = []
for f in QB_FEATURES:
    if f not in qb.columns:
        continue   # EWM cols not yet computed — skip, will be checked in Phase 3.1d
    n_all  = train[f].isna().mean() * 100
    n_e1   = era1[f].isna().mean()  * 100
    n_e2   = era2[f].isna().mean()  * 100
    n_e3   = era3[f].isna().mean()  * 100
    flag   = " <-- HIGH" if n_e3 > 70 else ""
    if n_e3 > 70:
        problem_features.append(f)
    if n_all > 5 or flag:
        print(f"  {f:<50} {n_all:>5.1f}% {n_e1:>8.1f}% {n_e2:>8.1f}% {n_e3:>8.1f}%{flag}")

if problem_features:
    print(f"\nFeatures >70% null in 2016-2022 era (candidates for removal): {problem_features}")
else:
    print("\nNo features >70% null in 2016-2022 era.")

# %%
# =============================================================================
# 2.3  TARGET-FEATURE CORRELATIONS
# =============================================================================

print("\n" + "="*60)
print("2.3  TARGET-FEATURE CORRELATIONS (Spearman, train 2016-2022)")
print("="*60)

# Use full-feature era only for correlation analysis
train_full = train[train["season"] >= 2016].copy()

print("\nTop 15 features correlated with each target (Spearman |r|):")
top_features_per_target = {}
for t in TARGETS:
    corrs = {}
    y = train_full[t].dropna()
    for f in QB_FEATURES:
        if f not in train_full.columns:
            continue   # EWM cols not yet computed at Phase 2 — skipped here, used in Phase 3+
        col = train_full.loc[y.index, f]
        valid = col.notna() & y.notna()
        if valid.sum() < 100:
            continue
        r, _ = stats.spearmanr(col[valid], y[valid])
        if not np.isnan(r):
            corrs[f] = r
    top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    top_features_per_target[t] = [f for f, _ in top]
    print(f"\n  {t}:")
    for f, r in top:
        bar = "#" * int(abs(r) * 30)
        print(f"    {f:<50} r={r:+.3f}  {bar}")

# %%
# --- Correlation heatmap: top features for passing_yards ---
top_py_feats = top_features_per_target.get("passing_yards", QB_FEATURES[:20])
corr_data = train_full[top_py_feats + ["passing_yards"]].dropna()
if len(corr_data) > 50:
    corr_matrix = corr_data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.3, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title("Spearman correlations — top passing_yards features (train 2016-2022)")
    plt.tight_layout()
    out = FIG_DIR / "corr_heatmap_passing_yards.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

# %%
# =============================================================================
# 2.4  TARGET-TARGET CORRELATIONS
# =============================================================================

print("\n" + "="*60)
print("2.4  TARGET-TARGET CORRELATIONS")
print("="*60)

target_corr = train_full[TARGETS].corr(method="spearman")
print("\nSpearman correlation matrix (targets):")
print(target_corr.to_string(float_format="{:.3f}".format))

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    target_corr, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5, ax=ax,
    annot_kws={"size": 10}
)
ax.set_title("QB Target-Target Spearman Correlations (train 2016-2022)")
plt.tight_layout()
out = FIG_DIR / "target_target_corr.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# %%
# =============================================================================
# 2.5  TEMPORAL TRENDS — league-wide pass volume drift
# =============================================================================

print("\n" + "="*60)
print("2.5  TEMPORAL TRENDS")
print("="*60)

season_stats = (
    train.groupby("season")[TARGETS]
    .mean()
    .reset_index()
)
print("\nMean per game by season:")
print(season_stats.to_string(float_format="{:.2f}".format, index=False))

# %%
# =============================================================================
# 2.6  COLD START / WEEK 1 ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("2.6  COLD START ANALYSIS")
print("="*60)

# Compare Week 1 predictions (L5 rolling) vs rest of season
train16 = train[train["season"] >= 2016].copy()

for t in ["passing_yards", "passing_tds"]:
    w1     = train16[train16["week"] == 1][t].dropna()
    w2plus = train16[train16["week"] > 1][t].dropna()
    l5_w1  = train16[train16["week"] == 1][f"{t}_L5"].dropna()
    l5_rest= train16[train16["week"] > 1][f"{t}_L5"].dropna()
    print(f"\n  {t}:")
    print(f"    Week 1  actual mean:  {w1.mean():.2f}  |  L5 predictor mean: {l5_w1.mean():.2f}")
    print(f"    Week 2+ actual mean: {w2plus.mean():.2f}  |  L5 predictor mean: {l5_rest.mean():.2f}")
    w1_l5_corr, _ = stats.spearmanr(
        train16[train16["week"]==1][[t, f"{t}_L5"]].dropna().iloc[:,0],
        train16[train16["week"]==1][[t, f"{t}_L5"]].dropna().iloc[:,1],
    )
    print(f"    Week 1 correlation (actual vs L5): {w1_l5_corr:.3f}")

# How sparse are L5/L10/L20 rolling features at Week 1?
print(f"\n  Rolling feature availability at Week 1 (2016+ train):")
wk1 = train16[train16["week"] == 1]
for w in [3, 5, 10, 20]:
    col = f"passing_yards_L{w}"
    pct = wk1[col].notna().mean() * 100
    print(f"    passing_yards_L{w}: {pct:.1f}% non-null at Week 1")

# %%
# =============================================================================
# SUMMARY: Loss function recommendations based on EDA
# =============================================================================

print("\n" + "="*60)
print("LOSS FUNCTION RECOMMENDATIONS (based on EDA)")
print("="*60)

for _, row in stats_df.iterrows():
    t    = row.name
    skew = row["skewness"]
    zero = row["pct_zero"]
    mu   = row["mean"]

    if zero < 15 and abs(skew) < 0.5:
        rec = "regression (MSE) -- near-normal, low zeros"
    elif zero < 15 and skew > 0.5:
        rec = "tweedie (power~1.5) -- continuous, right-skewed"
    elif zero >= 15 and zero < 50 and mu < 3:
        rec = "poisson -- count with moderate zero-inflation"
    elif zero >= 50 and mu < 2:
        rec = "poisson or tweedie (power~1.5) -- high zero-inflation, low mean"
    elif zero >= 75:
        rec = "tweedie (power~1.8) or poisson -- rare event, very high zeros"
    else:
        rec = "regression (MSE) -- default"

    print(f"  {t:<25}  skew={skew:+.2f}  zeros={zero:4.1f}%  -> {rec}")

print("\nEDA complete. Figures saved to:", FIG_DIR)

# %% [markdown]
# ## Phase 3 — Feature Preparation

# %%
# =============================================================================
# 3.1  LOCK DOWN FEATURE LIST
# =============================================================================
# Start from QB_FEATURES defined in Phase 2 and apply:
#   - Add `season` for temporal drift (QB rushing regime change)
#   - Remove features >70% null in the 2016+ era (modern era filter only)
#   - Encode game_location (home=1, neutral=0, away=-1)
# =============================================================================

print("\n" + "="*60)
print("PHASE 3 — FEATURE PREPARATION")
print("="*60)

# --- Loss functions confirmed from EDA ---
LOSS_FUNCTIONS = {
    "passing_yards":      "regression",        # near-normal, skew=-0.27
    "passing_tds":        "poisson",            # count, 33% zeros
    "interceptions":      "poisson",            # count, 51% zeros
    "rushing_yards":      "regression",          # 12.5% negatives — Tweedie invalid (requires y>=0); use MSE
    "rushing_tds":        "poisson",            # 90% zeros
    "fumbles_lost_total": "tweedie",            # 83% zeros, rare event
}
TWEEDIE_POWER = {
    "rushing_yards":      1.5,
    "fumbles_lost_total": 1.8,
}

# %%
# --- 3.0  Minimum attempts filter ---
# Remove mop-up / garbage-time appearances with < 10 pass attempts.
# Mean passing yards for <10 attempts is ~14 yds — pure noise for projection purposes.
MIN_ATTEMPTS = 15
before = len(qb)
qb = qb[qb["attempts"] >= MIN_ATTEMPTS].copy()
print(f"\nAttempts filter (>= {MIN_ATTEMPTS}): {before:,} -> {len(qb):,} rows  "
      f"(removed {before - len(qb):,} garbage-time appearances)")

# %%
# --- 3.1a Add season to feature list ---
# Captures long-run structural changes (QB rushing drift, passing inflation)
# Not in QB_FEATURES yet since it's an identifier in EDA — add explicitly here
FEATURE_COLS = ["season"] + QB_FEATURES.copy()
print(f"\nFeatures after adding season: {len(FEATURE_COLS)}")

# %%
# --- 3.1b Encode game_location ---
# home=1, neutral=0, away=-1
# Do this on the full qb dataframe before splitting
qb = qb.copy()
loc_map = {"home": 1, "neutral": 0, "away": -1}
qb["game_location"] = qb["game_location"].map(loc_map)

unmapped = qb["game_location"].isna().sum()
if unmapped > 0:
    print(f"  WARNING: {unmapped} game_location values unmapped — check raw values")
    print(f"  Unique raw values: {qb['game_location'].unique()}")
else:
    print(f"\ngame_location encoded: home=1, neutral=0, away=-1")

# %%
# --- 3.1b-EWMA  Exponentially weighted rolling means ---
# Computed here in QB.py — no master dataset rebuild needed.
# Added alongside simple rolling means so SHAP can show which memory regime
# the model prefers. If EWMA features dominate, replace simple ones next iteration.
#
# span=N: most recent game gets ~2x the weight of game N/2 back (mild recency bias).
# shift(1): strict pre-game-N information — no leakage.
# min_periods=2: EWM value available from game 2 onward (returns NaN for game 1).
#
# Only computed for stats where raw per-game values exist in qb.
# Team/opponent stats (off_epa_per_play etc.) skipped — no raw per-game column.

EWMA_STATS = {
    # ── Passing yards model (existing) ──────────────────────────────────────
    "passing_yards":            [5, 10, 20],
    "yards_per_attempt":        [5, 10, 20],
    "completion_pct":           [20],
    "epa_per_dropback":         [5, 10, 20],
    "qb_cpoe":                  [5, 20],
    "qb_air_yards_per_attempt": [5, 10, 20],
    "qb_pressure_rate":         [5, 10, 20],
    "epa_per_opportunity":      [5, 10, 20],

    # ── Rushing yards model ──────────────────────────────────────────────────
    "rushing_yards":            [5, 10, 20],   # extended from [20] only
    "carries":                  [5, 10, 20],   # carry volume
    "yards_per_carry":          [5, 10, 20],   # per-carry efficiency
    "rushing_epa":              [5, 10, 20],   # quality-adjusted rushing value
    "epa_per_carry":            [5, 10, 20],   # per-carry EPA
    "qb_scramble_rate":         [5, 10, 20],   # scramble tendency

    # ── Rushing TDs model ────────────────────────────────────────────────────
    "rushing_tds":              [5, 10, 20],   # TD history EWM
    # (rushing_yards, carries, epa_per_carry ewm already above)

    # ── Interceptions model ──────────────────────────────────────────────────
    "interceptions":            [5, 10, 20],   # raw INT count EWM
    "int_rate":                 [5, 10, 20],   # INT rate EWM (per dropback)
    "passing_epa":              [5, 10, 20],   # per-game passing EPA EWM
    "ngs_aggressiveness":       [5, 10, 20],   # throws into tight coverage — most INT-direct NGS stat
    # (qb_cpoe, qb_pressure_rate, epa_per_dropback ewm already above)

    # ── Fumbles model ────────────────────────────────────────────────────────
    "fumbles_lost_total":       [5, 10, 20],   # total fumbles lost EWM
    # (carries, qb_scramble_rate, qb_pressure_rate ewm already above)
}

# Sort qb chronologically within each player so ewm() sees games in order
qb = qb.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

ewm_cols_added = []
for stat, spans in EWMA_STATS.items():
    if stat not in qb.columns:
        print(f"  EWMA skip (col not found): {stat}")
        continue
    for span in spans:
        col = f"{stat}_ewm{span}"
        qb[col] = (
            qb.groupby("player_id")[stat]
            .transform(lambda x, s=span: x.ewm(span=s, min_periods=2).mean().shift(1))
        )
        ewm_cols_added.append(col)

# ── Fantasy points composite ─────────────────────────────────────────────────
# Computed here so EWM windows and career mean are available before QB_FEATURES
# is locked down in Phase 3.1d.
# Formula: 0.04*PY + 4*PTD + 0.1*RY + 6*RTD - 2*(INT + FUM)
# shift(1): strict pre-game info only — no leakage.
_fpts_raw = (
    0.04 * qb["passing_yards"].fillna(0)
    + 4   * qb["passing_tds"].fillna(0)
    + 0.1 * qb["rushing_yards"].fillna(0)
    + 6   * qb["rushing_tds"].fillna(0)
    - 2   * (qb["interceptions"].fillna(0) + qb["fumbles_lost_total"].fillna(0))
)
qb["_fantasy_pts_raw"] = _fpts_raw

for _span in [10, 20]:
    _col = f"fantasy_pts_ewm{_span}"
    qb[_col] = (
        qb.groupby("player_id")["_fantasy_pts_raw"]
        .transform(lambda x, s=_span: x.ewm(span=s, min_periods=2).mean().shift(1))
    )

# Career mean: expanding mean with min_periods=20; NaN if fewer than 20 prior games
qb["fantasy_pts_per_game_career"] = (
    qb.groupby("player_id")["_fantasy_pts_raw"]
    .transform(lambda x: x.expanding(min_periods=20).mean().shift(1))
)

qb.drop(columns=["_fantasy_pts_raw"], inplace=True)
print(f"  Fantasy points features computed: fantasy_pts_ewm10, fantasy_pts_ewm20, fantasy_pts_per_game_career")

# Career per-game stats (expanding mean, min 10 prior games, shift(1) — no leakage)
_CAREER_STATS = [
    ('passing_yards',      'passing_yards_per_game_career'),
    ('passing_tds',        'passing_tds_per_game_career'),
    ('rushing_yards',      'rushing_yards_per_game_career'),
    ('rushing_tds',        'rushing_tds_per_game_career'),
    ('carries',            'carries_per_game_career'),
    ('rushing_epa',        'rushing_epa_per_game_career'),
    ('interceptions',      'interceptions_per_game_career'),
    ('fumbles_lost_total', 'fumbles_lost_per_game_career'),
]
for _raw, _col in _CAREER_STATS:
    if _raw in qb.columns:
        qb[_col] = (
            qb.groupby('player_id')[_raw]
            .transform(lambda x: x.expanding(min_periods=20).mean().shift(1))
        )
print(f"  Career per-game stats computed: {[c for _, c in _CAREER_STATS]}")

# Regression-to-mean gap features: career baseline minus current EWM form.
# Positive gap = QB performing below career level (injury/slump → likely rebound).
# Gives the model a direct signal to identify injury-year bouncebacks.
if 'rushing_yards_per_game_career' in qb.columns and 'rushing_yards_ewm20' in qb.columns:
    qb['rushing_yards_career_vs_recent'] = (
        qb['rushing_yards_per_game_career'] - qb['rushing_yards_ewm20']
    )
if 'carries_per_game_career' in qb.columns and 'carries_ewm20' in qb.columns:
    qb['carries_career_vs_recent'] = (
        qb['carries_per_game_career'] - qb['carries_ewm20']
    )
if 'rushing_epa_per_game_career' in qb.columns and 'rushing_epa_ewm20' in qb.columns:
    qb['rushing_epa_career_vs_recent'] = (
        qb['rushing_epa_per_game_career'] - qb['rushing_epa_ewm20']
    )
if 'passing_yards_per_game_career' in qb.columns and 'passing_yards_ewm20' in qb.columns:
    qb['passing_yards_career_vs_recent'] = (
        qb['passing_yards_per_game_career'] - qb['passing_yards_ewm20']
    )
if 'passing_tds_per_game_career' in qb.columns and 'passing_tds_L20' in qb.columns:
    qb['passing_tds_career_vs_recent'] = (
        qb['passing_tds_per_game_career'] - qb['passing_tds_L20']
    )
print("  Gap features computed: rushing_yards_career_vs_recent, carries_career_vs_recent, "
      "rushing_epa_career_vs_recent, passing_yards_career_vs_recent, passing_tds_career_vs_recent")

# EWM columns are computed above for the full dataframe so future position models
# can reference them directly. QB_FEATURES is the authoritative list — do NOT
# auto-append EWM cols here. Any EWM col wanted in the QB model must be
# explicitly listed in QB_FEATURES above.
_ewm_in_features = [c for c in ewm_cols_added if c in FEATURE_COLS]
_ewm_computed_only = [c for c in ewm_cols_added if c not in FEATURE_COLS]
print(f"\n  EWMA cols computed: {len(ewm_cols_added)}")
print(f"    Used in QB model (in QB_FEATURES): {len(_ewm_in_features)}")
print(f"    Computed only (available for other models): {len(_ewm_computed_only)}")
print(f"  FEATURE_COLS total before null filter: {len(FEATURE_COLS)}")

# %%
# --- 3.1c Remove features >70% null in 2016+ era ---
# Apply only within 2016+ rows to avoid dropping NGS/snap features that are
# structurally unavailable pre-2016 but fully available in the modern era.
modern = qb[qb["season"] >= 2016]
high_null_modern = [
    f for f in FEATURE_COLS
    if f in qb.columns and modern[f].isna().mean() > 0.70
]
if high_null_modern:
    print(f"\nDropping {len(high_null_modern)} features >70% null in 2016+ era:")
    for f in high_null_modern:
        print(f"  {f}: {modern[f].isna().mean()*100:.1f}%")
    FEATURE_COLS = [f for f in FEATURE_COLS if f not in high_null_modern]
else:
    print("\nNo features >70% null in 2016+ era — none dropped.")

# %%
# --- 3.1d Confirm all feature cols exist ---
missing_cols = [f for f in FEATURE_COLS if f not in qb.columns]
if missing_cols:
    print(f"\nWARNING: {len(missing_cols)} features not found in dataset: {missing_cols}")
    FEATURE_COLS = [f for f in FEATURE_COLS if f in qb.columns]

print(f"\nFinal feature count: {len(FEATURE_COLS)}")

# %%
# --- 3.1e Leakage check ---
# Confirm no target columns or post-game columns are in FEATURE_COLS
POST_GAME = TARGETS + [
    "passing_epa", "rushing_epa", "receiving_epa",        # game EPA totals — known only after game
    "passing_air_yards", "passing_yac",                    # game totals
    "rushing_first_downs", "passing_first_downs",          # game totals
    "completions", "attempts", "carries",                  # current game counts
    "sacks", "sack_yards",
    "fumbles_total", "receiving_fumbles", "rushing_fumbles", "sack_fumbles",
    "receiving_fumbles_lost", "rushing_fumbles_lost", "sack_fumbles_lost",
]
leakage = [f for f in FEATURE_COLS if f in POST_GAME]
if leakage:
    print(f"\nLEAKAGE DETECTED — removing: {leakage}")
    FEATURE_COLS = [f for f in FEATURE_COLS if f not in leakage]
else:
    print("Leakage check passed — no post-game columns in feature set.")

print(f"Final feature count after leakage check: {len(FEATURE_COLS)}")

# Deduplicate FEATURE_COLS (preserves order, keeps first occurrence)
_seen = set()
_deduped = []
for _f in FEATURE_COLS:
    if _f not in _seen:
        _deduped.append(_f)
        _seen.add(_f)
if len(_deduped) < len(FEATURE_COLS):
    print(f"  WARNING: removed {len(FEATURE_COLS) - len(_deduped)} duplicate feature(s) from FEATURE_COLS")
FEATURE_COLS = _deduped
print(f"  Unique features: {len(FEATURE_COLS)}")

# %%
# =============================================================================
# 3.2  BUILD TRAIN / VAL / TEST SPLITS
# =============================================================================

print("\n" + "="*60)
print("3.2  BUILD SPLITS")
print("="*60)

df_train = qb[qb["season"] <= TRAIN_END].copy()
df_val   = qb[qb["season"].between(VAL_START, VAL_END)].copy()
df_test  = qb[qb["season"] == TEST_YEAR].copy()

X_train = df_train[FEATURE_COLS].reset_index(drop=True)
X_val   = df_val[FEATURE_COLS].reset_index(drop=True)
X_test  = df_test[FEATURE_COLS].reset_index(drop=True)

Y_train = df_train[TARGETS].reset_index(drop=True)
Y_val   = df_val[TARGETS].reset_index(drop=True)
Y_test  = df_test[TARGETS].reset_index(drop=True)

print(f"\n  X_train: {X_train.shape}  |  Y_train: {Y_train.shape}")
print(f"  X_val:   {X_val.shape}  |  Y_val:   {Y_val.shape}")
print(f"  X_test:  {X_test.shape}  |  Y_test:  {Y_test.shape}")

# %%
# =============================================================================
# 3.3  SAMPLE WEIGHTS
# =============================================================================

print("\n" + "="*60)
print("3.3  SAMPLE WEIGHTS")
print("="*60)

def make_sample_weights(seasons: pd.Series) -> np.ndarray:
    """
    Era-based sample weights for QB training rows.
    2006-2012: 0.65  (no snap counts, no off_* positional quality)
    2013-2015: 0.85  (no NGS)
    2016+:     1.00  (full feature set)
    """
    w = np.ones(len(seasons))
    w[seasons <= 2012] = 0.65
    w[(seasons >= 2013) & (seasons <= 2015)] = 0.85
    return w

sample_weights = make_sample_weights(df_train["season"])

print(f"\n  Weight distribution (train set):")
for era, lo, hi in [("2006-2012 (0.65)", 2006, 2012),
                     ("2013-2015 (0.85)", 2013, 2015),
                     ("2016-2023 (1.00)", 2016, 2023)]:
    mask = (df_train["season"] >= lo) & (df_train["season"] <= hi)
    n = mask.sum()
    w = sample_weights[mask.values].mean()
    print(f"    {era}: {n:,} rows  avg_weight={w:.2f}")

# %%
# =============================================================================
# 3.4  NULL RATE SUMMARY FOR FINAL FEATURE SET
# =============================================================================

print("\n" + "="*60)
print("3.4  NULL RATES IN FINAL FEATURE SET (train)")
print("="*60)

null_summary = X_train.isna().mean().sort_values(ascending=False)
has_nulls = null_summary[null_summary > 0]
if len(has_nulls) > 0:
    print(f"\n  {len(has_nulls)} features with nulls in train set:")
    for f, pct in has_nulls.items():
        era_note = ""
        if "ngs_" in f:
            era_note = "(NGS: null pre-2016, expected)"
        elif "off_" in f or "offense_pct" in f:
            era_note = "(snaps: null pre-2013, expected)"
        print(f"    {f:<52} {pct*100:5.1f}%  {era_note}")
else:
    print("  No nulls in training features (unexpected — check data).")

print(f"\n  LightGBM handles all nulls natively — no imputation needed.")

# %%
# =============================================================================
# 3.5  SAVE PREPARED DATA
# =============================================================================

print("\n" + "="*60)
print("3.5  SAVE PREPARED SPLITS")
print("="*60)

import joblib

PREP_DIR = DATA_DIR / "data" / "model_prep"
PREP_DIR.mkdir(parents=True, exist_ok=True)

prep = {
    "X_train":        X_train,
    "X_val":          X_val,
    "X_test":         X_test,
    "Y_train":        Y_train,
    "Y_val":          Y_val,
    "Y_test":         Y_test,
    "df_train":       df_train,       # needed for rolling CV splits in Phase 5
    "sample_weights": sample_weights,
    "feature_cols":   FEATURE_COLS,
    "targets":        TARGETS,
    "loss_functions": LOSS_FUNCTIONS,
    "tweedie_power":  TWEEDIE_POWER,
    "train_seasons":  list(range(TRAIN_START, TRAIN_END + 1)),
    "val_start":      VAL_START,
    "val_end":        VAL_END,
    "test_year":      TEST_YEAR,
}

out_path = PREP_DIR / "qb_model_prep.pkl"
joblib.dump(prep, out_path)
print(f"\n  Saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

# Sanity check: reload and verify shapes
loaded = joblib.load(out_path)
assert loaded["X_train"].shape == X_train.shape, "Reload mismatch"
print(f"  Reload verified.")

print(f"\nPhase 3 complete.")
print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Train rows: {len(X_train):,}  |  Val rows: {len(X_val):,}  |  Test rows: {len(X_test):,}")
print(f"  Next: Phase 4 — Baseline models")

# %%
# =============================================================================
# 3.6  2025 DATA CONSISTENCY CHECK
# =============================================================================
# Compares 2025 test set feature distributions against 2024 val set.
# Flags features where 2025 looks structurally different — high null rate,
# large mean shift, or near-zero variance — which would explain test degradation.
# Run this before Phase 5. Any flagged feature should be investigated in
# build_nfl_dataset.py / _weekly_from_pbp() before trusting 2025 test results.
# =============================================================================

print("\n" + "="*60)
print("3.6  2025 DATA CONSISTENCY CHECK")
print("="*60)

# Use 2024-only slice of val for comparison (apples-to-apples single season)
val_2024 = df_val[df_val["season"] == 2024][FEATURE_COLS].reset_index(drop=True)
test_2025 = X_test.copy()

flags = []
print(f"\n{'Feature':<45} {'null_val%':>9} {'null_tst%':>9} {'mean_val':>10} {'mean_tst':>10} {'flag'}")
print("-" * 100)

for col in FEATURE_COLS:
    if col not in val_2024.columns or col not in test_2025.columns:
        continue

    v_null = float(val_2024[col].isna().mean() * 100)
    t_null = float(test_2025[col].isna().mean() * 100)
    v_mean = float(val_2024[col].mean())
    t_mean = float(test_2025[col].mean())

    flag_parts = []

    # Flag: null rate spiked in 2025
    if t_null > v_null + 15:
        flag_parts.append(f"NULL+{t_null - v_null:.0f}%")

    # Flag: mean shifted by more than 1 std of the val distribution
    v_std = float(val_2024[col].std())
    if v_std > 0 and abs(t_mean - v_mean) > 1.5 * v_std:
        flag_parts.append(f"MEAN_SHIFT {(t_mean - v_mean)/v_std:+.1f}sd")

    # Flag: 2025 is entirely NaN (data missing completely)
    if float(t_null) == 100.0:
        flag_parts.append("ALL_NULL_2025")

    flag_str = " | ".join(flag_parts)
    if flag_str:
        flags.append((col, flag_str))
        print(f"  {col:<45} {v_null:>8.1f}% {t_null:>8.1f}% {v_mean:>10.3f} {t_mean:>10.3f}  *** {flag_str}")

if not flags:
    print("  No features flagged — 2025 distributions look consistent with 2024.")
else:
    print(f"\n  {len(flags)} feature(s) flagged. Investigate these before trusting test results.")
    print("  Likely causes: _weekly_from_pbp() fallback gaps, positional quality rollup issues.")

# Also check targets in 2025 vs 2024
print(f"\n  Target distribution check (2024 val vs 2025 test):")
print(f"  {'Target':<25} {'mean_2024':>10} {'mean_2025':>10} {'std_2024':>10} {'std_2025':>10}")
for tgt in TARGETS:
    v = df_val[df_val["season"] == 2024][tgt]
    t = Y_test[tgt]
    print(f"  {tgt:<25} {v.mean():>10.3f} {t.mean():>10.3f} {v.std():>10.3f} {t.std():>10.3f}")

# %%
# =============================================================================
# PHASE 4  —  BASELINES
# =============================================================================
# Four baselines define the floor LightGBM must beat in Phase 5:
#   B1  Global mean       — predict training mean for every game
#   B2  L1 (last game)    — use _L3 rolling as best available single-game proxy
#   B3  L5 rolling mean   — use _L5 rolling mean
#   B4  Ridge regression  — linear model on all 161 features (standardized)
#
# Metrics reported on VAL set (2024) for each target × baseline:
#   MAE, RMSE, R2, Bias (mean prediction - mean actual)
# =============================================================================

print("\n" + "="*60)
print("PHASE 4  —  BASELINES")
print("="*60)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
# --- Load prepared data (can run Phase 4 standalone) ---
import joblib

try:
    X_train  # already in memory from Phase 3
except NameError:
    PREP_DIR = DATA_DIR / "data" / "model_prep"
    prep     = joblib.load(PREP_DIR / "qb_model_prep.pkl")
    X_train        = prep["X_train"]
    X_val          = prep["X_val"]
    X_test         = prep["X_test"]
    Y_train        = prep["Y_train"]
    Y_val          = prep["Y_val"]
    Y_test         = prep["Y_test"]
    sample_weights = prep["sample_weights"]
    FEATURE_COLS   = prep["feature_cols"]
    TARGETS        = prep["targets"]

# %%
# --- Metric helper ---

def _metrics(y_true, y_pred, label):
    """Return dict of MAE / RMSE / R2 / Bias for a single target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    return {"baseline": label, "MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias}


# %%
# =============================================================================
# 4.1  BASELINE 1 — GLOBAL MEAN
# =============================================================================

print("\n--- B1: Global mean ---")

records = []
for tgt in TARGETS:
    train_mean = float(Y_train[tgt].mean())
    preds      = np.full(len(Y_val), train_mean)
    m = _metrics(Y_val[tgt], preds, "GlobalMean")
    m["target"] = tgt
    records.append(m)
    print(f"  {tgt:<25}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:+.3f}  Bias={m['Bias']:+.3f}")

df_b1 = pd.DataFrame(records)

# %%
# =============================================================================
# 4.2  BASELINE 2 — LAST GAME  (L3 proxy)
# =============================================================================
# The L3 rolling value is the closest available proxy for "most recent game"
# in this dataset.  For purely na cases (< 3 games ever) fall back to training mean.

print("\n--- B2: Last-game proxy (L3 rolling) ---")

records = []
for tgt in TARGETS:
    col = f"{tgt}_L3"
    train_mean = float(Y_train[tgt].mean())
    if col in X_val.columns:
        preds = X_val[col].fillna(train_mean).values
    else:
        preds = np.full(len(Y_val), train_mean)
        print(f"  {tgt}: L3 col missing — using global mean fallback")
    m = _metrics(Y_val[tgt], preds, "L1-proxy")
    m["target"] = tgt
    records.append(m)
    print(f"  {tgt:<25}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:+.3f}  Bias={m['Bias']:+.3f}")

df_b2 = pd.DataFrame(records)

# %%
# =============================================================================
# 4.3  BASELINE 3 — ROLLING L5 MEAN
# =============================================================================

print("\n--- B3: Rolling L5 mean ---")

records = []
for tgt in TARGETS:
    col = f"{tgt}_L5"
    train_mean = float(Y_train[tgt].mean())
    if col in X_val.columns:
        preds = X_val[col].fillna(train_mean).values
    else:
        preds = np.full(len(Y_val), train_mean)
        print(f"  {tgt}: L5 col missing — using global mean fallback")
    m = _metrics(Y_val[tgt], preds, "RollingL5")
    m["target"] = tgt
    records.append(m)
    print(f"  {tgt:<25}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:+.3f}  Bias={m['Bias']:+.3f}")

df_b3 = pd.DataFrame(records)

# %%
# =============================================================================
# 4.4  BASELINE 4 — RIDGE REGRESSION
# =============================================================================
# One Ridge model per target; StandardScaler fit on train, applied to val/test.
# NaN features filled with column median (Ridge can't handle NaN).

print("\n--- B4: Ridge regression ---")

# Fill NaNs with training column medians
col_medians = X_train.median()
X_tr_r = X_train.fillna(col_medians)
X_va_r = X_val.fillna(col_medians)

records = []
ridge_models = {}   # save for potential use in Phase 9 ensemble

for tgt in TARGETS:
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_tr_r)
    Xva_s  = scaler.transform(X_va_r)

    ridge = Ridge(alpha=10.0)
    ridge.fit(Xtr_s, Y_train[tgt].values, sample_weight=sample_weights)
    preds = ridge.predict(Xva_s)
    preds = np.clip(preds, 0, None)   # no negative projections

    m = _metrics(Y_val[tgt], preds, "Ridge")
    m["target"] = tgt
    records.append(m)
    ridge_models[tgt] = (scaler, ridge)
    print(f"  {tgt:<25}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:+.3f}  Bias={m['Bias']:+.3f}")

    # Print all coefficients sorted by absolute magnitude
    coef_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "coefficient": ridge.coef_})
        .reindex(pd.Series(ridge.coef_).abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
    coef_df.index = range(1, len(coef_df) + 1)
    print(f"\n  Ridge coefficients — {tgt} (standardized, sorted by |coef|):")
    print(f"  {'Rank':<5} {'Feature':<45} {'Coefficient':>12}")
    print(f"  {'-'*65}")
    for rank, row in coef_df.iterrows():
        print(f"  {rank:<5} {row['feature']:<45} {row['coefficient']:>+12.4f}")
    print()

df_b4 = pd.DataFrame(records)

# %%
# =============================================================================
# 4.5  COMBINED SUMMARY TABLE
# =============================================================================

print("\n" + "="*60)
print("BASELINE SUMMARY — VAL 2024")
print("="*60)

df_all = pd.concat([df_b1, df_b2, df_b3, df_b4], ignore_index=True)

for tgt in TARGETS:
    sub = df_all[df_all["target"] == tgt][["baseline", "MAE", "RMSE", "R2", "Bias"]].copy()
    sub = sub.set_index("baseline")
    print(f"\n  {tgt}")
    print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.1f}"))

# Best baseline per target (lowest MAE)
print("\n  Best baseline per target (lowest MAE on val):")
best = df_all.loc[df_all.groupby("target")["MAE"].idxmin(), ["target", "baseline", "MAE", "R2"]]
best = best.set_index("target")
print(best.to_string(float_format=lambda x: f"{x:.3f}"))

# %%
# =============================================================================
# 4.6  PLOT — BASELINE MAE COMPARISON
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
palette = {"GlobalMean": "#aec7e8", "L1-proxy": "#ffbb78", "RollingL5": "#98df8a", "Ridge": "#c5b0d5"}

for i, tgt in enumerate(TARGETS):
    ax  = axes[i]
    sub = df_all[df_all["target"] == tgt].copy()
    colors = [palette.get(b, "#888888") for b in sub["baseline"]]
    bars = ax.bar(sub["baseline"], sub["MAE"], color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(tgt.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_ylabel("MAE", fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=7, rotation=15)
    ax.tick_params(axis="y", labelsize=8)
    for bar, val in zip(bars, sub["MAE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

fig.suptitle("Baseline MAE — Val 2024", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

fig_path = FIG_DIR / "phase4_baseline_mae.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"\n  Saved: {fig_path}")

# %%
# =============================================================================
# 4.7  SAVE BASELINE RESULTS
# =============================================================================

baseline_path = PREP_DIR / "qb_baselines.pkl"
joblib.dump({
    "df_baselines":  df_all,
    "ridge_models":  ridge_models,
    "col_medians":   col_medians,
}, baseline_path)
print(f"  Saved baselines: {baseline_path}")

print(f"\nPhase 4 complete.")
print(f"  Best baselines saved. Ridge R2s are the floor LightGBM must beat in Phase 5.")


# %%
# =============================================================================
# PHASE 5  —  LightGBM: rolling forward CV + Optuna + final model
# =============================================================================
#
# ACTIVE_TARGET controls which model this block trains.
# To train the next target, change ACTIVE_TARGET and re-run phases 5-7.
#
# DESIGN:
#   Rolling forward CV folds:
#     Fold 1:  train 2006-2016, val 2017
#     Fold 2:  train 2006-2017, val 2018
#     ...
#     Fold 7:  train 2006-2022, val 2023   <- Optuna tuning fold
#   Optuna (60 trials, TPE): tunes on the last CV fold (train 2006-2022, val 2023).
#     Val 2024 is NEVER seen during hyperparameter search.
#   OOF predictions: collected across all folds (2017-2023) using best_params.
#   Final model: train 2006-2023 with best_params; n_trees = mean best_iteration
#     from CV folds. Val 2024 = honest holdout. Test 2025 = true held-out.
#   Labels clipped to >=0 for poisson/tweedie objectives.
# =============================================================================

ACTIVE_TARGET = "passing_yards"

OOF_FIRST_VAL_YEAR = 2017   # first val year; train fold = 2006-2016
OPTUNA_TUNE_YEAR   = 2023   # Optuna tunes on this val year (last CV fold)

print("\n" + "="*60)
print(f"PHASE 5  --  LightGBM: {ACTIVE_TARGET}")
print("="*60)

import lightgbm as lgb
import optuna
from lightgbm import early_stopping, log_evaluation
from tqdm import tqdm as _tqdm_std

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
# --- 5.0  Load prepared data ---

import joblib

try:
    X_train
except NameError:
    PREP_DIR = DATA_DIR / "data" / "model_prep"
    prep = joblib.load(PREP_DIR / "qb_model_prep.pkl")
    X_train        = prep["X_train"]
    X_val          = prep["X_val"]
    X_test         = prep["X_test"]
    Y_train        = prep["Y_train"]
    Y_val          = prep["Y_val"]
    Y_test         = prep["Y_test"]
    sample_weights = prep["sample_weights"]
    FEATURE_COLS   = prep["feature_cols"]
    TARGETS        = prep["targets"]
    LOSS_FUNCTIONS = prep["loss_functions"]
    TWEEDIE_POWER  = prep["tweedie_power"]

try:
    df_train
except NameError:
    PREP_DIR = DATA_DIR / "data" / "model_prep"
    _prep2 = joblib.load(PREP_DIR / "qb_model_prep.pkl")
    df_train = _prep2["df_train"]

try:
    df_all
except NameError:
    PREP_DIR = DATA_DIR / "data" / "model_prep"
    base_art = joblib.load(PREP_DIR / "qb_baselines.pkl")
    df_all = base_art["df_baselines"]

ARTIFACTS_DIR = DATA_DIR / "data" / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Load existing model registry if present (accumulates across targets)
registry_path = ARTIFACTS_DIR / "qb_lgb_models.pkl"
if registry_path.exists():
    _reg = joblib.load(registry_path)
    lgb_models      = _reg.get("lgb_models",    {})
    optuna_studies  = _reg.get("optuna_studies", {})
    phase5_records  = _reg.get("df_phase5", pd.DataFrame()).to_dict("records")
    best_params_all = _reg.get("best_params",    {})
    oof_store       = _reg.get("oof_store",       {})
else:
    lgb_models      = {}
    optuna_studies  = {}
    phase5_records  = []
    best_params_all = {}
    oof_store       = {}

# %%
# --- 5.1  Helpers ---

def _make_lgb_datasets(X_tr, y_tr, X_va, y_va, sw_tr, clip_labels=False):
    """Build LightGBM Dataset objects. clip_labels=True for poisson/tweedie."""
    if clip_labels:
        y_tr = np.clip(np.asarray(y_tr, dtype=float), 0, None)
        y_va = np.clip(np.asarray(y_va, dtype=float), 0, None)
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, free_raw_data=False)
    dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)
    return dtrain, dval


class _OptunaProgress:
    """Plain-text tqdm progress bar callback for Optuna."""
    def __init__(self, n_trials, target):
        self._pbar = _tqdm_std(
            total=n_trials,
            desc=f"  {target:<25}",
            unit="trial",
            ncols=90,
        )
    def __call__(self, study, trial):
        self._pbar.set_postfix_str(f"best={study.best_value:.4f}")
        self._pbar.update(1)
    def close(self):
        self._pbar.close()


def _make_objective(target, dtrain, dval, X_va, Y_va, objective, tweedie_power=None):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         objective,
            "metric":            "mae",
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs":            -1,
            "seed":              42,
        }
        if tweedie_power is not None:
            params["tweedie_variance_power"] = tweedie_power

        booster = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=-1),
            ],
        )
        preds = np.clip(booster.predict(X_va), 0, None)
        return mean_absolute_error(Y_va[target], preds)

    return objective_fn


def _cv_split(df_full, val_year, feature_cols, targets):
    """Return (X_tr, Y_tr, sw_tr, X_va, Y_va) for a single CV fold."""
    tr_mask = df_full["season"] < val_year
    va_mask = df_full["season"] == val_year
    X_tr = df_full.loc[tr_mask, feature_cols].reset_index(drop=True)
    Y_tr = df_full.loc[tr_mask, targets].reset_index(drop=True)
    X_va = df_full.loc[va_mask, feature_cols].reset_index(drop=True)
    Y_va = df_full.loc[va_mask, targets].reset_index(drop=True)
    sw   = make_sample_weights(df_full.loc[tr_mask, "season"])
    return X_tr, Y_tr, sw, X_va, Y_va


# %%
# --- 5.2  Optuna: tune on last CV fold (train 2006-2022, val 2023) ---

tgt = ACTIVE_TARGET
obj = LOSS_FUNCTIONS[tgt]
tp  = TWEEDIE_POWER.get(tgt, None)
_clip = obj in ("tweedie", "poisson")

print(f"\n  target={tgt}  objective={obj}"
      + (f"  tweedie_power={tp}" if tp else ""))
print(f"  Optuna tuning fold: train 2006-{OPTUNA_TUNE_YEAR - 1}, val {OPTUNA_TUNE_YEAR}")

_Xtr_opt, _Ytr_opt, _sw_opt, _Xva_opt, _Yva_opt = _cv_split(
    df_train, OPTUNA_TUNE_YEAR, FEATURE_COLS, TARGETS
)
_dtrain_opt, _dval_opt = _make_lgb_datasets(
    _Xtr_opt, _Ytr_opt[tgt], _Xva_opt, _Yva_opt[tgt], _sw_opt, clip_labels=_clip
)

N_TRIALS = 60
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
_progress = _OptunaProgress(N_TRIALS, tgt)
study.optimize(
    _make_objective(tgt, _dtrain_opt, _dval_opt, _Xva_opt, _Yva_opt, obj, tp),
    n_trials=N_TRIALS,
    show_progress_bar=False,
    callbacks=[_progress],
)
_progress.close()

best_params = study.best_params
print(f"\n  Optuna best MAE: {study.best_value:.4f}  (trial {study.best_trial.number}/{N_TRIALS})")
print(f"  Best params: {best_params}")

# %%
# --- 5.3  Rolling forward CV with best_params -> OOF predictions ---

cv_val_years = list(range(OOF_FIRST_VAL_YEAR, TRAIN_END + 1))   # 2017..2023

final_params = {
    "verbosity": -1,
    "objective": obj,
    "metric":    "mae",
    "n_jobs":    -1,
    "seed":      42,
    **{k: v for k, v in best_params.items() if k != "extra_trees"},
    "extra_trees": best_params["extra_trees"],
}
if tp is not None:
    final_params["tweedie_variance_power"] = tp

print(f"\n  Rolling forward CV: {len(cv_val_years)} folds  ({cv_val_years[0]}-{cv_val_years[-1]})")

oof_actual = []
oof_pred   = []
oof_years  = []
best_iters = []

for _yr in cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, FEATURE_COLS, TARGETS)
    _dt, _dv = _make_lgb_datasets(_Xtr, _Ytr[tgt], _Xva, _Yva[tgt], _sw, clip_labels=_clip)

    _b = lgb.train(
        final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=-1),
        ],
    )
    _p = np.clip(_b.predict(_Xva), 0, None)

    oof_actual.extend(_Yva[tgt].tolist())
    oof_pred.extend(_p.tolist())
    oof_years.extend([_yr] * len(_Yva))
    best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"MAE={mean_absolute_error(_Yva[tgt], _p):.3f}")

oof_actual = np.array(oof_actual)
oof_pred   = np.array(oof_pred)
oof_years  = np.array(oof_years)
mean_best_iter = int(np.mean(best_iters))
print(f"\n  Mean best_iteration across folds: {mean_best_iter}")

# %%
# --- 5.4  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"OOF METRICS (rolling forward CV)  --  {tgt}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in cv_val_years:
    _mask = oof_years == _yr
    _m    = _metrics(oof_actual[_mask], oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.2f}  {_m['RMSE']:>8.2f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.2f}")

_m_overall = _metrics(oof_actual, oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(oof_actual):>5}  "
      f"{_m_overall['MAE']:>8.2f}  {_m_overall['RMSE']:>8.2f}  "
      f"{_m_overall['R2']:>+8.3f}  {_m_overall['Bias']:>+8.2f}")

# %%
# --- 5.5  Final model: train 2006-2023, n_trees = mean best_iter, validate on 2024 ---

print(f"\n  Final model: train 2006-{TRAIN_END}, n_trees={mean_best_iter}")

_label_tr = np.clip(Y_train[tgt].values.astype(float), 0, None) if _clip else Y_train[tgt].values
dtrain_final = lgb.Dataset(X_train, label=_label_tr, weight=sample_weights, free_raw_data=False)

booster_final = lgb.train(
    {**final_params, "metric": "none"},
    dtrain_final,
    num_boost_round=mean_best_iter,
    callbacks=[log_evaluation(period=-1)],
)

# Honest val 2024 metrics (never seen during tuning or CV)
preds_val = np.clip(booster_final.predict(X_val), 0, None)
m = _metrics(Y_val[tgt], preds_val, "LightGBM")
m["target"]         = tgt
m["n_trees"]        = mean_best_iter
m["oof_mae"]        = float(_m_overall["MAE"])
m["oof_r2"]         = float(_m_overall["R2"])
print(f"  Val 2024 -> MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R2={m['R2']:+.3f}  Bias={m['Bias']:+.3f}")

# booster_cv alias for Phase 6 compatibility
booster_cv = booster_final

# %%
# --- 5.6  Save to registry ---

phase5_records = [r for r in phase5_records if r.get("target") != tgt]
phase5_records.append(m)
lgb_models[tgt]      = booster_final
optuna_studies[tgt]  = study
best_params_all[tgt] = best_params
oof_store[tgt] = {
    "oof_actual": oof_actual,
    "oof_pred":   oof_pred,
    "oof_years":  oof_years,
}

joblib.dump({
    "lgb_models":     lgb_models,
    "optuna_studies": optuna_studies,
    "df_phase5":      pd.DataFrame(phase5_records),
    "best_params":    best_params_all,
    "oof_store":      oof_store,
    "feature_cols":   FEATURE_COLS,
    "targets":        TARGETS,
    "loss_functions": LOSS_FUNCTIONS,
    "tweedie_power":  TWEEDIE_POWER,
}, registry_path)
print(f"\n  Registry saved: {registry_path}")
print(f"  Models in registry: {list(lgb_models.keys())}")

# %%
# --- 5.7  Summary vs baseline ---

print("\n" + "="*60)
print(f"PHASE 5 SUMMARY -- {tgt}")
print("="*60)

ridge_row = df_all[(df_all["baseline"] == "Ridge") & (df_all["target"] == tgt)].iloc[0]
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    lgb_oof  = float(_m_overall[metric])
    lgb_val  = m[metric]
    rdg_val  = ridge_row[metric]
    print(f"  {metric:<10}  {lgb_oof:>10.3f}  {lgb_val:>10.3f}  {rdg_val:>12.3f}")

print(f"\nPhase 5 complete for [{tgt}].")
print(f"  OOF covers {cv_val_years[0]}-{cv_val_years[-1]}  ({len(cv_val_years)} folds)")
print(f"  Val 2024 is a clean holdout — never used in tuning or CV.")

# %%
# =============================================================================
# PHASE 6  --  EVALUATION & DIAGNOSTICS: passing_yards
# =============================================================================
#
# Val (2024) : booster_cv predictions  (honest -- not trained on val)
# Test (2025): booster_final predictions (true held-out)
#
# 6.1  Full metrics table
# 6.2  Predicted vs actual scatter (val + test)
# 6.3  Residual distribution histogram + KDE + normal overlay
# 6.4  Calibration curve (decile buckets)
# 6.5  MAE by NFL week (test set)
# =============================================================================

print("\n" + "="*60)
print(f"PHASE 6  --  EVALUATION & DIAGNOSTICS: {ACTIVE_TARGET}")
print("="*60)

# Ensure _show is available when running this cell standalone
try:
    _show
except NameError:
    def _show(fig_path):
        try:
            from IPython.display import display, Image as _IPImage
            display(_IPImage(str(fig_path)))
        except Exception:
            pass

tgt = ACTIVE_TARGET
preds_val_p6  = np.clip(booster_cv.predict(X_val),     0, None)
preds_test_p6 = np.clip(booster_final.predict(X_test), 0, None)

# %%
# --- 6.1  Full metrics table ---

def _full_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]

    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    bias = float(np.mean(yp - yt))

    nonzero   = yt > 0
    mape      = float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero]))) if nonzero.sum() > 0 else np.nan
    abs_err   = np.abs(yt - yp)
    within_10 = float(np.mean(abs_err <= 0.10 * np.abs(yt).clip(1)))
    within_20 = float(np.mean(abs_err <= 0.20 * np.abs(yt).clip(1)))
    pearson_r = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else np.nan

    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias,
        "MAPE": mape, "Within10pct": within_10, "Within20pct": within_20,
        "Pearson_r": pearson_r,
    }

print(f"\n  {tgt}")
for split, yt, yp in [("Val 2024",  Y_val[tgt],  preds_val_p6),
                       ("Test 2025", Y_test[tgt], preds_test_p6)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15} {v:.4f}")

# %%
# --- 6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val[tgt].values,  preds_val_p6),
    ("Test 2025", Y_test[tgt].values, preds_test_p6),
]):
    ax.scatter(yt, yp, alpha=0.25, s=10, color="#1f77b4", rasterized=True)
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"{tgt.replace('_', ' ').title()} -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual (yards)", fontsize=9)
    ax.set_ylabel("Predicted (yards)", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig_path = FIG_DIR / f"phase6_{tgt}_pred_vs_actual.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"\n  Saved: {fig_path.name}")

# %%
# --- 6.3  Residual distribution ---

from scipy.stats import gaussian_kde, norm as sp_norm

_oof_resid_py = np.array(oof_pred) - np.array(oof_actual)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val[tgt].values,    preds_val_p6),
    ("Test 2025",   Y_test[tgt].values,   preds_test_p6),
    ("OOF 2017-23", np.array(oof_actual), np.array(oof_pred)),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, color="#1f77b4", alpha=0.55, density=True, edgecolor="none")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(kde_x, gaussian_kde(resid)(kde_x), "k-", lw=1.5, label="KDE")
    ax.plot(kde_x, sp_norm.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    ax.axvline(0, color="red", lw=1.5)
    ax.set_title(f"{tgt.replace('_', ' ').title()} Residuals -- {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual, yards)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig_path = FIG_DIR / f"phase6_{tgt}_residuals.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"  Saved: {fig_path.name}")

# %%
# --- 6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val[tgt].values,  preds_val_p6),
    ("Test 2025", Y_test[tgt].values, preds_test_p6),
]):
    df_cal = pd.DataFrame({"pred": yp, "actual": yt})
    df_cal["bucket"] = pd.qcut(df_cal["pred"], q=10, duplicates="drop")
    cal_agg = df_cal.groupby("bucket", observed=True).agg(
        mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), n=("actual", "count")
    ).reset_index()
    ax.plot(cal_agg["mean_pred"], cal_agg["mean_actual"], "o-", color="#1f77b4", lw=1.8, ms=6)
    lim = max(cal_agg["mean_pred"].max(), cal_agg["mean_actual"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
    for _, row in cal_agg.iterrows():
        ax.text(row["mean_pred"], row["mean_actual"] + 2, str(int(row["n"])),
                fontsize=6, ha="center", color="gray")
    ax.set_title(f"{tgt.replace('_', ' ').title()} Calibration -- {split}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
    ax.set_ylabel("Mean Actual", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig_path = FIG_DIR / f"phase6_{tgt}_calibration.png"
plt.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"  Saved: {fig_path.name}")

# %%
# --- 6.5  MAE by NFL week ---

try:
    df_raw = pd.read_parquet(MASTER_PATH, columns=["player_id", "season", "week", "position"])
    qb_test_rows = df_raw[(df_raw["season"] == TEST_YEAR) & (df_raw["position"] == "QB")].reset_index(drop=True)  # noqa
    week_col = qb_test_rows["week"].values[:len(X_test)]

    if len(week_col) == len(preds_test_p6):
        abs_err = np.abs(preds_test_p6 - Y_test[tgt].values)
        df_wk   = pd.DataFrame({"week": week_col, "abs_err": abs_err})
        wk_mae  = df_wk.groupby("week")["abs_err"].mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(wk_mae.index, wk_mae.values, color="#1f77b4", edgecolor="none")
        ax.axhline(abs_err.mean(), color="red", lw=1.5, linestyle="--",
                   label=f"Overall MAE = {abs_err.mean():.1f}")
        ax.set_title(f"{tgt.replace('_', ' ').title()} MAE by Week -- Test 2025",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Week", fontsize=9)
        ax.set_ylabel("MAE (yards)", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        fig_path = FIG_DIR / f"phase6_{tgt}_mae_by_week.png"
        plt.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close()
        _show(fig_path)
        print(f"  Saved: {fig_path.name}")
    else:
        print(f"  MAE-by-week: row count mismatch -- skipped")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- 6.6  Export 2025 test predictions to Excel ---
# Columns: player_display_name, week, attempts, depth_chart_rank, actual, predicted
preds_test_p6 = np.clip(booster_final.predict(X_test), 0, None)

_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_export_df["actual_passing_yards"]    = Y_test[tgt].values
_export_df["predicted_passing_yards"] = preds_test_p6
_export_df["residual"]                = _export_df["predicted_passing_yards"] - _export_df["actual_passing_yards"]
_export_df = _export_df.sort_values(["week", "actual_passing_yards"], ascending=[True, False]).reset_index(drop=True)

_export_path = DATA_DIR / f"test_predictions_2025_{tgt}.xlsx"
_export_df.to_excel(_export_path, index=False)
print(f"\n  2025 test predictions saved: {_export_path.name}  ({len(_export_df)} rows)")

print(f"\nPhase 6 complete for [{tgt}].")

# %%
# =============================================================================
# PHASE 7  --  FEATURE IMPORTANCE & SHAP: passing_yards
# =============================================================================
#
# 7.1  LightGBM native gain importance (top 25)
# 7.2  SHAP TreeExplainer on 500 test-set samples:
#        - Beeswarm plot (top 20 features, direction + magnitude)
#        - Bar plot (mean |SHAP|, top 20)
#        - Dependence plots for top 3 features
# 7.3  Save SHAP artifacts
# =============================================================================

print("\n" + "="*60)
print(f"PHASE 7  --  FEATURE IMPORTANCE & SHAP: {ACTIVE_TARGET}")
print("="*60)

try:
    import shap
    print(f"  shap version: {shap.__version__}")
except ImportError:
    raise ImportError("Run: pip install shap")

tgt     = ACTIVE_TARGET
booster = booster_final

SHAP_SAMPLE = 500
rng      = np.random.default_rng(42)
shap_idx = rng.choice(len(X_test), size=min(SHAP_SAMPLE, len(X_test)), replace=False)
X_shap   = X_test.iloc[shap_idx].reset_index(drop=True)

# %%
# --- 7.1  Native gain importance ---

imp_gain = pd.Series(
    booster.feature_importance(importance_type="gain"),
    index=FEATURE_COLS,
).sort_values(ascending=False)

top25 = imp_gain.head(25)
fig, ax = plt.subplots(figsize=(8, 8))
ax.barh(top25.index[::-1], top25.values[::-1], color="#1f77b4")
ax.set_title(f"{tgt.replace('_', ' ').title()} -- Top 25 Features (Gain)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Gain", fontsize=9)
ax.tick_params(axis="y", labelsize=7)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
fig_path = FIG_DIR / f"phase7_gain_{tgt}.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"\n  Gain chart saved: {fig_path.name}")
print(f"  Top 10 features (gain):")
for feat, val in imp_gain.head(10).items():
    print(f"    {feat:<35}  {val:,.0f}")

# %%
# --- 7.2  SHAP ---

print(f"\n  Computing SHAP on {len(X_shap)} test samples...")
explainer = shap.TreeExplainer(booster)
shap_vals = explainer.shap_values(X_shap)

mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)

print(f"  Top 10 features (mean |SHAP|):")
for feat, val in mean_abs_shap.head(10).items():
    print(f"    {feat:<35}  {val:.4f}")

# Beeswarm
fig, ax = plt.subplots(figsize=(9, 8))
shap.summary_plot(shap_vals, X_shap, feature_names=FEATURE_COLS,
                  max_display=20, show=False, plot_type="dot")
plt.title(f"{tgt.replace('_', ' ').title()} -- SHAP Beeswarm (Top 20)",
          fontsize=11, fontweight="bold")
plt.tight_layout()
fig_path = FIG_DIR / f"phase7_shap_beeswarm_{tgt}.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"\n  Beeswarm saved: {fig_path.name}")

# Bar
top20 = mean_abs_shap.head(20)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(top20.index[::-1], top20.values[::-1], color="#d62728")
ax.set_title(f"{tgt.replace('_', ' ').title()} -- SHAP Mean |Value| (Top 20)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(axis="y", labelsize=7)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
fig_path = FIG_DIR / f"phase7_shap_bar_{tgt}.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"  Bar chart saved: {fig_path.name}")

# Dependence plots -- top 3 features
top3 = list(mean_abs_shap.head(3).index)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, feat in zip(axes, top3):
    shap.dependence_plot(feat, shap_vals, X_shap, feature_names=FEATURE_COLS,
                         ax=ax, show=False)
    ax.set_title(feat, fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
fig.suptitle(f"{tgt.replace('_', ' ').title()} -- SHAP Dependence (Top 3 Features)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig_path = FIG_DIR / f"phase7_shap_dependence_{tgt}.png"
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
plt.close()
_show(fig_path)
print(f"  Dependence plots saved: {fig_path.name}")

# %%
# --- 7.3  Save SHAP artifacts ---

shap_path = ARTIFACTS_DIR / "qb_shap_values.pkl"
if shap_path.exists():
    shap_reg = joblib.load(shap_path)
else:
    shap_reg = {"shap_values": {}, "X_shap_per_target": {}, "feature_cols": FEATURE_COLS, "targets": TARGETS}

shap_reg["shap_values"][tgt]       = shap_vals
shap_reg["X_shap_per_target"][tgt] = X_shap
joblib.dump(shap_reg, shap_path)
print(f"\n  SHAP registry saved: {shap_path.name}")

# %%
# --- 7.4  Full feature tables (all features, printed + saved to Excel) ---

_full_gain = imp_gain.reset_index()
_full_gain.columns = ["feature", "gain"]
_full_gain["gain_rank"] = range(1, len(_full_gain) + 1)

_full_shap = mean_abs_shap.reset_index()
_full_shap.columns = ["feature", "mean_abs_shap"]
_full_shap["shap_rank"] = range(1, len(_full_shap) + 1)

_full_importance = _full_gain.merge(_full_shap, on="feature")
_full_importance = _full_importance[["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]]

print(f"\n  ALL FEATURES — Gain and SHAP ({len(_full_importance)} total):")
print(f"  {'Rank(G)':>8}  {'Rank(S)':>8}  {'Feature':<45}  {'Gain':>15}  {'Mean|SHAP|':>12}")
print("  " + "-"*95)
for _, row in _full_importance.iterrows():
    print(f"  {int(row.gain_rank):>8}  {int(row.shap_rank):>8}  {row.feature:<45}  {row.gain:>15,.0f}  {row.mean_abs_shap:>12.4f}")

_imp_path = DATA_DIR / f"feature_importance_{tgt}.xlsx"
_full_importance.to_excel(_imp_path, index=False)
print(f"\n  Full importance table saved: {_imp_path.name}")

print(f"\nPhase 7 complete for [{tgt}].")
print(f"  Figures saved to {FIG_DIR}")
print(f"\n  --> Next target: change ACTIVE_TARGET at the top of Phase 5 and re-run phases 5-7.")

# %%
# =============================================================================
# PHASE 7B  --  SEASON-LEVEL FIT DIAGNOSTICS: passing_yards
# =============================================================================
# Aggregate per-game predictions to season totals per QB, then evaluate.
# This is the relevant view for season-long fantasy projections.
# =============================================================================

print("\n" + "="*60)
print(f"PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {tgt}")
print("="*60)

# Load the per-game test predictions written in Phase 6.6
_pred_path = DATA_DIR / f"test_predictions_2025_{tgt}.xlsx"
_pg = pd.read_excel(_pred_path)

# Aggregate to season totals per QB
_szn = (
    _pg.groupby("player_display_name")
    .agg(
        games          =("actual_passing_yards",    "count"),
        actual_total   =("actual_passing_yards",    "sum"),
        predicted_total=("predicted_passing_yards", "sum"),
        depth_chart_rank_min=("depth_chart_rank",   "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_szn["residual"]     = _szn["predicted_total"] - _szn["actual_total"]
_szn["abs_residual"] = _szn["residual"].abs()
_szn["pct_error"]    = (_szn["residual"] / _szn["actual_total"].replace(0, np.nan)).abs()

# Fit diagnostics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_act  = _szn["actual_total"].values
_pred = _szn["predicted_total"].values

_mae  = mean_absolute_error(_act, _pred)
_rmse = np.sqrt(mean_squared_error(_act, _pred))
_r2   = r2_score(_act, _pred)
_bias = (_pred - _act).mean()
_mape = _szn["pct_error"].mean() * 100
_r    = np.corrcoef(_act, _pred)[0, 1]
_within10 = (_szn["pct_error"] <= 0.10).mean() * 100
_within20 = (_szn["pct_error"] <= 0.20).mean() * 100

print(f"\n  Season-level totals: {len(_szn)} QBs  ({_szn['games'].sum()} total game-rows)")
print(f"\n  {'Metric':<20}  {'Value':>10}")
print(f"  {'-'*32}")
print(f"  {'MAE (yards)':<20}  {_mae:>10.1f}")
print(f"  {'RMSE (yards)':<20}  {_rmse:>10.1f}")
print(f"  {'R2':<20}  {_r2:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_bias:>10.1f}")
print(f"  {'MAPE':<20}  {_mape:>10.2f}%")
print(f"  {'Pearson r':<20}  {_r:>10.4f}")
print(f"  {'Within 10%':<20}  {_within10:>10.1f}%")
print(f"  {'Within 20%':<20}  {_within20:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _szn.iterrows():
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.0f}  {row.residual:>+8.0f}  {row.pct_error*100:>7.1f}%  {int(row.depth_chart_rank_min):>7}")

# Save season-level summary to Excel
_szn_path = DATA_DIR / f"test_predictions_2025_{tgt}_season_totals.xlsx"
_szn.to_excel(_szn_path, index=False)
print(f"\n  Season totals saved: {_szn_path.name}")


# %%
# =============================================================================
# =============================================================================
# PASSING TDs MODEL
# =============================================================================
# =============================================================================

# %%
# =============================================================================
# TD-PHASE 4  —  BASELINES: passing_tds
# =============================================================================
# Baselines for passing_tds only. Ridge prints full coefficients.
# All data (qb, df_train, df_val, df_test, X_train etc.) already in memory
# from Phase 3 of the passing_yards model above.
# TD_FEATURES is the full broad feature set — all QB-relevant rolling windows,
# NGS, team context, game environment. SHAP will determine what matters here.
# =============================================================================

print("\n" + "="*70)
print("PASSING TDs MODEL")
print("="*70)

# %%
# --- TD Feature set: full broad set ---
# All QB-relevant columns available in the dataset.
# Start wide; use SHAP to prune in subsequent iteration.

TD_FEATURES = [
    # Identity / context
    "depth_chart_rank", "week",

    # Fantasy points history (elite QB prior)
    "fantasy_pts_ewm10",
    "fantasy_pts_ewm20",
    "fantasy_pts_per_game_career",
    "passing_tds_per_game_career",
    "passing_tds_career_vs_recent",

    # Passing volume — L20 + ewm20 only (L10/ewm10 cut: middle window redundant)
    "passing_yards_L20",
    "passing_yards_ewm20",

    # TD history — L5/L10/L20; td_rate L10/L20 (L5 cut: lowest SHAP of trio)
    *[f"passing_tds_L{w}" for w in [5, 10, 20]],
    "td_rate_L10", "td_rate_L20",

    # EPA efficiency — epa_per_opportunity ewm20; epa_per_dropback all cut (overlap)
    # passing_epa L5 (short-term) + L20 (long-term); L10 cut
    "epa_per_opportunity_ewm20",
    "passing_epa_L5", "passing_epa_L20",

    # Completion accuracy — ewm20 level; raw L5/L10/L20 cut (redundant with CPOE + yards/EPA)
    "completion_pct_ewm20",

    # INT rate — L5 only (L10/L20 cut: redundant, low SHAP)
    "int_rate_L5",

    # Air yards — vertical tendency (ewm20 only)
    "qb_air_yards_per_attempt_ewm20",

    # Rushing TDs — mobile QB signal
    "rushing_tds_L20",

    # NGS — accuracy above expectation L5/L20; intended air yards L20
    "ngs_avg_intended_air_yards_L20",
    "ngs_completion_pct_above_exp_L5",
    "ngs_completion_pct_above_exp_L20",

    # Own offense — L5/L20 only (L10 cut); WR target share
    "off_epa_per_play_L5", "off_epa_per_play_L20",
    "off_pass_rate_L5", "off_pass_rate_L20",
    "off_wr_wopr_L5",

    # Opponent defense — L20 dominant; L5 for td_rate + cpoe (unique signal)
    # epa_per_attempt L5 cut (L20 dominates); opp_def_team_team cut
    "opp_def_qb_qb_epa_per_attempt_L20",
    "opp_def_qb_qb_cpoe_L5",
    "opp_def_qb_qb_td_rate_L5", "opp_def_qb_qb_td_rate_L20",

    # Game environment
    "game_location", "is_dome", "game_temp", "game_wind", "game_precip_mm",
]

# Filter to columns that actually exist after EWM computation
TD_FEATURES = [f for f in TD_FEATURES if f in qb.columns]
# Deduplicate preserving order
_seen_td = set()
TD_FEATURES = [f for f in TD_FEATURES if not (_seen_td.add(f) or f in _seen_td - {f})]
# simpler dedup
_seen_td = set(); _td_deduped = []
for _f in TD_FEATURES:
    if _f not in _seen_td:
        _td_deduped.append(_f); _seen_td.add(_f)
TD_FEATURES = _td_deduped

print(f"\nTD_FEATURES: {len(TD_FEATURES)} features")

# %%
# --- Build TD train/val/test splits ---

TD_TARGET = "passing_tds"

X_train_td = df_train[TD_FEATURES].reset_index(drop=True)
X_val_td   = df_val[TD_FEATURES].reset_index(drop=True)
X_test_td  = df_test[TD_FEATURES].reset_index(drop=True)

Y_train_td = df_train[[TD_TARGET]].reset_index(drop=True)
Y_val_td   = df_val[[TD_TARGET]].reset_index(drop=True)
Y_test_td  = df_test[[TD_TARGET]].reset_index(drop=True)

print(f"  X_train_td: {X_train_td.shape}  X_val_td: {X_val_td.shape}  X_test_td: {X_test_td.shape}")

# %%
# --- TD-4.1  Global mean baseline ---
print("\n--- TD-B1: Global mean ---")
_td_train_mean = float(Y_train_td[TD_TARGET].mean())
_td_preds_mean = np.full(len(Y_val_td), _td_train_mean)
_td_b1 = _metrics(Y_val_td[TD_TARGET], _td_preds_mean, "GlobalMean")
_td_b1["target"] = TD_TARGET
print(f"  {TD_TARGET:<25}  MAE={_td_b1['MAE']:.3f}  RMSE={_td_b1['RMSE']:.3f}  R2={_td_b1['R2']:+.3f}  Bias={_td_b1['Bias']:+.3f}")

# %%
# --- TD-4.2  L1-proxy baseline (passing_tds_L3) ---
print("\n--- TD-B2: L1-proxy (passing_tds_L3) ---")
_col = "passing_tds_L3"
if _col in X_val_td.columns:
    _td_preds_l1 = X_val_td[_col].fillna(_td_train_mean).values
else:
    _td_preds_l1 = np.full(len(Y_val_td), _td_train_mean)
    print(f"  passing_tds_L3 missing — using global mean fallback")
_td_b2 = _metrics(Y_val_td[TD_TARGET], _td_preds_l1, "L1-proxy")
_td_b2["target"] = TD_TARGET
print(f"  {TD_TARGET:<25}  MAE={_td_b2['MAE']:.3f}  RMSE={_td_b2['RMSE']:.3f}  R2={_td_b2['R2']:+.3f}  Bias={_td_b2['Bias']:+.3f}")

# %%
# --- TD-4.3  Rolling L5 mean baseline ---
print("\n--- TD-B3: Rolling L5 mean (passing_tds_L5) ---")
_col = "passing_tds_L5"
if _col in X_val_td.columns:
    _td_preds_l5 = X_val_td[_col].fillna(_td_train_mean).values
else:
    _td_preds_l5 = np.full(len(Y_val_td), _td_train_mean)
    print(f"  passing_tds_L5 missing — using global mean fallback")
_td_b3 = _metrics(Y_val_td[TD_TARGET], _td_preds_l5, "RollingL5")
_td_b3["target"] = TD_TARGET
print(f"  {TD_TARGET:<25}  MAE={_td_b3['MAE']:.3f}  RMSE={_td_b3['RMSE']:.3f}  R2={_td_b3['R2']:+.3f}  Bias={_td_b3['Bias']:+.3f}")

# %%
# --- TD-4.4  Ridge baseline ---
print("\n--- TD-B4: Ridge regression ---")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_td_col_medians = X_train_td.median().fillna(0)   # fallback 0 for all-NaN cols (pre-NGS era)
_Xtr_td_r = X_train_td.fillna(_td_col_medians).fillna(0)
_Xva_td_r = X_val_td.fillna(_td_col_medians).fillna(0)

_td_scaler = StandardScaler()
_Xtr_td_s  = _td_scaler.fit_transform(_Xtr_td_r)
_Xva_td_s  = _td_scaler.transform(_Xva_td_r)

_td_ridge = Ridge(alpha=10.0)
_td_ridge.fit(_Xtr_td_s, Y_train_td[TD_TARGET].values, sample_weight=sample_weights)
_td_preds_ridge = np.clip(_td_ridge.predict(_Xva_td_s), 0, None)

_td_b4 = _metrics(Y_val_td[TD_TARGET], _td_preds_ridge, "Ridge")
_td_b4["target"] = TD_TARGET
print(f"  {TD_TARGET:<25}  MAE={_td_b4['MAE']:.3f}  RMSE={_td_b4['RMSE']:.3f}  R2={_td_b4['R2']:+.3f}  Bias={_td_b4['Bias']:+.3f}")

_td_coef_df = (
    pd.DataFrame({"feature": TD_FEATURES, "coefficient": _td_ridge.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
    .reset_index(drop=True)
)
_td_coef_df.index = range(1, len(_td_coef_df) + 1)
print(f"\n  Ridge coefficients — {TD_TARGET} (standardized, sorted by |coef|):")
print(f"  {'Rank':<5} {'Feature':<50} {'Coefficient':>12}")
print(f"  {'-'*70}")
for rank, row in _td_coef_df.iterrows():
    print(f"  {rank:<5} {row['feature']:<50} {row['coefficient']:>+12.4f}")

# %%
# --- TD-4.5  Baseline summary ---
df_td_baselines = pd.DataFrame([_td_b1, _td_b2, _td_b3, _td_b4])

print("\n" + "="*60)
print(f"TD BASELINE SUMMARY — VAL 2024")
print("="*60)
sub = df_td_baselines[["baseline", "MAE", "RMSE", "R2", "Bias"]].set_index("baseline")
print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.3f}"))

# %%
# =============================================================================
# TD-PHASE 5  —  LightGBM: passing_tds (Poisson, rolling forward CV + Optuna)
# =============================================================================

TD_OOF_FIRST_VAL_YEAR = 2017
TD_OPTUNA_TUNE_YEAR   = 2023
TD_OBJ                = "poisson"
TD_CLIP               = True     # clip labels >=0 for poisson

print("\n" + "="*60)
print(f"TD-PHASE 5  --  LightGBM: {TD_TARGET}")
print("="*60)
print(f"  objective={TD_OBJ}  rolling CV: val {TD_OOF_FIRST_VAL_YEAR}-{TD_OPTUNA_TUNE_YEAR}")

# %%
# --- TD-5.1  Optuna: tune on train 2006-2022, val 2023 ---

_Xtr_opt_td, _Ytr_opt_td, _sw_opt_td, _Xva_opt_td, _Yva_opt_td = _cv_split(
    df_train, TD_OPTUNA_TUNE_YEAR, TD_FEATURES, [TD_TARGET]
)

_y_tr_opt_td = np.clip(_Ytr_opt_td[TD_TARGET].values.astype(float), 0, None)
_y_va_opt_td = np.clip(_Yva_opt_td[TD_TARGET].values.astype(float), 0, None)

_dtrain_opt_td = lgb.Dataset(_Xtr_opt_td, label=_y_tr_opt_td, weight=_sw_opt_td, free_raw_data=False)
_dval_opt_td   = lgb.Dataset(_Xva_opt_td, label=_y_va_opt_td, reference=_dtrain_opt_td, free_raw_data=False)

def _make_td_objective(dtrain, dval, X_va, Y_va):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         TD_OBJ,
            "metric":            "mae",
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs": -1, "seed": 42,
        }
        _b = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        return mean_absolute_error(Y_va[TD_TARGET], np.clip(_b.predict(X_va), 0, None))
    return objective_fn

TD_N_TRIALS = 60
td_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
_td_progress = _OptunaProgress(TD_N_TRIALS, TD_TARGET)
td_study.optimize(
    _make_td_objective(_dtrain_opt_td, _dval_opt_td, _Xva_opt_td, _Yva_opt_td),
    n_trials=TD_N_TRIALS,
    show_progress_bar=False,
    callbacks=[_td_progress],
)
_td_progress.close()

td_best_params = td_study.best_params
print(f"\n  Optuna best MAE: {td_study.best_value:.4f}  (trial {td_study.best_trial.number}/{TD_N_TRIALS})")
print(f"  Best params: {td_best_params}")

# %%
# --- TD-5.2  Rolling forward CV with best_params ---

td_final_params = {
    "verbosity": -1, "objective": TD_OBJ, "metric": "mae",
    "n_jobs": -1, "seed": 42,
    **{k: v for k, v in td_best_params.items() if k != "extra_trees"},
    "extra_trees": td_best_params["extra_trees"],
}

td_cv_val_years = list(range(TD_OOF_FIRST_VAL_YEAR, TRAIN_END + 1))
print(f"\n  Rolling forward CV: {len(td_cv_val_years)} folds  ({td_cv_val_years[0]}-{td_cv_val_years[-1]})")

td_oof_actual = []; td_oof_pred = []; td_oof_years = []; td_best_iters = []

for _yr in td_cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, TD_FEATURES, [TD_TARGET])
    _y_tr = np.clip(_Ytr[TD_TARGET].values.astype(float), 0, None)
    _y_va = np.clip(_Yva[TD_TARGET].values.astype(float), 0, None)
    _dt = lgb.Dataset(_Xtr, label=_y_tr, weight=_sw, free_raw_data=False)
    _dv = lgb.Dataset(_Xva, label=_y_va, reference=_dt, free_raw_data=False)
    _b = lgb.train(
        td_final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
    )
    _p = np.clip(_b.predict(_Xva), 0, None)
    td_oof_actual.extend(_Yva[TD_TARGET].tolist())
    td_oof_pred.extend(_p.tolist())
    td_oof_years.extend([_yr] * len(_Yva))
    td_best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"MAE={mean_absolute_error(_Yva[TD_TARGET], _p):.3f}")

td_oof_actual    = np.array(td_oof_actual)
td_oof_pred      = np.array(td_oof_pred)
td_oof_years     = np.array(td_oof_years)
td_mean_best_iter = int(np.mean(td_best_iters))
print(f"\n  Mean best_iteration across folds: {td_mean_best_iter}")

# %%
# --- TD-5.3  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"TD OOF METRICS (rolling forward CV)  --  {TD_TARGET}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in td_cv_val_years:
    _mask = td_oof_years == _yr
    _m    = _metrics(td_oof_actual[_mask], td_oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.3f}  {_m['RMSE']:>8.3f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.3f}")

_td_m_overall = _metrics(td_oof_actual, td_oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(td_oof_actual):>5}  "
      f"{_td_m_overall['MAE']:>8.3f}  {_td_m_overall['RMSE']:>8.3f}  "
      f"{_td_m_overall['R2']:>+8.3f}  {_td_m_overall['Bias']:>+8.3f}")

# %%
# --- TD-5.4  Final model: train 2006-2023, n_trees = mean best_iter ---

print(f"\n  Final TD model: train 2006-{TRAIN_END}, n_trees={td_mean_best_iter}")

_y_tr_full = np.clip(Y_train_td[TD_TARGET].values.astype(float), 0, None)
_dtrain_td_final = lgb.Dataset(X_train_td, label=_y_tr_full, weight=sample_weights, free_raw_data=False)

td_booster_final = lgb.train(
    {**td_final_params, "metric": "none"},
    _dtrain_td_final,
    num_boost_round=td_mean_best_iter,
    callbacks=[log_evaluation(-1)],
)

# Val 2024 honest metrics
_td_preds_val = np.clip(td_booster_final.predict(X_val_td), 0, None)
_td_m_val = _metrics(Y_val_td[TD_TARGET], _td_preds_val, "LightGBM")
print(f"  Val 2024 -> MAE={_td_m_val['MAE']:.3f}  RMSE={_td_m_val['RMSE']:.3f}  "
      f"R2={_td_m_val['R2']:+.3f}  Bias={_td_m_val['Bias']:+.3f}")

# %%
# --- TD-5.5  Summary vs baselines ---
print("\n" + "="*60)
print(f"TD-PHASE 5 SUMMARY -- {TD_TARGET}")
print("="*60)
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<10}  {float(_td_m_overall[metric]):>10.3f}  "
          f"{_td_m_val[metric]:>10.3f}  {_td_b4[metric]:>12.3f}")

print(f"\nTD-Phase 5 complete.")

# Save to registry
lgb_models["passing_tds"]      = td_booster_final
best_params_all["passing_tds"] = td_best_params
oof_store["passing_tds"] = {
    "oof_actual": td_oof_actual,
    "oof_pred":   td_oof_pred,
    "oof_years":  td_oof_years,
}
joblib.dump({
    "lgb_models":     lgb_models,
    "optuna_studies": optuna_studies,
    "df_phase5":      pd.DataFrame(phase5_records),
    "best_params":    best_params_all,
    "oof_store":      oof_store,
    "feature_cols":   FEATURE_COLS,
    "td_feature_cols": TD_FEATURES,
    "targets":        TARGETS,
    "loss_functions": LOSS_FUNCTIONS,
    "tweedie_power":  TWEEDIE_POWER,
}, registry_path)
print(f"  Registry updated.")

# %%
# =============================================================================
# TD-PHASE 6  —  EVALUATION & DIAGNOSTICS: passing_tds
# =============================================================================

print("\n" + "="*60)
print(f"TD-PHASE 6  --  EVALUATION: {TD_TARGET}")
print("="*60)

_td_preds_test = np.clip(td_booster_final.predict(X_test_td), 0, None)

# %%
# --- TD-6.1  Full metrics table ---

print(f"\n  [{TD_TARGET}]")
for split, yt, yp in [("Val 2024", Y_val_td[TD_TARGET].values, _td_preds_val),
                       ("Test 2025", Y_test_td[TD_TARGET].values, _td_preds_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- TD-6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_td[TD_TARGET].values,  _td_preds_val),
    ("Test 2025", Y_test_td[TD_TARGET].values, _td_preds_test),
]):
    lim = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"Passing TDs -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual TDs", fontsize=9)
    ax.set_ylabel("Predicted TDs", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{TD_TARGET}_pred_vs_actual.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- TD-6.3  Residual distribution ---

from scipy.stats import gaussian_kde as _gkde, norm as _sp_norm
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val_td[TD_TARGET].values,  _td_preds_val),
    ("Test 2025",   Y_test_td[TD_TARGET].values, _td_preds_test),
    ("OOF 2017-23", td_oof_actual,               td_oof_pred),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals — Passing TDs {split}\nbias={resid.mean():+.3f}  sd={resid.std():.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{TD_TARGET}_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- TD-6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_td[TD_TARGET].values,  _td_preds_val),
    ("Test 2025", Y_test_td[TD_TARGET].values, _td_preds_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="steelblue", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Passing TDs Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{TD_TARGET}_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- TD-6.5  MAE by week (test 2025) ---

try:
    _td_week_df = pd.DataFrame({
        "week":   df_test["week"].values,
        "actual": Y_test_td[TD_TARGET].values,
        "pred":   _td_preds_test,
    })
    if len(_td_week_df) == len(df_test):
        _td_wk = (_td_week_df.groupby("week")
                  .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]))
                  .reset_index(name="MAE"))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(_td_wk["week"], _td_wk["MAE"], color="steelblue", edgecolor="white")
        ax.set_title(f"MAE by Week — Passing TDs (Test 2025)", fontsize=11, fontweight="bold")
        ax.set_xlabel("NFL Week", fontsize=9)
        ax.set_ylabel("MAE", fontsize=9)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        _fig_path = FIG_DIR / f"phase6_{TD_TARGET}_mae_by_week.png"
        plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
        plt.close()
        _show(_fig_path)
        print(f"  Saved: {_fig_path.name}")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- TD-6.6  Export test predictions ---

_td_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_td_export_df["actual_passing_tds"]    = Y_test_td[TD_TARGET].values
_td_export_df["predicted_passing_tds"] = _td_preds_test
_td_export_df["residual"]              = _td_export_df["predicted_passing_tds"] - _td_export_df["actual_passing_tds"]
_td_export_df = _td_export_df.sort_values(["week", "actual_passing_tds"], ascending=[True, False]).reset_index(drop=True)

_td_export_path = DATA_DIR / f"test_predictions_2025_{TD_TARGET}.xlsx"
_td_export_df.to_excel(_td_export_path, index=False)
print(f"\n  2025 test predictions saved: {_td_export_path.name}  ({len(_td_export_df)} rows)")

print(f"\nTD-Phase 6 complete.")

# %%
# =============================================================================
# TD-PHASE 7  —  FEATURE IMPORTANCE & SHAP: passing_tds
# =============================================================================

print("\n" + "="*60)
print(f"TD-PHASE 7  --  FEATURE IMPORTANCE: {TD_TARGET}")
print("="*60)

import shap as _shap_lib

# %%
# --- TD-7.1  LightGBM native gain importance ---

_td_imp_gain = (
    pd.Series(td_booster_final.feature_importance(importance_type="gain"),
              index=TD_FEATURES)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
_td_imp_gain.head(25).plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"LightGBM Gain Importance (top 25) — {TD_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Total Gain", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{TD_TARGET}_gain_importance.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- TD-7.2  SHAP ---

_td_sample_idx = np.random.default_rng(42).choice(len(X_test_td), size=min(500, len(X_test_td)), replace=False)
_td_X_shap     = X_test_td.iloc[_td_sample_idx].reset_index(drop=True)

_td_explainer   = _shap_lib.TreeExplainer(td_booster_final)
_td_shap_values = _td_explainer.shap_values(_td_X_shap)

_td_mean_abs_shap = pd.Series(
    np.abs(_td_shap_values).mean(axis=0),
    index=TD_FEATURES,
).sort_values(ascending=False)

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 9))
_shap_lib.summary_plot(_td_shap_values, _td_X_shap, max_display=20, show=False)
plt.title(f"SHAP Beeswarm — {TD_TARGET} (top 20)", fontsize=11, fontweight="bold")
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{TD_TARGET}_shap_beeswarm.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# Bar plot
fig, ax = plt.subplots(figsize=(10, 8))
_td_mean_abs_shap.head(20).plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"Mean |SHAP| (top 20) — {TD_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{TD_TARGET}_shap_bar.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- TD-7.3  Full importance table (all features) ---

_td_gain_df = _td_imp_gain.reset_index(); _td_gain_df.columns = ["feature", "gain"]
_td_shap_df = _td_mean_abs_shap.reset_index(); _td_shap_df.columns = ["feature", "mean_abs_shap"]
_td_gain_df["gain_rank"] = range(1, len(_td_gain_df) + 1)
_td_shap_df["shap_rank"] = range(1, len(_td_shap_df) + 1)

_td_full_imp = _td_gain_df.merge(_td_shap_df, on="feature")[
    ["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]
].sort_values("shap_rank").reset_index(drop=True)

print(f"\n  {'gain_rank':>10}  {'shap_rank':>10}  {'feature':<50}  {'gain':>14}  {'mean_abs_shap':>14}")
print(f"  {'-'*100}")
for _, r in _td_full_imp.iterrows():
    print(f"  {int(r['gain_rank']):>10}  {int(r['shap_rank']):>10}  {r['feature']:<50}  "
          f"{r['gain']:>14.2f}  {r['mean_abs_shap']:>14.6f}")

_td_imp_path = DATA_DIR / f"feature_importance_{TD_TARGET}.xlsx"
_td_full_imp.to_excel(_td_imp_path, index=False)
print(f"\n  Full importance saved: {_td_imp_path.name}")

print(f"\nTD-Phase 7 complete.")

# %%
# =============================================================================
# TD-PHASE 7B  —  SEASON-LEVEL DIAGNOSTICS: passing_tds
# =============================================================================

print("\n" + "="*60)
print(f"TD-PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {TD_TARGET}")
print("="*60)

_td_pg = _td_export_df.copy()

_td_szn = (
    _td_pg.groupby("player_display_name")
    .agg(
        games               =("actual_passing_tds",    "count"),
        actual_total        =("actual_passing_tds",    "sum"),
        predicted_total     =("predicted_passing_tds", "sum"),
        depth_chart_rank_min=("depth_chart_rank",      "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_td_szn["residual"]  = _td_szn["predicted_total"] - _td_szn["actual_total"]
_td_szn["pct_error"] = np.where(
    _td_szn["actual_total"] == 0, np.nan,
    (_td_szn["predicted_total"] - _td_szn["actual_total"]) / _td_szn["actual_total"]
)

_yt_s = _td_szn["actual_total"].values.astype(float)
_yp_s = _td_szn["predicted_total"].values.astype(float)
_td_mae_s    = mean_absolute_error(_yt_s, _yp_s)
_td_rmse_s   = float(np.sqrt(np.mean((_yp_s - _yt_s) ** 2)))
_td_r2_s     = float(r2_score(_yt_s, _yp_s))
_td_bias_s   = float(np.mean(_yp_s - _yt_s))
_td_mape_s   = float(np.mean(np.abs(_td_szn["pct_error"].dropna())))
_td_r_s      = float(np.corrcoef(_yt_s, _yp_s)[0, 1])
_td_w10_s    = float((_td_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_td_w20_s    = float((_td_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  Season-level fit diagnostics ({len(_td_szn)} QBs):")
print(f"  {'MAE (TDs)':<20}  {_td_mae_s:>10.3f}")
print(f"  {'RMSE (TDs)':<20}  {_td_rmse_s:>10.3f}")
print(f"  {'R2':<20}  {_td_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_td_bias_s:>10.3f}")
print(f"  {'MAPE':<20}  {_td_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_td_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_td_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_td_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _td_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.1f}  "
          f"{row.predicted_total:>8.2f}  {row.residual:>+8.2f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

_td_szn_path = DATA_DIR / f"test_predictions_2025_{TD_TARGET}_season_totals.xlsx"
_td_szn.to_excel(_td_szn_path, index=False)
print(f"\n  Season totals saved: {_td_szn_path.name}")
print(f"\nTD model complete.")

# %%
# =============================================================================
# =============================================================================
# RUSHING YARDS MODEL
# =============================================================================
# =============================================================================

# %%
# =============================================================================
# RY-PHASE 4  —  BASELINES: rushing_yards
# =============================================================================
# All data (qb, df_train, df_val, df_test) already in memory from Phase 3.
# Objective: Tweedie (right-skewed continuous, skew=2.55, many near-zero games)
# =============================================================================

print("\n" + "="*70)
print("RUSHING YARDS MODEL")
print("="*70)

# %%
# --- RY Feature set ---

RY_FEATURES = [
    # Identity / context (depth_chart_rank removed: near-zero SHAP)
    "week", "age",

    # Fantasy points history
    "fantasy_pts_ewm10",
    # fantasy_pts_ewm20 removed: 0.035 SHAP, redundant with ewm10 + career
    "fantasy_pts_per_game_career",

    # Career baselines — anchor against injury-year recency drag
    "rushing_yards_per_game_career",
    "carries_per_game_career",
    "rushing_epa_per_game_career",

    # Regression-to-mean gaps (career minus current EWM form)
    # Positive = performing below career level → rebound signal
    "rushing_yards_career_vs_recent",
    "carries_career_vs_recent",
    "rushing_epa_career_vs_recent",

    # Physical attributes
    "forty_yard_dash", "speed_score",

    # QB rushing volume — L windows (L5 removed: too noisy, recency bias)
    "rushing_yards_L10", "rushing_yards_L20",
    "carries_L20",

    # QB rushing efficiency — L windows
    "rushing_epa_L20",
    "rushing_tds_L10", "rushing_tds_L20",

    # QB rushing history — EWM
    "rushing_yards_ewm20",
    "carries_ewm10", "carries_ewm20",
    "rushing_epa_ewm20",

    # Pressure tendency
    "qb_pressure_rate_L20",
    "qb_pressure_rate_ewm20",

    # Passing efficiency as game-script signal
    "passing_epa_L20",
    "epa_per_dropback_ewm20",

    # Opponent run defense (opp_def_team_team_epa_per_rush_L20 removed: 0.031 SHAP;
    # opp_def_run_epa_L20 removed: 0.053 SHAP, redundant with L10)
    "opp_def_rb_rb_rush_epa_per_carry_L20",
    "opp_def_run_epa_L10",

    # Own offense run context (off_rb_carry_share_L20 removed: 0.035 SHAP, redundant with L10;
    # off_epa_per_play_L20 removed: 0.058 SHAP, captured by passing_epa + epa_per_dropback)
    "off_pass_rate_L20",
    "off_rb_carry_share_L10",
    "off_rb_yards_per_carry_L20",
    "off_rb_epa_per_carry_L10", "off_rb_epa_per_carry_L20",

    # Game environment
    "game_location", "game_temp", "game_precip_mm",
]

# Filter to columns that exist, deduplicate
RY_FEATURES = [f for f in RY_FEATURES if f in qb.columns]
_seen_ry = set(); _ry_deduped = []
for _f in RY_FEATURES:
    if _f not in _seen_ry:
        _ry_deduped.append(_f); _seen_ry.add(_f)
RY_FEATURES = _ry_deduped

print(f"\nRY_FEATURES: {len(RY_FEATURES)} features")

# %%
# --- Build RY train/val/test splits ---

RY_TARGET = "rushing_yards"

X_train_ry = df_train[RY_FEATURES].reset_index(drop=True)
X_val_ry   = df_val[RY_FEATURES].reset_index(drop=True)
X_test_ry  = df_test[RY_FEATURES].reset_index(drop=True)

Y_train_ry = df_train[[RY_TARGET]].reset_index(drop=True)
Y_val_ry   = df_val[[RY_TARGET]].reset_index(drop=True)
Y_test_ry  = df_test[[RY_TARGET]].reset_index(drop=True)

print(f"  X_train_ry: {X_train_ry.shape}  X_val_ry: {X_val_ry.shape}  X_test_ry: {X_test_ry.shape}")

# %%
# --- RY-4.1  Global mean baseline ---
print("\n--- RY-B1: Global mean ---")
_ry_train_mean = float(Y_train_ry[RY_TARGET].mean())
_ry_preds_mean = np.full(len(Y_val_ry), _ry_train_mean)
_ry_b1 = _metrics(Y_val_ry[RY_TARGET], _ry_preds_mean, "GlobalMean")
_ry_b1["target"] = RY_TARGET
print(f"  {RY_TARGET:<25}  MAE={_ry_b1['MAE']:.3f}  RMSE={_ry_b1['RMSE']:.3f}  R2={_ry_b1['R2']:+.3f}  Bias={_ry_b1['Bias']:+.3f}")

# %%
# --- RY-4.2  L1-proxy baseline (rushing_yards_L3) ---
print("\n--- RY-B2: L1-proxy (rushing_yards_L3) ---")
_col = "rushing_yards_L3"
if _col in df_val.columns:
    _ry_preds_l1 = df_val[_col].fillna(_ry_train_mean).reset_index(drop=True).values
else:
    _ry_preds_l1 = np.full(len(Y_val_ry), _ry_train_mean)
    print(f"  rushing_yards_L3 missing — using global mean fallback")
_ry_b2 = _metrics(Y_val_ry[RY_TARGET], _ry_preds_l1, "L1-proxy")
_ry_b2["target"] = RY_TARGET
print(f"  {RY_TARGET:<25}  MAE={_ry_b2['MAE']:.3f}  RMSE={_ry_b2['RMSE']:.3f}  R2={_ry_b2['R2']:+.3f}  Bias={_ry_b2['Bias']:+.3f}")

# %%
# --- RY-4.3  Rolling L5 mean baseline ---
print("\n--- RY-B3: Rolling L5 mean (rushing_yards_L5) ---")
_col = "rushing_yards_L5"
if _col in df_val.columns:
    _ry_preds_l5 = df_val[_col].fillna(_ry_train_mean).reset_index(drop=True).values
else:
    _ry_preds_l5 = np.full(len(Y_val_ry), _ry_train_mean)
    print(f"  rushing_yards_L5 missing — using global mean fallback")
_ry_b3 = _metrics(Y_val_ry[RY_TARGET], _ry_preds_l5, "RollingL5")
_ry_b3["target"] = RY_TARGET
print(f"  {RY_TARGET:<25}  MAE={_ry_b3['MAE']:.3f}  RMSE={_ry_b3['RMSE']:.3f}  R2={_ry_b3['R2']:+.3f}  Bias={_ry_b3['Bias']:+.3f}")

# %%
# --- RY-4.4  Ridge baseline ---
print("\n--- RY-B4: Ridge regression ---")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_ry_col_medians = X_train_ry.median().fillna(0)
_Xtr_ry_r = X_train_ry.fillna(_ry_col_medians).fillna(0)
_Xva_ry_r = X_val_ry.fillna(_ry_col_medians).fillna(0)

_ry_scaler = StandardScaler()
_Xtr_ry_s  = _ry_scaler.fit_transform(_Xtr_ry_r)
_Xva_ry_s  = _ry_scaler.transform(_Xva_ry_r)

_ry_ridge = Ridge(alpha=10.0)
_ry_ridge.fit(_Xtr_ry_s, Y_train_ry[RY_TARGET].values, sample_weight=sample_weights)
_ry_preds_ridge = np.clip(_ry_ridge.predict(_Xva_ry_s), 0, None)

_ry_b4 = _metrics(Y_val_ry[RY_TARGET], _ry_preds_ridge, "Ridge")
_ry_b4["target"] = RY_TARGET
print(f"  {RY_TARGET:<25}  MAE={_ry_b4['MAE']:.3f}  RMSE={_ry_b4['RMSE']:.3f}  R2={_ry_b4['R2']:+.3f}  Bias={_ry_b4['Bias']:+.3f}")

_ry_coef_df = (
    pd.DataFrame({"feature": RY_FEATURES, "coefficient": _ry_ridge.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
    .reset_index(drop=True)
)
_ry_coef_df.index = range(1, len(_ry_coef_df) + 1)
print(f"\n  Ridge coefficients — {RY_TARGET} (standardized, sorted by |coef|):")
print(f"  {'Rank':<5} {'Feature':<55} {'Coefficient':>12}")
print(f"  {'-'*75}")
for rank, row in _ry_coef_df.iterrows():
    print(f"  {rank:<5} {row['feature']:<55} {row['coefficient']:>+12.4f}")

# %%
# --- RY-4.5  Baseline summary ---
df_ry_baselines = pd.DataFrame([_ry_b1, _ry_b2, _ry_b3, _ry_b4])

print("\n" + "="*60)
print(f"RY BASELINE SUMMARY — VAL 2024")
print("="*60)
sub = df_ry_baselines[["baseline", "MAE", "RMSE", "R2", "Bias"]].set_index("baseline")
print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.3f}"))

# %%
# =============================================================================
# RY-PHASE 5  —  LightGBM: rushing_yards (Tweedie, rolling forward CV + Optuna)
# =============================================================================

RY_OOF_FIRST_VAL_YEAR = 2017
RY_OPTUNA_TUNE_YEAR   = 2023
RY_OBJ                = "regression"   # 12.5% negatives — Tweedie invalid; MSE handles negatives natively
RY_CLIP               = False          # do NOT clip — negatives are real and must be preserved

print("\n" + "="*60)
print(f"RY-PHASE 5  --  LightGBM: {RY_TARGET}")
print("="*60)
print(f"  objective={RY_OBJ}  rolling CV: val {RY_OOF_FIRST_VAL_YEAR}-{RY_OPTUNA_TUNE_YEAR}")

# %%
# --- RY-5.1  Optuna: tune on train 2006-2022, val 2023 ---

_Xtr_opt_ry, _Ytr_opt_ry, _sw_opt_ry, _Xva_opt_ry, _Yva_opt_ry = _cv_split(
    df_train, RY_OPTUNA_TUNE_YEAR, RY_FEATURES, [RY_TARGET]
)

_y_tr_opt_ry = _Ytr_opt_ry[RY_TARGET].values.astype(float)   # no clip — negatives are real
_y_va_opt_ry = _Yva_opt_ry[RY_TARGET].values.astype(float)

_dtrain_opt_ry = lgb.Dataset(_Xtr_opt_ry, label=_y_tr_opt_ry, weight=_sw_opt_ry, free_raw_data=False)
_dval_opt_ry   = lgb.Dataset(_Xva_opt_ry, label=_y_va_opt_ry, reference=_dtrain_opt_ry, free_raw_data=False)

def _make_ry_objective(dtrain, dval, X_va, Y_va):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         RY_OBJ,
            "metric":            "mae",
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs": -1, "seed": 42,
        }
        _b = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        return mean_absolute_error(Y_va[RY_TARGET], _b.predict(X_va))
    return objective_fn

RY_N_TRIALS = 60
ry_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
_ry_progress = _OptunaProgress(RY_N_TRIALS, RY_TARGET)
ry_study.optimize(
    _make_ry_objective(_dtrain_opt_ry, _dval_opt_ry, _Xva_opt_ry, _Yva_opt_ry),
    n_trials=RY_N_TRIALS,
    show_progress_bar=False,
    callbacks=[_ry_progress],
)
_ry_progress.close()

ry_best_params = ry_study.best_params
print(f"\n  Optuna best MAE: {ry_study.best_value:.4f}  (trial {ry_study.best_trial.number}/{RY_N_TRIALS})")
print(f"  Best params: {ry_best_params}")

# %%
# --- RY-5.2  Rolling forward CV with best_params ---

ry_final_params = {
    "verbosity": -1, "objective": RY_OBJ,
    "metric": "mae", "n_jobs": -1, "seed": 42,
    **{k: v for k, v in ry_best_params.items() if k != "extra_trees"},
    "extra_trees": ry_best_params["extra_trees"],
}

ry_cv_val_years = list(range(RY_OOF_FIRST_VAL_YEAR, TRAIN_END + 1))
print(f"\n  Rolling forward CV: {len(ry_cv_val_years)} folds  ({ry_cv_val_years[0]}-{ry_cv_val_years[-1]})")

ry_oof_actual = []; ry_oof_pred = []; ry_oof_years = []; ry_best_iters = []

for _yr in ry_cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, RY_FEATURES, [RY_TARGET])
    _y_tr = _Ytr[RY_TARGET].values.astype(float)   # no clip
    _y_va = _Yva[RY_TARGET].values.astype(float)
    _dt = lgb.Dataset(_Xtr, label=_y_tr, weight=_sw, free_raw_data=False)
    _dv = lgb.Dataset(_Xva, label=_y_va, reference=_dt, free_raw_data=False)
    _b = lgb.train(
        ry_final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
    )
    _p = _b.predict(_Xva)
    ry_oof_actual.extend(_Yva[RY_TARGET].tolist())
    ry_oof_pred.extend(_p.tolist())
    ry_oof_years.extend([_yr] * len(_Yva))
    ry_best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"MAE={mean_absolute_error(_Yva[RY_TARGET], _p):.3f}")

ry_oof_actual    = np.array(ry_oof_actual)
ry_oof_pred      = np.array(ry_oof_pred)
ry_oof_years     = np.array(ry_oof_years)
ry_mean_best_iter = int(np.mean(ry_best_iters))
print(f"\n  Mean best_iteration across folds: {ry_mean_best_iter}")

# %%
# --- RY-5.3  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"RY OOF METRICS (rolling forward CV)  --  {RY_TARGET}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in ry_cv_val_years:
    _mask = ry_oof_years == _yr
    _m    = _metrics(ry_oof_actual[_mask], ry_oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.3f}  {_m['RMSE']:>8.3f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.3f}")

_ry_m_overall = _metrics(ry_oof_actual, ry_oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(ry_oof_actual):>5}  "
      f"{_ry_m_overall['MAE']:>8.3f}  {_ry_m_overall['RMSE']:>8.3f}  "
      f"{_ry_m_overall['R2']:>+8.3f}  {_ry_m_overall['Bias']:>+8.3f}")

# %%
# --- RY-5.4  Final model: train 2006-2023, n_trees = mean best_iter ---

print(f"\n  Final RY model: train 2006-{TRAIN_END}, n_trees={ry_mean_best_iter}")

_y_tr_ry_full = Y_train_ry[RY_TARGET].values.astype(float)   # no clip — negatives are real
_dtrain_ry_final = lgb.Dataset(X_train_ry, label=_y_tr_ry_full, weight=sample_weights, free_raw_data=False)

ry_booster_final = lgb.train(
    {**ry_final_params, "metric": "none"},
    _dtrain_ry_final,
    num_boost_round=ry_mean_best_iter,
    callbacks=[log_evaluation(-1)],
)

_ry_preds_val = ry_booster_final.predict(X_val_ry)
_ry_m_val = _metrics(Y_val_ry[RY_TARGET], _ry_preds_val, "LightGBM")
print(f"  Val 2024 -> MAE={_ry_m_val['MAE']:.3f}  RMSE={_ry_m_val['RMSE']:.3f}  "
      f"R2={_ry_m_val['R2']:+.3f}  Bias={_ry_m_val['Bias']:+.3f}")

# %%
# --- RY-5.5  Summary vs baselines ---
print("\n" + "="*60)
print(f"RY-PHASE 5 SUMMARY -- {RY_TARGET}")
print("="*60)
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<10}  {float(_ry_m_overall[metric]):>10.3f}  "
          f"{_ry_m_val[metric]:>10.3f}  {_ry_b4[metric]:>12.3f}")

print(f"\nRY-Phase 5 complete.")

# Save to registry
lgb_models["rushing_yards"]      = ry_booster_final
best_params_all["rushing_yards"] = ry_best_params
oof_store["rushing_yards"] = {
    "oof_actual": ry_oof_actual,
    "oof_pred":   ry_oof_pred,
    "oof_years":  ry_oof_years,
}
joblib.dump({
    "lgb_models":     lgb_models,
    "optuna_studies": optuna_studies,
    "df_phase5":      pd.DataFrame(phase5_records),
    "best_params":    best_params_all,
    "oof_store":      oof_store,
    "feature_cols":   FEATURE_COLS,
    "td_feature_cols": TD_FEATURES,
    "ry_feature_cols": RY_FEATURES,
    "targets":        TARGETS,
    "loss_functions": LOSS_FUNCTIONS,
    "tweedie_power":  TWEEDIE_POWER,
}, registry_path)
print(f"  Registry updated.")

# %%
# =============================================================================
# RY-PHASE 6  —  EVALUATION & DIAGNOSTICS: rushing_yards
# =============================================================================

print("\n" + "="*60)
print(f"RY-PHASE 6  --  EVALUATION: {RY_TARGET}")
print("="*60)

_ry_preds_test = ry_booster_final.predict(X_test_ry)

# %%
# --- RY-6.1  Full metrics table ---

print(f"\n  [{RY_TARGET}]")
for split, yt, yp in [("Val 2024", Y_val_ry[RY_TARGET].values, _ry_preds_val),
                      ("Test 2025", Y_test_ry[RY_TARGET].values, _ry_preds_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- RY-6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_ry[RY_TARGET].values,  _ry_preds_val),
    ("Test 2025", Y_test_ry[RY_TARGET].values, _ry_preds_test),
]):
    lim = max(yt.max(), yp.max()) * 1.05
    ax.scatter(yt, yp, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"Rushing Yards -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Rushing Yards", fontsize=9)
    ax.set_ylabel("Predicted Rushing Yards", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RY_TARGET}_pred_vs_actual.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RY-6.3  Residual distribution ---

from scipy.stats import gaussian_kde as _gkde_ry, norm as _sp_norm_ry
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val_ry[RY_TARGET].values,  _ry_preds_val),
    ("Test 2025",   Y_test_ry[RY_TARGET].values, _ry_preds_test),
    ("OOF 2017-23", ry_oof_actual,               ry_oof_pred),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde_ry(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm_ry.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals — Rushing Yards {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RY_TARGET}_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RY-6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_ry[RY_TARGET].values,  _ry_preds_val),
    ("Test 2025", Y_test_ry[RY_TARGET].values, _ry_preds_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="steelblue", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Rushing Yards Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RY_TARGET}_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RY-6.5  MAE by week (test 2025) ---

try:
    _ry_week_df = pd.DataFrame({
        "week":   df_test["week"].values,
        "actual": Y_test_ry[RY_TARGET].values,
        "pred":   _ry_preds_test,
    })
    _ry_wk = (_ry_week_df.groupby("week")
              .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]))
              .reset_index(name="MAE"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(_ry_wk["week"], _ry_wk["MAE"], color="steelblue", edgecolor="white")
    ax.set_title(f"MAE by Week — Rushing Yards (Test 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("NFL Week", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _fig_path = FIG_DIR / f"phase6_{RY_TARGET}_mae_by_week.png"
    plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    _show(_fig_path)
    print(f"  Saved: {_fig_path.name}")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- RY-6.6  Export test predictions ---

_ry_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_ry_export_df["actual_rushing_yards"]    = Y_test_ry[RY_TARGET].values
_ry_export_df["predicted_rushing_yards"] = _ry_preds_test
_ry_export_df["residual"]               = _ry_export_df["predicted_rushing_yards"] - _ry_export_df["actual_rushing_yards"]
_ry_export_df = _ry_export_df.sort_values(["week", "actual_rushing_yards"], ascending=[True, False]).reset_index(drop=True)

_ry_export_path = DATA_DIR / f"test_predictions_2025_{RY_TARGET}.xlsx"
_ry_export_df.to_excel(_ry_export_path, index=False)
print(f"\n  2025 test predictions saved: {_ry_export_path.name}  ({len(_ry_export_df)} rows)")
print(f"\nRY-Phase 6 complete.")

# %%
# =============================================================================
# RY-PHASE 7  —  FEATURE IMPORTANCE & SHAP: rushing_yards
# =============================================================================

print("\n" + "="*60)
print(f"RY-PHASE 7  --  FEATURE IMPORTANCE: {RY_TARGET}")
print("="*60)

import shap as _shap_lib

# %%
# --- RY-7.1  LightGBM native gain importance ---

_ry_imp_gain = (
    pd.Series(ry_booster_final.feature_importance(importance_type="gain"),
              index=RY_FEATURES)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
_ry_imp_gain.head(25).plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"LightGBM Gain Importance (top 25) — {RY_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Total Gain", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RY_TARGET}_gain_importance.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RY-7.2  SHAP ---

_ry_sample_idx = np.random.default_rng(42).choice(len(X_test_ry), size=min(500, len(X_test_ry)), replace=False)
_ry_X_shap     = X_test_ry.iloc[_ry_sample_idx].reset_index(drop=True)

_ry_explainer   = _shap_lib.TreeExplainer(ry_booster_final)
_ry_shap_values = _ry_explainer.shap_values(_ry_X_shap)

_ry_mean_abs_shap = pd.Series(
    np.abs(_ry_shap_values).mean(axis=0),
    index=RY_FEATURES,
).sort_values(ascending=False)

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 9))
_shap_lib.summary_plot(_ry_shap_values, _ry_X_shap, max_display=20, show=False)
plt.title(f"SHAP Beeswarm — {RY_TARGET} (top 20)", fontsize=11, fontweight="bold")
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RY_TARGET}_shap_beeswarm.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# Bar plot
fig, ax = plt.subplots(figsize=(10, 8))
_ry_mean_abs_shap.head(20).plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"Mean |SHAP| (top 20) — {RY_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RY_TARGET}_shap_bar.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RY-7.3  Full importance table (all features) ---

_ry_gain_df = _ry_imp_gain.reset_index(); _ry_gain_df.columns = ["feature", "gain"]
_ry_shap_df = _ry_mean_abs_shap.reset_index(); _ry_shap_df.columns = ["feature", "mean_abs_shap"]
_ry_gain_df["gain_rank"] = range(1, len(_ry_gain_df) + 1)
_ry_shap_df["shap_rank"] = range(1, len(_ry_shap_df) + 1)

_ry_full_imp = _ry_gain_df.merge(_ry_shap_df, on="feature")[
    ["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]
].sort_values("shap_rank").reset_index(drop=True)

print(f"\n  {'gain_rank':>10}  {'shap_rank':>10}  {'feature':<55}  {'gain':>14}  {'mean_abs_shap':>14}")
print(f"  {'-'*105}")
for _, r in _ry_full_imp.iterrows():
    print(f"  {int(r['gain_rank']):>10}  {int(r['shap_rank']):>10}  {r['feature']:<55}  "
          f"{r['gain']:>14.2f}  {r['mean_abs_shap']:>14.6f}")

_ry_imp_path = DATA_DIR / f"feature_importance_{RY_TARGET}.xlsx"
_ry_full_imp.to_excel(_ry_imp_path, index=False)
print(f"\n  Full importance saved: {_ry_imp_path.name}")
print(f"\nRY-Phase 7 complete.")

# %%
# =============================================================================
# RY-PHASE 7B  —  SEASON-LEVEL DIAGNOSTICS: rushing_yards
# =============================================================================

print("\n" + "="*60)
print(f"RY-PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {RY_TARGET}")
print("="*60)

_ry_pg = _ry_export_df.copy()

_ry_szn = (
    _ry_pg.groupby("player_display_name")
    .agg(
        games               =("actual_rushing_yards",    "count"),
        actual_total        =("actual_rushing_yards",    "sum"),
        predicted_total     =("predicted_rushing_yards", "sum"),
        depth_chart_rank_min=("depth_chart_rank",        "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_ry_szn["residual"]  = _ry_szn["predicted_total"] - _ry_szn["actual_total"]
_ry_szn["pct_error"] = np.where(
    _ry_szn["actual_total"] == 0, np.nan,
    (_ry_szn["predicted_total"] - _ry_szn["actual_total"]) / _ry_szn["actual_total"]
)

_yt_s = _ry_szn["actual_total"].values.astype(float)
_yp_s = _ry_szn["predicted_total"].values.astype(float)
_ry_mae_s  = mean_absolute_error(_yt_s, _yp_s)
_ry_rmse_s = float(np.sqrt(np.mean((_yp_s - _yt_s) ** 2)))
_ry_r2_s   = float(r2_score(_yt_s, _yp_s))
_ry_bias_s = float(np.mean(_yp_s - _yt_s))
_ry_mape_s = float(np.mean(np.abs(_ry_szn["pct_error"].dropna())))
_ry_r_s    = float(np.corrcoef(_yt_s, _yp_s)[0, 1])
_ry_w10_s  = float((_ry_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_ry_w20_s  = float((_ry_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  Season-level fit diagnostics ({len(_ry_szn)} QBs):")
print(f"  {'MAE (yards)':<20}  {_ry_mae_s:>10.1f}")
print(f"  {'RMSE (yards)':<20}  {_ry_rmse_s:>10.1f}")
print(f"  {'R2':<20}  {_ry_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_ry_bias_s:>10.1f}")
print(f"  {'MAPE':<20}  {_ry_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_ry_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_ry_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_ry_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _ry_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.0f}  {row.residual:>+8.0f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

_ry_szn_path = DATA_DIR / f"test_predictions_2025_{RY_TARGET}_season_totals.xlsx"
_ry_szn.to_excel(_ry_szn_path, index=False)
print(f"\n  Season totals saved: {_ry_szn_path.name}")
print(f"\nRushing yards model complete.")

# %%
# =============================================================================
# RT-PHASE 4-7B  —  rushing_tds  (Poisson, same pipeline as rushing_yards)
# =============================================================================

# %%
# --- RT Feature set ---

# RT_FEATURES_RIDGE: lean 28-feature set, multicollinearity-clean for linear model
RT_FEATURES_RIDGE = [
    "depth_chart_rank", "age",
    "fantasy_pts_ewm10", "fantasy_pts_ewm20", "fantasy_pts_per_game_career",
    "rushing_tds_per_game_career",
    "forty_yard_dash", "speed_score",
    "rushing_tds_L20",
    "rushing_tds_ewm10", "rushing_tds_ewm20",
    "rushing_yards_L10", "rushing_yards_L20",
    "carries_L10", "carries_L20",
    "rushing_yards_ewm20",
    "carries_ewm10", "carries_ewm20",
    "rushing_epa_L5", "rushing_epa_L20",
    "rushing_epa_ewm20",
    "qb_pressure_rate_L5", "qb_pressure_rate_L10",
    "qb_pressure_rate_ewm20",
    "passing_epa_L5",
    "opp_def_run_epa_L10", "opp_def_run_epa_L20",
    "off_rb_carry_share_L5", "off_rb_carry_share_L20",
    "game_location", "game_temp", "game_precip_mm", "is_dome",
]

# RT_FEATURES_LGB: full set — adds short windows that gave LGB nonlinear signal
RT_FEATURES_LGB = RT_FEATURES_RIDGE + [
    "season",
    "rushing_tds_L5", "rushing_tds_L10", "rushing_tds_ewm5",
    "rushing_yards_L5", "rushing_yards_ewm10",
    "carries_L5",
    "rushing_epa_L10",
    "qb_pressure_rate_L20",
]

def _dedup(feats, df):
    seen = set()
    out = []
    for f in feats:
        if f in df.columns and f not in seen:
            out.append(f); seen.add(f)
    return out

RT_FEATURES_RIDGE = _dedup(RT_FEATURES_RIDGE, qb)
RT_FEATURES_LGB   = _dedup(RT_FEATURES_LGB,   qb)

print(f"\nRT_FEATURES_RIDGE: {len(RT_FEATURES_RIDGE)} features")
print(f"RT_FEATURES_LGB:   {len(RT_FEATURES_LGB)} features")

# %%
# --- Build RT train/val/test splits ---

RT_TARGET = "rushing_tds"

# LGB uses the full feature set
X_train_rt = df_train[RT_FEATURES_LGB].reset_index(drop=True)
X_val_rt   = df_val[RT_FEATURES_LGB].reset_index(drop=True)
X_test_rt  = df_test[RT_FEATURES_LGB].reset_index(drop=True)

# Ridge uses the lean feature set
X_train_rt_ridge = df_train[RT_FEATURES_RIDGE].reset_index(drop=True)
X_val_rt_ridge   = df_val[RT_FEATURES_RIDGE].reset_index(drop=True)
X_test_rt_ridge  = df_test[RT_FEATURES_RIDGE].reset_index(drop=True)

Y_train_rt = df_train[[RT_TARGET]].reset_index(drop=True)
Y_val_rt   = df_val[[RT_TARGET]].reset_index(drop=True)
Y_test_rt  = df_test[[RT_TARGET]].reset_index(drop=True)

print(f"  LGB:   X_train={X_train_rt.shape}  X_val={X_val_rt.shape}  X_test={X_test_rt.shape}")
print(f"  Ridge: X_train={X_train_rt_ridge.shape}  X_val={X_val_rt_ridge.shape}  X_test={X_test_rt_ridge.shape}")

# %%
# --- RT-4.1  Global mean baseline ---
print("\n--- RT-B1: Global mean ---")
_rt_train_mean = float(Y_train_rt[RT_TARGET].mean())
_rt_preds_mean = np.full(len(Y_val_rt), _rt_train_mean)
_rt_b1 = _metrics(Y_val_rt[RT_TARGET], _rt_preds_mean, "GlobalMean")
_rt_b1["target"] = RT_TARGET
print(f"  {RT_TARGET:<25}  MAE={_rt_b1['MAE']:.3f}  RMSE={_rt_b1['RMSE']:.3f}  R2={_rt_b1['R2']:+.3f}  Bias={_rt_b1['Bias']:+.3f}")

# %%
# --- RT-4.2  L1-proxy baseline (rushing_tds_L3) ---
print("\n--- RT-B2: L1-proxy (rushing_tds_L3) ---")
_col = "rushing_tds_L3"
if _col in df_val.columns:
    _rt_preds_l1 = df_val[_col].fillna(_rt_train_mean).reset_index(drop=True).values
else:
    _rt_preds_l1 = np.full(len(Y_val_rt), _rt_train_mean)
    print(f"  rushing_tds_L3 missing — using global mean fallback")
_rt_b2 = _metrics(Y_val_rt[RT_TARGET], _rt_preds_l1, "L1-proxy")
_rt_b2["target"] = RT_TARGET
print(f"  {RT_TARGET:<25}  MAE={_rt_b2['MAE']:.3f}  RMSE={_rt_b2['RMSE']:.3f}  R2={_rt_b2['R2']:+.3f}  Bias={_rt_b2['Bias']:+.3f}")

# %%
# --- RT-4.3  Rolling L5 mean baseline ---
print("\n--- RT-B3: Rolling L5 mean (rushing_tds_L5) ---")
_col = "rushing_tds_L5"
if _col in df_val.columns:
    _rt_preds_l5 = df_val[_col].fillna(_rt_train_mean).reset_index(drop=True).values
else:
    _rt_preds_l5 = np.full(len(Y_val_rt), _rt_train_mean)
    print(f"  rushing_tds_L5 missing — using global mean fallback")
_rt_b3 = _metrics(Y_val_rt[RT_TARGET], _rt_preds_l5, "RollingL5")
_rt_b3["target"] = RT_TARGET
print(f"  {RT_TARGET:<25}  MAE={_rt_b3['MAE']:.3f}  RMSE={_rt_b3['RMSE']:.3f}  R2={_rt_b3['R2']:+.3f}  Bias={_rt_b3['Bias']:+.3f}")

# %%
# --- RT-4.4  Ridge baseline ---
print("\n--- RT-B4: Ridge regression ---")

_rt_col_medians = X_train_rt_ridge.median().fillna(0)
_Xtr_rt_r = X_train_rt_ridge.fillna(_rt_col_medians).fillna(0)
_Xva_rt_r = X_val_rt_ridge.fillna(_rt_col_medians).fillna(0)

_rt_scaler = StandardScaler()
_Xtr_rt_s  = _rt_scaler.fit_transform(_Xtr_rt_r)
_Xva_rt_s  = _rt_scaler.transform(_Xva_rt_r)

_rt_ridge = Ridge(alpha=10.0)
_rt_ridge.fit(_Xtr_rt_s, Y_train_rt[RT_TARGET].values, sample_weight=sample_weights)
_rt_preds_ridge = np.clip(_rt_ridge.predict(_Xva_rt_s), 0, None)

_rt_b4 = _metrics(Y_val_rt[RT_TARGET], _rt_preds_ridge, "Ridge")
_rt_b4["target"] = RT_TARGET
print(f"  {RT_TARGET:<25}  MAE={_rt_b4['MAE']:.3f}  RMSE={_rt_b4['RMSE']:.3f}  R2={_rt_b4['R2']:+.3f}  Bias={_rt_b4['Bias']:+.3f}")

_rt_coef_df = (
    pd.DataFrame({"feature": RT_FEATURES_RIDGE, "coefficient": _rt_ridge.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
    .reset_index(drop=True)
)
_rt_coef_df.index = range(1, len(_rt_coef_df) + 1)
print(f"\n  Ridge coefficients — {RT_TARGET} (standardized, sorted by |coef|):")
print(f"  {'Rank':<5} {'Feature':<55} {'Coefficient':>12}")
print(f"  {'-'*75}")
for rank, row in _rt_coef_df.iterrows():
    print(f"  {rank:<5} {row['feature']:<55} {row['coefficient']:>+12.4f}")

# %%
# --- RT-4.5  Baseline summary ---
df_rt_baselines = pd.DataFrame([_rt_b1, _rt_b2, _rt_b3, _rt_b4])

print("\n" + "="*60)
print(f"RT BASELINE SUMMARY — VAL 2024")
print("="*60)
sub = df_rt_baselines[["baseline", "MAE", "RMSE", "R2", "Bias"]].set_index("baseline")
print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.3f}"))

# %%
# =============================================================================
# RT-PHASE 5  —  LightGBM: rushing_tds (Poisson, rolling forward CV + Optuna)
# =============================================================================

RT_OOF_FIRST_VAL_YEAR = 2017
RT_OPTUNA_TUNE_YEAR   = 2023
RT_CLIP               = True  # clip predictions >= 0

print("\n" + "="*60)
print(f"RT-PHASE 5  --  LightGBM: {RT_TARGET}")
print("="*60)
print(f"  objective=Optuna[regression|tweedie]  rolling CV: val {RT_OOF_FIRST_VAL_YEAR}-{RT_OPTUNA_TUNE_YEAR}")

# %%
# --- RT-5.1  Optuna: tune on train 2006-2022, val 2023 ---

_Xtr_opt_rt, _Ytr_opt_rt, _sw_opt_rt, _Xva_opt_rt, _Yva_opt_rt = _cv_split(
    df_train, RT_OPTUNA_TUNE_YEAR, RT_FEATURES_LGB, [RT_TARGET]
)

_y_tr_opt_rt     = np.clip(_Ytr_opt_rt[RT_TARGET].values.astype(float), 0, None)
_y_va_opt_rt_raw = np.clip(_Yva_opt_rt[RT_TARGET].values.astype(float), 0, None)
_dtrain_opt_rt = lgb.Dataset(_Xtr_opt_rt, label=_y_tr_opt_rt, weight=_sw_opt_rt, free_raw_data=False)
_dval_opt_rt   = lgb.Dataset(_Xva_opt_rt, label=_y_va_opt_rt_raw, reference=_dtrain_opt_rt, free_raw_data=False)

def _make_rt_objective(dtrain, dval, X_va, Y_va_raw):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         "regression",   # MSE — squares errors, penalizes missed TD games heavily
            "metric":            "mse",           # consistent with objective; early stopping on MSE
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs": -1, "seed": 42,
        }
        _b = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        _preds_orig = np.clip(_b.predict(X_va), 0, None)
        # Return RMSE — squares errors so Optuna can't game it by predicting near-zero
        return float(np.sqrt(np.mean((_preds_orig - Y_va_raw) ** 2)))
    return objective_fn

RT_N_TRIALS = 60
rt_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
_rt_progress = _OptunaProgress(RT_N_TRIALS, RT_TARGET)
rt_study.optimize(
    _make_rt_objective(_dtrain_opt_rt, _dval_opt_rt, _Xva_opt_rt, _y_va_opt_rt_raw),
    n_trials=RT_N_TRIALS,
    show_progress_bar=False,
    callbacks=[_rt_progress],
)
_rt_progress.close()

rt_best_params = rt_study.best_params
print(f"\n  Optuna best RMSE: {rt_study.best_value:.4f}  (trial {rt_study.best_trial.number}/{RT_N_TRIALS})")
print(f"  Best params: {rt_best_params}")

# %%
# --- RT-5.2  Rolling forward CV with best_params ---

rt_final_params = {
    "verbosity": -1,
    "objective": "regression",
    "metric": "mse",
    "n_jobs": -1, "seed": 42,
    **{k: v for k, v in rt_best_params.items() if k != "extra_trees"},
    "extra_trees": rt_best_params["extra_trees"],
}

rt_cv_val_years = list(range(RT_OOF_FIRST_VAL_YEAR, TRAIN_END + 1))
print(f"\n  Rolling forward CV: {len(rt_cv_val_years)} folds  ({rt_cv_val_years[0]}-{rt_cv_val_years[-1]})")

rt_oof_actual = []; rt_oof_pred = []; rt_oof_years = []; rt_best_iters = []

for _yr in rt_cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, RT_FEATURES_LGB, [RT_TARGET])
    _y_tr_raw = np.clip(_Ytr[RT_TARGET].values.astype(float), 0, None)
    _y_va_raw = np.clip(_Yva[RT_TARGET].values.astype(float), 0, None)
    _dt = lgb.Dataset(_Xtr, label=_y_tr_raw, weight=_sw, free_raw_data=False)
    _dv = lgb.Dataset(_Xva, label=_y_va_raw, reference=_dt, free_raw_data=False)
    _b = lgb.train(
        rt_final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
    )
    _p = np.clip(_b.predict(_Xva), 0, None)
    rt_oof_actual.extend(_y_va_raw.tolist())
    rt_oof_pred.extend(_p.tolist())
    rt_oof_years.extend([_yr] * len(_Yva))
    rt_best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"RMSE={np.sqrt(mean_squared_error(_y_va_raw, _p)):.3f}")

rt_oof_actual    = np.array(rt_oof_actual)
rt_oof_pred      = np.array(rt_oof_pred)
rt_oof_years     = np.array(rt_oof_years)
rt_mean_best_iter = int(np.mean(rt_best_iters))
print(f"\n  Mean best_iteration across folds: {rt_mean_best_iter}")

# %%
# --- RT-5.3  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"RT OOF METRICS (rolling forward CV)  --  {RT_TARGET}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in rt_cv_val_years:
    _mask = rt_oof_years == _yr
    _m    = _metrics(rt_oof_actual[_mask], rt_oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.3f}  {_m['RMSE']:>8.3f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.3f}")

_rt_m_overall = _metrics(rt_oof_actual, rt_oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(rt_oof_actual):>5}  "
      f"{_rt_m_overall['MAE']:>8.3f}  {_rt_m_overall['RMSE']:>8.3f}  "
      f"{_rt_m_overall['R2']:>+8.3f}  {_rt_m_overall['Bias']:>+8.3f}")

# %%
# --- RT-5.4  Final model: train 2006-2023, n_trees = mean best_iter ---

print(f"\n  Final RT model: train 2006-{TRAIN_END}, n_trees={rt_mean_best_iter}")

_y_tr_rt_full  = np.clip(Y_train_rt[RT_TARGET].values.astype(float), 0, None)
_dtrain_rt_final = lgb.Dataset(X_train_rt, label=_y_tr_rt_full, weight=sample_weights, free_raw_data=False)

rt_booster_final = lgb.train(
    {**rt_final_params, "metric": "none"},
    _dtrain_rt_final,
    num_boost_round=rt_mean_best_iter,
    callbacks=[log_evaluation(-1)],
)

_rt_preds_val = np.clip(rt_booster_final.predict(X_val_rt), 0, None)
_rt_m_val = _metrics(Y_val_rt[RT_TARGET], _rt_preds_val, "LightGBM")
print(f"  Val 2024 -> MAE={_rt_m_val['MAE']:.3f}  RMSE={_rt_m_val['RMSE']:.3f}  "
      f"R2={_rt_m_val['R2']:+.3f}  Bias={_rt_m_val['Bias']:+.3f}")

# %%
# --- RT-5.5  Summary vs baselines ---
print("\n" + "="*60)
print(f"RT-PHASE 5 SUMMARY -- {RT_TARGET}")
print("="*60)
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<10}  {float(_rt_m_overall[metric]):>10.3f}  "
          f"{_rt_m_val[metric]:>10.3f}  {_rt_b4[metric]:>12.3f}")

print(f"\nRT-Phase 5 complete.")

# Save to registry
lgb_models["rushing_tds"]      = rt_booster_final
best_params_all["rushing_tds"] = rt_best_params
oof_store["rushing_tds"] = {
    "oof_actual": rt_oof_actual,
    "oof_pred":   rt_oof_pred,
    "oof_years":  rt_oof_years,
}
joblib.dump({
    "lgb_models":     lgb_models,
    "optuna_studies": optuna_studies,
    "df_phase5":      pd.DataFrame(phase5_records),
    "best_params":    best_params_all,
    "oof_store":      oof_store,
    "feature_cols":   FEATURE_COLS,
    "td_feature_cols": TD_FEATURES,
    "ry_feature_cols": RY_FEATURES,
    "targets":        TARGETS,
    "loss_functions": LOSS_FUNCTIONS,
    "tweedie_power":  TWEEDIE_POWER,
}, registry_path)
print(f"  Registry updated.")

# %%
# =============================================================================
# RT-PHASE 6  —  EVALUATION & DIAGNOSTICS: rushing_tds
# =============================================================================

print("\n" + "="*60)
print(f"RT-PHASE 6  --  EVALUATION: {RT_TARGET}")
print("="*60)

_rt_preds_test = np.clip(rt_booster_final.predict(X_test_rt), 0, None)

# %%
# --- RT-6.1  Full metrics table ---

print(f"\n  [{RT_TARGET}]")
for split, yt, yp in [("Val 2024", Y_val_rt[RT_TARGET].values, _rt_preds_val),
                      ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- RT-6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_rt[RT_TARGET].values,  _rt_preds_val),
    ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_test),
]):
    lim = max(yt.max(), yp.max()) * 1.15 + 0.5
    ax.scatter(yt, yp, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"Rushing TDs -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Rushing TDs", fontsize=9)
    ax.set_ylabel("Predicted Rushing TDs", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RT_TARGET}_pred_vs_actual.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-6.3  Residual distribution ---

from scipy.stats import gaussian_kde as _gkde_rt, norm as _sp_norm_rt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val_rt[RT_TARGET].values,  _rt_preds_val),
    ("Test 2025",   Y_test_rt[RT_TARGET].values, _rt_preds_test),
    ("OOF 2017-23", rt_oof_actual,               rt_oof_pred),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde_rt(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm_rt.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals — Rushing TDs {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RT_TARGET}_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_rt[RT_TARGET].values,  _rt_preds_val),
    ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="steelblue", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Rushing TDs Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{RT_TARGET}_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-6.5  MAE by week (test 2025) ---

try:
    _rt_week_df = pd.DataFrame({
        "week":   df_test["week"].values,
        "actual": Y_test_rt[RT_TARGET].values,
        "pred":   _rt_preds_test,
    })
    _rt_wk = (_rt_week_df.groupby("week")
              .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]))
              .reset_index(name="MAE"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(_rt_wk["week"], _rt_wk["MAE"], color="steelblue", edgecolor="white")
    ax.set_title(f"MAE by Week — Rushing TDs (Test 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("NFL Week", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _fig_path = FIG_DIR / f"phase6_{RT_TARGET}_mae_by_week.png"
    plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    _show(_fig_path)
    print(f"  Saved: {_fig_path.name}")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- RT-6.6  Export test predictions ---

_rt_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_rt_export_df["actual_rushing_tds"]    = Y_test_rt[RT_TARGET].values
_rt_export_df["predicted_rushing_tds"] = _rt_preds_test
_rt_export_df["residual"]              = _rt_export_df["predicted_rushing_tds"] - _rt_export_df["actual_rushing_tds"]
_rt_export_df = _rt_export_df.sort_values(["week", "actual_rushing_tds"], ascending=[True, False]).reset_index(drop=True)

_rt_export_path = DATA_DIR / f"test_predictions_2025_{RT_TARGET}.xlsx"
_rt_export_df.to_excel(_rt_export_path, index=False)
print(f"\n  2025 test predictions saved: {_rt_export_path.name}  ({len(_rt_export_df)} rows)")
print(f"\nRT-Phase 6 complete.")

# %%
# =============================================================================
# RT-PHASE 7  —  FEATURE IMPORTANCE & SHAP: rushing_tds
# =============================================================================

print("\n" + "="*60)
print(f"RT-PHASE 7  --  FEATURE IMPORTANCE: {RT_TARGET}")
print("="*60)

# %%
# --- RT-7.1  LightGBM native gain importance ---

_rt_imp_gain = (
    pd.Series(rt_booster_final.feature_importance(importance_type="gain"),
              index=RT_FEATURES_LGB)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
_rt_imp_gain.head(25).plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"LightGBM Gain Importance (top 25) — {RT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Total Gain", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RT_TARGET}_gain_importance.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-7.2  SHAP ---

_rt_sample_idx = np.random.default_rng(42).choice(len(X_test_rt), size=min(500, len(X_test_rt)), replace=False)
_rt_X_shap     = X_test_rt.iloc[_rt_sample_idx].reset_index(drop=True)

_rt_explainer   = _shap_lib.TreeExplainer(rt_booster_final)
_rt_shap_values = _rt_explainer.shap_values(_rt_X_shap)

_rt_mean_abs_shap = pd.Series(
    np.abs(_rt_shap_values).mean(axis=0),
    index=RT_FEATURES_LGB,
).sort_values(ascending=False)

# Beeswarm
fig, ax = plt.subplots(figsize=(10, 9))
_shap_lib.summary_plot(_rt_shap_values, _rt_X_shap, max_display=20, show=False)
plt.title(f"SHAP Beeswarm — {RT_TARGET} (top 20)", fontsize=11, fontweight="bold")
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RT_TARGET}_shap_beeswarm.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# Bar plot
fig, ax = plt.subplots(figsize=(10, 8))
_rt_mean_abs_shap.head(20).plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"Mean |SHAP| (top 20) — {RT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{RT_TARGET}_shap_bar.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-7.3  Full importance table (all features) ---

_rt_gain_df = _rt_imp_gain.reset_index(); _rt_gain_df.columns = ["feature", "gain"]
_rt_shap_df = _rt_mean_abs_shap.reset_index(); _rt_shap_df.columns = ["feature", "mean_abs_shap"]
_rt_gain_df["gain_rank"] = range(1, len(_rt_gain_df) + 1)
_rt_shap_df["shap_rank"] = range(1, len(_rt_shap_df) + 1)

_rt_full_imp = _rt_gain_df.merge(_rt_shap_df, on="feature")[
    ["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]
].sort_values("shap_rank").reset_index(drop=True)

print(f"\n  {'gain_rank':>10}  {'shap_rank':>10}  {'feature':<55}  {'gain':>14}  {'mean_abs_shap':>14}")
print(f"  {'-'*105}")
for _, r in _rt_full_imp.iterrows():
    print(f"  {int(r['gain_rank']):>10}  {int(r['shap_rank']):>10}  {r['feature']:<55}  "
          f"{r['gain']:>14.2f}  {r['mean_abs_shap']:>14.6f}")

_rt_imp_path = DATA_DIR / f"feature_importance_{RT_TARGET}.xlsx"
_rt_full_imp.to_excel(_rt_imp_path, index=False)
print(f"\n  Full importance saved: {_rt_imp_path.name}")
print(f"\nRT-Phase 7 complete.")

# %%
# =============================================================================
# RT-PHASE 7B  —  SEASON-LEVEL DIAGNOSTICS: rushing_tds (Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"RT-PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {RT_TARGET}")
print("="*60)

_rt_pg = _rt_export_df.copy()

_rt_szn = (
    _rt_pg.groupby("player_display_name")
    .agg(
        games               =("actual_rushing_tds",    "count"),
        actual_total        =("actual_rushing_tds",    "sum"),
        predicted_total     =("predicted_rushing_tds", "sum"),
        depth_chart_rank_min=("depth_chart_rank",      "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_rt_szn["residual"]  = _rt_szn["predicted_total"] - _rt_szn["actual_total"]
_rt_szn["pct_error"] = np.where(
    _rt_szn["actual_total"] == 0, np.nan,
    (_rt_szn["predicted_total"] - _rt_szn["actual_total"]) / _rt_szn["actual_total"]
)

_yt_rt_s = _rt_szn["actual_total"].values.astype(float)
_yp_rt_s = _rt_szn["predicted_total"].values.astype(float)
_rt_mae_s  = mean_absolute_error(_yt_rt_s, _yp_rt_s)
_rt_rmse_s = float(np.sqrt(np.mean((_yp_rt_s - _yt_rt_s) ** 2)))
_rt_r2_s   = float(r2_score(_yt_rt_s, _yp_rt_s))
_rt_bias_s = float(np.mean(_yp_rt_s - _yt_rt_s))
_rt_mape_s = float(np.mean(np.abs(_rt_szn["pct_error"].dropna())))
_rt_r_s    = float(np.corrcoef(_yt_rt_s, _yp_rt_s)[0, 1])
_rt_w10_s  = float((_rt_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_rt_w20_s  = float((_rt_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  Season-level fit diagnostics ({len(_rt_szn)} QBs):")
print(f"  {'MAE (TDs)':<20}  {_rt_mae_s:>10.2f}")
print(f"  {'RMSE (TDs)':<20}  {_rt_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_rt_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_rt_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_rt_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_rt_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_rt_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_rt_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _rt_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

_rt_szn_path = DATA_DIR / f"test_predictions_2025_{RT_TARGET}_season_totals.xlsx"
_rt_szn.to_excel(_rt_szn_path, index=False)
print(f"\n  Season totals saved: {_rt_szn_path.name}")
print(f"\nRushing TDs model complete.")

# %%
# =============================================================================
# RT-PHASE 7C  —  RIDGE FULL DIAGNOSTICS: rushing_tds (Val 2024 + Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"RT-PHASE 7C  --  RIDGE FULL DIAGNOSTICS: {RT_TARGET}")
print("="*60)

# Ridge test-2025 predictions (scaler already fit on train in PHASE 4)
_Xte_rt_r        = X_test_rt_ridge.fillna(_rt_col_medians).fillna(0)
_Xte_rt_s        = _rt_scaler.transform(_Xte_rt_r)
_rt_preds_ridge_test = np.clip(_rt_ridge.predict(_Xte_rt_s), 0, None)

# %%
# --- RT-7C.1  Full metrics table (Ridge, both splits) ---

print(f"\n  [Ridge — {RT_TARGET}]")
for split, yt, yp in [("Val 2024",  Y_val_rt[RT_TARGET].values,  _rt_preds_ridge),
                      ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_ridge_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- RT-7C.2  LGB vs Ridge side-by-side summary ---

print("\n" + "="*60)
print(f"LGB vs Ridge Comparison — {RT_TARGET}")
print("="*60)
_rdg_v_m = _metrics(Y_val_rt[RT_TARGET], _rt_preds_ridge, "R")
_lgb_t_m = _metrics(Y_test_rt[RT_TARGET], _rt_preds_test, "L")
_rdg_t_m = _metrics(Y_test_rt[RT_TARGET], _rt_preds_ridge_test, "R")

def _pearson(yt, yp):
    yt, yp = np.asarray(yt, dtype=float), np.asarray(yp, dtype=float)
    return float(np.corrcoef(yt, yp)[0, 1])

_pearson_row = {
    "LGB Val24":  _pearson(Y_val_rt[RT_TARGET].values,  _rt_preds_val),
    "Rdg Val24":  _pearson(Y_val_rt[RT_TARGET].values,  _rt_preds_ridge),
    "LGB Tst25":  _pearson(Y_test_rt[RT_TARGET].values, _rt_preds_test),
    "Rdg Tst25":  _pearson(Y_test_rt[RT_TARGET].values, _rt_preds_ridge_test),
}

print(f"  {'Metric':<12}  {'LGB Val24':>10}  {'Rdg Val24':>10}  {'LGB Tst25':>10}  {'Rdg Tst25':>10}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<12}  {_rt_m_val[metric]:>10.3f}  {_rdg_v_m[metric]:>10.3f}  "
          f"{_lgb_t_m[metric]:>10.3f}  {_rdg_t_m[metric]:>10.3f}")
print(f"  {'Pearson_r':<12}  {_pearson_row['LGB Val24']:>10.3f}  {_pearson_row['Rdg Val24']:>10.3f}  "
      f"  {_pearson_row['LGB Tst25']:>10.3f}  {_pearson_row['Rdg Tst25']:>10.3f}")

# %%
# --- RT-7C.3  Calibration: Ridge val 2024 and test 2025 ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_rt[RT_TARGET].values,  _rt_preds_ridge),
    ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_ridge_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="darkorange", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Ridge Calibration — Rushing TDs {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase7c_{RT_TARGET}_ridge_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-7C.4  Residuals: Ridge val 2024 and test 2025 ---

from scipy.stats import gaussian_kde as _gkde_rdg, norm as _sp_norm_rdg
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_rt[RT_TARGET].values,  _rt_preds_ridge),
    ("Test 2025", Y_test_rt[RT_TARGET].values, _rt_preds_ridge_test),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="darkorange", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde_rdg(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm_rdg.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Ridge Residuals — Rushing TDs {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase7c_{RT_TARGET}_ridge_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- RT-7C.5  Ridge season-level diagnostics (Test 2025) ---

_rt_ridge_pg = df_test[["player_display_name", "week", "depth_chart_rank"]].copy().reset_index(drop=True)
_rt_ridge_pg["actual_rushing_tds"]    = Y_test_rt[RT_TARGET].values
_rt_ridge_pg["predicted_rushing_tds"] = _rt_preds_ridge_test

_rt_ridge_szn = (
    _rt_ridge_pg.groupby("player_display_name")
    .agg(
        games               =("actual_rushing_tds",    "count"),
        actual_total        =("actual_rushing_tds",    "sum"),
        predicted_total     =("predicted_rushing_tds", "sum"),
        depth_chart_rank_min=("depth_chart_rank",      "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_rt_ridge_szn["residual"]  = _rt_ridge_szn["predicted_total"] - _rt_ridge_szn["actual_total"]
_rt_ridge_szn["pct_error"] = np.where(
    _rt_ridge_szn["actual_total"] == 0, np.nan,
    (_rt_ridge_szn["predicted_total"] - _rt_ridge_szn["actual_total"]) / _rt_ridge_szn["actual_total"]
)

_yt_rdg = _rt_ridge_szn["actual_total"].values.astype(float)
_yp_rdg = _rt_ridge_szn["predicted_total"].values.astype(float)
_rdg_mae_s  = mean_absolute_error(_yt_rdg, _yp_rdg)
_rdg_rmse_s = float(np.sqrt(np.mean((_yp_rdg - _yt_rdg) ** 2)))
_rdg_r2_s   = float(r2_score(_yt_rdg, _yp_rdg))
_rdg_bias_s = float(np.mean(_yp_rdg - _yt_rdg))
_rdg_mape_s = float(np.mean(np.abs(_rt_ridge_szn["pct_error"].dropna())))
_rdg_r_s    = float(np.corrcoef(_yt_rdg, _yp_rdg)[0, 1])
_rdg_w10_s  = float((_rt_ridge_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_rdg_w20_s  = float((_rt_ridge_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  [Ridge] Season-level diagnostics ({len(_rt_ridge_szn)} QBs) — Test 2025:")
print(f"  {'MAE (TDs)':<20}  {_rdg_mae_s:>10.2f}")
print(f"  {'RMSE (TDs)':<20}  {_rdg_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_rdg_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_rdg_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_rdg_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_rdg_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_rdg_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_rdg_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _rt_ridge_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

print(f"\nRT-Phase 7C complete.")

# %%
# =============================================================================
# IT-PHASE 4-7C  —  interceptions  (MSE + RMSE, same pipeline as rushing_tds)
# =============================================================================

# %%
# --- IT Feature set ---

# IT_FEATURES_RIDGE: lean, multicollinearity-clean for Ridge
# One window per concept (L10/L20/ewm20 only), based on prior SHAP analysis
IT_FEATURES_RIDGE = [
    # INT rate
    "int_rate_L10", "int_rate_L20",
    "int_rate_ewm10", "int_rate_ewm20",

    # Aggressiveness / deep passing
    "ngs_aggressiveness_ewm5", "ngs_aggressiveness_ewm20",
    "ngs_avg_intended_air_yards_L10", "ngs_avg_intended_air_yards_L20",
    "qb_air_yards_per_attempt_ewm20",

    # Time to throw
    "ngs_avg_time_to_throw_L10",

    # Pressure
    "qb_pressure_rate_ewm5", "qb_pressure_rate_ewm20",

    # Receiver / separation
    "off_wr_separation_L10", "off_wr_separation_L20",
    "off_wr_epa_per_target_L10",
    "off_te_epa_per_target_L10", "off_te_epa_per_target_L20",

    # Efficiency
    "epa_per_opportunity_L20",

    # Volume
    "off_pass_rate_L10",

    # Opponent pass defense
    "opp_def_qb_qb_int_rate_L10", "opp_def_qb_qb_int_rate_L20",
    "opp_def_qb_qb_int_rate_ewm20",
    "opp_def_qb_qb_cpoe_L10", "opp_def_qb_qb_cpoe_L20",
    "opp_def_pass_epa_L20",
    "opp_def_wr_wr_catch_rate_L20",
    "opp_def_team_team_epa_per_pass_L20",

    # Game environment
    "game_temp", "game_wind", "game_precip_mm",

    # Context
    "depth_chart_rank", "age",
    "fantasy_pts_ewm10", "fantasy_pts_ewm20", "fantasy_pts_per_game_career",
    "interceptions_per_game_career",
]

# IT_FEATURES_LGB: 37 features — baseline INT model
IT_FEATURES_LGB = [
    # === INT HISTORY (rate only) ===
    "int_rate_L10", "int_rate_L20",
    "int_rate_ewm10", "int_rate_ewm20",

    # === AGGRESSIVENESS ===
    "ngs_aggressiveness_ewm5", "ngs_aggressiveness_ewm10", "ngs_aggressiveness_ewm20",

    # === DEEP PASSING TENDENCY ===
    "ngs_avg_intended_air_yards_L10", "ngs_avg_intended_air_yards_L20",
    "qb_air_yards_per_attempt_ewm5", "qb_air_yards_per_attempt_ewm20",

    # === TIME TO THROW ===
    "ngs_avg_time_to_throw_L10",

    # === PRESSURE (EWM only) ===
    "qb_pressure_rate_ewm5", "qb_pressure_rate_ewm10", "qb_pressure_rate_ewm20",

    # === RECEIVER / SEPARATION ===
    "off_wr_separation_L10", "off_wr_separation_L20",
    "off_wr_epa_per_target_L10",
    "off_te_epa_per_target_L10", "off_te_epa_per_target_L20",

    # === PASSING EFFICIENCY ===
    "epa_per_opportunity_L10", "epa_per_opportunity_L20",

    # === VOLUME CONTEXT ===
    "off_pass_rate_L10",

    # === OPPONENT PASS DEFENSE ===
    "opp_def_qb_qb_int_rate_L10", "opp_def_qb_qb_int_rate_L20",
    "opp_def_qb_qb_int_rate_ewm5", "opp_def_qb_qb_int_rate_ewm20",
    "opp_def_qb_qb_cpoe_L10", "opp_def_qb_qb_cpoe_L20",
    "opp_def_pass_epa_L20",
    "opp_def_wr_wr_catch_rate_L20",
    "opp_def_team_team_epa_per_pass_L10", "opp_def_team_team_epa_per_pass_L20",

    # === GAME ENVIRONMENT ===
    "game_temp", "game_wind", "game_precip_mm",

    # === PLAYER CONTEXT ===
    "depth_chart_rank", "age",
    "fantasy_pts_ewm10", "fantasy_pts_ewm20", "fantasy_pts_per_game_career",
    "interceptions_per_game_career",
]

IT_FEATURES_RIDGE = _dedup(IT_FEATURES_RIDGE, qb)
IT_FEATURES_LGB   = _dedup(IT_FEATURES_LGB,   qb)

print(f"\nIT_FEATURES_RIDGE: {len(IT_FEATURES_RIDGE)} features")
print(f"IT_FEATURES_LGB:   {len(IT_FEATURES_LGB)} features")

# %%
# --- Build IT train/val/test splits ---

IT_TARGET = "interceptions"

X_train_it       = df_train[IT_FEATURES_LGB].reset_index(drop=True)
X_val_it         = df_val[IT_FEATURES_LGB].reset_index(drop=True)
X_test_it        = df_test[IT_FEATURES_LGB].reset_index(drop=True)
X_train_it_ridge = df_train[IT_FEATURES_RIDGE].reset_index(drop=True)
X_val_it_ridge   = df_val[IT_FEATURES_RIDGE].reset_index(drop=True)
X_test_it_ridge  = df_test[IT_FEATURES_RIDGE].reset_index(drop=True)

Y_train_it = df_train[[IT_TARGET]].reset_index(drop=True)
Y_val_it   = df_val[[IT_TARGET]].reset_index(drop=True)
Y_test_it  = df_test[[IT_TARGET]].reset_index(drop=True)

print(f"  LGB:   X_train={X_train_it.shape}  X_val={X_val_it.shape}  X_test={X_test_it.shape}")
print(f"  Ridge: X_train={X_train_it_ridge.shape}  X_val={X_val_it_ridge.shape}  X_test={X_test_it_ridge.shape}")

# %%
# --- IT distribution ---

print("\n--- INT distribution (train set) ---")
_it_counts = df_train[IT_TARGET].value_counts().sort_index()
for v, n in _it_counts.items():
    print(f"  {int(v)} INT: {n:5,}  ({n/len(df_train):5.1%})")
print(f"  Mean: {df_train[IT_TARGET].mean():.3f}  Median: {df_train[IT_TARGET].median():.1f}")
print(f"  Zero games: {(df_train[IT_TARGET] == 0).mean():.1%}")

# %%
# --- IT-4.1  Global mean baseline ---
print("\n--- IT-B1: Global mean ---")
_it_train_mean = float(Y_train_it[IT_TARGET].mean())
_it_preds_mean = np.full(len(Y_val_it), _it_train_mean)
_it_b1 = _metrics(Y_val_it[IT_TARGET], _it_preds_mean, "GlobalMean")
_it_b1["target"] = IT_TARGET
print(f"  {IT_TARGET:<25}  MAE={_it_b1['MAE']:.3f}  RMSE={_it_b1['RMSE']:.3f}  R2={_it_b1['R2']:+.3f}  Bias={_it_b1['Bias']:+.3f}")

# %%
# --- IT-4.2  L1-proxy baseline (interceptions_L3) ---
print("\n--- IT-B2: L1-proxy (interceptions_L3) ---")
_col = "interceptions_L3"
if _col in df_val.columns:
    _it_preds_l1 = df_val[_col].fillna(_it_train_mean).reset_index(drop=True).values
else:
    _it_preds_l1 = np.full(len(Y_val_it), _it_train_mean)
    print(f"  interceptions_L3 missing -- using global mean fallback")
_it_preds_l1 = np.clip(_it_preds_l1, 0, None)
_it_b2 = _metrics(Y_val_it[IT_TARGET], _it_preds_l1, "L1-proxy")
_it_b2["target"] = IT_TARGET
print(f"  {IT_TARGET:<25}  MAE={_it_b2['MAE']:.3f}  RMSE={_it_b2['RMSE']:.3f}  R2={_it_b2['R2']:+.3f}  Bias={_it_b2['Bias']:+.3f}")

# %%
# --- IT-4.3  Rolling L5 mean baseline ---
print("\n--- IT-B3: Rolling L5 mean (interceptions_L5) ---")
_col = "interceptions_L5"
if _col in df_val.columns:
    _it_preds_l5 = df_val[_col].fillna(_it_train_mean).reset_index(drop=True).values
else:
    _it_preds_l5 = np.full(len(Y_val_it), _it_train_mean)
    print(f"  interceptions_L5 missing -- using global mean fallback")
_it_preds_l5 = np.clip(_it_preds_l5, 0, None)
_it_b3 = _metrics(Y_val_it[IT_TARGET], _it_preds_l5, "RollingL5")
_it_b3["target"] = IT_TARGET
print(f"  {IT_TARGET:<25}  MAE={_it_b3['MAE']:.3f}  RMSE={_it_b3['RMSE']:.3f}  R2={_it_b3['R2']:+.3f}  Bias={_it_b3['Bias']:+.3f}")

# %%
# --- IT-4.4  Ridge baseline ---
print("\n--- IT-B4: Ridge regression ---")

_it_col_medians  = X_train_it_ridge.median().fillna(0)
_Xtr_it_r        = X_train_it_ridge.fillna(_it_col_medians).fillna(0)
_Xva_it_r        = X_val_it_ridge.fillna(_it_col_medians).fillna(0)

_it_scaler  = StandardScaler()
_Xtr_it_s   = _it_scaler.fit_transform(_Xtr_it_r)
_Xva_it_s   = _it_scaler.transform(_Xva_it_r)

_it_ridge = Ridge(alpha=10.0)
_it_ridge.fit(_Xtr_it_s, Y_train_it[IT_TARGET].values, sample_weight=sample_weights)
_it_preds_ridge = np.clip(_it_ridge.predict(_Xva_it_s), 0, None)

_it_b4 = _metrics(Y_val_it[IT_TARGET], _it_preds_ridge, "Ridge")
_it_b4["target"] = IT_TARGET
print(f"  {IT_TARGET:<25}  MAE={_it_b4['MAE']:.3f}  RMSE={_it_b4['RMSE']:.3f}  R2={_it_b4['R2']:+.3f}  Bias={_it_b4['Bias']:+.3f}")

_it_coef_df = (
    pd.DataFrame({"feature": IT_FEATURES_RIDGE, "coefficient": _it_ridge.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
    .reset_index(drop=True)
)
_it_coef_df.index = range(1, len(_it_coef_df) + 1)
print(f"\n  Ridge coefficients -- {IT_TARGET} (standardized, sorted by |coef|):")
print(f"  {'Rank':<5} {'Feature':<55} {'Coefficient':>12}")
print(f"  {'-'*75}")
for rank, row in _it_coef_df.iterrows():
    print(f"  {rank:<5} {row['feature']:<55} {row['coefficient']:>+12.4f}")

# %%
# --- IT-4.5  Baseline summary ---
df_it_baselines = pd.DataFrame([_it_b1, _it_b2, _it_b3, _it_b4])

print("\n" + "="*60)
print(f"IT BASELINE SUMMARY -- VAL 2024")
print("="*60)
sub = df_it_baselines[["baseline", "MAE", "RMSE", "R2", "Bias"]].set_index("baseline")
print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.3f}"))

# %%
# =============================================================================
# IT-PHASE 5  —  LightGBM: interceptions (MSE, rolling forward CV + Optuna)
# =============================================================================

IT_OOF_FIRST_VAL_YEAR = 2017
IT_OPTUNA_TUNE_YEAR   = 2023
IT_N_TRIALS           = 60
IT_CLIP               = True

print("\n" + "="*60)
print(f"IT-PHASE 5  --  LightGBM: {IT_TARGET}")
print("="*60)
print(f"  objective=regression(MSE)  rolling CV: val {IT_OOF_FIRST_VAL_YEAR}-{IT_OPTUNA_TUNE_YEAR}")

# %%
# --- IT-5.1  Optuna: tune on train 2006-2022, val 2023 ---

_Xtr_opt_it, _Ytr_opt_it, _sw_opt_it, _Xva_opt_it, _Yva_opt_it = _cv_split(
    df_train, IT_OPTUNA_TUNE_YEAR, IT_FEATURES_LGB, [IT_TARGET]
)

_y_tr_opt_it     = np.clip(_Ytr_opt_it[IT_TARGET].values.astype(float), 0, None)
_y_va_opt_it_raw = np.clip(_Yva_opt_it[IT_TARGET].values.astype(float), 0, None)
_dtrain_opt_it = lgb.Dataset(_Xtr_opt_it, label=_y_tr_opt_it, weight=_sw_opt_it, free_raw_data=False)
_dval_opt_it   = lgb.Dataset(_Xva_opt_it, label=_y_va_opt_it_raw, reference=_dtrain_opt_it, free_raw_data=False)

def _make_it_objective(dtrain, dval, X_va, Y_va_raw):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         "regression",
            "metric":            "mse",
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs": -1, "seed": 42,
        }
        _b = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        _preds = np.clip(_b.predict(X_va), 0, None)
        return float(np.sqrt(np.mean((_preds - Y_va_raw) ** 2)))  # RMSE — cannot be gamed by near-zero predictions
    return objective_fn

_it_progress = _OptunaProgress(IT_N_TRIALS, IT_TARGET)
it_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
it_study.optimize(
    _make_it_objective(_dtrain_opt_it, _dval_opt_it, _Xva_opt_it, _y_va_opt_it_raw),
    n_trials=IT_N_TRIALS,
    show_progress_bar=False,
    callbacks=[_it_progress],
)
_it_progress.close()

it_best_params = it_study.best_params
print(f"\n  Optuna best RMSE: {it_study.best_value:.4f}  (trial {it_study.best_trial.number}/{IT_N_TRIALS})")
print(f"  Best params: {it_best_params}")

# %%
# --- IT-5.2  Rolling forward CV with best_params ---

it_final_params = {
    "verbosity": -1,
    "objective": "regression",
    "metric": "mse",
    "n_jobs": -1, "seed": 42,
    **{k: v for k, v in it_best_params.items() if k != "extra_trees"},
    "extra_trees": it_best_params["extra_trees"],
}

it_cv_val_years = list(range(IT_OOF_FIRST_VAL_YEAR, TRAIN_END + 1))
print(f"\n  Rolling forward CV: {len(it_cv_val_years)} folds  ({it_cv_val_years[0]}-{it_cv_val_years[-1]})")

it_oof_actual = []; it_oof_pred = []; it_oof_years = []; it_best_iters = []
for _yr in it_cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, IT_FEATURES_LGB, [IT_TARGET])
    _y_tr_raw = np.clip(_Ytr[IT_TARGET].values.astype(float), 0, None)
    _y_va_raw = np.clip(_Yva[IT_TARGET].values.astype(float), 0, None)
    _dt = lgb.Dataset(_Xtr, label=_y_tr_raw, weight=_sw, free_raw_data=False)
    _dv = lgb.Dataset(_Xva, label=_y_va_raw, reference=_dt, free_raw_data=False)
    _b = lgb.train(
        it_final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
    )
    _p = np.clip(_b.predict(_Xva), 0, None)
    it_oof_actual.extend(_y_va_raw.tolist())
    it_oof_pred.extend(_p.tolist())
    it_oof_years.extend([_yr] * len(_Yva))
    it_best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"RMSE={np.sqrt(mean_squared_error(_y_va_raw, _p)):.3f}")

it_oof_actual     = np.array(it_oof_actual)
it_oof_pred       = np.array(it_oof_pred)
it_oof_years      = np.array(it_oof_years)
it_mean_best_iter = int(np.mean(it_best_iters))
print(f"\n  Mean best_iteration across folds: {it_mean_best_iter}")

# %%
# --- IT-5.3  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"IT OOF METRICS (rolling forward CV)  --  {IT_TARGET}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in it_cv_val_years:
    _mask = it_oof_years == _yr
    _m    = _metrics(it_oof_actual[_mask], it_oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.3f}  {_m['RMSE']:>8.3f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.3f}")

_it_m_overall = _metrics(it_oof_actual, it_oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(it_oof_actual):>5}  "
      f"{_it_m_overall['MAE']:>8.3f}  {_it_m_overall['RMSE']:>8.3f}  "
      f"{_it_m_overall['R2']:>+8.3f}  {_it_m_overall['Bias']:>+8.3f}")

# %%
# --- IT-5.4  Final model: train 2006-2023, n_trees = mean best_iter ---

print(f"\n  Final IT model: train 2006-{TRAIN_END}, n_trees={it_mean_best_iter}")

_y_tr_it_full  = np.clip(Y_train_it[IT_TARGET].values.astype(float), 0, None)
_dtrain_it_final = lgb.Dataset(X_train_it, label=_y_tr_it_full, weight=sample_weights, free_raw_data=False)

it_booster_final = lgb.train(
    {**it_final_params, "metric": "none"},
    _dtrain_it_final,
    num_boost_round=it_mean_best_iter,
    callbacks=[log_evaluation(-1)],
)

_it_preds_val = np.clip(it_booster_final.predict(X_val_it), 0, None)
_it_m_val = _metrics(Y_val_it[IT_TARGET], _it_preds_val, "LightGBM")
print(f"  Val 2024 -> MAE={_it_m_val['MAE']:.3f}  RMSE={_it_m_val['RMSE']:.3f}  "
      f"R2={_it_m_val['R2']:+.3f}  Bias={_it_m_val['Bias']:+.3f}")

# %%
# --- IT-5.5  Summary vs baselines ---
print("\n" + "="*60)
print(f"IT-PHASE 5 SUMMARY -- {IT_TARGET}")
print("="*60)
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<10}  {float(_it_m_overall[metric]):>10.3f}  "
          f"{_it_m_val[metric]:>10.3f}  {_it_b4[metric]:>12.3f}")

print(f"\nIT-Phase 5 complete.")

# %%
# =============================================================================
# IT-PHASE 6  —  EVALUATION & DIAGNOSTICS: interceptions
# =============================================================================

print("\n" + "="*60)
print(f"IT-PHASE 6  --  EVALUATION: {IT_TARGET}")
print("="*60)

_it_preds_test = np.clip(it_booster_final.predict(X_test_it), 0, None)

# %%
# --- IT-6.1  Full metrics table ---

print(f"\n  [{IT_TARGET}]")
for split, yt, yp in [("Val 2024",  Y_val_it[IT_TARGET].values,  _it_preds_val),
                      ("Test 2025", Y_test_it[IT_TARGET].values, _it_preds_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- IT-6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_it[IT_TARGET].values,  _it_preds_val),
    ("Test 2025", Y_test_it[IT_TARGET].values, _it_preds_test),
]):
    lim = max(yt.max(), yp.max()) * 1.15 + 0.5
    ax.scatter(yt, yp, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"Interceptions -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Interceptions", fontsize=9)
    ax.set_ylabel("Predicted Interceptions", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{IT_TARGET}_pred_vs_actual.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-6.3  Residual distribution ---

from scipy.stats import gaussian_kde as _gkde_it, norm as _sp_norm_it
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val_it[IT_TARGET].values,  _it_preds_val),
    ("Test 2025",   Y_test_it[IT_TARGET].values, _it_preds_test),
    ("OOF 2017-23", it_oof_actual,               it_oof_pred),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde_it(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm_it.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals -- Interceptions {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{IT_TARGET}_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_it[IT_TARGET].values,  _it_preds_val),
    ("Test 2025", Y_test_it[IT_TARGET].values, _it_preds_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="steelblue", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Interceptions Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{IT_TARGET}_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-6.5  MAE by week (test 2025) ---

try:
    _it_week_df = pd.DataFrame({
        "week":   df_test["week"].values,
        "actual": Y_test_it[IT_TARGET].values,
        "pred":   _it_preds_test,
    })
    _it_wk = (_it_week_df.groupby("week")
              .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]))
              .reset_index(name="MAE"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(_it_wk["week"], _it_wk["MAE"], color="steelblue", edgecolor="white")
    ax.set_title(f"MAE by Week -- Interceptions (Test 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("NFL Week", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _fig_path = FIG_DIR / f"phase6_{IT_TARGET}_mae_by_week.png"
    plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    _show(_fig_path)
    print(f"  Saved: {_fig_path.name}")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- IT-6.6  Export test predictions ---

_it_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_it_export_df["actual_interceptions"]    = Y_test_it[IT_TARGET].values
_it_export_df["predicted_interceptions"] = _it_preds_test
_it_export_df["residual"]                = _it_export_df["predicted_interceptions"] - _it_export_df["actual_interceptions"]
_it_export_df = _it_export_df.sort_values(["week", "actual_interceptions"], ascending=[True, False]).reset_index(drop=True)

_it_export_path = DATA_DIR / f"test_predictions_2025_{IT_TARGET}.xlsx"
_it_export_df.to_excel(_it_export_path, index=False)
print(f"\n  2025 test predictions saved: {_it_export_path.name}  ({len(_it_export_df)} rows)")
print(f"\nIT-Phase 6 complete.")

# %%
# =============================================================================
# IT-PHASE 7  —  FEATURE IMPORTANCE & SHAP: interceptions
# =============================================================================

print("\n" + "="*60)
print(f"IT-PHASE 7  --  FEATURE IMPORTANCE: {IT_TARGET}")
print("="*60)

# %%
# --- IT-7.1  LightGBM native gain importance ---

_it_imp_gain = (
    pd.Series(it_booster_final.feature_importance(importance_type="gain"),
              index=IT_FEATURES_LGB)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
_it_imp_gain.head(25).plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"LightGBM Gain Importance (top 25) -- {IT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Total Gain", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{IT_TARGET}_gain_importance.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-7.2  SHAP ---

_it_sample_idx = np.random.default_rng(42).choice(len(X_test_it), size=min(500, len(X_test_it)), replace=False)
_it_X_shap     = X_test_it.iloc[_it_sample_idx].reset_index(drop=True)

_it_explainer   = _shap_lib.TreeExplainer(it_booster_final)
_it_shap_values = _it_explainer.shap_values(_it_X_shap)

_it_mean_abs_shap = pd.Series(
    np.abs(_it_shap_values).mean(axis=0),
    index=IT_FEATURES_LGB,
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 9))
_shap_lib.summary_plot(_it_shap_values, _it_X_shap, max_display=20, show=False)
plt.title(f"SHAP Beeswarm -- {IT_TARGET} (top 20)", fontsize=11, fontweight="bold")
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{IT_TARGET}_shap_beeswarm.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

fig, ax = plt.subplots(figsize=(10, 8))
_it_mean_abs_shap.head(20).plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"Mean |SHAP| (top 20) -- {IT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{IT_TARGET}_shap_bar.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-7.3  Full importance table ---

_it_gain_df = _it_imp_gain.reset_index(); _it_gain_df.columns = ["feature", "gain"]
_it_shap_df = _it_mean_abs_shap.reset_index(); _it_shap_df.columns = ["feature", "mean_abs_shap"]
_it_gain_df["gain_rank"] = range(1, len(_it_gain_df) + 1)
_it_shap_df["shap_rank"] = range(1, len(_it_shap_df) + 1)

_it_full_imp = _it_gain_df.merge(_it_shap_df, on="feature")[
    ["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]
].sort_values("shap_rank").reset_index(drop=True)

print(f"\n  {'gain_rank':>10}  {'shap_rank':>10}  {'feature':<55}  {'gain':>14}  {'mean_abs_shap':>14}")
print(f"  {'-'*105}")
for _, r in _it_full_imp.iterrows():
    print(f"  {int(r['gain_rank']):>10}  {int(r['shap_rank']):>10}  {r['feature']:<55}  "
          f"{r['gain']:>14.2f}  {r['mean_abs_shap']:>14.6f}")

_it_imp_path = DATA_DIR / f"feature_importance_{IT_TARGET}.xlsx"
_it_full_imp.to_excel(_it_imp_path, index=False)
print(f"\n  Full importance saved: {_it_imp_path.name}")
print(f"\nIT-Phase 7 complete.")

# %%
# =============================================================================
# IT-PHASE 7B  —  SEASON-LEVEL DIAGNOSTICS: interceptions (Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"IT-PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {IT_TARGET}")
print("="*60)

_it_szn = (
    _it_export_df.groupby("player_display_name")
    .agg(
        games               =("actual_interceptions",    "count"),
        actual_total        =("actual_interceptions",    "sum"),
        predicted_total     =("predicted_interceptions", "sum"),
        depth_chart_rank_min=("depth_chart_rank",        "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_it_szn["residual"]  = _it_szn["predicted_total"] - _it_szn["actual_total"]
_it_szn["pct_error"] = np.where(
    _it_szn["actual_total"] == 0, np.nan,
    (_it_szn["predicted_total"] - _it_szn["actual_total"]) / _it_szn["actual_total"]
)

_yt_it_s = _it_szn["actual_total"].values.astype(float)
_yp_it_s = _it_szn["predicted_total"].values.astype(float)
_it_mae_s  = mean_absolute_error(_yt_it_s, _yp_it_s)
_it_rmse_s = float(np.sqrt(np.mean((_yp_it_s - _yt_it_s) ** 2)))
_it_r2_s   = float(r2_score(_yt_it_s, _yp_it_s))
_it_bias_s = float(np.mean(_yp_it_s - _yt_it_s))
_it_mape_s = float(np.mean(np.abs(_it_szn["pct_error"].dropna())))
_it_r_s    = float(np.corrcoef(_yt_it_s, _yp_it_s)[0, 1])
_it_w10_s  = float((_it_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_it_w20_s  = float((_it_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  Season-level fit diagnostics ({len(_it_szn)} QBs):")
print(f"  {'MAE (INTs)':<20}  {_it_mae_s:>10.2f}")
print(f"  {'RMSE (INTs)':<20}  {_it_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_it_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_it_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_it_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_it_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_it_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_it_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _it_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

_it_szn_path = DATA_DIR / f"test_predictions_2025_{IT_TARGET}_season_totals.xlsx"
_it_szn.to_excel(_it_szn_path, index=False)
print(f"\n  Season totals saved: {_it_szn_path.name}")
print(f"\nIT-Phase 7B complete.")

# %%
# =============================================================================
# IT-PHASE 7C  —  RIDGE FULL DIAGNOSTICS: interceptions (Val 2024 + Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"IT-PHASE 7C  --  RIDGE FULL DIAGNOSTICS: {IT_TARGET}")
print("="*60)

_Xte_it_r        = X_test_it_ridge.fillna(_it_col_medians).fillna(0)
_Xte_it_s        = _it_scaler.transform(_Xte_it_r)
_it_preds_ridge_test = np.clip(_it_ridge.predict(_Xte_it_s), 0, None)

print(f"\n  [Ridge -- {IT_TARGET}]")
for split, yt, yp in [("Val 2024",  Y_val_it[IT_TARGET].values,  _it_preds_ridge),
                      ("Test 2025", Y_test_it[IT_TARGET].values, _it_preds_ridge_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- IT-7C.1  Calibration: Ridge ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_it[IT_TARGET].values,  _it_preds_ridge),
    ("Test 2025", Y_test_it[IT_TARGET].values, _it_preds_ridge_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="tomato", lw=1.5)
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Ridge Interceptions Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase7c_{IT_TARGET}_ridge_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- IT-7C.2  LGB vs Ridge comparison ---

_it_lgb_val  = _full_metrics(Y_val_it[IT_TARGET].values,  _it_preds_val)
_it_lgb_test = _full_metrics(Y_test_it[IT_TARGET].values, _it_preds_test)
_it_rdg_val  = _full_metrics(Y_val_it[IT_TARGET].values,  _it_preds_ridge)
_it_rdg_test = _full_metrics(Y_test_it[IT_TARGET].values, _it_preds_ridge_test)

print(f"\n{'='*60}")
print(f"LGB vs Ridge Comparison -- {IT_TARGET}")
print(f"{'='*60}")
print(f"  {'Metric':<12}  {'LGB Val24':>10}  {'Rdg Val24':>10}  {'LGB Tst25':>10}  {'Rdg Tst25':>10}")
for metric in ["MAE", "RMSE", "R2", "Pearson_r"]:
    print(f"  {metric:<12}  {_it_lgb_val[metric]:>10.3f}  {_it_rdg_val[metric]:>10.3f}  "
          f"  {_it_lgb_test[metric]:>10.3f}  {_it_rdg_test[metric]:>10.3f}")

# %%
# --- IT-7C.3  Ridge season-level diagnostics (Test 2025) ---

_it_ridge_szn_df = df_test[["player_display_name", "week", "depth_chart_rank"]].copy().reset_index(drop=True)
_it_ridge_szn_df["actual_interceptions"]    = Y_test_it[IT_TARGET].values
_it_ridge_szn_df["predicted_interceptions"] = _it_preds_ridge_test

_it_ridge_szn = (
    _it_ridge_szn_df.groupby("player_display_name")
    .agg(
        games               =("actual_interceptions",    "count"),
        actual_total        =("actual_interceptions",    "sum"),
        predicted_total     =("predicted_interceptions", "sum"),
        depth_chart_rank_min=("depth_chart_rank",        "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_it_ridge_szn["residual"]  = _it_ridge_szn["predicted_total"] - _it_ridge_szn["actual_total"]
_it_ridge_szn["pct_error"] = np.where(
    _it_ridge_szn["actual_total"] == 0, np.nan,
    (_it_ridge_szn["predicted_total"] - _it_ridge_szn["actual_total"]) / _it_ridge_szn["actual_total"]
)

_yt_rdg_it = _it_ridge_szn["actual_total"].values.astype(float)
_yp_rdg_it = _it_ridge_szn["predicted_total"].values.astype(float)
_rdg_it_mae_s  = mean_absolute_error(_yt_rdg_it, _yp_rdg_it)
_rdg_it_rmse_s = float(np.sqrt(np.mean((_yp_rdg_it - _yt_rdg_it) ** 2)))
_rdg_it_r2_s   = float(r2_score(_yt_rdg_it, _yp_rdg_it))
_rdg_it_bias_s = float(np.mean(_yp_rdg_it - _yt_rdg_it))
_rdg_it_mape_s = float(np.mean(np.abs(_it_ridge_szn["pct_error"].dropna())))
_rdg_it_r_s    = float(np.corrcoef(_yt_rdg_it, _yp_rdg_it)[0, 1])
_rdg_it_w10_s  = float((_it_ridge_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_rdg_it_w20_s  = float((_it_ridge_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  [Ridge] Season-level diagnostics ({len(_it_ridge_szn)} QBs) -- Test 2025:")
print(f"  {'MAE (INTs)':<20}  {_rdg_it_mae_s:>10.2f}")
print(f"  {'RMSE (INTs)':<20}  {_rdg_it_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_rdg_it_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_rdg_it_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_rdg_it_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_rdg_it_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_rdg_it_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_rdg_it_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _it_ridge_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

print(f"\nIT-Phase 7C complete.")

# %%
# =============================================================================
# FT-PHASE 4-7C  —  fumbles_lost_total  (MSE + RMSE, same pipeline as IT)
# =============================================================================

# %%
# --- FT Feature set ---

FT_FEATURES_RIDGE = [
    # Fumble history
    "fumbles_lost_total_L10", "fumbles_lost_total_ewm10", "fumbles_lost_total_ewm20",

    # Volume / scramble
    "carries_ewm10",
    "qb_scramble_rate_ewm10",

    # Pressure
    "qb_pressure_rate_ewm10",
    "off_qb_pressure_rate_L10",

    # Efficiency
    "epa_per_dropback_ewm10",

    # Opponent fumble defense
    "opp_def_team_team_fumbles_forced_L10", "opp_def_team_team_fumbles_forced_ewm10",

    # Opponent pass defense context
    "opp_def_team_team_epa_per_pass_L20",

    # Game environment
    "game_precip_mm", "game_temp", "game_wind",

    # Player context
    "depth_chart_rank", "age",
    "fantasy_pts_ewm10", "fantasy_pts_ewm20", "fantasy_pts_per_game_career",
    "fumbles_lost_per_game_career",
]

FT_FEATURES_LGB = [
    # === FUMBLE HISTORY ===
    "fumbles_lost_total_L10", "fumbles_lost_total_L20",
    "fumbles_lost_total_ewm5", "fumbles_lost_total_ewm10", "fumbles_lost_total_ewm20",

    # === CARRIES (exposure) ===
    "carries_L10",
    "carries_ewm5", "carries_ewm10", "carries_ewm20",

    # === RUSHING YARDS ===
    "rushing_yards_L5", "rushing_yards_L10",
    "rushing_yards_ewm5", "rushing_yards_ewm10",

    # === RUSHING EPA ===
    "rushing_epa_L5", "rushing_epa_L10", "rushing_epa_L20",
    "rushing_epa_ewm5", "rushing_epa_ewm10", "rushing_epa_ewm20",

    # === PRESSURE (EWM only for QB, all windows for team) ===
    "qb_pressure_rate_ewm5", "qb_pressure_rate_ewm10", "qb_pressure_rate_ewm20",
    "off_qb_pressure_rate_L5", "off_qb_pressure_rate_L10", "off_qb_pressure_rate_L20",

    # === VOLUME / SCHEME ===
    "off_pass_rate_L10",

    # === EFFICIENCY ===
    "epa_per_dropback_L10", "epa_per_dropback_L20",
    "epa_per_dropback_ewm5", "epa_per_dropback_ewm10", "epa_per_dropback_ewm20",

    # === OPPONENT FUMBLE DEFENSE (EWM only — rolling windows were dead) ===
    "opp_def_team_team_fumbles_forced_ewm5", "opp_def_team_team_fumbles_forced_ewm10",
    "opp_def_team_team_fumbles_forced_ewm20",

    # === OPPONENT PASS/RUSH DEFENSE ===
    "opp_def_team_team_epa_per_pass_L10", "opp_def_team_team_epa_per_pass_L20",
    "opp_def_team_team_epa_per_rush_L20",

    # === GAME ENVIRONMENT ===
    "game_precip_mm", "game_temp", "game_wind",

    # === PLAYER CONTEXT ===
    "depth_chart_rank", "age",
    "fantasy_pts_ewm10", "fantasy_pts_ewm20", "fantasy_pts_per_game_career",
    "fumbles_lost_per_game_career",
]

FT_FEATURES_RIDGE = _dedup(FT_FEATURES_RIDGE, qb)
FT_FEATURES_LGB   = _dedup(FT_FEATURES_LGB,   qb)

print(f"\nFT_FEATURES_RIDGE: {len(FT_FEATURES_RIDGE)} features")
print(f"FT_FEATURES_LGB:   {len(FT_FEATURES_LGB)} features")

# %%
# --- Build FT train/val/test splits ---

FT_TARGET = "fumbles_lost_total"

X_train_ft       = df_train[FT_FEATURES_LGB].reset_index(drop=True)
X_val_ft         = df_val[FT_FEATURES_LGB].reset_index(drop=True)
X_test_ft        = df_test[FT_FEATURES_LGB].reset_index(drop=True)
X_train_ft_ridge = df_train[FT_FEATURES_RIDGE].reset_index(drop=True)
X_val_ft_ridge   = df_val[FT_FEATURES_RIDGE].reset_index(drop=True)
X_test_ft_ridge  = df_test[FT_FEATURES_RIDGE].reset_index(drop=True)

Y_train_ft = df_train[[FT_TARGET]].reset_index(drop=True)
Y_val_ft   = df_val[[FT_TARGET]].reset_index(drop=True)
Y_test_ft  = df_test[[FT_TARGET]].reset_index(drop=True)

print(f"  LGB:   X_train={X_train_ft.shape}  X_val={X_val_ft.shape}  X_test={X_test_ft.shape}")
print(f"  Ridge: X_train={X_train_ft_ridge.shape}  X_val={X_val_ft_ridge.shape}  X_test={X_test_ft_ridge.shape}")

# %%
# --- FT distribution ---

print("\n--- Fumble distribution (train set) ---")
_ft_counts = df_train[FT_TARGET].value_counts().sort_index()
for v, n in _ft_counts.items():
    print(f"  {int(v)} fumbles: {n:5,}  ({n/len(df_train):5.1%})")
print(f"  Mean: {df_train[FT_TARGET].mean():.3f}  Median: {df_train[FT_TARGET].median():.1f}")
print(f"  Zero games: {(df_train[FT_TARGET] == 0).mean():.1%}")

# %%
# --- FT-4.1  Global mean baseline ---
print("\n--- FT-B1: Global mean ---")
_ft_train_mean = float(Y_train_ft[FT_TARGET].mean())
_ft_preds_mean = np.full(len(Y_val_ft), _ft_train_mean)
_ft_b1 = _metrics(Y_val_ft[FT_TARGET], _ft_preds_mean, "GlobalMean")
_ft_b1["target"] = FT_TARGET
print(f"  {FT_TARGET:<25}  MAE={_ft_b1['MAE']:.3f}  RMSE={_ft_b1['RMSE']:.3f}  R2={_ft_b1['R2']:+.3f}  Bias={_ft_b1['Bias']:+.3f}")

# %%
# --- FT-4.2  L1-proxy baseline (fumbles_lost_total_L3) ---
print("\n--- FT-B2: L1-proxy (fumbles_lost_total_L3) ---")
_col = "fumbles_lost_total_L3"
if _col in df_val.columns:
    _ft_preds_l1 = df_val[_col].fillna(_ft_train_mean).reset_index(drop=True).values
else:
    _ft_preds_l1 = np.full(len(Y_val_ft), _ft_train_mean)
    print(f"  fumbles_lost_total_L3 missing -- using global mean fallback")
_ft_preds_l1 = np.clip(_ft_preds_l1, 0, None)
_ft_b2 = _metrics(Y_val_ft[FT_TARGET], _ft_preds_l1, "L1-proxy")
_ft_b2["target"] = FT_TARGET
print(f"  {FT_TARGET:<25}  MAE={_ft_b2['MAE']:.3f}  RMSE={_ft_b2['RMSE']:.3f}  R2={_ft_b2['R2']:+.3f}  Bias={_ft_b2['Bias']:+.3f}")

# %%
# --- FT-4.3  Rolling L5 mean baseline ---
print("\n--- FT-B3: Rolling L5 mean (fumbles_lost_total_L5) ---")
_col = "fumbles_lost_total_L5"
if _col in df_val.columns:
    _ft_preds_l5 = df_val[_col].fillna(_ft_train_mean).reset_index(drop=True).values
else:
    _ft_preds_l5 = np.full(len(Y_val_ft), _ft_train_mean)
    print(f"  fumbles_lost_total_L5 missing -- using global mean fallback")
_ft_preds_l5 = np.clip(_ft_preds_l5, 0, None)
_ft_b3 = _metrics(Y_val_ft[FT_TARGET], _ft_preds_l5, "RollingL5")
_ft_b3["target"] = FT_TARGET
print(f"  {FT_TARGET:<25}  MAE={_ft_b3['MAE']:.3f}  RMSE={_ft_b3['RMSE']:.3f}  R2={_ft_b3['R2']:+.3f}  Bias={_ft_b3['Bias']:+.3f}")

# %%
# --- FT-4.4  Ridge baseline ---
print("\n--- FT-B4: Ridge regression ---")

_ft_col_medians  = X_train_ft_ridge.median().fillna(0)
_Xtr_ft_r        = X_train_ft_ridge.fillna(_ft_col_medians).fillna(0)
_Xva_ft_r        = X_val_ft_ridge.fillna(_ft_col_medians).fillna(0)

_ft_scaler  = StandardScaler()
_Xtr_ft_s   = _ft_scaler.fit_transform(_Xtr_ft_r)
_Xva_ft_s   = _ft_scaler.transform(_Xva_ft_r)

_ft_ridge = Ridge(alpha=10.0)
_ft_ridge.fit(_Xtr_ft_s, Y_train_ft[FT_TARGET].values, sample_weight=sample_weights)
_ft_preds_ridge = np.clip(_ft_ridge.predict(_Xva_ft_s), 0, None)

_ft_b4 = _metrics(Y_val_ft[FT_TARGET], _ft_preds_ridge, "Ridge")
_ft_b4["target"] = FT_TARGET
print(f"  {FT_TARGET:<25}  MAE={_ft_b4['MAE']:.3f}  RMSE={_ft_b4['RMSE']:.3f}  R2={_ft_b4['R2']:+.3f}  Bias={_ft_b4['Bias']:+.3f}")

_ft_coef_df = (
    pd.DataFrame({"feature": FT_FEATURES_RIDGE, "coefficient": _ft_ridge.coef_})
    .assign(abs_coef=lambda d: d["coefficient"].abs())
    .sort_values("abs_coef", ascending=False)
    .drop(columns="abs_coef")
    .reset_index(drop=True)
)
_ft_coef_df.index = range(1, len(_ft_coef_df) + 1)
print(f"\n  Ridge coefficients -- {FT_TARGET} (standardized, sorted by |coef|):")
print(f"  {'Rank':<5} {'Feature':<55} {'Coefficient':>12}")
print(f"  {'-'*75}")
for rank, row in _ft_coef_df.iterrows():
    print(f"  {rank:<5} {row['feature']:<55} {row['coefficient']:>+12.4f}")

# %%
# --- FT-4.5  Baseline summary ---
df_ft_baselines = pd.DataFrame([_ft_b1, _ft_b2, _ft_b3, _ft_b4])

print("\n" + "="*60)
print(f"FT BASELINE SUMMARY -- VAL 2024")
print("="*60)
sub = df_ft_baselines[["baseline", "MAE", "RMSE", "R2", "Bias"]].set_index("baseline")
print(sub.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:.3f}"))

# %%
# =============================================================================
# FT-PHASE 5  —  LightGBM: fumbles_lost_total (MSE, rolling forward CV + Optuna)
# =============================================================================

FT_OOF_FIRST_VAL_YEAR = 2017
FT_OPTUNA_TUNE_YEAR   = 2023
FT_N_TRIALS           = 60
FT_CLIP               = True

print("\n" + "="*60)
print(f"FT-PHASE 5  --  LightGBM: {FT_TARGET}")
print("="*60)
print(f"  objective=regression(MSE)  rolling CV: val {FT_OOF_FIRST_VAL_YEAR}-{FT_OPTUNA_TUNE_YEAR}")

# %%
# --- FT-5.1  Optuna: tune on train 2006-2022, val 2023 ---

_Xtr_opt_ft, _Ytr_opt_ft, _sw_opt_ft, _Xva_opt_ft, _Yva_opt_ft = _cv_split(
    df_train, FT_OPTUNA_TUNE_YEAR, FT_FEATURES_LGB, [FT_TARGET]
)

_y_tr_opt_ft     = np.clip(_Ytr_opt_ft[FT_TARGET].values.astype(float), 0, None)
_y_va_opt_ft_raw = np.clip(_Yva_opt_ft[FT_TARGET].values.astype(float), 0, None)
_dtrain_opt_ft = lgb.Dataset(_Xtr_opt_ft, label=_y_tr_opt_ft, weight=_sw_opt_ft, free_raw_data=False)
_dval_opt_ft   = lgb.Dataset(_Xva_opt_ft, label=_y_va_opt_ft_raw, reference=_dtrain_opt_ft, free_raw_data=False)

def _make_ft_objective(dtrain, dval, X_va, Y_va_raw):
    def objective_fn(trial):
        params = {
            "verbosity":         -1,
            "objective":         "regression",
            "metric":            "mse",
            "num_leaves":        trial.suggest_int("num_leaves", 31, 512),
            "max_depth":         trial.suggest_int("max_depth", 4, 12),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 1.0),
            "extra_trees":       trial.suggest_categorical("extra_trees", [True, False]),
            "n_jobs": -1, "seed": 42,
        }
        _b = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)],
        )
        _preds = np.clip(_b.predict(X_va), 0, None)
        return float(np.sqrt(np.mean((_preds - Y_va_raw) ** 2)))
    return objective_fn

_ft_progress = _OptunaProgress(FT_N_TRIALS, FT_TARGET)
ft_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
ft_study.optimize(
    _make_ft_objective(_dtrain_opt_ft, _dval_opt_ft, _Xva_opt_ft, _y_va_opt_ft_raw),
    n_trials=FT_N_TRIALS,
    show_progress_bar=False,
    callbacks=[_ft_progress],
)
_ft_progress.close()

ft_best_params = ft_study.best_params
print(f"\n  Optuna best RMSE: {ft_study.best_value:.4f}  (trial {ft_study.best_trial.number}/{FT_N_TRIALS})")
print(f"  Best params: {ft_best_params}")

# %%
# --- FT-5.2  Rolling forward CV with best_params ---

ft_final_params = {
    "verbosity": -1,
    "objective": "regression",
    "metric": "mse",
    "n_jobs": -1, "seed": 42,
    **{k: v for k, v in ft_best_params.items() if k != "extra_trees"},
    "extra_trees": ft_best_params["extra_trees"],
}

ft_cv_val_years = list(range(FT_OOF_FIRST_VAL_YEAR, TRAIN_END + 1))
print(f"\n  Rolling forward CV: {len(ft_cv_val_years)} folds  ({ft_cv_val_years[0]}-{ft_cv_val_years[-1]})")

ft_oof_actual = []; ft_oof_pred = []; ft_oof_years = []; ft_best_iters = []
for _yr in ft_cv_val_years:
    _Xtr, _Ytr, _sw, _Xva, _Yva = _cv_split(df_train, _yr, FT_FEATURES_LGB, [FT_TARGET])
    _y_tr_raw = np.clip(_Ytr[FT_TARGET].values.astype(float), 0, None)
    _y_va_raw = np.clip(_Yva[FT_TARGET].values.astype(float), 0, None)
    _dt = lgb.Dataset(_Xtr, label=_y_tr_raw, weight=_sw, free_raw_data=False)
    _dv = lgb.Dataset(_Xva, label=_y_va_raw, reference=_dt, free_raw_data=False)
    _b = lgb.train(
        ft_final_params, _dt,
        num_boost_round=2000,
        valid_sets=[_dv],
        callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)],
    )
    _p = np.clip(_b.predict(_Xva), 0, None)
    ft_oof_actual.extend(_y_va_raw.tolist())
    ft_oof_pred.extend(_p.tolist())
    ft_oof_years.extend([_yr] * len(_Yva))
    ft_best_iters.append(_b.best_iteration)
    print(f"    fold val={_yr}  n_train={len(_Xtr):,}  n_val={len(_Xva):,}  "
          f"best_iter={_b.best_iteration}  "
          f"RMSE={np.sqrt(mean_squared_error(_y_va_raw, _p)):.3f}")

ft_oof_actual     = np.array(ft_oof_actual)
ft_oof_pred       = np.array(ft_oof_pred)
ft_oof_years      = np.array(ft_oof_years)
ft_mean_best_iter = int(np.mean(ft_best_iters))
print(f"\n  Mean best_iteration across folds: {ft_mean_best_iter}")

# %%
# --- FT-5.3  OOF metrics: per-year + overall ---

print("\n" + "="*60)
print(f"FT OOF METRICS (rolling forward CV)  --  {FT_TARGET}")
print("="*60)
print(f"  {'Year':<6}  {'N':>5}  {'MAE':>8}  {'RMSE':>8}  {'R2':>8}  {'Bias':>8}")
print(f"  {'-'*52}")

for _yr in ft_cv_val_years:
    _mask = ft_oof_years == _yr
    _m    = _metrics(ft_oof_actual[_mask], ft_oof_pred[_mask], "OOF")
    print(f"  {_yr:<6}  {_mask.sum():>5}  "
          f"{_m['MAE']:>8.3f}  {_m['RMSE']:>8.3f}  {_m['R2']:>+8.3f}  {_m['Bias']:>+8.3f}")

_ft_m_overall = _metrics(ft_oof_actual, ft_oof_pred, "OOF-Overall")
print(f"  {'-'*52}")
print(f"  {'TOTAL':<6}  {len(ft_oof_actual):>5}  "
      f"{_ft_m_overall['MAE']:>8.3f}  {_ft_m_overall['RMSE']:>8.3f}  "
      f"{_ft_m_overall['R2']:>+8.3f}  {_ft_m_overall['Bias']:>+8.3f}")

# %%
# --- FT-5.4  Final model: train 2006-2023, n_trees = mean best_iter ---

print(f"\n  Final FT model: train 2006-{TRAIN_END}, n_trees={ft_mean_best_iter}")

_y_tr_ft_full  = np.clip(Y_train_ft[FT_TARGET].values.astype(float), 0, None)
_dtrain_ft_final = lgb.Dataset(X_train_ft, label=_y_tr_ft_full, weight=sample_weights, free_raw_data=False)

ft_booster_final = lgb.train(
    {**ft_final_params, "metric": "none"},
    _dtrain_ft_final,
    num_boost_round=ft_mean_best_iter,
    callbacks=[log_evaluation(-1)],
)

_ft_preds_val = np.clip(ft_booster_final.predict(X_val_ft), 0, None)
_ft_m_val = _metrics(Y_val_ft[FT_TARGET], _ft_preds_val, "LightGBM")
print(f"  Val 2024 -> MAE={_ft_m_val['MAE']:.3f}  RMSE={_ft_m_val['RMSE']:.3f}  "
      f"R2={_ft_m_val['R2']:+.3f}  Bias={_ft_m_val['Bias']:+.3f}")

# %%
# --- FT-5.5  Summary vs baselines ---
print("\n" + "="*60)
print(f"FT-PHASE 5 SUMMARY -- {FT_TARGET}")
print("="*60)
print(f"  {'Metric':<10}  {'LGB OOF':>10}  {'LGB Val24':>10}  {'Ridge Val24':>12}")
for metric in ["MAE", "RMSE", "R2"]:
    print(f"  {metric:<10}  {float(_ft_m_overall[metric]):>10.3f}  "
          f"{_ft_m_val[metric]:>10.3f}  {_ft_b4[metric]:>12.3f}")

print(f"\nFT-Phase 5 complete.")

# %%
# =============================================================================
# FT-PHASE 6  —  EVALUATION & DIAGNOSTICS: fumbles_lost_total
# =============================================================================

print("\n" + "="*60)
print(f"FT-PHASE 6  --  EVALUATION: {FT_TARGET}")
print("="*60)

_ft_preds_test = np.clip(ft_booster_final.predict(X_test_ft), 0, None)

# %%
# --- FT-6.1  Full metrics table ---

print(f"\n  [{FT_TARGET}]")
for split, yt, yp in [("Val 2024",  Y_val_ft[FT_TARGET].values,  _ft_preds_val),
                      ("Test 2025", Y_test_ft[FT_TARGET].values, _ft_preds_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- FT-6.2  Predicted vs actual scatter ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_ft[FT_TARGET].values,  _ft_preds_val),
    ("Test 2025", Y_test_ft[FT_TARGET].values, _ft_preds_test),
]):
    lim = max(yt.max(), yp.max()) * 1.15 + 0.5
    ax.scatter(yt, yp, alpha=0.25, s=12, color="steelblue", edgecolors="none")
    ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect")
    m_c, b_c = np.polyfit(yt, yp, 1)
    xs = np.linspace(0, lim, 100)
    ax.plot(xs, m_c * xs + b_c, "k-", lw=1.2, alpha=0.7, label="OLS fit")
    r2v   = r2_score(yt, yp)
    pearr = np.corrcoef(yt, yp)[0, 1]
    ax.set_title(f"Fumbles Lost -- {split}\nR2={r2v:.3f}  r={pearr:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual Fumbles Lost", fontsize=9)
    ax.set_ylabel("Predicted Fumbles Lost", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{FT_TARGET}_pred_vs_actual.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-6.3  Residual distribution ---

from scipy.stats import gaussian_kde as _gkde_ft, norm as _sp_norm_ft
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",    Y_val_ft[FT_TARGET].values,  _ft_preds_val),
    ("Test 2025",   Y_test_ft[FT_TARGET].values, _ft_preds_test),
    ("OOF 2017-23", ft_oof_actual,               ft_oof_pred),
]):
    resid = yp - yt
    ax.hist(resid, bins=40, density=True, color="steelblue", alpha=0.6, edgecolor="white")
    kde_x = np.linspace(resid.min(), resid.max(), 300)
    try:
        ax.plot(kde_x, _gkde_ft(resid)(kde_x), "k-", lw=1.5, label="KDE")
        ax.plot(kde_x, _sp_norm_ft.pdf(kde_x, resid.mean(), resid.std()), "r--", lw=1.2, alpha=0.8, label="Normal")
    except Exception:
        pass
    ax.axvline(0, color="red", linestyle="--", lw=1.5)
    ax.set_title(f"Residuals -- Fumbles Lost {split}\nbias={resid.mean():+.2f}  sd={resid.std():.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Residual (pred - actual)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{FT_TARGET}_residuals.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-6.4  Calibration curve ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_ft[FT_TARGET].values,  _ft_preds_val),
    ("Test 2025", Y_test_ft[FT_TARGET].values, _ft_preds_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="steelblue", lw=1.5)
        for _, r in _cal_grp.iterrows():
            ax.annotate(f"{r.name}", (r["mean_pred"], r["mean_actual"]), fontsize=7,
                        textcoords="offset points", xytext=(4, 2))
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Fumbles Lost Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase6_{FT_TARGET}_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-6.5  MAE by week (test 2025) ---

try:
    _ft_week_df = pd.DataFrame({
        "week":   df_test["week"].values,
        "actual": Y_test_ft[FT_TARGET].values,
        "pred":   _ft_preds_test,
    })
    _ft_wk = (_ft_week_df.groupby("week")
              .apply(lambda g: mean_absolute_error(g["actual"], g["pred"]))
              .reset_index(name="MAE"))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(_ft_wk["week"], _ft_wk["MAE"], color="steelblue", edgecolor="white")
    ax.set_title(f"MAE by Week -- Fumbles Lost (Test 2025)", fontsize=11, fontweight="bold")
    ax.set_xlabel("NFL Week", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    _fig_path = FIG_DIR / f"phase6_{FT_TARGET}_mae_by_week.png"
    plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    _show(_fig_path)
    print(f"  Saved: {_fig_path.name}")
except Exception as e:
    print(f"  MAE-by-week skipped: {e}")

# %%
# --- FT-6.6  Export test predictions ---

_ft_export_df = df_test[["player_display_name", "week", "attempts", "depth_chart_rank"]].copy().reset_index(drop=True)
_ft_export_df["actual_fumbles_lost"]    = Y_test_ft[FT_TARGET].values
_ft_export_df["predicted_fumbles_lost"] = _ft_preds_test
_ft_export_df["residual"]               = _ft_export_df["predicted_fumbles_lost"] - _ft_export_df["actual_fumbles_lost"]
_ft_export_df = _ft_export_df.sort_values(["week", "actual_fumbles_lost"], ascending=[True, False]).reset_index(drop=True)

_ft_export_path = DATA_DIR / f"test_predictions_2025_{FT_TARGET}.xlsx"
_ft_export_df.to_excel(_ft_export_path, index=False)
print(f"\n  2025 test predictions saved: {_ft_export_path.name}  ({len(_ft_export_df)} rows)")
print(f"\nFT-Phase 6 complete.")

# %%
# =============================================================================
# FT-PHASE 7  —  FEATURE IMPORTANCE & SHAP: fumbles_lost_total
# =============================================================================

print("\n" + "="*60)
print(f"FT-PHASE 7  --  FEATURE IMPORTANCE: {FT_TARGET}")
print("="*60)

# %%
# --- FT-7.1  LightGBM native gain importance ---

_ft_imp_gain = (
    pd.Series(ft_booster_final.feature_importance(importance_type="gain"),
              index=FT_FEATURES_LGB)
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
_ft_imp_gain.head(25).plot.barh(ax=ax, color="steelblue", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"LightGBM Gain Importance (top 25) -- {FT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Total Gain", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{FT_TARGET}_gain_importance.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-7.2  SHAP ---

_ft_sample_idx = np.random.default_rng(42).choice(len(X_test_ft), size=min(500, len(X_test_ft)), replace=False)
_ft_X_shap     = X_test_ft.iloc[_ft_sample_idx].reset_index(drop=True)

_ft_explainer   = _shap_lib.TreeExplainer(ft_booster_final)
_ft_shap_values = _ft_explainer.shap_values(_ft_X_shap)

_ft_mean_abs_shap = pd.Series(
    np.abs(_ft_shap_values).mean(axis=0),
    index=FT_FEATURES_LGB,
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 9))
_shap_lib.summary_plot(_ft_shap_values, _ft_X_shap, max_display=20, show=False)
plt.title(f"SHAP Beeswarm -- {FT_TARGET} (top 20)", fontsize=11, fontweight="bold")
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{FT_TARGET}_shap_beeswarm.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

fig, ax = plt.subplots(figsize=(10, 8))
_ft_mean_abs_shap.head(20).plot.barh(ax=ax, color="darkorange", edgecolor="white")
ax.invert_yaxis()
ax.set_title(f"Mean |SHAP| (top 20) -- {FT_TARGET}", fontsize=11, fontweight="bold")
ax.set_xlabel("Mean |SHAP value|", fontsize=9)
ax.tick_params(labelsize=8)
plt.tight_layout()
_fig_path = FIG_DIR / f"phase7_{FT_TARGET}_shap_bar.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-7.3  Full importance table ---

_ft_gain_df = _ft_imp_gain.reset_index(); _ft_gain_df.columns = ["feature", "gain"]
_ft_shap_df = _ft_mean_abs_shap.reset_index(); _ft_shap_df.columns = ["feature", "mean_abs_shap"]
_ft_gain_df["gain_rank"] = range(1, len(_ft_gain_df) + 1)
_ft_shap_df["shap_rank"] = range(1, len(_ft_shap_df) + 1)

_ft_full_imp = _ft_gain_df.merge(_ft_shap_df, on="feature")[
    ["gain_rank", "shap_rank", "feature", "gain", "mean_abs_shap"]
].sort_values("shap_rank").reset_index(drop=True)

print(f"\n  {'gain_rank':>10}  {'shap_rank':>10}  {'feature':<55}  {'gain':>14}  {'mean_abs_shap':>14}")
print(f"  {'-'*105}")
for _, r in _ft_full_imp.iterrows():
    print(f"  {int(r['gain_rank']):>10}  {int(r['shap_rank']):>10}  {r['feature']:<55}  "
          f"{r['gain']:>14.2f}  {r['mean_abs_shap']:>14.6f}")

_ft_imp_path = DATA_DIR / f"feature_importance_{FT_TARGET}.xlsx"
_ft_full_imp.to_excel(_ft_imp_path, index=False)
print(f"\n  Full importance saved: {_ft_imp_path.name}")
print(f"\nFT-Phase 7 complete.")

# %%
# =============================================================================
# FT-PHASE 7B  —  SEASON-LEVEL DIAGNOSTICS: fumbles_lost_total (Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"FT-PHASE 7B  --  SEASON-LEVEL DIAGNOSTICS: {FT_TARGET}")
print("="*60)

_ft_szn = (
    _ft_export_df.groupby("player_display_name")
    .agg(
        games               =("actual_fumbles_lost",    "count"),
        actual_total        =("actual_fumbles_lost",    "sum"),
        predicted_total     =("predicted_fumbles_lost", "sum"),
        depth_chart_rank_min=("depth_chart_rank",       "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_ft_szn["residual"]  = _ft_szn["predicted_total"] - _ft_szn["actual_total"]
_ft_szn["pct_error"] = np.where(
    _ft_szn["actual_total"] == 0, np.nan,
    (_ft_szn["predicted_total"] - _ft_szn["actual_total"]) / _ft_szn["actual_total"]
)

_yt_ft_s = _ft_szn["actual_total"].values.astype(float)
_yp_ft_s = _ft_szn["predicted_total"].values.astype(float)
_ft_mae_s  = mean_absolute_error(_yt_ft_s, _yp_ft_s)
_ft_rmse_s = float(np.sqrt(np.mean((_yp_ft_s - _yt_ft_s) ** 2)))
_ft_r2_s   = float(r2_score(_yt_ft_s, _yp_ft_s))
_ft_bias_s = float(np.mean(_yp_ft_s - _yt_ft_s))
_ft_mape_s = float(np.mean(np.abs(_ft_szn["pct_error"].dropna())))
_ft_r_s    = float(np.corrcoef(_yt_ft_s, _yp_ft_s)[0, 1])
_ft_w10_s  = float((_ft_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_ft_w20_s  = float((_ft_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  Season-level fit diagnostics ({len(_ft_szn)} QBs):")
print(f"  {'MAE (Fumbles)':<20}  {_ft_mae_s:>10.2f}")
print(f"  {'RMSE (Fumbles)':<20}  {_ft_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_ft_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_ft_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_ft_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_ft_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_ft_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_ft_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _ft_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

_ft_szn_path = DATA_DIR / f"test_predictions_2025_{FT_TARGET}_season_totals.xlsx"
_ft_szn.to_excel(_ft_szn_path, index=False)
print(f"\n  Season totals saved: {_ft_szn_path.name}")
print(f"\nFT-Phase 7B complete.")

# %%
# =============================================================================
# FT-PHASE 7C  —  RIDGE FULL DIAGNOSTICS: fumbles_lost_total (Val 2024 + Test 2025)
# =============================================================================

print("\n" + "="*60)
print(f"FT-PHASE 7C  --  RIDGE FULL DIAGNOSTICS: {FT_TARGET}")
print("="*60)

_Xte_ft_r        = X_test_ft_ridge.fillna(_ft_col_medians).fillna(0)
_Xte_ft_s        = _ft_scaler.transform(_Xte_ft_r)
_ft_preds_ridge_test = np.clip(_ft_ridge.predict(_Xte_ft_s), 0, None)

print(f"\n  [Ridge -- {FT_TARGET}]")
for split, yt, yp in [("Val 2024",  Y_val_ft[FT_TARGET].values,  _ft_preds_ridge),
                      ("Test 2025", Y_test_ft[FT_TARGET].values, _ft_preds_ridge_test)]:
    fm = _full_metrics(yt, yp)
    print(f"\n  [{split}]")
    for k, v in fm.items():
        print(f"    {k:<15}  {v:>10.4f}")

# %%
# --- FT-7C.1  Calibration: Ridge ---

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (split, yt, yp) in zip(axes, [
    ("Val 2024",  Y_val_ft[FT_TARGET].values,  _ft_preds_ridge),
    ("Test 2025", Y_test_ft[FT_TARGET].values, _ft_preds_ridge_test),
]):
    _cal_df = pd.DataFrame({"actual": yt, "pred": yp})
    try:
        _cal_df["decile"] = pd.qcut(_cal_df["pred"], q=10, labels=False, duplicates="drop")
        _cal_grp = _cal_df.groupby("decile").agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"))
        lim = max(_cal_grp["mean_pred"].max(), _cal_grp["mean_actual"].max()) * 1.2
        ax.plot(_cal_grp["mean_pred"], _cal_grp["mean_actual"], "o-", color="tomato", lw=1.5)
        ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect calibration")
        ax.set_title(f"Ridge Fumbles Lost Calibration -- {split}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted (decile)", fontsize=9)
        ax.set_ylabel("Mean Actual", fontsize=9)
        ax.legend(fontsize=8)
    except Exception as e:
        ax.set_title(f"Calibration skipped: {e}")

plt.tight_layout()
_fig_path = FIG_DIR / f"phase7c_{FT_TARGET}_ridge_calibration.png"
plt.savefig(_fig_path, dpi=130, bbox_inches="tight")
plt.close()
_show(_fig_path)
print(f"  Saved: {_fig_path.name}")

# %%
# --- FT-7C.2  LGB vs Ridge comparison ---

_ft_lgb_val  = _full_metrics(Y_val_ft[FT_TARGET].values,  _ft_preds_val)
_ft_lgb_test = _full_metrics(Y_test_ft[FT_TARGET].values, _ft_preds_test)
_ft_rdg_val  = _full_metrics(Y_val_ft[FT_TARGET].values,  _ft_preds_ridge)
_ft_rdg_test = _full_metrics(Y_test_ft[FT_TARGET].values, _ft_preds_ridge_test)

print(f"\n{'='*60}")
print(f"LGB vs Ridge Comparison -- {FT_TARGET}")
print(f"{'='*60}")
print(f"  {'Metric':<12}  {'LGB Val24':>10}  {'Rdg Val24':>10}  {'LGB Tst25':>10}  {'Rdg Tst25':>10}")
for metric in ["MAE", "RMSE", "R2", "Pearson_r"]:
    print(f"  {metric:<12}  {_ft_lgb_val[metric]:>10.3f}  {_ft_rdg_val[metric]:>10.3f}  "
          f"  {_ft_lgb_test[metric]:>10.3f}  {_ft_rdg_test[metric]:>10.3f}")

# %%
# --- FT-7C.3  Ridge season-level diagnostics (Test 2025) ---

_ft_ridge_szn_df = df_test[["player_display_name", "week", "depth_chart_rank"]].copy().reset_index(drop=True)
_ft_ridge_szn_df["actual_fumbles_lost"]    = Y_test_ft[FT_TARGET].values
_ft_ridge_szn_df["predicted_fumbles_lost"] = _ft_preds_ridge_test

_ft_ridge_szn = (
    _ft_ridge_szn_df.groupby("player_display_name")
    .agg(
        games               =("actual_fumbles_lost",    "count"),
        actual_total        =("actual_fumbles_lost",    "sum"),
        predicted_total     =("predicted_fumbles_lost", "sum"),
        depth_chart_rank_min=("depth_chart_rank",       "min"),
    )
    .reset_index()
    .sort_values("actual_total", ascending=False)
)
_ft_ridge_szn["residual"]  = _ft_ridge_szn["predicted_total"] - _ft_ridge_szn["actual_total"]
_ft_ridge_szn["pct_error"] = np.where(
    _ft_ridge_szn["actual_total"] == 0, np.nan,
    (_ft_ridge_szn["predicted_total"] - _ft_ridge_szn["actual_total"]) / _ft_ridge_szn["actual_total"]
)

_yt_rdg_ft = _ft_ridge_szn["actual_total"].values.astype(float)
_yp_rdg_ft = _ft_ridge_szn["predicted_total"].values.astype(float)
_rdg_ft_mae_s  = mean_absolute_error(_yt_rdg_ft, _yp_rdg_ft)
_rdg_ft_rmse_s = float(np.sqrt(np.mean((_yp_rdg_ft - _yt_rdg_ft) ** 2)))
_rdg_ft_r2_s   = float(r2_score(_yt_rdg_ft, _yp_rdg_ft))
_rdg_ft_bias_s = float(np.mean(_yp_rdg_ft - _yt_rdg_ft))
_rdg_ft_mape_s = float(np.mean(np.abs(_ft_ridge_szn["pct_error"].dropna())))
_rdg_ft_r_s    = float(np.corrcoef(_yt_rdg_ft, _yp_rdg_ft)[0, 1])
_rdg_ft_w10_s  = float((_ft_ridge_szn["pct_error"].abs().dropna() <= 0.10).mean() * 100)
_rdg_ft_w20_s  = float((_ft_ridge_szn["pct_error"].abs().dropna() <= 0.20).mean() * 100)

print(f"\n  [Ridge] Season-level diagnostics ({len(_ft_ridge_szn)} QBs) -- Test 2025:")
print(f"  {'MAE (Fumbles)':<20}  {_rdg_ft_mae_s:>10.2f}")
print(f"  {'RMSE (Fumbles)':<20}  {_rdg_ft_rmse_s:>10.2f}")
print(f"  {'R2':<20}  {_rdg_ft_r2_s:>10.4f}")
print(f"  {'Bias (pred-act)':<20}  {_rdg_ft_bias_s:>10.2f}")
print(f"  {'MAPE':<20}  {_rdg_ft_mape_s:>10.2%}")
print(f"  {'Pearson r':<20}  {_rdg_ft_r_s:>10.4f}")
print(f"  {'Within 10%':<20}  {_rdg_ft_w10_s:>10.1f}%")
print(f"  {'Within 20%':<20}  {_rdg_ft_w20_s:>10.1f}%")

print(f"\n  {'QB':<25}  {'G':>3}  {'Actual':>8}  {'Pred':>8}  {'Resid':>8}  {'Pct Err':>8}  {'DC Rank':>7}")
print(f"  {'-'*75}")
for _, row in _ft_ridge_szn.iterrows():
    pct_str = f"{row.pct_error*100:+7.1f}%" if pd.notna(row.pct_error) else "     N/A"
    print(f"  {row.player_display_name:<25}  {int(row.games):>3}  {row.actual_total:>8.0f}  "
          f"{row.predicted_total:>8.1f}  {row.residual:>+8.1f}  {pct_str}  {int(row.depth_chart_rank_min):>7}")

print(f"\nFT-Phase 7C complete.")

# %%
# =============================================================================
# COMBINED FANTASY ANALYSIS  —  All2025 / AllVal2024 / AllOOF / AllCombined
# =============================================================================

# %%
# --- Shared helpers ---

_STAT_COLS = [
    ("passing_yards",  "Pass Yds"),
    ("passing_tds",    "Pass TDs"),
    ("rushing_yards",  "Rush Yds"),
    ("rushing_tds",    "Rush TDs"),
    ("interceptions",  "INTs"),
    ("fumbles_lost",   "Fumbles"),
]

def _add_fantasy(df):
    df = df.copy()
    df["PredictedFantasy"] = (
          0.04 * df["predicted_passing_yards"]
        + 4.0  * df["predicted_passing_tds"]
        + 0.1  * df["predicted_rushing_yards"]
        + 6.0  * df["predicted_rushing_tds"]
        - 2.0  * (df["predicted_interceptions"] + df["predicted_fumbles_lost"])
    )
    df["ActualFantasy"] = (
          0.04 * df["actual_passing_yards"]
        + 4.0  * df["actual_passing_tds"]
        + 0.1  * df["actual_rushing_yards"]
        + 6.0  * df["actual_rushing_tds"]
        - 2.0  * (df["actual_interceptions"] + df["actual_fumbles_lost"])
    )
    df["residual"] = df["PredictedFantasy"] - df["ActualFantasy"]
    return df

def _season_totals(df):
    szn = (
        df.groupby("player_display_name")
        .agg(
            games                   =("week",                    "count"),
            actual_passing_yards    =("actual_passing_yards",    "sum"),
            predicted_passing_yards =("predicted_passing_yards", "sum"),
            actual_passing_tds      =("actual_passing_tds",      "sum"),
            predicted_passing_tds   =("predicted_passing_tds",   "sum"),
            actual_rushing_yards    =("actual_rushing_yards",    "sum"),
            predicted_rushing_yards =("predicted_rushing_yards", "sum"),
            actual_rushing_tds      =("actual_rushing_tds",      "sum"),
            predicted_rushing_tds   =("predicted_rushing_tds",   "sum"),
            actual_interceptions    =("actual_interceptions",    "sum"),
            predicted_interceptions =("predicted_interceptions", "sum"),
            actual_fumbles_lost     =("actual_fumbles_lost",     "sum"),
            predicted_fumbles_lost  =("predicted_fumbles_lost",  "sum"),
        )
        .reset_index()
    )
    return _add_fantasy(szn).sort_values("ActualFantasy", ascending=False).reset_index(drop=True)

def _print_analysis(label, df, min_games=1):
    print(f"\n{'='*65}")
    print(f"{label}")
    print(f"{'='*65}")

    yt = df["ActualFantasy"].values
    yp = df["PredictedFantasy"].values

    print(f"\n  [Game-level -- {len(df):,} observations]")
    print(f"  {'MAE (pts)':<22}  {mean_absolute_error(yt, yp):>10.3f}")
    print(f"  {'RMSE (pts)':<22}  {float(np.sqrt(np.mean((yp-yt)**2))):>10.3f}")
    print(f"  {'StdDev residual':<22}  {float(np.std(yp-yt)):>10.3f}")
    print(f"  {'R2':<22}  {float(r2_score(yt, yp)):>10.4f}")
    print(f"  {'Bias (pred-act)':<22}  {float(np.mean(yp-yt)):>10.3f}")
    print(f"  {'Pearson r':<22}  {float(np.corrcoef(yt, yp)[0,1]):>10.4f}")

    print(f"\n  Per-category bias / RMSE (game-level):")
    print(f"  {'Stat':<14}  {'MeanAct':>9}  {'MeanPred':>9}  {'Bias':>9}  {'RMSE':>9}  {'R2':>8}")
    print(f"  {'-'*65}")
    for col, name in _STAT_COLS:
        a = df[f"actual_{col}"].values.astype(float)
        p = df[f"predicted_{col}"].values.astype(float)
        print(f"  {name:<14}  {a.mean():>9.3f}  {p.mean():>9.3f}  "
              f"{np.mean(p-a):>+9.3f}  {np.sqrt(np.mean((p-a)**2)):>9.3f}  "
              f"{r2_score(a, p):>+8.4f}")

    szn = _season_totals(df)
    szn_full = szn[szn["games"] >= min_games]
    yt_s = szn_full["ActualFantasy"].values
    yp_s = szn_full["PredictedFantasy"].values

    print(f"\n  [Season-level -- {len(szn_full)} QBs (>={min_games}g)]")
    print(f"  {'MAE (pts)':<22}  {mean_absolute_error(yt_s, yp_s):>10.3f}")
    print(f"  {'RMSE (pts)':<22}  {float(np.sqrt(np.mean((yp_s-yt_s)**2))):>10.3f}")
    print(f"  {'R2':<22}  {float(r2_score(yt_s, yp_s)):>10.4f}")
    print(f"  {'Bias (pred-act)':<22}  {float(np.mean(yp_s-yt_s)):>10.3f}")
    print(f"  {'Pearson r':<22}  {float(np.corrcoef(yt_s, yp_s)[0,1]):>10.4f}")

    print(f"\n  [Season starters >=12g -- {len(szn[szn['games']>=12])} QBs]")
    szn12 = szn[szn["games"] >= 12]
    if len(szn12) >= 3:
        yt12 = szn12["ActualFantasy"].values
        yp12 = szn12["PredictedFantasy"].values
        print(f"  {'MAE (pts)':<22}  {mean_absolute_error(yt12, yp12):>10.3f}")
        print(f"  {'RMSE (pts)':<22}  {float(np.sqrt(np.mean((yp12-yt12)**2))):>10.3f}")
        print(f"  {'R2':<22}  {float(r2_score(yt12, yp12)):>10.4f}")
        print(f"  {'Bias (pred-act)':<22}  {float(np.mean(yp12-yt12)):>10.3f}")
        print(f"  {'Pearson r':<22}  {float(np.corrcoef(yt12, yp12)[0,1]):>10.4f}")

    print(f"\n  {'QB':<25}  {'G':>3}  {'ActFP':>8}  {'PredFP':>8}  {'Resid':>8}")
    print(f"  {'-'*60}")
    for _, row in szn_full.sort_values("ActualFantasy", ascending=False).iterrows():
        print(f"  {row.player_display_name:<25}  {int(row.games):>3}  "
              f"{row.ActualFantasy:>8.1f}  {row.PredictedFantasy:>8.1f}  "
              f"{row.residual:>+8.1f}")

    return szn

# %%
# --- BUILD TEST 2025 combo ---

_all25 = (
    _export_df[["player_display_name", "week", "attempts",
                "actual_passing_yards", "predicted_passing_yards"]]
    .merge(_td_export_df[["player_display_name", "week",
                          "actual_passing_tds", "predicted_passing_tds"]],
           on=["player_display_name", "week"], how="inner")
    .merge(_ry_export_df[["player_display_name", "week",
                          "actual_rushing_yards", "predicted_rushing_yards"]],
           on=["player_display_name", "week"], how="inner")
    .merge(_rt_export_df[["player_display_name", "week",
                          "actual_rushing_tds", "predicted_rushing_tds"]],
           on=["player_display_name", "week"], how="inner")
    .merge(_it_export_df[["player_display_name", "week",
                          "actual_interceptions", "predicted_interceptions"]],
           on=["player_display_name", "week"], how="inner")
    .merge(_ft_export_df[["player_display_name", "week",
                          "actual_fumbles_lost", "predicted_fumbles_lost"]],
           on=["player_display_name", "week"], how="inner")
    .sort_values(["player_display_name", "week"])
    .reset_index(drop=True)
)
_all25 = _add_fantasy(_all25)
_all25["split"] = "Test2025"

# %%
# --- BUILD VAL 2024 combo ---

_val_info = df_val[["player_display_name", "week", "attempts"]].reset_index(drop=True)

_all_val = _val_info.copy()
_all_val["actual_passing_yards"]    = Y_val[ACTIVE_TARGET].values
_all_val["predicted_passing_yards"] = preds_val_p6
_all_val["actual_passing_tds"]      = Y_val_td[TD_TARGET].values
_all_val["predicted_passing_tds"]   = _td_preds_val
_all_val["actual_rushing_yards"]    = Y_val_ry[RY_TARGET].values
_all_val["predicted_rushing_yards"] = _ry_preds_val
_all_val["actual_rushing_tds"]      = Y_val_rt[RT_TARGET].values
_all_val["predicted_rushing_tds"]   = _rt_preds_val
_all_val["actual_interceptions"]    = Y_val_it[IT_TARGET].values
_all_val["predicted_interceptions"] = _it_preds_val
_all_val["actual_fumbles_lost"]     = Y_val_ft[FT_TARGET].values
_all_val["predicted_fumbles_lost"]  = _ft_preds_val
_all_val = _add_fantasy(_all_val)
_all_val["split"] = "Val2024"

# %%
# --- BUILD OOF combo (2017-2023) ---
# Rows ordered identically to the OOF arrays: year-by-year in df_train order

_oof_info = pd.concat([
    df_train.loc[df_train["season"] == _yr,
                 ["player_display_name", "week", "attempts", "season"]]
    .reset_index(drop=True)
    for _yr in range(2017, TRAIN_END + 1)
], ignore_index=True)

_all_oof = _oof_info.copy()
_all_oof["actual_passing_yards"]    = oof_actual
_all_oof["predicted_passing_yards"] = oof_pred
_all_oof["actual_passing_tds"]      = td_oof_actual
_all_oof["predicted_passing_tds"]   = td_oof_pred
_all_oof["actual_rushing_yards"]    = ry_oof_actual
_all_oof["predicted_rushing_yards"] = ry_oof_pred
_all_oof["actual_rushing_tds"]      = rt_oof_actual
_all_oof["predicted_rushing_tds"]   = rt_oof_pred
_all_oof["actual_interceptions"]    = it_oof_actual
_all_oof["predicted_interceptions"] = it_oof_pred
_all_oof["actual_fumbles_lost"]     = ft_oof_actual
_all_oof["predicted_fumbles_lost"]  = ft_oof_pred
_all_oof = _add_fantasy(_all_oof)
_all_oof["split"] = "OOF"

print(f"\n  OOF rows: {len(_all_oof)}  |  Val rows: {len(_all_val)}  |  Test rows: {len(_all25)}")

# %%
# --- BUILD combined (OOF + Val + Test) ---

_all_combined = pd.concat(
    [_all_oof, _all_val[_all_oof.columns.intersection(_all_val.columns)],
     _all25[_all_oof.columns.intersection(_all25.columns)]],
    ignore_index=True,
)

# %%
# --- PRINT analysis for each split ---

_szn25  = _print_analysis("Test 2025 -- All QBs", _all25)
_sznval = _print_analysis("Val 2024 -- All QBs",  _all_val)
_sznoof = _print_analysis("OOF 2017-2023 -- All QBs", _all_oof)
_szncmb = _print_analysis("Combined (OOF+Val+Test) -- All QBs", _all_combined)

# %%
# --- SAVE 4 Excel files ---

_game_cols = [
    "player_display_name", "week", "attempts",
    "actual_passing_yards",    "predicted_passing_yards",
    "actual_passing_tds",      "predicted_passing_tds",
    "actual_rushing_yards",    "predicted_rushing_yards",
    "actual_rushing_tds",      "predicted_rushing_tds",
    "actual_interceptions",    "predicted_interceptions",
    "actual_fumbles_lost",     "predicted_fumbles_lost",
    "ActualFantasy", "PredictedFantasy", "residual",
]

for fname, df_game, df_szn in [
    ("All2025.xlsx",    _all25,    _szn25),
    ("AllVal2024.xlsx", _all_val,  _sznval),
    ("AllOOF.xlsx",     _all_oof,  _sznoof),
]:
    _path = DATA_DIR / fname
    _gcols = [c for c in _game_cols if c in df_game.columns]
    with pd.ExcelWriter(_path, engine="openpyxl") as _w:
        df_game[_gcols].to_excel(_w, sheet_name="game_level", index=False)
        df_szn.to_excel(_w, sheet_name="season_totals", index=False)
    print(f"  Saved: {fname}  ({len(df_game):,} game rows  |  {len(df_szn)} QBs)")

# AllCombined.xlsx gets a split column
_cmb_game_cols = ["split"] + _game_cols
_cmb_gcols = [c for c in _cmb_game_cols if c in _all_combined.columns]
_cmb_path = DATA_DIR / "AllCombined.xlsx"
with pd.ExcelWriter(_cmb_path, engine="openpyxl") as _w:
    _all_combined[_cmb_gcols].to_excel(_w, sheet_name="game_level", index=False)
    _szncmb.to_excel(_w, sheet_name="season_totals", index=False)
print(f"  Saved: AllCombined.xlsx  ({len(_all_combined):,} game rows  |  {len(_szncmb)} QBs)")

print(f"\nCombined Fantasy Analysis complete.")
