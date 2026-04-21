# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Install dependencies (run once, then restart kernel)

# %%
# Uncomment and run this cell once if nfl_data_py is not installed in your kernel,
# then restart the kernel before proceeding.
# import sys
# !{sys.executable} -m pip install nfl_data_py pyarrow pandas

# %% [markdown]
# # Build NFL Dataset
#
# Master data pipeline — all 15 steps in sequence.
# Run cells top to bottom to produce the final nfl_master_dataset.parquet.
#
# ## Step 1: Pull Weekly Player Stats
#
# Pulls base game-level rows per player from nfl_data_py for all seasons 2002-2025.
# Produces one row per player per game with raw target variables and basic usage stats.
# This is the foundation that all subsequent pipeline steps join onto.
#
# Output: data/raw/weekly_stats_2002_2025.parquet

# %%
import io
import sys
import warnings
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

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

RAW_DIR = DATA_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = list(range(2002, 2026))  # 1999-2001 excluded: no roster data, structurally different NFL

# %%
# --- PBP fallback for years where import_weekly_data is unavailable (e.g. 404) ---
# Derives per-player per-game box-score stats directly from play-by-play.
# Returns a DataFrame with the same column names as import_weekly_data so it
# flows through the same KEEP_COLS processing step below.
# Columns not derivable from PBP: pacr (not in master), dakota (QB composite, 88% null anyway).
# 2pt conversions and special_teams_tds ARE derived from PBP below.

def _weekly_from_pbp(years):
    """Build weekly player stats from PBP for seasons where import_weekly_data unavailable."""
    all_parts = []

    for yr in years:
        print(f"    PBP fallback: pulling {yr}...", end=" ", flush=True)
        try:
            pbp_raw = nfl.import_pbp_data(years=[yr])
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        pbp = pbp_raw[pbp_raw["season_type"] == "REG"].copy()
        del pbp_raw
        print(f"{len(pbp):,} plays")

        meta_cols = ["game_id", "season", "week", "season_type"]

        def _base(id_col, name_col, df):
            """Extract unique (player_id, game_id) rows with team/opponent metadata."""
            sub = (
                df[[id_col, name_col, "posteam", "defteam"] + meta_cols]
                .dropna(subset=[id_col])
                .drop_duplicates(subset=[id_col, "game_id"])
            )
            return sub.rename(columns={
                id_col:   "player_id",
                name_col: "player_name",
                "posteam": "recent_team",
                "defteam": "opponent_team",
            })

        pp = pbp[pbp["pass_attempt"] == 1].copy()
        pp["is_sack"] = pp["sack"].fillna(0).astype(int)
        rp = pbp[pbp["rush_attempt"] == 1].copy()
        tp = pbp[
            (pbp["pass_attempt"] == 1) &
            (pbp["sack"].fillna(0) == 0) &
            pbp["receiver_player_id"].notna()
        ].copy()

        # Build player-game base (one row per unique player × game)
        base_parts = []
        if pp["passer_player_id"].notna().any():
            base_parts.append(_base("passer_player_id",  "passer_player_name",   pp))
        if rp["rusher_player_id"].notna().any():
            base_parts.append(_base("rusher_player_id",  "rusher_player_name",   rp))
        if tp["receiver_player_id"].notna().any():
            base_parts.append(_base("receiver_player_id","receiver_player_name", tp))

        if not base_parts:
            print(f"    {yr}: no player data found, skipped")
            continue

        # Concat and dedup — passer name wins for QBs appearing in both rush and pass
        base = (
            pd.concat(base_parts, ignore_index=True)
            .drop_duplicates(subset=["player_id", "game_id"], keep="first")
        )

        # === PASSING STATS ===
        if pp["passer_player_id"].notna().any():
            p = pp[pp["passer_player_id"].notna()]
            p_stats = (
                p.groupby(["passer_player_id", "game_id"], dropna=True)
                .agg(
                    passing_yards              =("passing_yards",      "sum"),
                    passing_tds                =("pass_touchdown",     "sum"),
                    interceptions              =("interception",       "sum"),
                    sacks                      =("is_sack",            "sum"),
                    passing_air_yards          =("air_yards",          "sum"),
                    passing_yards_after_catch  =("yards_after_catch",  "sum"),
                    passing_first_downs        =("first_down_pass",    "sum"),
                    passing_epa                =("epa",                "sum"),
                    completions                =("complete_pass",      "sum"),
                    _dropbacks                 =("pass_attempt",       "sum"),
                )
                .reset_index()
                .rename(columns={"passer_player_id": "player_id"})
            )
            p_stats["attempts"] = p_stats["_dropbacks"] - p_stats["sacks"]
            p_stats.drop(columns=["_dropbacks"], inplace=True)

            # Sack yards — import_weekly_data stores as POSITIVE magnitude (e.g. 7.0 = 7 yards lost).
            # PBP yards_gained on sack plays is NEGATIVE (e.g. -7.0). Negate to match weekly convention.
            sack_y = (
                p[p["is_sack"] == 1]
                .groupby(["passer_player_id", "game_id"], dropna=True)["yards_gained"]
                .sum().reset_index()
                .rename(columns={"passer_player_id": "player_id", "yards_gained": "sack_yards"})
            )
            sack_y["sack_yards"] = -sack_y["sack_yards"]   # negate: PBP is negative, weekly is positive
            p_stats = p_stats.merge(sack_y, on=["player_id", "game_id"], how="left")
            p_stats["sack_yards"] = p_stats["sack_yards"].fillna(0)

            # Sack fumbles
            sf = (
                p[(p["is_sack"] == 1) & (p["fumble"].fillna(0) == 1)]
                .groupby(["passer_player_id", "game_id"], dropna=True)
                .agg(sack_fumbles=("fumble", "sum"), sack_fumbles_lost=("fumble_lost", "sum"))
                .reset_index()
                .rename(columns={"passer_player_id": "player_id"})
            )
            p_stats = p_stats.merge(sf, on=["player_id", "game_id"], how="left")
            p_stats[["sack_fumbles", "sack_fumbles_lost"]] = (
                p_stats[["sack_fumbles", "sack_fumbles_lost"]].fillna(0)
            )
            base = base.merge(p_stats, on=["player_id", "game_id"], how="left")

        # === RUSHING STATS ===
        if rp["rusher_player_id"].notna().any():
            r = rp[rp["rusher_player_id"].notna()]
            r_stats = (
                r.groupby(["rusher_player_id", "game_id"], dropna=True)
                .agg(
                    rushing_yards       =("rushing_yards",   "sum"),
                    rushing_tds         =("rush_touchdown",  "sum"),
                    carries             =("rush_attempt",    "sum"),
                    rushing_first_downs =("first_down_rush", "sum"),
                    rushing_epa         =("epa",             "sum"),
                )
                .reset_index()
                .rename(columns={"rusher_player_id": "player_id"})
            )
            rf = (
                r[(r["fumble"].fillna(0) == 1)]
                .groupby(["rusher_player_id", "game_id"], dropna=True)
                .agg(rushing_fumbles=("fumble", "sum"), rushing_fumbles_lost=("fumble_lost", "sum"))
                .reset_index()
                .rename(columns={"rusher_player_id": "player_id"})
            )
            r_stats = r_stats.merge(rf, on=["player_id", "game_id"], how="left")
            r_stats[["rushing_fumbles", "rushing_fumbles_lost"]] = (
                r_stats[["rushing_fumbles", "rushing_fumbles_lost"]].fillna(0)
            )
            base = base.merge(r_stats, on=["player_id", "game_id"], how="left")

        # === RECEIVING STATS ===
        if tp["receiver_player_id"].notna().any():
            # Team volume totals for target_share / air_yards_share
            team_vol = (
                tp.groupby(["posteam", "game_id"], dropna=True)
                .agg(_team_targets=("pass_attempt", "sum"), _team_air_yards=("air_yards", "sum"))
                .reset_index()
                .rename(columns={"posteam": "recent_team"})
            )
            t = tp[tp["receiver_player_id"].notna()]
            rec_stats = (
                t.groupby(["receiver_player_id", "game_id", "posteam"], dropna=True)
                .agg(
                    targets                      =("pass_attempt",      "sum"),
                    receptions                   =("complete_pass",     "sum"),
                    receiving_yards              =("receiving_yards",   "sum"),
                    receiving_tds                =("pass_touchdown",    "sum"),
                    receiving_air_yards          =("air_yards",         "sum"),
                    receiving_yards_after_catch  =("yards_after_catch", "sum"),
                    receiving_first_downs        =("first_down_pass",   "sum"),
                    receiving_epa                =("epa",               "sum"),
                )
                .reset_index()
                .rename(columns={"receiver_player_id": "player_id", "posteam": "recent_team"})
            )
            rec_stats = rec_stats.merge(team_vol, on=["recent_team", "game_id"], how="left")
            rec_stats["target_share"] = (
                rec_stats["targets"] / rec_stats["_team_targets"].replace(0, float("nan"))
            )
            rec_stats["air_yards_share"] = (
                rec_stats["receiving_air_yards"] / rec_stats["_team_air_yards"].replace(0, float("nan"))
            )
            rec_stats["wopr"] = 1.5 * rec_stats["target_share"] + 0.7 * rec_stats["air_yards_share"]
            rec_stats["racr"] = (
                rec_stats["receiving_yards"] / rec_stats["receiving_air_yards"].replace(0, float("nan"))
            )
            rec_stats.drop(columns=["_team_targets", "_team_air_yards", "recent_team"], inplace=True)

            recv_fum = (
                t[(t["fumble"].fillna(0) == 1)]
                .groupby(["receiver_player_id", "game_id"], dropna=True)
                .agg(receiving_fumbles=("fumble","sum"), receiving_fumbles_lost=("fumble_lost","sum"))
                .reset_index()
                .rename(columns={"receiver_player_id": "player_id"})
            )
            rec_stats = rec_stats.merge(recv_fum, on=["player_id", "game_id"], how="left")
            rec_stats[["receiving_fumbles", "receiving_fumbles_lost"]] = (
                rec_stats[["receiving_fumbles", "receiving_fumbles_lost"]].fillna(0)
            )
            base = base.merge(rec_stats, on=["player_id", "game_id"], how="left")

        # === 2PT CONVERSIONS ===
        if "two_point_conv_result" in pbp.columns and "two_point_attempt" in pbp.columns:
            twopc = pbp[
                (pbp["two_point_attempt"].fillna(0) == 1) &
                (pbp["two_point_conv_result"] == "success")
            ].copy()
            if len(twopc) > 0:
                pass2pt = (
                    twopc[twopc["passer_player_id"].notna()]
                    .groupby(["passer_player_id", "game_id"], dropna=True)
                    .size().reset_index(name="passing_2pt_conversions")
                    .rename(columns={"passer_player_id": "player_id"})
                )
                rush2pt = (
                    twopc[twopc["rusher_player_id"].notna()]
                    .groupby(["rusher_player_id", "game_id"], dropna=True)
                    .size().reset_index(name="rushing_2pt_conversions")
                    .rename(columns={"rusher_player_id": "player_id"})
                )
                rec2pt = (
                    twopc[twopc["receiver_player_id"].notna()]
                    .groupby(["receiver_player_id", "game_id"], dropna=True)
                    .size().reset_index(name="receiving_2pt_conversions")
                    .rename(columns={"receiver_player_id": "player_id"})
                )
                for twodf in [pass2pt, rush2pt, rec2pt]:
                    base = base.merge(twodf, on=["player_id", "game_id"], how="left")
                for col in ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]:
                    if col in base.columns:
                        base[col] = base[col].fillna(0)

        # === SPECIAL TEAMS TDS ===
        if "td_player_id" in pbp.columns and "special_teams_play" in pbp.columns:
            st_tds = (
                pbp[
                    (pbp["special_teams_play"].fillna(0) == 1) &
                    pbp["td_player_id"].notna()
                ]
                .groupby(["td_player_id", "game_id"], dropna=True)
                .size().reset_index(name="special_teams_tds")
                .rename(columns={"td_player_id": "player_id"})
            )
            if len(st_tds) > 0:
                base = base.merge(st_tds, on=["player_id", "game_id"], how="left")
                base["special_teams_tds"] = base["special_teams_tds"].fillna(0)

        # === POSITION lookup from players registry ===
        players_lkp = (
            nfl.import_players()[["gsis_id", "display_name", "position", "position_group"]]
            .dropna(subset=["gsis_id"])
            .drop_duplicates(subset=["gsis_id"])
            .rename(columns={"gsis_id": "player_id", "display_name": "player_display_name"})
        )
        base = base.merge(players_lkp, on="player_id", how="left")
        base = base[base["position"].isin(["QB", "RB", "WR", "TE", "K"])].copy()

        # Fill NaN -> 0 for raw count/volume columns only.
        # import_weekly_data returns 0 for non-applicable counts (e.g. WR passing_yards=0,
        # QB receiving_fumbles=0). PBP derivation only populates on actual plays -> NaN otherwise.
        # Zero-fill matches import_weekly_data behavior for counts.
        #
        # Do NOT zero-fill rate/share columns (target_share, air_yards_share, wopr, racr,
        # passing_epa, rushing_epa, receiving_epa) — import_weekly_data returns NaN for
        # these when the denominator is 0 (e.g. target_share NaN when targets=0).
        # Our PBP derivation already sets these to NaN when no plays occurred — correct.
        COUNT_COLS = [
            "passing_yards", "passing_tds", "interceptions", "sacks", "sack_yards",
            "passing_air_yards", "passing_yards_after_catch", "passing_first_downs",
            "completions", "attempts",
            "rushing_yards", "rushing_tds", "carries", "rushing_first_downs",
            "receptions", "targets", "receiving_yards", "receiving_tds",
            "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs",
            "receiving_fumbles", "receiving_fumbles_lost",
            "rushing_fumbles", "rushing_fumbles_lost",
            "sack_fumbles", "sack_fumbles_lost",
            "special_teams_tds",
            "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
        ]
        for col in COUNT_COLS:
            if col in base.columns:
                base[col] = base[col].fillna(0)

        print(f"    {yr} PBP fallback: {len(base):,} player-game rows, "
              f"{base['position'].value_counts().to_dict()}")
        all_parts.append(base)

    if not all_parts:
        return pd.DataFrame()
    return pd.concat(all_parts, ignore_index=True)

# %%
# --- Pull weekly player stats ---
# nfl.import_weekly_data returns one row per player per game with
# receiving, rushing, passing, and misc stats for all skill positions.
# Also includes snap counts where available (2012+).

print(f"Pulling weekly stats for {len(SEASONS)} seasons: {SEASONS[0]}-{SEASONS[-1]}")

# import_weekly_data may 404 for the most recent season if nflverse hasn't
# published the pre-built file yet. Fall back to year-by-year; for any year
# that still fails, derive stats from PBP (fully available for all seasons).
try:
    weekly = nfl.import_weekly_data(years=SEASONS)
    print(f"Bulk pull succeeded: {len(weekly):,} rows x {weekly.shape[1]} columns")
except Exception as bulk_err:
    print(f"Bulk pull failed ({bulk_err}), retrying year by year...")
    parts = []
    failed_years = []
    for yr in SEASONS:
        try:
            parts.append(nfl.import_weekly_data(years=[yr]))
            print(f"  {yr}: OK ({len(parts[-1]):,} rows)")
        except Exception as yr_err:
            print(f"  {yr}: weekly_data unavailable ({yr_err}) -> will use PBP fallback")
            failed_years.append(yr)
    if failed_years:
        print(f"  Running PBP fallback for: {failed_years}")
        pbp_fallback = _weekly_from_pbp(failed_years)
        if len(pbp_fallback) > 0:
            parts.append(pbp_fallback)
            print(f"  PBP fallback produced {len(pbp_fallback):,} rows for {failed_years}")
    weekly = pd.concat(parts, ignore_index=True)
    print(f"Year-by-year pull complete: {len(weekly):,} rows x {weekly.shape[1]} columns")

print(f"Raw pull: {len(weekly):,} rows x {weekly.shape[1]} columns")

# %%
# --- Inspect available columns ---
print("\nAll columns returned by import_weekly_data:")
for col in sorted(weekly.columns):
    print(f"  {col}: {weekly[col].dtype}")

# %%
# --- Select and rename columns we care about ---
# Keep identifiers, target variables for all positions, and usage stats.
# We drop fantasy points, redundant aggregates, and anything we will
# compute ourselves from play-by-play in Step 2.

KEEP_COLS = {
    # Identifiers — game_id is not in weekly pull; constructed below from season+week+team+opponent
    "player_id":            "player_id",
    "player_name":          "player_name",
    "player_display_name":  "player_display_name",
    "position":             "position",
    "position_group":       "position_group",
    # headshot_url excluded — not a model feature, 100K+ URLs break Excel's hyperlink limit
    "recent_team":          "team",
    "season":               "season",
    "week":                 "week",
    "season_type":          "season_type",
    "opponent_team":        "opponent",

    # QB target variables
    "passing_yards":        "passing_yards",
    "passing_tds":          "passing_tds",
    "interceptions":        "interceptions",
    "sacks":                "sacks",                     # QB sacks taken (also useful for DST step)
    "sack_yards":           "sack_yards",
    "passing_air_yards":    "passing_air_yards",
    "passing_yards_after_catch": "passing_yac",
    "passing_first_downs":  "passing_first_downs",
    "passing_epa":          "passing_epa",               # raw game EPA, per-play version in Step 2
    "pacr":                 "pacr",                      # passing air conversion ratio
    # dakota excluded: QB-only composite, 88% null, unavailable for 2025 via PBP
    "completions":          "completions",
    "attempts":             "attempts",
    # completion_percentage not in pull — computed below from completions / attempts

    # QB rushing (scrambles + designed runs)
    "rushing_yards":        "rushing_yards",             # for QB rows: QB rushing yards
    "rushing_tds":          "rushing_tds",
    "rushing_first_downs":  "rushing_first_downs",
    "rushing_epa":          "rushing_epa",

    # WR / TE / RB receiving target variables
    "receptions":           "receptions",
    "targets":              "targets",
    "receiving_yards":      "receiving_yards",
    "receiving_tds":        "receiving_tds",
    "receiving_air_yards":  "receiving_air_yards",
    "receiving_yards_after_catch": "receiving_yac",
    "receiving_first_downs": "receiving_first_downs",
    "receiving_epa":        "receiving_epa",
    "racr":                 "racr",                      # receiver air conversion ratio
    "air_yards_share":      "air_yards_share",
    "target_share":         "target_share",
    "wopr":                 "wopr",                      # weighted opportunity rating (pre-computed)
    # adot (average depth of target) not in weekly pull — computed from play-by-play in Step 2

    # Fumbles — split by play type in nfl_data_py, consolidated below
    "receiving_fumbles":        "receiving_fumbles",
    "receiving_fumbles_lost":   "receiving_fumbles_lost",
    "rushing_fumbles":          "rushing_fumbles",
    "rushing_fumbles_lost":     "rushing_fumbles_lost",
    "sack_fumbles":             "sack_fumbles",
    "sack_fumbles_lost":        "sack_fumbles_lost",

    # Usage / opportunity
    "carries":              "carries",
    "special_teams_tds":    "special_teams_tds",

    # 2pt conversions (small signal for scoring context)
    "passing_2pt_conversions":   "passing_2pt_conversions",
    "rushing_2pt_conversions":   "rushing_2pt_conversions",
    "receiving_2pt_conversions": "receiving_2pt_conversions",
}

# Keep only columns that actually exist in this pull
# (some columns are not present in older seasons)
available = {k: v for k, v in KEEP_COLS.items() if k in weekly.columns}
missing = [k for k in KEEP_COLS if k not in weekly.columns]
if missing:
    print(f"\nColumns not found in pull (will be NaN): {missing}")

weekly_clean = weekly[list(available.keys())].rename(columns=available)

print(f"\nAfter column selection: {weekly_clean.shape[1]} columns kept")

# %%
# --- Construct game_id and derived columns ---
# game_id is not in the weekly pull. Construct as SEASON_WEEK_TEAM_OPP
# using sorted team abbreviations so the same game has the same ID
# regardless of which player's row we are looking at.
# Format matches nflfastR convention: 2024_09_PHI_CIN (away_home sorted alphabetically)

weekly_clean["game_id"] = (
    weekly_clean["season"].astype(str)
    + "_"
    + weekly_clean["week"].astype(str).str.zfill(2)
    + "_"
    + weekly_clean.apply(
        lambda r: "_".join(sorted([str(r["team"]), str(r["opponent"])])), axis=1
    )
)

# completion_percentage: computed from completions and attempts
# Guard against division by zero for non-QB rows where attempts = 0
weekly_clean["completion_pct"] = (
    weekly_clean["completions"] / weekly_clean["attempts"].replace(0, float("nan"))
)

# fumbles_total and fumbles_lost_total: sum across all play types
# keeps position-specific columns too for granularity
fumble_cols     = ["receiving_fumbles",      "rushing_fumbles",      "sack_fumbles"]
fumble_lost_cols = ["receiving_fumbles_lost", "rushing_fumbles_lost", "sack_fumbles_lost"]

existing_fumble      = [c for c in fumble_cols      if c in weekly_clean.columns]
existing_fumble_lost = [c for c in fumble_lost_cols if c in weekly_clean.columns]

weekly_clean["fumbles_total"]      = weekly_clean[existing_fumble].sum(axis=1)
weekly_clean["fumbles_lost_total"] = weekly_clean[existing_fumble_lost].sum(axis=1)

print("Derived columns added: game_id, completion_pct, fumbles_total, fumbles_lost_total")

# %%
# --- Filter to regular season only ---
# Playoffs excluded per dataset design — we model regular season games only.
# season_type == 'REG' is the regular season flag in nfl_data_py.

if "season_type" in weekly_clean.columns:
    n_before = len(weekly_clean)
    weekly_clean = weekly_clean[weekly_clean["season_type"] == "REG"].copy()
    print(f"Regular season filter: {n_before:,} -> {len(weekly_clean):,} rows")
else:
    print("Warning: season_type column not found, no playoff filter applied")

# %%
# --- Filter to skill positions we model ---
# QB, RB, WR, TE, K are the positions we build projection rows for.
# DST rows are built separately from team-level aggregates.
# Exclude OL, DL, LB, CB, S, P, LS, etc.

SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

if "position" in weekly_clean.columns:
    n_before = len(weekly_clean)
    weekly_clean = weekly_clean[weekly_clean["position"].isin(SKILL_POSITIONS)].copy()
    print(f"Skill position filter: {n_before:,} -> {len(weekly_clean):,} rows")
    print(f"\nRow counts by position:")
    print(weekly_clean["position"].value_counts().to_string())
else:
    print("Warning: position column not found, no position filter applied")

# %%
# --- Basic data quality checks ---

print(f"\nSeason range: {weekly_clean['season'].min()} - {weekly_clean['season'].max()}")
print(f"Week range: {weekly_clean['week'].min()} - {weekly_clean['week'].max()}")
print(f"Total rows: {len(weekly_clean):,}")
print(f"Unique players: {weekly_clean['player_id'].nunique():,}")
print(f"Unique games: {weekly_clean['game_id'].nunique():,}")

print(f"\nNull counts for key columns:")
key_cols = [
    "player_id", "game_id", "season", "week", "team", "opponent",
    "position", "passing_yards", "rushing_yards", "receiving_yards",
    "targets", "carries", "receptions", "fumbles_total", "fumbles_lost_total"
]
for col in key_cols:
    if col in weekly_clean.columns:
        n_null = weekly_clean[col].isna().sum()
        pct = 100 * n_null / len(weekly_clean)
        print(f"  {col}: {n_null:,} nulls ({pct:.1f}%)")

# %%
# --- Check snap count availability ---
# Snap counts only available from 2012 onward in nfl_data_py.
# We note this here for the snap filter applied in Step 3.

snap_cols = [c for c in weekly_clean.columns if "snap" in c.lower()]
if snap_cols:
    print(f"\nSnap columns found: {snap_cols}")
    for col in snap_cols:
        n_nonull = weekly_clean[col].notna().sum()
        print(f"  {col}: {n_nonull:,} non-null rows")
else:
    print("\nNo snap columns in weekly stats — will be joined from import_snap_counts in Step 3")

# %%
# --- Save Step 1 raw output ---

out_path = RAW_DIR / f"weekly_stats_{SEASONS[0]}_{SEASONS[-1]}.parquet"
weekly_clean.to_parquet(out_path, index=False)
print(f"Saved raw: {out_path}  ({out_path.stat().st_size / 1_048_576:.1f} MB)")

# %%
# --- Initialise master dataset and save helper ---
# Master is saved to parquet + Excel only after the most recently completed step.
# Intermediate step outputs are saved to data/raw/ for inspection if needed.
# Excel row limit is ~1M rows; at 122K rows we are well within bounds.

MASTER_DIR = DATA_DIR / "data" / "master"
MASTER_DIR.mkdir(parents=True, exist_ok=True)

def save_master(df: pd.DataFrame, step: int) -> None:
    """Save the current master dataset to parquet and Excel."""
    parquet_path = MASTER_DIR / "nfl_master_dataset.parquet"
    excel_path   = MASTER_DIR / "nfl_master_dataset.xlsx"

    df.to_parquet(parquet_path, index=False)

    # xlsxwriter with strings_to_urls=False prevents URL hyperlink detection,
    # which corrupts files when there are more than 65,530 URL-like strings.
    # Convert datetime columns to strings first — xlsxwriter doesn't handle them natively.
    # Use xlsxwriter in constant_memory mode — streams rows directly to disk,
    # produces Excel-native XML. strings_to_urls=False prevents hyperlink corruption.
    # Write to local temp first, then copy atomically to avoid OneDrive mid-write interference.
    import tempfile, shutil
    import xlsxwriter

    df_excel = df.copy()
    for col in df_excel.select_dtypes(include=['datetime64[ns]', 'datetimetz']).columns:
        df_excel[col] = df_excel[col].astype(str)

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    workbook  = xlsxwriter.Workbook(str(tmp_path),
                                    {'constant_memory': True, 'strings_to_urls': False})
    worksheet = workbook.add_worksheet()

    # Header
    for col_idx, col_name in enumerate(df_excel.columns):
        worksheet.write(0, col_idx, col_name)

    # Data rows — write NaN as blank
    for row_idx, row in enumerate(df_excel.itertuples(index=False, name=None), start=1):
        for col_idx, val in enumerate(row):
            if val is None or (isinstance(val, float) and val != val):
                worksheet.write_blank(row_idx, col_idx, None)
            else:
                worksheet.write(row_idx, col_idx, val)

    workbook.close()
    shutil.copy2(tmp_path, excel_path)
    tmp_path.unlink(missing_ok=True)

    print(f"  Step {step} master saved — {len(df):,} rows x {df.shape[1]} cols")
    print(f"    Parquet : {parquet_path.stat().st_size / 1_048_576:.1f} MB")
    print(f"    Excel   : {excel_path.stat().st_size / 1_048_576:.1f} MB")

# --- Step 1 clean-up: apply fixes before initialising master ---

# Fix 1: Drop kickers — nfl_data_py weekly data has only 45 K rows across 25 seasons
# (should be ~13,000+). Kicker stats will be sourced separately in a dedicated step.
n_before = len(weekly_clean)
weekly_clean = weekly_clean[weekly_clean["position"] != "K"].copy()
print(f"Kicker rows removed: {n_before - len(weekly_clean)} rows dropped "
      f"({n_before:,} -> {len(weekly_clean):,})")

# Fix 2: Drop PACR — redundant with qb_air_yards_per_attempt computed from PBP,
# and contains zeros for QBs in older seasons where air_yards was not tracked.
if "pacr" in weekly_clean.columns:
    weekly_clean.drop(columns=["pacr"], inplace=True)
    print("PACR column dropped (redundant with PBP-derived qb_air_yards_per_attempt)")

# Master dataset after Step 1
master = weekly_clean.copy()
print(f"\nMaster after Step 1: {master.shape}")

# %% [markdown]
# ## Step 2: Play-by-Play EPA Aggregation
#
# Pulls raw play-by-play from nfl_data_py for each season and aggregates
# per player per game. Produces the following features not available in the
# weekly stats pull:
#
# QB (passer):   epa_per_dropback, cpoe, air_yards_per_attempt, pressure_rate, scramble_rate
# Receiver:      epa_per_target, adot, yac_per_reception, catch_rate, breakaway_rate (WR/TE/RB)
# Rusher:        epa_per_carry, breakaway_rate (rushes >= 15 yards / total carries)
#
# PBP is pulled one season at a time to manage memory — each season is ~45K rows x 372 cols.
# Only the columns we need are retained before aggregation.
# All aggregations are per (player_id, season, week) to match master join keys.
#
# Output: data/raw/pbp_aggregated_2002_2025.parquet

# %%
# Columns to retain from raw PBP before aggregation
PBP_COLS = [
    "game_id", "season", "week", "season_type",
    "passer_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "play_type",
    "pass_attempt",      # 1 on any pass play including sacks
    "rush_attempt",      # 1 on any rush play
    "sack",              # 1 when QB was sacked
    "qb_scramble",       # 1 when QB scrambled
    "qb_hit",            # 1 when QB was hit (pressure proxy even without sack)
    "epa",               # expected points added on the play
    "cpoe",              # completion % over expected (pass plays only)
    "air_yards",         # depth of target / intended throw
    "yards_after_catch", # yards gained after the catch
    "yards_gained",      # total yards on the play
    "complete_pass",     # 1 on completions
    "incomplete_pass",   # 1 on incompletions
    "first_down",        # 1 if play resulted in a first down
    "touchdown",         # 1 if play resulted in a TD
    "interception",      # 1 if play resulted in an INT
    "fumble",            # 1 if fumble occurred
    "fumble_lost",       # 1 if fumble was lost
]

# %%
# --- Pull PBP one season at a time and aggregate ---
# Pulling all seasons at once would require ~10GB RAM.
# Season-by-season aggregation keeps peak memory to ~500MB.

pbp_agg_list = []

for season in SEASONS:
    print(f"  PBP {season}...", end=" ", flush=True)

    try:
        pbp_raw = nfl.import_pbp_data(years=[season])
    except Exception as e:
        print(f"FAILED ({e})")
        continue

    # Keep only the columns we need and regular season plays
    existing = [c for c in PBP_COLS if c in pbp_raw.columns]
    pbp = pbp_raw[existing].copy()
    del pbp_raw  # free memory immediately

    pbp = pbp[pbp["season_type"] == "REG"].copy()

    # --- QB / Passer aggregations ---
    # A dropback is any pass_attempt (includes sacks and scrambles called as pass)
    # We compute per passer per (season, week)
    pass_plays = pbp[pbp["pass_attempt"] == 1].copy()

    # Pressure = qb_hit OR sack (either counts as a pressured dropback)
    pass_plays["pressured"] = (
        (pass_plays["qb_hit"].fillna(0) == 1) |
        (pass_plays["sack"].fillna(0) == 1)
    ).astype(int)

    qb_agg = (
        pass_plays[pass_plays["passer_player_id"].notna()]
        .groupby(["passer_player_id", "season", "week"])
        .agg(
            qb_dropbacks       = ("pass_attempt",    "sum"),    # total dropbacks
            qb_epa_per_dropback= ("epa",             "mean"),   # EPA per dropback
            qb_cpoe            = ("cpoe",            "mean"),   # CPOE (pre-computed in nflfastR)
            qb_air_yards_total = ("air_yards",       "sum"),    # for per-attempt calc below
            qb_completions_pbp = ("complete_pass",   "sum"),    # completions from PBP
            qb_attempts_pbp    = ("pass_attempt",    "sum"),    # attempts from PBP
            qb_pressure_count  = ("pressured",       "sum"),    # pressured dropbacks
            qb_scrambles       = ("qb_scramble",     "sum"),    # scrambles
        )
        .reset_index()
        .rename(columns={"passer_player_id": "player_id"})
    )

    # Per-attempt rates (guard against zero denominator)
    qb_agg["qb_air_yards_per_attempt"] = (
        qb_agg["qb_air_yards_total"] /
        qb_agg["qb_attempts_pbp"].replace(0, float("nan"))
    )
    qb_agg["qb_pressure_rate"] = (
        qb_agg["qb_pressure_count"] /
        qb_agg["qb_dropbacks"].replace(0, float("nan"))
    )
    qb_agg["qb_scramble_rate"] = (
        qb_agg["qb_scrambles"] /
        qb_agg["qb_dropbacks"].replace(0, float("nan"))
    )

    # Drop intermediate count columns used only for rate computation
    qb_agg.drop(columns=[
        "qb_air_yards_total", "qb_completions_pbp",
        "qb_attempts_pbp", "qb_pressure_count", "qb_scrambles",
    ], inplace=True)

    # --- Receiver aggregations ---
    # Targeted plays: pass_attempt == 1 and receiver_player_id is not null
    # (includes completions, incompletions, INTs — any pass with an intended target)
    rec_plays = pass_plays[pass_plays["receiver_player_id"].notna()].copy()

    # Breakaway on receiving: reception gaining 15+ yards counts for WR/TE/RB receiving
    rec_plays["rec_breakaway"] = (
        (rec_plays["complete_pass"] == 1) &
        (rec_plays["yards_gained"] >= 15)
    ).astype(int)

    rec_agg = (
        rec_plays
        .groupby(["receiver_player_id", "season", "week"])
        .agg(
            rec_targets        = ("pass_attempt",      "sum"),   # targets
            rec_epa_total      = ("epa",               "sum"),   # total EPA as receiver
            rec_air_yards_sum  = ("air_yards",         "sum"),   # for aDOT
            rec_yac_sum        = ("yards_after_catch",  "sum"),  # for YAC per reception
            rec_completions    = ("complete_pass",      "sum"),  # receptions
            rec_breakaways     = ("rec_breakaway",      "sum"),  # 15+ yard catches
        )
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )

    # Per-target / per-reception rates
    rec_agg["rec_epa_per_target"] = (
        rec_agg["rec_epa_total"] /
        rec_agg["rec_targets"].replace(0, float("nan"))
    )
    rec_agg["adot"] = (                                      # average depth of target
        rec_agg["rec_air_yards_sum"] /
        rec_agg["rec_targets"].replace(0, float("nan"))
    )
    rec_agg["yac_per_reception_pbp"] = (
        rec_agg["rec_yac_sum"] /
        rec_agg["rec_completions"].replace(0, float("nan"))
    )
    rec_agg["catch_rate_pbp"] = (                            # receptions / targets (std catch rate)
        rec_agg["rec_completions"] /
        rec_agg["rec_targets"].replace(0, float("nan"))
    )
    rec_agg["rec_breakaway_rate"] = (                        # 15+ yard catches / targets
        rec_agg["rec_breakaways"] /
        rec_agg["rec_targets"].replace(0, float("nan"))
    )

    # Drop intermediate sum columns
    rec_agg.drop(columns=[
        "rec_epa_total", "rec_air_yards_sum", "rec_yac_sum",
        "rec_completions", "rec_breakaways",
    ], inplace=True)

    # --- Rusher aggregations ---
    rush_plays = pbp[pbp["rush_attempt"] == 1].copy()

    rush_plays["rush_breakaway"] = (rush_plays["yards_gained"] >= 15).astype(int)

    rush_agg = (
        rush_plays[rush_plays["rusher_player_id"].notna()]
        .groupby(["rusher_player_id", "season", "week"])
        .agg(
            rush_carries       = ("rush_attempt",    "sum"),
            rush_epa_total     = ("epa",             "sum"),
            rush_yards_sum     = ("yards_gained",    "sum"),
            rush_breakaways    = ("rush_breakaway",  "sum"),
            rush_first_downs   = ("first_down_rush", "sum") if "first_down_rush" in pbp.columns else ("first_down", "sum"),
        )
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )

    rush_agg["rush_epa_per_carry"] = (
        rush_agg["rush_epa_total"] /
        rush_agg["rush_carries"].replace(0, float("nan"))
    )
    rush_agg["rush_yards_per_carry_pbp"] = (
        rush_agg["rush_yards_sum"] /
        rush_agg["rush_carries"].replace(0, float("nan"))
    )
    rush_agg["rush_breakaway_rate"] = (
        rush_agg["rush_breakaways"] /
        rush_agg["rush_carries"].replace(0, float("nan"))
    )

    rush_agg.drop(columns=[
        "rush_epa_total", "rush_yards_sum", "rush_breakaways",
    ], inplace=True)

    # --- Combine all three aggregations for this season ---
    # Outer merge so a player who threw and ran (QB) or caught and ran (RB) gets all columns
    season_agg = (
        qb_agg
        .merge(rec_agg,  on=["player_id", "season", "week"], how="outer")
        .merge(rush_agg, on=["player_id", "season", "week"], how="outer")
    )

    pbp_agg_list.append(season_agg)
    print(f"done ({len(season_agg):,} player-game rows)")
    del pbp, pass_plays, rec_plays, rush_plays  # free memory

# %%
# --- Concatenate all seasons ---

pbp_agg = pd.concat(pbp_agg_list, ignore_index=True)
del pbp_agg_list

print(f"\nPBP aggregated: {len(pbp_agg):,} rows x {pbp_agg.shape[1]} cols")
print(f"Columns: {pbp_agg.columns.tolist()}")

# %%
# --- Save PBP aggregation ---

pbp_out = RAW_DIR / f"pbp_aggregated_{SEASONS[0]}_{SEASONS[-1]}.parquet"
pbp_agg.to_parquet(pbp_out, index=False)
print(f"Saved: {pbp_out}  ({pbp_out.stat().st_size / 1_048_576:.1f} MB)")

# %%
# --- Fix 3: Remove the 1 known duplicate before joining ---
# Matthew Stafford 2010 Week 8 appears twice in the weekly pull.
# Deduplicate on (player_id, season, week) keeping the first occurrence.
n_before = len(master)
master = master.drop_duplicates(subset=["player_id", "season", "week"], keep="first").copy()
if len(master) < n_before:
    print(f"Duplicates removed: {n_before - len(master)} row(s) dropped")

# %%
# --- Join PBP aggregates onto master dataset ---
# Join key: (player_id, season, week) — unique per player per game.
# Left join: keep all master rows; PBP columns will be NaN for players
# with no PBP activity in a given game (e.g. WR with 0 targets).

n_before = len(master)
master = master.merge(pbp_agg, on=["player_id", "season", "week"], how="left")
assert len(master) == n_before, "Row count changed after PBP join — check for duplicates in pbp_agg"

print(f"\nMaster after Step 2 join: {master.shape}")

# Null audit — high null rates for position-irrelevant columns are expected:
# QB rows will have null rec_* columns; WR/TE/RB rows will have null qb_* columns.
pbp_new_cols = [
    "qb_epa_per_dropback", "qb_cpoe", "qb_air_yards_per_attempt",
    "qb_pressure_rate", "qb_scramble_rate",
    "rec_epa_per_target", "adot", "yac_per_reception_pbp", "catch_rate_pbp",
    "rush_epa_per_carry", "rush_yards_per_carry_pbp", "rush_breakaway_rate",
]
print("\nNull rates for new PBP columns (expected high for cross-position columns):")
for col in pbp_new_cols:
    if col in master.columns:
        null_pct = 100 * master[col].isna().mean()
        print(f"  {col}: {null_pct:.1f}% null overall")

# %%
# --- Save master after Step 2 ---
save_master(master, step=2)

# %% [markdown]
# ## Step 3: Snap Counts
#
# Pulls offensive snap counts from nfl_data_py for seasons 2013+.
# (Data not available before 2013. Pre-2013 rows get NaN for snap columns.)
#
# Adds three columns to master:
#   offense_snaps     — raw offensive snap count for the player in that game
#   offense_pct       — fraction of team offensive snaps the player was on field (0.0-1.0)
#   snap_count_source — 'snap_data' for 2013+ rows, 'unavailable' for 1999-2012 rows
#
# Key engineering challenge: snap counts use PFR player IDs (e.g. 'BrowSp00'),
# but master uses GSIS player IDs (e.g. '00-0026498'). We build a crosswalk
# from nfl_data_py's player table which contains both ID systems.
#
# Inclusion filter: drop rows where targets = 0 AND carries = 0 AND attempts = 0.
# This removes true noise rows (mis-tagged linemen, QB2s who never snapped,
# data errors) while retaining all fantasy-relevant players regardless of snap share.
# A percentage threshold is intentionally not used — low-snap players (red zone
# specialists, split backfield RBs, injured starters returning) are fantasy-scoreable
# and must be present in training so the model covers that region of feature space.
#
# Output: snap counts joined to master; zero-involvement rows removed.

# %%
SNAP_SEASONS = list(range(2013, 2026))  # snap data available 2013+

# --- Build PFR -> GSIS player ID crosswalk ---
print("Building PFR -> GSIS player ID crosswalk...")
players_df = nfl.import_players()

# Keep only rows where both IDs are present
crosswalk = (
    players_df[["gsis_id", "pfr_id"]]
    .dropna(subset=["gsis_id", "pfr_id"])
    .drop_duplicates(subset=["pfr_id"])   # pfr_id must be unique for a clean join
    .copy()
)
print(f"  Crosswalk size: {len(crosswalk):,} players with both GSIS and PFR IDs")

# %%
# --- Pull snap counts for all available seasons ---
print(f"\nPulling snap counts for {len(SNAP_SEASONS)} seasons: "
      f"{SNAP_SEASONS[0]}-{SNAP_SEASONS[-1]}")

snap_raw = nfl.import_snap_counts(years=SNAP_SEASONS)

# Regular season only
snap_raw = snap_raw[snap_raw["game_type"] == "REG"].copy()
print(f"Raw snap pull (REG only): {len(snap_raw):,} rows")

# %%
# --- Join GSIS player_id onto snap data via crosswalk ---
snap_raw = snap_raw.merge(
    crosswalk.rename(columns={"pfr_id": "pfr_player_id", "gsis_id": "player_id"}),
    on="pfr_player_id",
    how="left",
)

n_matched   = snap_raw["player_id"].notna().sum()
n_unmatched = snap_raw["player_id"].isna().sum()
print(f"  Crosswalk match: {n_matched:,} matched, {n_unmatched:,} unmatched "
      f"({100*n_unmatched/len(snap_raw):.1f}% unmatched)")

# Unmatched rows cannot be joined to master — drop them
snap_raw = snap_raw[snap_raw["player_id"].notna()].copy()

# %%
# --- Keep only offensive skill positions ---
# Snap data covers all positions including OL, DL, LB, CB, S.
# We only need offensive snap counts for QB, RB, WR, TE.
# (DST snap counts handled separately in the DST step.)
SNAP_SKILL = {"QB", "RB", "WR", "TE", "HB", "FB"}   # HB and FB map to RB in master
snap_skill = snap_raw[snap_raw["position"].isin(SNAP_SKILL)].copy()
print(f"After skill position filter: {len(snap_skill):,} rows")
print(f"Position breakdown:\n{snap_skill['position'].value_counts().to_string()}")

# %%
# --- Select and rename columns needed for master join ---
snap_clean = (
    snap_skill[["player_id", "season", "week", "offense_snaps", "offense_pct"]]
    .rename(columns={
        "offense_snaps": "offense_snaps",   # raw count
        "offense_pct":   "offense_pct",     # fraction 0.0-1.0
    })
    .copy()
)

# Deduplicate: a player should have exactly one row per (player_id, season, week)
n_before = len(snap_clean)
snap_clean = snap_clean.drop_duplicates(subset=["player_id", "season", "week"], keep="first")
if len(snap_clean) < n_before:
    print(f"Snap duplicates removed: {n_before - len(snap_clean)} rows")

# %%
# --- Save raw snap output ---
snap_out = RAW_DIR / f"snap_counts_{SNAP_SEASONS[0]}_{SNAP_SEASONS[-1]}.parquet"
snap_clean.to_parquet(snap_out, index=False)
print(f"Saved: {snap_out}  ({snap_out.stat().st_size / 1_048_576:.2f} MB)")

# %%
# --- Join snap counts onto master ---
n_before = len(master)
master = master.merge(snap_clean, on=["player_id", "season", "week"], how="left")
assert len(master) == n_before, "Row count changed after snap join"

# Add source flag so the model knows which rows have real snap data
master["snap_count_source"] = master["offense_pct"].apply(
    lambda x: "snap_data" if pd.notna(x) else "unavailable"
)

print(f"\nSnap data coverage:")
print(master.groupby("snap_count_source").size().to_string())

# %%
# --- Apply involvement filter ---
# Goal: remove rows where a player had zero meaningful involvement in the game.
# These are true noise rows — linemen mis-tagged as skill positions, QB2s who
# never took a snap, jersey-number errors — not low-snap fantasy-relevant players.
#
# Criterion: drop row if targets = 0 AND carries = 0 AND attempts = 0.
# Applied uniformly across all seasons regardless of snap data availability.
#
# We do NOT use a snap share percentage threshold because:
#   1. Fantasy-relevant players can have well under 25% snap share (red zone
#      specialists, receiving backs in split backfields, injured starters returning)
#   2. Filtering by snap % would remove valid low-usage rows from training,
#      leaving the model blind to that region of feature space at prediction time
#   3. The involvement criterion already catches true noise rows — a player
#      with zero touches has nothing to teach the model about their position

n_before = len(master)

zero_involvement = (
    (master["targets"].fillna(0)  == 0) &
    (master["carries"].fillna(0)  == 0) &
    (master["attempts"].fillna(0) == 0)
)

master = master[~zero_involvement].copy()

print(f"\nInvolvement filter:")
print(f"  Rows before : {n_before:,}")
print(f"  Dropped (zero targets + carries + attempts): {zero_involvement.sum():,}")
print(f"  Rows after  : {len(master):,}")

print(f"\nMaster after Step 3: {master.shape}")
print(f"Position breakdown:\n{master['position'].value_counts().to_string()}")
print(f"\nSnap data coverage after filter:")
print(master.groupby("snap_count_source").size().to_string())

# %%
# --- Save master after Step 3 ---
save_master(master, step=3)

# %% [markdown]
# ## Step 4: NextGen Stats (NGS) + Pre-2006 Air Yards Fix
#
# Two things happen in this step:
#
# **Part A — Pre-2006 air yards fix:**
# Air yards were not tracked before 2006. nfl_data_py stores these as 0.0
# rather than NaN, which would corrupt rolling averages (a QB with 30 attempts
# showing 0 air yards per attempt is wrong, not zero). We NaN-ify:
#   - passing_air_yards where season < 2006 and attempts > 0
#   - receiving_air_yards, adot, air_yards_share, wopr where season < 2006
#   - racr where season < 2006 (derived from air yards, same issue)
#   - receiving_yac, passing_yac where season < 2006 (also untracked pre-2006)
#
# **Part B — NextGen Stats (2016+):**
# Pulls NGS weekly data for passing, receiving, and rushing.
# Adds the following columns to master (all NaN for pre-2016 rows):
#
#   QB / passing:
#     ngs_avg_time_to_throw        — seconds from snap to release
#     ngs_avg_intended_air_yards   — average air yards per attempt (NGS version)
#     ngs_aggressiveness           — % of passes into tight coverage
#     ngs_completion_pct_above_exp — CPOE equivalent from NGS
#
#   WR / TE / receiving:
#     ngs_avg_cushion              — yards between receiver and nearest defender at snap
#     ngs_avg_separation           — yards of separation at time of target
#     ngs_avg_yac_above_expectation — YAC above expected given situation
#
#   RB / rushing:
#     ngs_rush_yards_over_expected_per_att — yards gained vs model expectation per carry
#     ngs_rush_pct_over_expected           — % of rushes exceeding expected yards
#     ngs_avg_time_to_los                  — seconds from snap to line of scrimmage
#
# Sets has_nextgen = 1 for all 2016+ rows, 0 otherwise.
#
# Output: NGS columns joined to master; pre-2006 air yards corrected.

# %%
# --- Part A: Fix pre-2006 air yards zeros ---
# These columns contain 0.0 where the true value is unknown (not tracked).
# Replacing with NaN prevents rolling averages from being anchored to 0.

AIR_YARDS_COLS_QB = ["passing_air_yards", "passing_yac", "qb_air_yards_per_attempt"]
AIR_YARDS_COLS_SKILL = [
    "receiving_air_yards", "receiving_yac",
    "air_yards_share", "wopr", "racr",
    "adot", "yac_per_reception_pbp",
]

pre2006 = master["season"] < 2006

for col in AIR_YARDS_COLS_QB:
    if col in master.columns:
        # Only zero-out QB rows where attempts > 0 (confirms they were actually tracking)
        mask = pre2006 & (master["position"] == "QB") & (master["attempts"] > 0)
        n_fixed = mask.sum()
        master.loc[mask, col] = float("nan")
        print(f"  {col}: {n_fixed:,} pre-2006 QB zeros -> NaN")

for col in AIR_YARDS_COLS_SKILL:
    if col in master.columns:
        mask = pre2006 & (master["position"].isin(["WR", "TE", "RB"]))
        n_fixed = (master.loc[mask, col] == 0).sum()
        master.loc[mask, col] = master.loc[mask, col].replace(0.0, float("nan"))
        print(f"  {col}: {n_fixed:,} pre-2006 skill zeros -> NaN")

print(f"\nPre-2006 air yards fix complete.")

# %%
# --- Part B: Pull NextGen Stats ---
NGS_SEASONS = list(range(2016, 2026))  # NGS available from 2016

print(f"\nPulling NGS data for {len(NGS_SEASONS)} seasons: "
      f"{NGS_SEASONS[0]}-{NGS_SEASONS[-1]}")

# --- Passing NGS ---
ngs_pass_raw = nfl.import_ngs_data(stat_type="passing", years=NGS_SEASONS)

# Keep weekly rows only (week=0 are season totals)
ngs_pass = ngs_pass_raw[
    (ngs_pass_raw["week"] > 0) &
    (ngs_pass_raw["season_type"] == "REG")
].copy()

ngs_pass = ngs_pass[[
    "player_gsis_id", "season", "week",
    "avg_time_to_throw",
    "avg_intended_air_yards",
    "aggressiveness",
    "completion_percentage_above_expectation",
]].rename(columns={
    "player_gsis_id":                         "player_id",
    "avg_time_to_throw":                      "ngs_avg_time_to_throw",
    "avg_intended_air_yards":                 "ngs_avg_intended_air_yards",
    "aggressiveness":                         "ngs_aggressiveness",
    "completion_percentage_above_expectation":"ngs_completion_pct_above_exp",
})

print(f"  Passing NGS: {len(ngs_pass):,} weekly rows")

# --- Receiving NGS ---
ngs_rec_raw = nfl.import_ngs_data(stat_type="receiving", years=NGS_SEASONS)

ngs_rec = ngs_rec_raw[
    (ngs_rec_raw["week"] > 0) &
    (ngs_rec_raw["season_type"] == "REG")
].copy()

ngs_rec = ngs_rec[[
    "player_gsis_id", "season", "week",
    "avg_cushion",
    "avg_separation",
    "avg_yac_above_expectation",
]].rename(columns={
    "player_gsis_id":            "player_id",
    "avg_cushion":               "ngs_avg_cushion",
    "avg_separation":            "ngs_avg_separation",
    "avg_yac_above_expectation": "ngs_avg_yac_above_expectation",
})

print(f"  Receiving NGS: {len(ngs_rec):,} weekly rows")

# --- Rushing NGS ---
ngs_rush_raw = nfl.import_ngs_data(stat_type="rushing", years=NGS_SEASONS)

ngs_rush = ngs_rush_raw[
    (ngs_rush_raw["week"] > 0) &
    (ngs_rush_raw["season_type"] == "REG")
].copy()

ngs_rush = ngs_rush[[
    "player_gsis_id", "season", "week",
    "rush_yards_over_expected_per_att",
    "rush_pct_over_expected",
    "avg_time_to_los",
]].rename(columns={
    "player_gsis_id":                  "player_id",
    "rush_yards_over_expected_per_att":"ngs_rush_yards_over_expected_per_att",
    "rush_pct_over_expected":          "ngs_rush_pct_over_expected",
    "avg_time_to_los":                 "ngs_avg_time_to_los",
})

print(f"  Rushing NGS: {len(ngs_rush):,} weekly rows")

# %%
# --- Deduplicate NGS tables ---
# NGS can have duplicate rows in edge cases (player on multiple teams same week)
for name, df in [("ngs_pass", ngs_pass), ("ngs_rec", ngs_rec), ("ngs_rush", ngs_rush)]:
    n_before = len(df)
    df.drop_duplicates(subset=["player_id", "season", "week"], keep="first", inplace=True)
    if len(df) < n_before:
        print(f"  {name}: {n_before - len(df)} duplicates removed")

# %%
# --- Save raw NGS outputs ---
ngs_pass.to_parquet(RAW_DIR / f"ngs_passing_{NGS_SEASONS[0]}_{NGS_SEASONS[-1]}.parquet", index=False)
ngs_rec.to_parquet(RAW_DIR  / f"ngs_receiving_{NGS_SEASONS[0]}_{NGS_SEASONS[-1]}.parquet", index=False)
ngs_rush.to_parquet(RAW_DIR / f"ngs_rushing_{NGS_SEASONS[0]}_{NGS_SEASONS[-1]}.parquet", index=False)
print("NGS raw files saved.")

# %%
# --- Join NGS onto master ---
# All three are left joins — NaN for pre-2016 rows and for players
# below the NGS minimum snap threshold (~20 snaps for NGS eligibility).

n_before = len(master)

master = master.merge(ngs_pass,  on=["player_id", "season", "week"], how="left")
master = master.merge(ngs_rec,   on=["player_id", "season", "week"], how="left")
master = master.merge(ngs_rush,  on=["player_id", "season", "week"], how="left")

assert len(master) == n_before, "Row count changed after NGS join"

# %%
# --- Set has_nextgen flag ---
# 1 for rows from 2016 onward where NGS data exists in principle.
# 0 for pre-2016 rows. The model uses this to interpret NaN NGS columns correctly.
master["has_nextgen"] = (master["season"] >= 2016).astype(int)

print(f"\nNGS join complete.")
print(f"has_nextgen distribution:\n{master['has_nextgen'].value_counts().to_string()}")

# Null rate check for NGS columns among 2016+ rows only
ngs_cols = [
    "ngs_avg_time_to_throw", "ngs_avg_intended_air_yards",
    "ngs_aggressiveness", "ngs_completion_pct_above_exp",
    "ngs_avg_cushion", "ngs_avg_separation", "ngs_avg_yac_above_expectation",
    "ngs_rush_yards_over_expected_per_att", "ngs_rush_pct_over_expected",
    "ngs_avg_time_to_los",
]
post2016 = master[master["season"] >= 2016]
print(f"\nNGS null rates among 2016+ rows (n={len(post2016):,}):")
for col in ngs_cols:
    if col in master.columns:
        pct = 100 * post2016[col].isna().mean()
        print(f"  {col}: {pct:.1f}% null")

print(f"\nMaster after Step 4: {master.shape}")

# %% [markdown]
# ## Step 5: Rosters & Physical Traits
#
# Two data pulls joined in sequence:
#
# **Part A — Weekly rosters (2002–2025):**
# Pulls birth_date, height (inches), weight (lbs), years_exp, pfr_id per player per week.
# Deduplicated to one row per (player_id, season) — physical traits don't change within a season.
# Computes:
#   - age: continuous years at game_date — primary aging curve input
#   - bmi: (weight / height^2) × 703 — body type composite
#   - games_played_current_season: games played before this game in the current season (0-indexed)
#
# **Part B — Combine data (2000–2025 draft classes):**
# Pulls forty_yard_dash, vertical_jump_inches, three_cone_drill, broad_jump_inches.
# Joined via pfr_id crosswalk (present in both roster and combine tables).
# Computes speed_score = (weight × 200) / (forty_yard_dash^4).
#
# NaN handling:
#   - Players missing from roster data (very old/obscure): physical cols NaN
#   - Players never at combine (UDFAs, pre-2000 draft): all combine metrics NaN
#   - LightGBM handles all NaN natively — no imputation needed
#
# Output: +11 columns (age, height, weight, bmi, years_exp, games_played_current_season,
#          forty_yard_dash, vertical_jump_inches, three_cone_drill, broad_jump_inches, speed_score)

# %%
# --- Part A: Weekly rosters ---
ROSTER_SEASONS = list(range(2002, 2026))  # weekly rosters available from 2002 — matches dataset start year

print(f"Pulling weekly rosters for {len(ROSTER_SEASONS)} seasons: "
      f"{ROSTER_SEASONS[0]}-{ROSTER_SEASONS[-1]}")

rosters_raw = nfl.import_weekly_rosters(years=ROSTER_SEASONS)
print(f"  Raw roster rows: {len(rosters_raw):,}")

# %%
# --- Filter to skill positions and keep needed columns ---
ROSTER_COLS = ['player_id', 'season', 'week', 'birth_date', 'height', 'weight', 'years_exp', 'pfr_id']
roster = rosters_raw[
    rosters_raw['position'].isin(['QB', 'RB', 'WR', 'TE', 'FB', 'HB'])
][ROSTER_COLS].copy()

# Deduplicate to one row per (player_id, season) — take first non-null per season
# (physical traits don't change within a season; weekly roster repeats same values)
roster_season = (
    roster
    .sort_values(['player_id', 'season', 'week'])
    .groupby(['player_id', 'season'], as_index=False)
    .first()
    .drop(columns=['week'])
)
print(f"  After dedup to (player_id, season): {len(roster_season):,} rows")

# %%
# --- Join rosters to master ---
# Drop these columns if they already exist (idempotent re-run safety)
ROSTER_JOIN_COLS = ['birth_date', 'height', 'weight', 'years_exp', 'pfr_id']
master.drop(columns=[c for c in ROSTER_JOIN_COLS if c in master.columns], inplace=True)

n_before = len(master)
master = master.merge(
    roster_season[['player_id', 'season', 'birth_date', 'height', 'weight', 'years_exp', 'pfr_id']],
    on=['player_id', 'season'],
    how='left'
)
assert len(master) == n_before, "Row count changed after roster join"

print(f"  Roster join complete. Null rates:")
for col in ['height', 'weight', 'birth_date', 'years_exp']:
    print(f"    {col}: {master[col].isna().mean()*100:.1f}%")

# %%
# --- Pull game_date from schedules and join to master ---
# import_weekly_data has no date column — schedules is the authoritative source for game dates.
# Pull once here; Step 6 will pull schedules again for home/away, rest days, stadium metadata.
# IMPORTANT: schedule game_id uses away_home order (e.g. PIT_ATL) but master sorts alphabetically
# (ATL_PIT). Rebuild a sorted game_id from schedule's away/home team columns to match master format.
sched_raw = nfl.import_schedules(years=SEASONS)

# import_schedules keeps historical abbreviations (SD, STL, OAK) while import_weekly_data
# normalizes to current names (LAC, LA, LV). Map before building sorted game_id.
TEAM_ABBREV_MAP = {'SD': 'LAC', 'STL': 'LA', 'OAK': 'LV'}
sched_raw['away_team'] = sched_raw['away_team'].replace(TEAM_ABBREV_MAP)
sched_raw['home_team'] = sched_raw['home_team'].replace(TEAM_ABBREV_MAP)

sched_raw['game_id_sorted'] = (
    sched_raw['season'].astype(str)
    + '_'
    + sched_raw['week'].astype(str).str.zfill(2)
    + '_'
    + sched_raw.apply(lambda r: '_'.join(sorted([str(r['away_team']), str(r['home_team'])])), axis=1)
)
game_dates = sched_raw[['game_id_sorted', 'gameday']].copy()
game_dates['gameday'] = pd.to_datetime(game_dates['gameday'])
game_dates = game_dates.rename(columns={'gameday': 'game_date', 'game_id_sorted': 'game_id'})
game_dates = game_dates.drop_duplicates(subset=['game_id'])

if 'game_date' in master.columns:
    master.drop(columns=['game_date'], inplace=True)

n_before = len(master)
master = master.merge(game_dates, on='game_id', how='left')
assert len(master) == n_before, "Row count changed after game_date join"
print(f"  game_date joined. Null: {master['game_date'].isna().mean()*100:.1f}%")

# %%
# --- Compute age at game_date ---
master['birth_date'] = pd.to_datetime(master['birth_date'])
master['age'] = (master['game_date'] - master['birth_date']).dt.days / 365.25

print(f"\n  Age computed.")
print(f"    Range: {master['age'].dropna().min():.1f} - {master['age'].dropna().max():.1f} years")
print(f"    Null: {master['age'].isna().mean()*100:.1f}%")

# %%
# --- Compute BMI ---
# bmi = (weight_lbs / height_inches^2) * 703
master['bmi'] = (master['weight'] / (master['height'] ** 2)) * 703

print(f"\n  BMI computed.")
print(f"    Range: {master['bmi'].dropna().min():.1f} - {master['bmi'].dropna().max():.1f}")
print(f"    Null: {master['bmi'].isna().mean()*100:.1f}%")

# %%
# --- Compute games_played_current_season ---
# Count of games played in the current season BEFORE this game.
# Sort chronologically then cumcount within (player_id, season) gives 0 for first game,
# 1 for second, etc. — i.e., "how many games has this player already played this season".
master = master.sort_values(['player_id', 'season', 'week']).copy()
master['games_played_current_season'] = (
    master.groupby(['player_id', 'season']).cumcount()
)
master = master.sort_index()

print(f"\n  games_played_current_season computed.")
print(f"    Max value: {master['games_played_current_season'].max()}")
print(f"    Distribution:\n{master['games_played_current_season'].value_counts().sort_index().head(10).to_string()}")

# %%
# --- Part B: Combine data ---
COMBINE_SEASONS = list(range(2000, 2026))

print(f"\nPulling combine data for {len(COMBINE_SEASONS)} draft classes: "
      f"{COMBINE_SEASONS[0]}-{COMBINE_SEASONS[-1]}")

combine_raw = nfl.import_combine_data(years=COMBINE_SEASONS)

# Filter to skill positions and keep only what we need
combine = combine_raw[
    combine_raw['pos'].isin(['QB', 'RB', 'WR', 'TE', 'FB'])
][['pfr_id', 'forty', 'vertical', 'cone', 'broad_jump']].copy()

combine = combine.rename(columns={
    'forty':      'forty_yard_dash',
    'vertical':   'vertical_jump_inches',
    'cone':       'three_cone_drill',
    'broad_jump': 'broad_jump_inches',
})

# Remove duplicates — some players appear twice in combine data (re-tested)
n_before_combine = len(combine)
combine.drop_duplicates(subset=['pfr_id'], keep='first', inplace=True)
if len(combine) < n_before_combine:
    print(f"  {n_before_combine - len(combine)} combine duplicates removed")

print(f"  Combine rows: {len(combine):,}")

# %%
# --- Join combine to master via pfr_id ---
COMBINE_JOIN_COLS = ['forty_yard_dash', 'vertical_jump_inches', 'three_cone_drill', 'broad_jump_inches']
master.drop(columns=[c for c in COMBINE_JOIN_COLS if c in master.columns], inplace=True)

n_before = len(master)
master = master.merge(combine, on='pfr_id', how='left')
assert len(master) == n_before, "Row count changed after combine join"

print(f"  Combine join complete. Null rates:")
for col in ['forty_yard_dash', 'vertical_jump_inches', 'three_cone_drill', 'broad_jump_inches']:
    print(f"    {col}: {master[col].isna().mean()*100:.1f}%")

# %%
# --- Compute speed_score ---
# speed_score = (weight × 200) / (forty_yard_dash^4)
# Uses current-season roster weight (more accurate than combine weight for multi-year players)
master['speed_score'] = (master['weight'] * 200) / (master['forty_yard_dash'] ** 4)

print(f"\n  Speed score computed.")
print(f"    Range: {master['speed_score'].dropna().min():.1f} - {master['speed_score'].dropna().max():.1f}")
print(f"    Null: {master['speed_score'].isna().mean()*100:.1f}%")

# %%
# --- Drop working columns not needed as model features ---
# pfr_id: crosswalk only, not a model feature
# birth_date: used to compute age, not a raw feature
master.drop(columns=['pfr_id', 'birth_date'], inplace=True, errors='ignore')

print(f"\nMaster after Step 5: {master.shape}")
print(f"Columns added: age, height, weight, bmi, years_exp, games_played_current_season,")
print(f"               forty_yard_dash, vertical_jump_inches, three_cone_drill, broad_jump_inches, speed_score")

# %%
# --- Save master after Step 5 ---
save_master(master, step=5)

# %%
# =============================================================================
# STEP 6 — Schedule context: home/away, rest days, dome/turf, altitude,
#           temperature, wind, divisional game flag
# =============================================================================
#
# Source: nfl.import_schedules — one row per game, contains both teams.
# We join on sorted game_id (same pattern as the game_date join in Step 5).
# Then derive per-player features based on whether their team was home or away.
#
# Features added:
#   game_location   — 'home', 'away', or 'neutral' (neutral = international/SB/neutral site)
#   rest_days       — days since last game for the player's team (7 = normal week,
#                     <7 = short week e.g. Thursday game, >7 = after bye)
#   is_dome         — 1 if game played under a fixed or retractable closed roof
#   is_turf         — 1 if artificial surface, 0 if natural grass
#   stadium_altitude— stadium elevation in feet (0 for sea-level venues;
#                     only Denver ~5280 ft and Mexico City ~7350 ft are material)
#   game_temp       — outdoor temperature in °F (NaN for dome games)
#   game_wind       — wind speed in mph (NaN for dome games)
#   div_game        — 1 if divisional matchup

print("=" * 60)
print("STEP 6 — Schedule context features")
print("=" * 60)

# %%
# --- Pull schedules and normalise team abbreviations ---
STEP6_ABBREV_MAP = {'SD': 'LAC', 'STL': 'LA', 'OAK': 'LV'}

sched6 = nfl.import_schedules(years=list(SEASONS))
sched6['away_team'] = sched6['away_team'].replace(STEP6_ABBREV_MAP)
sched6['home_team'] = sched6['home_team'].replace(STEP6_ABBREV_MAP)
sched6 = sched6[sched6['game_type'] == 'REG'].copy()

# Build sorted game_id to match master format
sched6['game_id'] = (
    sched6['season'].astype(str) + '_' +
    sched6['week'].astype(str).str.zfill(2) + '_' +
    sched6.apply(lambda r: '_'.join(sorted([str(r['away_team']), str(r['home_team'])])), axis=1)
)

sched6 = sched6[['game_id', 'away_team', 'home_team',
                  'away_rest', 'home_rest',
                  'roof', 'surface',
                  'div_game', 'temp', 'wind',
                  'stadium_id', 'location']].drop_duplicates(subset=['game_id']).copy()

print(f"  Schedule rows (REG only): {len(sched6):,}")

# %%
# --- Derive is_dome and is_turf from roof/surface ---
DOME_ROOFS    = {'dome', 'closed'}
GRASS_SURFACES = {'grass', 'dessograss'}

sched6['is_dome'] = sched6['roof'].str.lower().isin(DOME_ROOFS).astype(int)
sched6['is_turf'] = (~sched6['surface'].str.lower().isin(GRASS_SURFACES) &
                     sched6['surface'].notna()).astype(int)

# %%
# --- Stadium altitude lookup (feet above sea level) ---
# Only venues materially above sea level are listed; all others default to 0.
STADIUM_ALTITUDE = {
    'DEN00': 5280,   # Empower Field at Mile High, Denver CO
    'MEX00': 7349,   # Estadio Azteca, Mexico City
    'PHO00': 1100,   # State Farm Stadium, Glendale AZ
    'PHO99': 1135,   # Sun Devil Stadium, Tempe AZ
    'NAS00':  597,   # Nissan Stadium, Nashville TN
    'ATL00':  997,   # Georgia Dome, Atlanta GA
    'ATL97': 1050,   # Mercedes-Benz Stadium, Atlanta GA
    'DAL00': 600,    # AT&T Stadium, Arlington TX
    'DAL99': 530,    # Texas Stadium, Irving TX
}
sched6['stadium_altitude'] = sched6['stadium_id'].map(STADIUM_ALTITUDE).fillna(0).astype(float)

# %%
# --- Join schedule features to master ---
STEP6_JOIN_COLS = ['game_location', 'rest_days', 'rest_days_opponent', 'is_dome', 'is_turf',
                   'stadium_altitude', 'game_temp', 'game_wind', 'div_game']
master.drop(columns=[c for c in STEP6_JOIN_COLS if c in master.columns], inplace=True)

n_before = len(master)
master = master.merge(
    sched6[['game_id', 'away_team', 'home_team',
            'away_rest', 'home_rest',
            'is_dome', 'is_turf', 'stadium_altitude',
            'div_game', 'temp', 'wind', 'location']],
    on='game_id', how='left'
)
assert len(master) == n_before, f"Row count changed after Step 6 join: {n_before} -> {len(master)}"

# %%
# --- Derive player-level features from the joined team columns ---
# game_location: 'neutral' if schedule location == 'Neutral', else 'home'/'away'
is_home_flag = (master['team'] == master['home_team'])
is_neutral   = master['location'].str.lower().str.strip() == 'neutral'
master['game_location'] = 'away'
master.loc[is_home_flag & ~is_neutral, 'game_location'] = 'home'
master.loc[is_neutral, 'game_location'] = 'neutral'

master['rest_days']      = master.apply(
    lambda r: r['home_rest'] if r['team'] == r['home_team'] else r['away_rest'], axis=1
)
master['rest_days_opponent'] = master.apply(
    lambda r: r['away_rest'] if r['team'] == r['home_team'] else r['home_rest'], axis=1
)

# Temperature and wind are only meaningful outdoors; dome games get NaN
master['game_temp'] = master['temp'].where(master['is_dome'] == 0)
master['game_wind'] = master['wind'].where(master['is_dome'] == 0)

# div_game: cast to int (may come as float after merge)
master['div_game'] = master['div_game'].fillna(0).astype(int)

# Drop the intermediate team/rest columns used for derivation
master.drop(columns=['away_team', 'home_team', 'away_rest', 'home_rest', 'temp', 'wind', 'location'],
            inplace=True)

# %%
# --- Validation ---
print(f"\n  Null rates after Step 6:")
for col in STEP6_JOIN_COLS:
    pct = master[col].isna().mean() * 100
    print(f"    {col}: {pct:.1f}%")

print(f"\n  game_location distribution:")
print(master['game_location'].value_counts().to_string())

print(f"\n  rest_days distribution (top 10):")
print(master['rest_days'].value_counts().sort_index().to_string())

print(f"\n  is_dome: {master['is_dome'].mean()*100:.1f}% dome games")
print(f"  is_turf: {master['is_turf'].mean()*100:.1f}% turf games")
print(f"  div_game: {master['div_game'].mean()*100:.1f}% divisional games")
print(f"  altitude > 1000 ft: {(master['stadium_altitude'] > 1000).mean()*100:.1f}% of games")
print(f"\n  Master after Step 6: {master.shape}")

# %%
# --- Save master after Step 6 ---
save_master(master, step=6)

# %%
# =============================================================================
# STEP 7 — Weather (temperature, precipitation, weather_type categorical)
# =============================================================================
#
# Source: Open-Meteo historical weather API (free, no key required).
# For each unique outdoor (stadium, game_date) pair we fetch hourly
# temperature and precipitation, then extract the 4-hour window starting
# at kickoff to characterise game conditions.
#
# Dome games skip the API and get weather_type = 'dome'.
#
# Precipitation threshold: 2.5 mm total over the 4-hour game window.
# This is ~0.1 inches — the standard US threshold for "measurable precipitation"
# that would actually affect gameplay (light drizzle < 2.5 mm stays 'clear').
#
# weather_type categories (from plan):
#   'dome'       — fixed or retractable-closed roof, weather irrelevant
#   'snow'       — temp < 32°F AND precip >= 2.5 mm
#   'cold_clear' — temp < 32°F AND precip <  2.5 mm
#   'rain'       — temp >= 32°F AND precip >= 2.5 mm
#   'clear'      — temp >= 32°F AND precip <  2.5 mm
#
# Columns added: weather_type (categorical string)
# game_temp and game_wind already added in Step 6 from schedule data.

print("=" * 60)
print("STEP 7 — Weather")
print("=" * 60)

# %%
import time
import requests

# Stadium coordinates keyed by stadium_id.
# Dome stadiums are included for completeness; they won't be queried.
STADIUM_COORDS = {
    'ATL00': (33.757, -84.401),   # Georgia Dome (dome)
    'ATL97': (33.755, -84.401),   # Mercedes-Benz Stadium (dome)
    'BAL00': (39.278, -76.623),   # M&T Bank Stadium
    'BOS00': (42.091, -71.264),   # Gillette Stadium
    'BRG00': (30.412, -91.183),   # Tiger Stadium LSU
    'BUF00': (42.774, -78.787),   # Highmark Stadium
    'BUF01': (43.641, -79.389),   # Rogers Centre (dome)
    'CAR00': (35.225, -80.853),   # Bank of America Stadium
    'CHI98': (41.862, -87.617),   # Soldier Field
    'CHI99': (40.096, -88.236),   # Memorial Stadium Champaign
    'CIN00': (39.095, -84.516),   # Paycor Stadium
    'CLE00': (41.506, -81.699),   # Cleveland Browns Stadium
    'DAL00': (32.748, -97.093),   # AT&T Stadium (retractable dome)
    'DAL99': (32.843, -96.958),   # Texas Stadium (open hole in roof — outdoor)
    'DEN00': (39.744, -105.020),  # Empower Field at Mile High
    'DET00': (42.340, -83.046),   # Ford Field (dome)
    'FRA00': (50.069,   8.645),   # Deutsche Bank Park, Frankfurt
    'GER00': (48.219,  11.625),   # Allianz Arena, Munich
    'GNB00': (44.501, -88.062),   # Lambeau Field
    'HOU00': (29.685, -95.411),   # NRG Stadium (dome)
    'IND00': (39.760, -86.164),   # Lucas Oil Stadium (dome)
    'IND99': (39.762, -86.157),   # RCA Dome (dome)
    'JAX00': (30.324, -81.638),   # TIAA Bank Field
    'KAN00': (39.049, -94.484),   # Arrowhead Stadium
    'LAX01': (33.953, -118.339),  # SoFi Stadium (dome)
    'LAX97': (33.864, -118.261),  # Dignity Health Sports Park
    'LAX99': (34.014, -118.288),  # LA Memorial Coliseum
    'LON00': (51.556,  -0.280),   # Wembley Stadium
    'LON01': (51.456,  -0.341),   # Twickenham Stadium
    'LON02': (51.604,  -0.066),   # Tottenham Hotspur Stadium
    'MEX00': (19.303, -99.151),   # Estadio Azteca
    'MIA00': (25.958, -80.239),   # Hard Rock Stadium
    'MIN00': (44.974, -93.258),   # Metrodome (dome)
    'MIN01': (44.974, -93.258),   # U.S. Bank Stadium (dome)
    'MIN98': (44.978, -93.228),   # TCF Bank Stadium
    'NAS00': (36.166, -86.771),   # Nissan Stadium
    'NOR00': (29.951, -90.081),   # Superdome (dome)
    'NYC00': (40.814, -74.074),   # Giants Stadium
    'NYC01': (40.813, -74.074),   # MetLife Stadium
    'OAK00': (37.751, -122.201),  # Oakland Coliseum
    'PHI00': (39.901, -75.168),   # Lincoln Financial Field
    'PHI99': (39.902, -75.171),   # Veterans Stadium
    'PHO00': (33.528, -112.263),  # State Farm Stadium (retractable dome)
    'PHO99': (33.426, -111.933),  # Sun Devil Stadium
    'PIT00': (40.447, -80.016),   # Acrisure Stadium
    'SAN00': (29.419, -98.473),   # Alamodome (dome)
    'SAO00': (-23.545, -46.474),  # Arena Corinthians, São Paulo
    'SDG00': (32.783, -117.120),  # Qualcomm Stadium
    'SEA00': (47.595, -122.332),  # Lumen Field
    'SFO00': (37.714, -122.386),  # Candlestick Park
    'SFO01': (37.403, -121.970),  # Levi's Stadium
    'STL00': (38.633, -90.188),   # Edward Jones Dome (dome)
    'TAM00': (27.976, -82.503),   # Raymond James Stadium
    'VEG00': (36.091, -115.184),  # Allegiant Stadium (dome)
    'WAS00': (38.908, -76.864),   # FedExField
}

PRECIP_THRESHOLD_MM = 2.5  # total precipitation over game window to classify as rain/snow

# %%
# Build a lookup: (stadium_id, game_date_str) → precipitation_mm, for outdoor games only.
# One API call per outdoor stadium covering the full 2002-2025 date range.

# Pull the schedule data we need (game_id, stadium_id, gameday, gametime, is_dome)
sched_weather = nfl.import_schedules(years=list(SEASONS))
STEP7_ABBREV_MAP = {'SD': 'LAC', 'STL': 'LA', 'OAK': 'LV'}
sched_weather['away_team'] = sched_weather['away_team'].replace(STEP7_ABBREV_MAP)
sched_weather['home_team'] = sched_weather['home_team'].replace(STEP7_ABBREV_MAP)
sched_weather = sched_weather[sched_weather['game_type'] == 'REG'].copy()
sched_weather['game_id_sorted'] = (
    sched_weather['season'].astype(str) + '_' +
    sched_weather['week'].astype(str).str.zfill(2) + '_' +
    sched_weather.apply(
        lambda r: '_'.join(sorted([str(r['away_team']), str(r['home_team'])])), axis=1)
)

# Determine dome status per game (reuse is_dome logic from Step 6)
DOME_ROOFS_W = {'dome', 'closed'}
sched_weather['is_dome_w'] = sched_weather['roof'].str.lower().isin(DOME_ROOFS_W).astype(int)

# Only outdoor games need weather lookup
outdoor = sched_weather[sched_weather['is_dome_w'] == 0].copy()
outdoor['gameday'] = pd.to_datetime(outdoor['gameday'])

print(f"  Outdoor games to fetch: {len(outdoor):,}")
print(f"  Unique stadiums:        {outdoor['stadium_id'].nunique()}")

# %%
# Fetch historical hourly precipitation from Open-Meteo, one call per stadium.
# Returns {(stadium_id, date_str): precip_mm_over_game_window}

def fetch_stadium_precip(stadium_id, lat, lon, dates_and_times, max_retries=5):
    """
    Fetch hourly precipitation for a stadium over all its game dates.
    dates_and_times: list of (date_str 'YYYY-MM-DD', kickoff_hour int)
    Returns dict: date_str -> precip_mm (sum over 4-hour game window)
    Retries up to max_retries times with exponential backoff on 429.
    """
    if not dates_and_times:
        return {}
    all_dates = sorted({d for d, _ in dates_and_times})
    start = all_dates[0]
    end   = all_dates[-1]
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "hourly":     "precipitation",
        "timezone":   "auto",
    }
    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=params,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 5  # 5, 10, 20, 40, 80 seconds
                print(f"    429 rate-limit for {stadium_id}, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 5
                print(f"    WARNING: API error for {stadium_id} (attempt {attempt+1}): {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                print(f"    WARNING: API error for {stadium_id} after {max_retries} attempts: {e}")
                return {}
    if data is None:
        print(f"    WARNING: {stadium_id} exhausted all retries — skipping")
        return {}

    # Build date → hourly precip lookup
    hours = data["hourly"]["time"]          # list of "YYYY-MM-DDTHH:00"
    precip = data["hourly"]["precipitation"]
    hourly = {}
    for t, p in zip(hours, precip):
        date_part, hour_part = t.split("T")
        h = int(hour_part.split(":")[0])
        hourly.setdefault(date_part, {})[h] = p if p is not None else 0.0

    results = {}
    for date_str, kickoff_hour in dates_and_times:
        # Sum precipitation over 4-hour game window starting at kickoff
        day_data = hourly.get(date_str, {})
        total = sum(day_data.get(kickoff_hour + i, 0.0) for i in range(4))
        results[date_str] = total
    return results

# Group outdoor games by stadium and fetch
precip_lookup = {}  # (game_id_sorted) -> precip_mm

stadiums_to_fetch = outdoor['stadium_id'].unique()
print(f"\n  Fetching precipitation for {len(stadiums_to_fetch)} stadiums...")

for i, sid in enumerate(stadiums_to_fetch):
    coords = STADIUM_COORDS.get(sid)
    if coords is None:
        print(f"    SKIP {sid} — no coordinates")
        continue

    lat, lon = coords
    games = outdoor[outdoor['stadium_id'] == sid]

    # Parse kickoff hour (gametime format "HH:MM", local time)
    dates_and_times = []
    for _, row in games.iterrows():
        date_str = row['gameday'].strftime('%Y-%m-%d')
        try:
            kickoff_hour = int(str(row['gametime']).split(':')[0])
        except Exception:
            kickoff_hour = 13  # default to 1pm if unknown
        dates_and_times.append((date_str, kickoff_hour))

    stadium_precip = fetch_stadium_precip(sid, lat, lon, dates_and_times)

    # Map back to game_id
    for _, row in games.iterrows():
        date_str = row['gameday'].strftime('%Y-%m-%d')
        gid = row['game_id_sorted']
        precip_lookup[gid] = stadium_precip.get(date_str, float('nan'))

    if (i + 1) % 10 == 0:
        print(f"    {i+1}/{len(stadiums_to_fetch)} stadiums done...")
    time.sleep(0.3)  # polite rate limit

print(f"  Precipitation fetched for {len(precip_lookup):,} outdoor games")

# %%
# Add game_precip_mm as a continuous feature (primary) and derive weather_type
# as a categorical (secondary). Dome games get NaN for all weather values —
# is_dome=1 already signals indoor conditions to the model.
#
# Continuous features (all NaN for dome games):
#   game_temp      — already added in Step 6 (°F at kickoff)
#   game_wind      — already added in Step 6 (mph at kickoff)
#   game_precip_mm — total precipitation over 4-hour game window (mm)
#
# Categorical feature (secondary):
#   weather_type   — captures temp×precip interaction explicitly:
#                    'dome' | 'snow' | 'cold_clear' | 'rain' | 'clear' | 'unknown'

# Map continuous precipitation to master
master['game_precip_mm'] = master['game_id'].map(precip_lookup)
# Dome games: set precip to NaN (consistent with game_temp / game_wind)
master.loc[master['is_dome'] == 1, 'game_precip_mm'] = float('nan')

# Derive weather_type from the continuous values already in master
def derive_weather_type(row):
    if row['is_dome'] == 1:
        return 'dome'
    precip = row['game_precip_mm']
    temp   = row['game_temp']
    if pd.isna(precip) or pd.isna(temp):
        return 'unknown'
    is_precip = precip >= PRECIP_THRESHOLD_MM
    is_cold   = temp < 32
    if   is_cold and     is_precip: return 'snow'
    elif is_cold and not is_precip: return 'cold_clear'
    elif not is_cold and is_precip: return 'rain'
    else:                           return 'clear'

master['weather_type'] = master.apply(derive_weather_type, axis=1)

# %%
# --- Validation ---
print(f"\n  game_precip_mm: {master['game_precip_mm'].notna().sum():,} non-null "
      f"({master['game_precip_mm'].isna().mean()*100:.1f}% null)")
print(f"  precip range (outdoor): "
      f"{master['game_precip_mm'].min():.1f} – {master['game_precip_mm'].max():.1f} mm")
print(f"\n  weather_type distribution:")
print(master['weather_type'].value_counts().to_string())
print(f"\n  unknown rate: {(master['weather_type']=='unknown').mean()*100:.1f}%")
print(f"\n  Master after Step 7: {master.shape}")

# %%
# --- Save master after Step 7 ---
save_master(master, step=7)

# %%
# =============================================================================
# STEP 7b (plan Step 12) — Rolling lagged features (L3 / L5 / L10 / L20)
# =============================================================================
# NOTE: In the original 15-step plan this is Step 12. It is implemented here
# early because the raw base stats are already available. Steps 8-11 (DVOA,
# depth charts, IR, trades) will add new base columns that get their own
# rolling pass in the final Step 12. The windows below cover the core target
# and usage stats only.
#
# For each key stat, compute rolling means over the last 3, 5, 10, and 20 games
# for each player. Features are STRICTLY prior to the current game (shift by 1
# before rolling) to prevent any lookahead leakage.
#
# Design decisions:
#   - Cross-season continuity: we group only by player_id, NOT by (player_id, season).
#     A player's form carries over; the model learns that if needed.
#   - min_periods=1: avoid NaN for a player's first few games; early windows use
#     whatever games are available rather than dropping rows.
#   - Sort order: player_id → season → week (ascending) ensures chronological order
#     within each player across all seasons.
#
# Stats rolled (11 core stats covering all positions):
#   Passing : passing_yards, passing_tds, passing_epa
#   Rushing : rushing_yards, rushing_tds, carries, rushing_epa
#   Receiving: receiving_yards, receiving_tds, targets, receiving_epa
#   Usage   : offense_pct  (snap share — strong usage signal)
#
# Windows: L3, L5, L10, L20  →  48 new columns total (12 stats × 4 windows)

print("=" * 60)
print("STEP 7 — Rolling lagged features")
print("=" * 60)

# %%
# Sort chronologically within each player across all seasons
master = master.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

ROLL_STATS = [
    'passing_yards', 'passing_tds', 'passing_epa', 'interceptions',
    'rushing_yards',  'rushing_tds',  'carries', 'rushing_epa',
    'receiving_yards','receiving_tds', 'targets', 'receiving_epa', 'receptions',
    'offense_pct',
    'fumbles_lost_total',   # QB/skill position ball security — L5/L10/L20 used in models
]
WINDOWS = [3, 5, 10, 20]

# %%
# Compute rolling means — shift(1) ensures current game is excluded
for stat in ROLL_STATS:
    for w in WINDOWS:
        col = f'{stat}_L{w}'
        master[col] = (
            master.groupby('player_id')[stat]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )

print(f"  Rolling columns added: {len(ROLL_STATS) * len(WINDOWS)}")
print(f"  Master shape: {master.shape}")

# %%
# --- Validation ---
# Spot-check: a specific player's rolling passing_yards should lag by one game
sample_player = master[master['position'] == 'QB'].iloc[0]['player_id']
sample = master[master['player_id'] == sample_player][
    ['player_display_name', 'season', 'week', 'passing_yards',
     'passing_yards_L3', 'passing_yards_L5']
].head(8)
print(f"\n  Spot-check rolling (QB player_id={sample_player}):")
print(sample.to_string(index=False))

# Null rate on rolling cols: only first game(s) of a player's career should be NaN
# (min_periods=1 means L3 is never NaN after game 1, just uses fewer games)
null_roll = master[[f'{s}_L{w}' for s in ROLL_STATS for w in WINDOWS]].isnull().mean() * 100
print(f"\n  Max null rate across all rolling cols: {null_roll.max():.1f}%")
print(f"  Mean null rate across all rolling cols: {null_roll.mean():.1f}%")

# %%
# --- Save master after Step 7 ---
save_master(master, step=7)

# %%
# =============================================================================
# STEP 8 — Opponent defensive context (rolling yards/TDs/EPA allowed per position)
# =============================================================================
#
# "How tough is the matchup?" — a WR facing the league's best secondary has a
# very different outlook than one facing a porous defense.
#
# Approach:
#   1. For each (team-as-defense, season, week, position), sum ALL yards/TDs/EPA
#      allowed to that position group from the master dataset.
#   2. Sort by (team, season, week), then shift(1) + rolling to get PRIOR-game
#      defensive averages — strictly no lookahead.
#   3. Join back to master on (opponent, season, week) so each player gets the
#      rolling defensive stats of the team they're facing.
#
# Stats per position group:
#   QB  : passing_yards, passing_tds, passing_epa
#   RB  : rushing_yards, rushing_tds, receiving_yards
#   WR  : receiving_yards, targets, receiving_tds
#   TE  : receiving_yards, targets, receiving_tds
#
# Windows: L3, L5  (24 new columns total: 4 positions × 3 stats × 2 windows)
# All columns are NaN for inapplicable positions — LightGBM handles this natively.

print("=" * 60)
print("STEP 8 — Opponent defensive context")
print("=" * 60)

# %%
POS_DEF_CONFIG = {
    'QB': ['passing_yards', 'passing_tds', 'passing_epa'],
    'RB': ['rushing_yards', 'rushing_tds', 'receiving_yards'],
    'WR': ['receiving_yards', 'targets', 'receiving_tds'],
    'TE': ['receiving_yards', 'targets', 'receiving_tds'],
}
DEF_WINDOWS = [5, 10, 20]

# Ensure master is sorted (Step 7 sorted it, but be explicit)
master = master.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

all_def_frames = []

for pos, stats in POS_DEF_CONFIG.items():
    prefix = f'opp_{pos.lower()}'

    # Aggregate: sum all players of this position vs each team's defense per game
    pos_rows = master.loc[master['position'] == pos, ['opponent', 'season', 'week'] + stats]
    def_game = (
        pos_rows
        .groupby(['opponent', 'season', 'week'])[stats]
        .sum()
        .reset_index()
        .rename(columns={'opponent': 'team'})
        .sort_values(['team', 'season', 'week'])
        .reset_index(drop=True)
    )

    # Rolling lags: shift(1) so current game is excluded
    for stat in stats:
        for w in DEF_WINDOWS:
            col = f'{prefix}_{stat}_L{w}'
            def_game[col] = (
                def_game.groupby('team')[stat]
                .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            )

    roll_cols = [f'{prefix}_{s}_L{w}' for s in stats for w in DEF_WINDOWS]
    all_def_frames.append((pos, def_game[['team', 'season', 'week'] + roll_cols], roll_cols))
    print(f"  {pos}: {len(def_game):,} game-defense rows, {len(roll_cols)} rolling cols")

# %%
# Drop any pre-existing Step 8 columns (idempotent re-run guard)
all_step8_cols = [c for _, _, cols in all_def_frames for c in cols]
master.drop(columns=[c for c in all_step8_cols if c in master.columns], inplace=True)

# Initialise all Step 8 columns as NaN
for col in all_step8_cols:
    master[col] = float('nan')

# Join each position's defensive rolling stats back to master
# master['opponent'] is the defending team; join on (opponent=team, season, week)
for pos, def_join, roll_cols in all_def_frames:
    pos_mask = master['position'] == pos
    merged = (
        master.loc[pos_mask, ['opponent', 'season', 'week']]
        .merge(def_join, left_on=['opponent', 'season', 'week'],
               right_on=['team', 'season', 'week'], how='left')
    )
    master.loc[pos_mask, roll_cols] = merged[roll_cols].values

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")
print(f"\n  Null rates for Step 8 columns:")
for col in all_step8_cols:
    pct = master[col].isna().mean() * 100
    print(f"    {col}: {pct:.1f}%")

# Spot-check: QB facing a specific opponent — opp_qb stats should be from prior games
sample_qb = master[(master['position']=='QB') & master['opp_qb_passing_yards_L5'].notna()].head(5)
print(f"\n  Spot-check QB opp defensive stats:")
print(sample_qb[['player_display_name','season','week','opponent',
                  'passing_yards','opp_qb_passing_yards_L5','opp_qb_passing_yards_L10']].to_string(index=False))

# %%
# =============================================================================
# STEP 8b — Team-level defensive EPA composites
# =============================================================================
#
# Aggregate EPA allowed across all skill-position players facing each defense
# per game, then compute rolling L5/L10/L20 averages (shift(1), no lookahead).
#
# Three composite columns per window:
#   opp_def_pass_epa_L{w}  — avg pass EPA allowed per game (QB passing_epa +
#                             all receiving_epa from WR/TE/RB)
#   opp_def_run_epa_L{w}   — avg rush EPA allowed per game (all rushing_epa)
#   opp_def_epa_L{w}       — total offensive EPA allowed (pass + run combined)
#
# Higher value = defense allows more EPA = weaker defense.
# Join: master['opponent'] -> defending team
#
# All non-skill-position rows get NaN — model treats them as missing naturally.
# =============================================================================

print("  Building team-level defensive EPA composites...")

TEAM_DEF_WINDOWS = [5, 10, 20]

# --- Pass EPA: QB passing_epa + any position's receiving_epa ---
pass_epa_rows = []
# QB passing
qb_pass = master.loc[master['position'] == 'QB', ['opponent', 'season', 'week', 'passing_epa']].copy()
qb_pass.rename(columns={'passing_epa': '_pass_epa'}, inplace=True)
pass_epa_rows.append(qb_pass)
# All receiver EPA (WR, TE, RB as receivers)
rec_epa = master.loc[master['receiving_epa'].notna(), ['opponent', 'season', 'week', 'receiving_epa']].copy()
rec_epa.rename(columns={'receiving_epa': '_pass_epa'}, inplace=True)
pass_epa_rows.append(rec_epa)

pass_epa_game = (
    pd.concat(pass_epa_rows, ignore_index=True)
    .groupby(['opponent', 'season', 'week'])['_pass_epa']
    .sum()
    .reset_index()
    .rename(columns={'opponent': 'team', '_pass_epa': 'pass_epa_sum'})
    .sort_values(['team', 'season', 'week'])
    .reset_index(drop=True)
)

# --- Run EPA: all rushing_epa ---
run_epa_game = (
    master.loc[master['rushing_epa'].notna(), ['opponent', 'season', 'week', 'rushing_epa']]
    .groupby(['opponent', 'season', 'week'])['rushing_epa']
    .sum()
    .reset_index()
    .rename(columns={'opponent': 'team', 'rushing_epa': 'run_epa_sum'})
    .sort_values(['team', 'season', 'week'])
    .reset_index(drop=True)
)

# --- Combine into one team-game table ---
team_def = (
    pass_epa_game
    .merge(run_epa_game, on=['team', 'season', 'week'], how='outer')
    .fillna({'pass_epa_sum': 0.0, 'run_epa_sum': 0.0})
)
team_def['total_epa_sum'] = team_def['pass_epa_sum'] + team_def['run_epa_sum']
team_def = team_def.sort_values(['team', 'season', 'week']).reset_index(drop=True)

# --- Rolling lags ---
team_def_roll_cols = []
for w in TEAM_DEF_WINDOWS:
    for src_col, prefix in [('pass_epa_sum', 'opp_def_pass_epa'),
                             ('run_epa_sum',  'opp_def_run_epa'),
                             ('total_epa_sum','opp_def_epa')]:
        col = f'{prefix}_L{w}'
        team_def[col] = (
            team_def.groupby('team')[src_col]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )
        team_def_roll_cols.append(col)

print(f"    team_def rows: {len(team_def):,}  |  rolling cols: {len(team_def_roll_cols)}")

# --- Drop pre-existing cols (idempotent) ---
master.drop(columns=[c for c in team_def_roll_cols if c in master.columns], inplace=True)

# --- Join to master on (opponent -> team, season, week) ---
master = master.merge(
    team_def[['team', 'season', 'week'] + team_def_roll_cols],
    left_on=['opponent', 'season', 'week'],
    right_on=['team', 'season', 'week'],
    how='left'
).drop(columns=['team_y']).rename(columns={'team_x': 'team'})

print(f"    Master shape after team def composites: {master.shape}")
print(f"    Null rates for team def EPA cols:")
for col in team_def_roll_cols:
    pct = master[col].isna().mean() * 100
    print(f"      {col}: {pct:.1f}%")

# %%
# --- Save master after Step 8 ---
save_master(master, step=8)

# %%
# =============================================================================
# STEP 9 — Depth chart rank
# =============================================================================
#
# Adds depth_chart_rank: a player's positional depth on their team for that
# week (1 = starter, 2 = first backup, 3 = second backup, etc.).
# Published pre-game by teams, so no lookahead — safe to join current week.
#
# Source: nfl.import_depth_charts()
# Key column: depth_team (integer rank within position group)
#
# Strategy:
#   - Pull all depth charts for SEASONS
#   - Keep one row per (player_id, season, week): take the minimum depth_team
#     if duplicates exist (a player occasionally appears twice in edge cases)
#   - Left-join onto master on (player_id, season, week)
#   - Fill NaN with 0 — "not on depth chart" is informative (injured reserve,
#     practice squad, inactive) and should be distinct from rank 1
# =============================================================================

print("=" * 60)
print("STEP 9 — Depth chart rank")
print("=" * 60)

# %%
# nfl.import_depth_charts() returns two different schemas depending on season:
#   Pre-2025 (old schema): season, week, depth_team, gsis_id, position  (game-week based)
#   2025+    (new schema): dt (UTC timestamp), pos_rank, gsis_id, pos_abb (snapshot based)
#
# We handle both schemas, normalise to (player_id, season, week, depth_chart_rank),
# then union before joining to master.

# --- Old schema: seasons where depth_charts returns season/week columns ---
_old_seasons = [s for s in SEASONS if s <= 2024]
_new_seasons  = [s for s in SEASONS if s >= 2025]

dc_parts = []

if _old_seasons:
    print(f"  Importing depth charts (old schema) for {_old_seasons[0]}-{_old_seasons[-1]}...")
    dc_old = nfl.import_depth_charts(years=_old_seasons)
    print(f"    Rows: {len(dc_old):,}  Cols: {list(dc_old.columns)}")
    dc_old = (
        dc_old[['gsis_id', 'season', 'week', 'depth_team']]
        .rename(columns={'gsis_id': 'player_id'})
        .dropna(subset=['player_id', 'depth_team'])
    )
    dc_old['depth_team'] = dc_old['depth_team'].astype(int)
    dc_old = (
        dc_old.groupby(['player_id', 'season', 'week'])['depth_team']
        .min().reset_index()
        .rename(columns={'depth_team': 'depth_chart_rank'})
    )
    dc_parts.append(dc_old)
    print(f"    Deduplicated old-schema rows: {len(dc_old):,}")

if _new_seasons:
    # New schema: map UTC timestamps -> NFL weeks via schedule
    print(f"  Importing depth charts (new schema) for {_new_seasons}...")
    dc_new_raw = nfl.import_depth_charts(years=_new_seasons)
    print(f"    Rows: {len(dc_new_raw):,}  Cols: {list(dc_new_raw.columns)}")

    # Build week -> last-game-date lookup across all new-schema seasons
    _sched_new = nfl.import_schedules(_new_seasons)
    _reg_new   = _sched_new[_sched_new['game_type'] == 'REG'][['season', 'week', 'gameday']].copy()
    _reg_new['gameday'] = pd.to_datetime(_reg_new['gameday'])
    _week_end_new = (
        _reg_new.groupby(['season', 'week'])['gameday']
        .max()
        .reset_index()
        .rename(columns={'gameday': 'week_end_date'})
    )

    def _dt_to_season_week(dt, week_end_df):
        """Return (season, week) for a tz-naive dt by finding the earliest
        week_end_date >= dt within each season."""
        for _, row in week_end_df.sort_values(['season', 'week']).iterrows():
            if dt <= row['week_end_date'] + pd.Timedelta(days=1):
                return int(row['season']), int(row['week'])
        return None, None

    dc_new_raw['dt_parsed'] = pd.to_datetime(dc_new_raw['dt'], utc=True).dt.tz_localize(None)

    # Vectorised mapping using merge_asof per season for speed
    dc_new_rows = []
    for _szn in _new_seasons:
        _we = _week_end_new[_week_end_new['season'] == _szn].sort_values('week_end_date')
        # Filter snapshots that fall within the regular season date range
        _lo = _we['week_end_date'].min() - pd.Timedelta(days=60)  # include pre-season builds
        _hi = _we['week_end_date'].max() + pd.Timedelta(days=1)
        _dc_s = dc_new_raw[dc_new_raw['dt_parsed'].between(_lo, _hi)].copy()
        if _dc_s.empty:
            continue
        _dc_s = _dc_s.sort_values('dt_parsed')
        # Assign week via merge_asof: each snapshot gets the earliest week whose end_date >= snapshot
        _we_sorted = _we.sort_values('week_end_date').rename(columns={'week_end_date': 'dt_parsed'})
        _merged = pd.merge_asof(
            _dc_s[['gsis_id', 'pos_abb', 'pos_rank', 'dt_parsed']].sort_values('dt_parsed'),
            _we_sorted[['dt_parsed', 'week']],
            on='dt_parsed',
            direction='forward',  # find next week_end >= snapshot date
        )
        _merged['season'] = _szn
        dc_new_rows.append(_merged)

    if dc_new_rows:
        dc_new = pd.concat(dc_new_rows, ignore_index=True)
        dc_new = dc_new.dropna(subset=['week'])
        dc_new['week'] = dc_new['week'].astype(int)
        dc_new = (
            dc_new.groupby(['gsis_id', 'season', 'week'])['pos_rank']
            .min().reset_index()
            .rename(columns={'gsis_id': 'player_id', 'pos_rank': 'depth_chart_rank'})
        )
        dc_parts.append(dc_new)
        print(f"    Deduplicated new-schema rows: {len(dc_new):,}")
    else:
        print("    WARNING: no new-schema depth chart rows mapped to regular-season weeks")

dc = pd.concat(dc_parts, ignore_index=True) if dc_parts else pd.DataFrame(
    columns=['player_id', 'season', 'week', 'depth_chart_rank']
)
print(f"\n  Total depth chart rows: {len(dc):,}")
print(f"  depth_chart_rank distribution:\n{dc['depth_chart_rank'].value_counts().sort_index().head(10)}")

# %%
# Drop pre-existing column (idempotent re-run guard)
master.drop(columns=['depth_chart_rank'], errors='ignore', inplace=True)

# Left-join onto master
master = master.merge(dc, on=['player_id', 'season', 'week'], how='left')

# Fill NaN with 0 — not on depth chart (IR, practice squad, inactive)
master['depth_chart_rank'] = master['depth_chart_rank'].fillna(0).astype(int)

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")
null_pct = (master['depth_chart_rank'] == 0).mean() * 100
print(f"  Players not on depth chart (rank=0): {null_pct:.1f}%")
print(f"  depth_chart_rank distribution (top 6):")
print(master['depth_chart_rank'].value_counts().sort_index().head(6))

# Spot-check: starters for a known team/week should be rank 1
sample = master[
    (master['team'] == 'KC') &
    (master['season'] == 2023) &
    (master['week'] == 1) &
    (master['position'].isin(['QB', 'WR', 'RB', 'TE']))
][['player_display_name', 'position', 'depth_chart_rank']].sort_values(['position', 'depth_chart_rank'])
print(f"\n  Spot-check KC Week 1 2023 skill positions:")
print(sample.to_string(index=False))

# %%
# --- Save master after Step 9 ---
save_master(master, step=9)

# %%
# =============================================================================
# STEP 10 — IR return tracking
# =============================================================================
#
# Two features capturing injured reserve history:
#
#   games_since_ir_return
#     0  = not in a return window
#     1  = first game back from IR, 2 = second game back, etc.
#     Capped at IR_RETURN_CAP (8) then resets to 0
#     Captures rustiness / snap-count risk in games after return
#
#   weeks_on_ir
#     0  = not in a return window
#     N  = consecutive weeks on IR before this return
#     Constant throughout the return window
#     Captures injury severity — 8-week absence vs 1-week absence
#
# Source: nfl.import_weekly_rosters() — roster status field
#   status == 'IR' means player is on Injured Reserve that week
#   This is the correct source; import_injuries() only has game-week
#   designations (Questionable/Probable/Out/Doubtful) and never contains 'IR'
#   because IR players are off the active roster entirely.
# =============================================================================

print("=" * 60)
print("STEP 10 — IR return tracking")
print("=" * 60)

IR_RETURN_CAP = 17  # one full season — covers ACL/serious injury return seasons

# %%
print(f"  Importing weekly rosters for {SEASONS[0]}-{SEASONS[-1]}...")
rosters_raw = nfl.import_weekly_rosters(years=SEASONS)
print(f"  Raw rosters: {len(rosters_raw):,} rows x {rosters_raw.shape[1]} cols")
print(f"  status values: {rosters_raw['status'].dropna().unique()}")

# %%
# Build weekly IR flag per player from roster status
rosters = rosters_raw[['season', 'week', 'player_id', 'status']].copy()
rosters = rosters.dropna(subset=['player_id'])

IR_STATUSES = {'RES', 'PUP', 'RSN'}  # Reserve/Injured, Physically Unable to Perform, Reserve/Non-Football Injury
rosters['on_ir'] = (rosters['status'].fillna('').str.upper().isin(IR_STATUSES)).astype(int)

# One row per (player_id, season, week)
inj = (
    rosters.groupby(['player_id', 'season', 'week'])['on_ir']
    .max()
    .reset_index()
)
print(f"  Deduplicated roster rows: {len(inj):,}")
print(f"  IR-flagged rows: {inj['on_ir'].sum():,}")

# Build the set of weeks each player actually played — a week where a player
# appeared in master cannot be an IR week even if the roster still lists RES
# (the roster file sometimes carries the prior-week status into game week)
played_weeks = (
    master[['player_id', 'season', 'week']]
    .drop_duplicates()
    .assign(played=1)
)

# Pre-build per-player IR week sets, excluding any week the player actually played
inj_ir_only = inj.loc[inj['on_ir'] == 1].copy()
inj_ir_only = inj_ir_only.merge(
    played_weeks, on=['player_id', 'season', 'week'], how='left'
)
inj_ir_only = inj_ir_only[inj_ir_only['played'].isna()].drop(columns='played')
print(f"  IR weeks after removing played weeks: {len(inj_ir_only):,}")

player_ir_map = (
    inj_ir_only
    .groupby('player_id')
    .apply(lambda g: set(zip(g['season'], g['week'])))
    .to_dict()
)

# %%
# Compute both IR features per player
played = (
    master[['player_id', 'season', 'week']]
    .drop_duplicates()
    .sort_values(['player_id', 'season', 'week'])
    .reset_index(drop=True)
)

def compute_ir_features(group):
    pid          = group['player_id'].iloc[0]
    player_ir    = player_ir_map.get(pid, set())

    games_since  = []
    weeks_on     = []
    counter      = 0
    weeks_inj    = 0
    in_window    = False

    for _, row in group.iterrows():
        s, w   = row['season'], row['week']
        prev   = (s, w - 1)
        on_ir  = prev in player_ir

        if on_ir:
            # Count consecutive prior IR weeks for severity
            stint    = 0
            check_w  = w - 1
            while (s, check_w) in player_ir and check_w >= 1:
                stint   += 1
                check_w -= 1
            counter   = 1
            weeks_inj = stint
            in_window = True
        elif in_window:
            counter += 1
            if counter > IR_RETURN_CAP:
                counter   = 0
                weeks_inj = 0
                in_window = False
        else:
            counter   = 0
            weeks_inj = 0

        games_since.append(counter)
        weeks_on.append(weeks_inj)

    out = group.copy()
    out['games_since_ir_return'] = games_since
    out['weeks_on_ir']           = weeks_on
    return out

print("  Computing IR features (may take ~45s)...")
played_ir = (
    played
    .groupby('player_id', group_keys=False)
    .apply(compute_ir_features)
)
print(f"  Done.")
print(f"  IR return rows (games_since>=1): {(played_ir['games_since_ir_return'] >= 1).sum():,}")
print(f"  weeks_on_ir > 0 rows:            {(played_ir['weeks_on_ir'] > 0).sum():,}")

# %%
# Drop pre-existing columns (idempotent re-run guard)
master.drop(columns=['games_since_ir_return', 'weeks_on_ir'], errors='ignore', inplace=True)

master = master.merge(
    played_ir[['player_id', 'season', 'week', 'games_since_ir_return', 'weeks_on_ir']],
    on=['player_id', 'season', 'week'],
    how='left'
)
master['games_since_ir_return'] = master['games_since_ir_return'].fillna(0).astype(int)
master['weeks_on_ir']           = master['weeks_on_ir'].fillna(0).astype(int)

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")
n_ret = (master['games_since_ir_return'] >= 1).sum()
print(f"  Player-games in IR return window: {n_ret:,} ({n_ret/len(master)*100:.1f}%)")
print(f"\n  games_since_ir_return distribution:")
print(master['games_since_ir_return'].value_counts().sort_index().head(10))
print(f"\n  weeks_on_ir distribution (return window only):")
print(master.loc[master['weeks_on_ir'] > 0, 'weeks_on_ir'].value_counts().sort_index().head(10))

# Spot-check: pick a player with a long return window and show all their games
# to verify the counter increments correctly and weeks_on_ir stays constant
sample_player = (
    master[master['games_since_ir_return'] >= 5]['player_id'].value_counts().index[0]
)
sample_ir = master[master['player_id'] == sample_player][
    ['player_display_name', 'position', 'team', 'season', 'week',
     'games_since_ir_return', 'weeks_on_ir']
].sort_values(['season', 'week'])
print(f"\n  Full return window trace for player_id={sample_player}:")
print(sample_ir.to_string(index=False))

# %%
# --- Save master after Step 10 ---
save_master(master, step=10)

# %%
# =============================================================================
# STEP 11 — Trade / mid-season team change flags
# =============================================================================
#
# A player who changes teams mid-season faces a real adjustment period:
# new playbook, new teammates, new scheme. This step quantifies that.
#
# Features:
#   traded_this_season  — binary, 1 from the first game after a team change
#                         through the rest of the season; 0 otherwise
#   games_since_trade   — 0 if no recent trade, 1 = first game after trade,
#                         2 = second game, etc. Capped at TRADE_CAP (8) then
#                         resets to 0. Captures the acute adjustment window.
#
# Detection: within a season, compare each player's team in master week-by-week.
# A team change between consecutive played weeks = mid-season transaction.
# Catches trades, waiver claims, and free agent signings equally — all involve
# joining a new team mid-season with no full training camp adjustment.
#
# Source: master['team'] — already populated from weekly stats in Step 1.
# No additional data pull needed.
# =============================================================================

print("=" * 60)
print("STEP 11 — Trade / mid-season team change flags")
print("=" * 60)

TRADE_CAP = 8  # games after which acute adjustment window closes

# %%
# Build per-player team-by-week sequence from master
# We only need player_id, season, week, team — one row per played game
player_games = (
    master[['player_id', 'season', 'week', 'team']]
    .drop_duplicates()
    .sort_values(['player_id', 'season', 'week'])
    .reset_index(drop=True)
)

# Detect team changes within a season:
# shift(1) within (player_id, season) gives the prior game's team
# A change = current team != prior team AND prior team is not null (first game of season)
player_games['prev_team'] = (
    player_games.groupby(['player_id', 'season'])['team'].shift(1)
)
player_games['team_changed'] = (
    player_games['prev_team'].notna() &
    (player_games['team'] != player_games['prev_team'])
).astype(int)

n_changes = player_games['team_changed'].sum()
print(f"  Mid-season team changes detected: {n_changes:,}")

# %%
# Compute traded_this_season and games_since_trade per player per season
def compute_trade_features(group):
    traded    = []
    since     = []
    counter   = 0
    in_window = False
    ever_traded = False

    for _, row in group.iterrows():
        if row['team_changed'] == 1:
            ever_traded = True
            counter     = 1
            in_window   = True
        elif in_window:
            counter += 1
            if counter > TRADE_CAP:
                counter   = 0
                in_window = False
        else:
            counter = 0

        traded.append(1 if ever_traded else 0)
        since.append(counter)

    out = group.copy()
    out['traded_this_season'] = traded
    out['games_since_trade']  = since
    return out

print("  Computing trade features...")
player_games_trade = (
    player_games
    .groupby(['player_id', 'season'], group_keys=False)
    .apply(compute_trade_features)
)

n_traded = (player_games_trade['traded_this_season'] == 1).sum()
n_window = (player_games_trade['games_since_trade'] >= 1).sum()
print(f"  Player-games with traded_this_season=1: {n_traded:,}")
print(f"  Player-games in acute trade window:     {n_window:,}")

# %%
# Drop pre-existing columns (idempotent re-run guard)
master.drop(columns=['traded_this_season', 'games_since_trade'], errors='ignore', inplace=True)

# Join back to master on (player_id, season, week)
master = master.merge(
    player_games_trade[['player_id', 'season', 'week', 'traded_this_season', 'games_since_trade']],
    on=['player_id', 'season', 'week'],
    how='left'
)
master['traded_this_season'] = master['traded_this_season'].fillna(0).astype(int)
master['games_since_trade']  = master['games_since_trade'].fillna(0).astype(int)

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")
print(f"  traded_this_season distribution:")
print(master['traded_this_season'].value_counts().sort_index())
print(f"\n  games_since_trade distribution:")
print(master['games_since_trade'].value_counts().sort_index().head(10))

# Spot-check: find a player with a known mid-season trade and trace them
sample_traded_id = (
    player_games_trade[player_games_trade['traded_this_season'] == 1]
    ['player_id'].value_counts().index[0]
)
sample_trade = master[master['player_id'] == sample_traded_id][
    ['player_display_name', 'position', 'team', 'season', 'week',
     'traded_this_season', 'games_since_trade']
].sort_values(['season', 'week'])
print(f"\n  Full trade trace for player_id={sample_traded_id}:")
print(sample_trade.to_string(index=False))

# %%
# --- Save master after Step 11 ---
save_master(master, step=11)

# %%
# =============================================================================
# STEP 12 — Advanced derived metrics + rolling windows
# =============================================================================
#
# Computes per-game derived efficiency and opportunity metrics by position,
# then rolls them using L3/L5/L10/L20 windows (shift(1), cross-season).
#
# Metrics:
#   QB  : yards_per_attempt, completion_pct, td_rate, int_rate, epa_per_dropback
#   WR/TE: target_share, air_yards_share, wopr, adot, catch_rate,
#           yac_per_reception, epa_per_target
#   RB  : carry_share, yards_per_carry, epa_per_carry, yards_per_reception,
#           opportunity_share
#
# Team-level denominators (team_targets, team_carries, team_air_yards) are
# computed per (team, season, week) and joined before metric calculation.
# Division by zero produces NaN — correct and handled natively by LightGBM.
# =============================================================================

print("=" * 60)
print("STEP 12 — Advanced derived metrics")
print("=" * 60)

ROLL_WINDOWS = [3, 5, 10, 20]

# %%
# --- Step 12a: Team-game aggregates for denominators ---
print("  Computing team-game aggregates...")

team_game = (
    master
    .groupby(['team', 'season', 'week'])
    .agg(
        team_targets   = ('targets',          'sum'),
        team_carries   = ('carries',          'sum'),
        team_air_yards = ('receiving_air_yards', 'sum'),
    )
    .reset_index()
)
print(f"  Team-game rows: {len(team_game):,}")

# Join team aggregates back to master
master.drop(columns=['team_targets', 'team_carries', 'team_air_yards'],
            errors='ignore', inplace=True)
master = master.merge(team_game, on=['team', 'season', 'week'], how='left')

# %%
# --- Step 12b: Per-game derived metrics ---
print("  Computing per-game derived metrics...")

# QB metrics
master['yards_per_attempt']  = master['passing_yards']  / master['attempts'].replace(0, float('nan'))
master['completion_pct']     = master['completions']     / master['attempts'].replace(0, float('nan'))
master['td_rate']            = master['passing_tds']     / master['attempts'].replace(0, float('nan'))
master['int_rate']           = master['interceptions']   / master['attempts'].replace(0, float('nan'))
master['epa_per_dropback']   = master['passing_epa']     / master['attempts'].replace(0, float('nan'))

# Mask QB metrics to QB rows only
qb_mask = master['position'] != 'QB'
for col in ['yards_per_attempt', 'completion_pct', 'td_rate', 'int_rate', 'epa_per_dropback']:
    master.loc[qb_mask, col] = float('nan')

# WR/TE metrics
# target_share and air_yards_share already exist from Step 1 (nfl_data_py)
# Recompute from raw to fill any gaps (older seasons where NGS data is missing)
master['target_share']     = master['target_share'].fillna(
    master['targets'] / master['team_targets'].replace(0, float('nan'))
)
master['air_yards_share']  = master['air_yards_share'].fillna(
    master['receiving_air_yards'] / master['team_air_yards'].replace(0, float('nan'))
)
master['wopr']             = 1.5 * master['target_share'] + 0.7 * master['air_yards_share']
master['adot']             = master['receiving_air_yards'] / master['targets'].replace(0, float('nan'))
master['catch_rate']       = master['receptions']         / master['targets'].replace(0, float('nan'))
master['yac_per_reception']= master['receiving_yac']      / master['receptions'].replace(0, float('nan'))
master['epa_per_target']   = master['receiving_epa']      / master['targets'].replace(0, float('nan'))

# Mask WR/TE metrics to WR/TE rows only
# adot also applies to RBs — receiving backs have meaningful air target depths
wr_te_mask = ~master['position'].isin(['WR', 'TE'])
for col in ['target_share', 'air_yards_share', 'wopr',
            'catch_rate', 'yac_per_reception', 'epa_per_target']:
    master.loc[wr_te_mask, col] = float('nan')
wr_te_rb_mask = ~master['position'].isin(['WR', 'TE', 'RB'])
master.loc[wr_te_rb_mask, 'adot'] = float('nan')

# RB metrics
master['carry_share']        = master['carries']        / master['team_carries'].replace(0, float('nan'))
master['yards_per_carry']    = master['rushing_yards']  / master['carries'].replace(0, float('nan'))
master['epa_per_carry']      = master['rushing_epa']    / master['carries'].replace(0, float('nan'))
master['opportunity_share']  = (
    (master['carries'] + master['targets']) /
    (master['team_carries'] + master['team_targets']).replace(0, float('nan'))
)
# rb_target_share: targets / team_targets (parallel to WR target_share but for RBs)
master['rb_target_share']    = master['targets'] / master['team_targets'].replace(0, float('nan'))

# Mask RB metrics to RB rows only
rb_mask = master['position'] != 'RB'
for col in ['carry_share', 'yards_per_carry', 'epa_per_carry',
            'opportunity_share', 'rb_target_share']:
    master.loc[rb_mask, col] = float('nan')

# yards_per_reception — applies to WR, TE, and RB receiving
master['yards_per_reception'] = master['receiving_yards'] / master['receptions'].replace(0, float('nan'))
rec_mask = ~master['position'].isin(['WR', 'TE', 'RB'])
master.loc[rec_mask, 'yards_per_reception'] = float('nan')

# EPA per opportunity — universal across positions
# QB: passing_epa / attempts; WR/TE: receiving_epa / targets; RB: (rushing_epa+receiving_epa) / (carries+targets)
# Compute for each position then combine — no masking needed since each is position-specific
epa_opp_qb = master['passing_epa']   / master['attempts'].replace(0, float('nan'))
epa_opp_wr = master['receiving_epa'] / master['targets'].replace(0, float('nan'))
epa_opp_rb = (
    (master['rushing_epa'].fillna(0) + master['receiving_epa'].fillna(0)) /
    (master['carries'].fillna(0) + master['targets'].fillna(0)).replace(0, float('nan'))
)
master['epa_per_opportunity'] = float('nan')
master.loc[master['position'] == 'QB', 'epa_per_opportunity'] = epa_opp_qb
master.loc[master['position'].isin(['WR', 'TE']), 'epa_per_opportunity'] = epa_opp_wr
master.loc[master['position'] == 'RB', 'epa_per_opportunity'] = epa_opp_rb

# NGS-sourced columns rolled as derived metrics (2016+ non-null; pre-2016 NaN is expected)
# qb_air_yards_per_attempt, qb_pressure_rate, qb_scramble_rate — from PBP Step 2
# ngs_avg_time_to_throw, ngs_avg_separation — from NGS Step 4 (2016+ only)
# racr — from weekly stats Step 1 (available 2006+)
# These columns already exist in master; we just include them in DERIVED_COLS to get rolled.

# All derived game-level columns
DERIVED_COLS = [
    # QB
    'yards_per_attempt', 'completion_pct', 'td_rate', 'int_rate', 'epa_per_dropback',
    'qb_cpoe',                          # PBP Step 2, QB only
    'qb_air_yards_per_attempt',         # PBP Step 2, QB only
    'qb_pressure_rate',                 # PBP Step 2, QB only
    'qb_scramble_rate',                 # PBP Step 2, QB only
    'ngs_avg_time_to_throw',            # NGS 2016+, QB only
    'ngs_avg_intended_air_yards',       # NGS 2016+, QB only
    'ngs_aggressiveness',               # NGS 2016+, QB only (% passes into tight coverage)
    'ngs_completion_pct_above_exp',     # NGS 2016+, QB only
    # WR/TE
    'target_share', 'air_yards_share', 'wopr', 'catch_rate',
    'yac_per_reception', 'epa_per_target',
    'racr',                             # weekly stats, WR/TE
    'ngs_avg_separation',               # NGS 2016+, WR/TE
    'ngs_avg_cushion',                  # NGS 2016+, WR/TE (yards to nearest defender at snap)
    'ngs_avg_yac_above_expectation',    # NGS 2016+, WR/TE
    # WR/TE/RB — adot now applies to all receiving skill positions
    'adot',
    'yards_per_reception',
    # RB
    'carry_share', 'yards_per_carry', 'epa_per_carry',
    'rb_target_share', 'opportunity_share',
    'ngs_rush_yards_over_expected_per_att',  # NGS 2016+, RB only
    'ngs_rush_pct_over_expected',            # NGS 2016+, RB only
    'ngs_avg_time_to_los',                   # NGS 2016+, RB only (time from snap to LOS)
    # Universal
    'epa_per_opportunity',
]
print(f"  Derived game-level cols: {len(DERIVED_COLS)}")

# %%
# --- Step 12c: Rolling windows ---
print("  Computing rolling windows...")

master = master.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

roll_cols_12 = []
for col in DERIVED_COLS:
    for w in ROLL_WINDOWS:
        rcol = f'{col}_L{w}'
        master[rcol] = (
            master.groupby('player_id')[col]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )
        roll_cols_12.append(rcol)

print(f"  Rolling cols added: {len(roll_cols_12)}")

# Drop team aggregate helper columns — no longer needed
master.drop(columns=['team_targets', 'team_carries', 'team_air_yards'],
            errors='ignore', inplace=True)

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")

# Null rates on rolling cols — expect high for non-applicable positions
sample_cols = [
    'yards_per_attempt_L5', 'completion_pct_L5',       # QB
    'target_share_L5', 'wopr_L5', 'epa_per_target_L5', # WR/TE
    'carry_share_L5', 'epa_per_carry_L5',               # RB
]
print(f"\n  Null rates on sample rolling cols:")
for col in sample_cols:
    pct = master[col].isna().mean() * 100
    print(f"    {col}: {pct:.1f}%")

# Spot-check: WR target_share values should be between 0 and 1
ts = master.loc[master['position'] == 'WR', 'target_share'].dropna()
print(f"\n  WR target_share range: {ts.min():.3f} - {ts.max():.3f} (should be 0-1)")

# Spot-check: rolling wopr for a known WR
sample_wr = master.loc[
    (master['position'] == 'WR') & master['wopr_L5'].notna()
][['player_display_name', 'season', 'week', 'target_share', 'wopr', 'wopr_L5']].head(6)
print(f"\n  Sample WR wopr rolling:")
print(sample_wr.to_string(index=False))

# %%
# --- Save master after Step 12 ---
save_master(master, step=12)

# %%
# =============================================================================
# STEP 13 — Own offense positional quality (Section E offense)
# =============================================================================
#
# Roster-aware positional group quality scores — analogous to NBA BPM roster
# quality. For each player's game row, look at who ACTUALLY PLAYED for their
# team in that game, rank by cumulative snap_share through game N-1, take the
# top N per position group, average their rolling efficiency metrics, and
# exclude the player themselves from their own group average.
#
# If an elite WR misses a game, their rolling metrics do NOT enter the WR group
# score for that game — the score reflects only who was on the field.
#
# Position groups and top-N:
#   QB  : top 1 by snap_share (starter)
#   RB  : top 3 by snap_share (also serves as OL proxy via carry efficiency)
#   WR  : top 6 by target_share (opportunity-based rank)
#   TE  : top 3 by snap_share
#
# Metrics averaged per group:
#   QB  : epa_per_dropback_L5/L10/L20, qb_cpoe_L5/L10/L20,
#          adot_L5/L10/L20, yards_per_attempt_L5/L10/L20
#   RB  : epa_per_carry_L5/L10/L20, yards_per_carry_L5/L10/L20,
#          carry_share_L5/L10/L20, receiving_yards_L5/L10/L20, receptions_L5/L10/L20
#   WR  : wopr_L5/L10/L20, adot_L5/L10/L20, racr_L5/L10/L20,
#          epa_per_target_L5/L10/L20, ngs_avg_separation_L5/L10/L20 (2016+)
#   TE  : epa_per_target_L5/L10/L20, target_share_L5/L10/L20
#
# Also computes team composite rolling metrics:
#   off_epa_per_play_L5/L10/L20, off_pass_rate_L5/L10/L20,
#   off_success_rate_L5/L10/L20
#
# Windows: L5, L10, L20 only (no L3 — group quality doesn't move on 3-game scale)
# All windows end strictly at game N-1 (rolling metrics already computed in Step 12)
# =============================================================================

print("=" * 60)
print("STEP 13 — Own offense positional quality (roster-aware)")
print("=" * 60)

POS_WINDOWS = [5, 10, 20]

# %%
# --- Step 13a: Cumulative season snap share rank per player per team ---
# Used to determine group membership (top N by snap share through game N-1)
# Rate stat: average snap_share across games played this season up to game N-1

print("  Computing season-to-date snap share ranks...")

# Sort master chronologically
master = master.sort_values(['season', 'week', 'team', 'player_id']).reset_index(drop=True)

# Cumulative mean snap_share per player per season through the PRIOR game
# We compute expanding mean within (player_id, season), shifted by 1
master['_snap_share_ytd'] = (
    master.groupby(['player_id', 'season'])['offense_pct']
    .transform(lambda x: x.shift(1).expanding().mean())
)

# Fallback: if no season-to-date data (week 1 of season), use prior season avg
prior_season_snap = (
    master.groupby(['player_id', 'season'])['offense_pct']
    .mean()
    .reset_index()
    .rename(columns={'season': '_prior_season', 'offense_pct': '_snap_prior'})
)
prior_season_snap['season'] = prior_season_snap['_prior_season'] + 1

master = master.merge(
    prior_season_snap[['player_id', 'season', '_snap_prior']],
    on=['player_id', 'season'], how='left'
)
master['_snap_rank_val'] = master['_snap_share_ytd'].fillna(master['_snap_prior'])

print(f"  Snap rank values: {master['_snap_rank_val'].notna().sum():,} non-null")

# %%
# --- Step 13b: Define group configs ---

# Rolling metrics available from Steps 7b and 12
# Map: (position, metric_prefix, top_N, rank_by_col, output_prefix)
GROUP_CONFIGS = {
    'QB': {
        'top_n':    1,
        'rank_col': '_snap_rank_val',
        'metrics': {
            'epa_per_dropback':  'off_qb_epa_per_dropback',
            'qb_cpoe':           'off_qb_cpoe',
            'adot':              'off_qb_adot',
            'yards_per_attempt': 'off_qb_yards_per_attempt',
            'qb_pressure_rate':  'off_qb_pressure_rate',
        }
    },
    'RB': {
        'top_n':    3,
        'rank_col': '_snap_rank_val',
        'metrics': {
            'epa_per_carry':    'off_rb_epa_per_carry',
            'yards_per_carry':  'off_rb_yards_per_carry',
            'carry_share':      'off_rb_carry_share',
            'receiving_yards':  'off_rb_receiving_yards',  # receiving threat proxy
            'receptions':       'off_rb_receptions',
        }
    },
    'WR': {
        'top_n':    6,
        'rank_col': '_snap_rank_val',
        'metrics': {
            'wopr':             'off_wr_wopr',
            'adot':             'off_wr_adot',
            'racr':             'off_wr_racr',
            'epa_per_target':   'off_wr_epa_per_target',
            'ngs_avg_separation':'off_wr_separation',  # NGS 2016+, NaN pre-2016
        }
    },
    'TE': {
        'top_n':    3,
        'rank_col': '_snap_rank_val',
        'metrics': {
            'epa_per_target': 'off_te_epa_per_target',
            'target_share':   'off_te_target_share',
        }
    },
}

# %%
# --- Step 13c: Compute group quality scores per (team, season, week) ---
# For each game, find the players who appeared (have a row in master),
# rank them by _snap_rank_val, take top N, average their rolling metrics.

print("  Computing positional group quality scores...")

# We need rolling metric columns at each window
# Map from metric base name -> actual column names in master
def get_roll_col(metric, w):
    """Return the column name for a rolling metric at window w."""
    # These come from Step 12 derived cols or Step 7b base rolling cols
    candidates = [f'{metric}_L{w}']
    for c in candidates:
        if c in master.columns:
            return c
    return None

# Build the full output column list upfront for idempotent drop
all_step13_cols = []
for pos, cfg in GROUP_CONFIGS.items():
    for metric, prefix in cfg['metrics'].items():
        for w in POS_WINDOWS:
            all_step13_cols.append(f'{prefix}_L{w}')

master.drop(columns=[c for c in all_step13_cols if c in master.columns],
            errors='ignore', inplace=True)

# Process each position group
for pos, cfg in GROUP_CONFIGS.items():
    top_n    = cfg['top_n']
    rank_col = cfg['rank_col']
    metrics  = cfg['metrics']

    print(f"  Processing {pos} group (top {top_n})...")

    # All rows for this position
    pos_rows = master[master['position'] == pos].copy()

    # Get unique (team, season, week) games that have this position
    games = pos_rows[['team', 'season', 'week']].drop_duplicates()

    result_records = []

    for _, game in games.iterrows():
        t, s, w = game['team'], game['season'], game['week']

        # All players of this position on this team in this game
        grp = pos_rows[
            (pos_rows['team'] == t) &
            (pos_rows['season'] == s) &
            (pos_rows['week'] == w)
        ].copy()

        if len(grp) == 0:
            continue

        # Rank by snap_rank_val descending, take top N
        # Player is INCLUDED in their own group — group quality reflects the
        # full positional corps that played, including the player themselves.
        grp_sorted = grp.nlargest(top_n, rank_col, keep='all')

        # Snap-share weighted average — CMC at 75% snap share gets 7x the
        # weight of a backup at 10%. Simple average would be illogical.
        weights = grp_sorted[rank_col].fillna(0).values
        weight_sum = weights.sum()

        rec_base = {}
        for metric, prefix in metrics.items():
            for ww in POS_WINDOWS:
                src_col = get_roll_col(metric, ww)
                if src_col and src_col in grp_sorted.columns:
                    vals = grp_sorted[src_col].values
                    # Only include players with non-null metric values in weighted avg
                    valid_mask = ~pd.isna(vals)
                    if valid_mask.sum() == 0 or weight_sum == 0:
                        val = float('nan')
                    else:
                        w_valid = weights[valid_mask]
                        v_valid = vals[valid_mask]
                        val = (v_valid * w_valid).sum() / w_valid.sum()
                else:
                    val = float('nan')
                rec_base[f'{prefix}_L{ww}'] = val

        # One record per (team, season, week) — same group score for all
        # players on that team that game, regardless of their own position.
        # A WR gets the QB group score, RB group score, TE group score, etc.
        rec = {'team': t, 'season': s, 'week': w}
        rec.update(rec_base)
        result_records.append(rec)

    if not result_records:
        continue

    results_df = pd.DataFrame(result_records)
    roll_cols_pos = [f'{prefix}_L{ww}'
                     for metric, prefix in metrics.items()
                     for ww in POS_WINDOWS]

    # Join on (team, season, week) — ALL players on the team get this group's
    # quality scores, not just players of the matching position
    master.drop(columns=[c for c in roll_cols_pos if c in master.columns],
                errors='ignore', inplace=True)
    master = master.merge(
        results_df[['team', 'season', 'week'] + roll_cols_pos],
        on=['team', 'season', 'week'],
        how='left'
    )

    print(f"    {pos}: {len(results_df):,} team-game rows -> joined to all {len(master):,} player rows")

# %%
# --- Step 13d: Team offense composites (rolling, not roster-aware) ---
# off_epa_per_play, off_pass_rate, off_success_rate
# Sourced from play-by-play aggregates already in master (qb_epa_per_dropback
# as pass proxy, rushing_epa as run proxy)

print("  Computing team offense composites...")

# Use existing EPA columns to build per-game team totals
# pass plays = QB attempts + scrambles (approx = attempts)
# run plays = team carries

team_comp = (
    master.groupby(['team', 'season', 'week'])
    .agg(
        _pass_epa   = ('passing_epa',  'sum'),
        _rush_epa   = ('rushing_epa',  'sum'),
        _attempts   = ('attempts',     'sum'),
        _carries    = ('carries',      'sum'),
    )
    .reset_index()
)
team_comp['_total_plays'] = (
    team_comp['_attempts'].fillna(0) + team_comp['_carries'].fillna(0)
)
team_comp['_total_epa'] = (
    team_comp['_pass_epa'].fillna(0) + team_comp['_rush_epa'].fillna(0)
)
team_comp['_pass_rate']  = (
    team_comp['_attempts'] / team_comp['_total_plays'].replace(0, float('nan'))
)
team_comp['_epa_per_play'] = (
    team_comp['_total_epa'] / team_comp['_total_plays'].replace(0, float('nan'))
)
team_comp = team_comp.sort_values(['team', 'season', 'week']).reset_index(drop=True)

comp_roll_cols = []
for src, prefix in [('_epa_per_play', 'off_epa_per_play'),
                    ('_pass_rate',    'off_pass_rate')]:
    for w in POS_WINDOWS:
        col = f'{prefix}_L{w}'
        team_comp[col] = (
            team_comp.groupby('team')[src]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )
        comp_roll_cols.append(col)

master.drop(columns=[c for c in comp_roll_cols if c in master.columns],
            errors='ignore', inplace=True)
master = master.merge(
    team_comp[['team', 'season', 'week'] + comp_roll_cols],
    on=['team', 'season', 'week'], how='left'
)
print(f"  Team composite cols added: {len(comp_roll_cols)}")

# %%
# Clean up helper columns
master.drop(columns=['_snap_share_ytd', '_snap_prior', '_snap_rank_val'],
            errors='ignore', inplace=True)

# %%
# --- Validation ---
print(f"\n  Master shape: {master.shape}")

sample_check = all_step13_cols[:6] + comp_roll_cols[:4]
print(f"\n  Null rates on sample Step 13 cols:")
for col in sample_check:
    if col in master.columns:
        pct = master[col].isna().mean() * 100
        print(f"    {col}: {pct:.1f}%")

# Spot-check: Ja'Marr Chase 2024 — WR group should exclude Chase himself
chase = master[
    (master['player_display_name'].str.contains("Chase", na=False)) &
    (master['season'] == 2024) &
    (master['position'] == 'WR')
][['player_display_name', 'season', 'week',
   'wopr_L5', 'off_wr_wopr_L5', 'off_wr_racr_L5', 'off_wr_epa_per_target_L5']].head(6)
print(f"\n  Spot-check Ja'Marr Chase 2024 (own wopr vs WR group wopr):")
print(chase.to_string(index=False))

# %%
# --- Save master after Step 13 ---
save_master(master, step=13)

# %%
# =============================================================================
# STEP 14 — Opponent defense positional quality (Section E defense)
# =============================================================================
#
# For each player-game row, compute rolling stats describing how well the
# OPPONENT's defense has performed against each offensive positional group
# (QB, WR, RB, TE) in their prior L3/L5/L10/L20 games.
#
# Approach: team-level "allowed" stats, not individual defender tracking.
# For each (defteam, season, week), aggregate what the opposing offense's
# position groups produced — then roll those per-game rates over prior games.
#
# Aggregation: volume-weighted rates via sum/sum (normalizes for game volume).
#   QB  : passing_epa/attempts, qb_cpoe mean, passing_yards/attempts,
#          passing_tds/attempts
#   WR  : receiving_epa/targets, receiving_yards/targets,
#          receiving_tds/targets, receptions/targets (catch rate)
#   RB  : rushing_epa/carries, rushing_yards/carries,
#          rushing_tds/carries (run defense)
#          + receiving_epa/targets, receiving_yards/targets (pass defense vs RB)
#   TE  : receiving_epa/targets, receiving_yards/targets,
#          receiving_tds/targets, receptions/targets (catch rate)
#
# Team-level composites:
#   pass defense: passing_epa / passing_attempts (all QBs)
#   rush defense: rushing_epa / carries (all RBs + QBs)
#
# Windows: L3/L5/L10/L20 (cross-season, no reset — same as individual players)
# Join key: master['opponent'] == defteam
# =============================================================================

print("=" * 60)
print("STEP 14 — Opponent defense positional quality (allowed stats)")
print("=" * 60)

DEF_WINDOWS = [3, 5, 10, 20]

# %%
# --- Step 14a: Build per-game positional group allowed stats ---
# For each (opponent, season, week, position_group), compute volume-weighted
# rate stats using raw game columns.
# "opponent" here is the DEFENSIVE team — these are the stats they allowed.

print("  Building per-game allowed stats per position group...")

# We need the raw game columns (not rolled):
#   QB  : attempts, passing_epa, qb_cpoe, passing_yards, passing_tds
#   WR  : targets, receiving_epa, receiving_yards, receiving_tds, receptions
#   RB  : carries, rushing_epa, rushing_yards, rushing_tds,
#          targets (receiving), receiving_epa (as rec_epa), receiving_yards (as rec_yds)
#   TE  : targets, receiving_epa, receiving_yards, receiving_tds, receptions

# Sort master to ensure consistent ordering
master = master.sort_values(['season', 'week', 'team', 'player_id']).reset_index(drop=True)

# --- QB allowed ---
qb_rows = master[master['position'] == 'QB'].copy()
qb_game = (
    qb_rows.groupby(['opponent', 'season', 'week'])
    .agg(
        _qb_attempts     = ('attempts',       'sum'),
        _qb_pass_epa     = ('passing_epa',    'sum'),
        _qb_pass_yards   = ('passing_yards',  'sum'),
        _qb_pass_tds     = ('passing_tds',    'sum'),
        _qb_interceptions= ('interceptions',  'sum'),
        _qb_cpoe_sum     = ('qb_cpoe',        'sum'),    # will divide by games below
        _qb_cpoe_count   = ('qb_cpoe',        'count'),  # for mean
    )
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
# Volume-weighted rates (safe divide — 0 attempts → NaN)
safe = qb_game['_qb_attempts'].replace(0, float('nan'))
qb_game['_qb_epa_per_attempt']   = qb_game['_qb_pass_epa']   / safe
qb_game['_qb_yards_per_attempt'] = qb_game['_qb_pass_yards'] / safe
qb_game['_qb_td_rate']           = qb_game['_qb_pass_tds']   / safe
qb_game['_qb_int_rate']          = qb_game['_qb_interceptions'] / safe
# CPOE: mean across QBs (weighted by count)
safe_c = qb_game['_qb_cpoe_count'].replace(0, float('nan'))
qb_game['_qb_cpoe'] = qb_game['_qb_cpoe_sum'] / safe_c

print(f"  QB allowed: {len(qb_game):,} defteam-game rows")

# --- WR allowed ---
wr_rows = master[master['position'] == 'WR'].copy()
wr_game = (
    wr_rows.groupby(['opponent', 'season', 'week'])
    .agg(
        _wr_targets       = ('targets',          'sum'),
        _wr_rec_epa       = ('receiving_epa',    'sum'),
        _wr_rec_yards     = ('receiving_yards',  'sum'),
        _wr_rec_tds       = ('receiving_tds',    'sum'),
        _wr_receptions    = ('receptions',       'sum'),
    )
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
safe = wr_game['_wr_targets'].replace(0, float('nan'))
wr_game['_wr_epa_per_target']   = wr_game['_wr_rec_epa']   / safe
wr_game['_wr_yards_per_target'] = wr_game['_wr_rec_yards'] / safe
wr_game['_wr_td_rate']          = wr_game['_wr_rec_tds']   / safe
wr_game['_wr_catch_rate']       = wr_game['_wr_receptions'] / safe

print(f"  WR allowed: {len(wr_game):,} defteam-game rows")

# --- RB allowed ---
rb_rows = master[master['position'] == 'RB'].copy()
rb_game = (
    rb_rows.groupby(['opponent', 'season', 'week'])
    .agg(
        _rb_carries       = ('carries',          'sum'),
        _rb_rush_epa      = ('rushing_epa',      'sum'),
        _rb_rush_yards    = ('rushing_yards',    'sum'),
        _rb_rush_tds      = ('rushing_tds',      'sum'),
        _rb_rec_targets   = ('targets',          'sum'),
        _rb_rec_epa       = ('receiving_epa',    'sum'),
        _rb_rec_yards     = ('receiving_yards',  'sum'),
    )
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
safe_c = rb_game['_rb_carries'].replace(0, float('nan'))
rb_game['_rb_rush_epa_per_carry']   = rb_game['_rb_rush_epa']   / safe_c
rb_game['_rb_rush_yards_per_carry'] = rb_game['_rb_rush_yards'] / safe_c
rb_game['_rb_rush_td_rate']         = rb_game['_rb_rush_tds']   / safe_c
safe_t = rb_game['_rb_rec_targets'].replace(0, float('nan'))
rb_game['_rb_rec_epa_per_target']   = rb_game['_rb_rec_epa']    / safe_t
rb_game['_rb_rec_yards_per_target'] = rb_game['_rb_rec_yards']  / safe_t

print(f"  RB allowed: {len(rb_game):,} defteam-game rows")

# --- TE allowed ---
te_rows = master[master['position'] == 'TE'].copy()
te_game = (
    te_rows.groupby(['opponent', 'season', 'week'])
    .agg(
        _te_targets       = ('targets',          'sum'),
        _te_rec_epa       = ('receiving_epa',    'sum'),
        _te_rec_yards     = ('receiving_yards',  'sum'),
        _te_rec_tds       = ('receiving_tds',    'sum'),
        _te_receptions    = ('receptions',       'sum'),
    )
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
safe = te_game['_te_targets'].replace(0, float('nan'))
te_game['_te_epa_per_target']   = te_game['_te_rec_epa']   / safe
te_game['_te_yards_per_target'] = te_game['_te_rec_yards'] / safe
te_game['_te_td_rate']          = te_game['_te_rec_tds']   / safe
te_game['_te_catch_rate']       = te_game['_te_receptions'] / safe

print(f"  TE allowed: {len(te_game):,} defteam-game rows")

# --- Team-level defense composites ---
# pass EPA: all QB passing EPA / all QB attempts
# rush EPA: all RB + QB rushing EPA / all carries
team_pass = (
    master[master['position'] == 'QB']
    .groupby(['opponent', 'season', 'week'])
    .agg(_team_pass_epa=('passing_epa','sum'), _team_attempts=('attempts','sum'))
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
safe = team_pass['_team_attempts'].replace(0, float('nan'))
team_pass['_team_epa_per_pass'] = team_pass['_team_pass_epa'] / safe

team_rush = (
    master[master['position'].isin(['RB', 'QB'])]
    .groupby(['opponent', 'season', 'week'])
    .agg(_team_rush_epa=('rushing_epa','sum'), _team_carries=('carries','sum'))
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)
safe = team_rush['_team_carries'].replace(0, float('nan'))
team_rush['_team_epa_per_rush'] = team_rush['_team_rush_epa'] / safe

# fumbles forced: all skill-position fumbles lost against each defense, per game
team_fumbles = (
    master[master['position'].isin(['QB', 'RB', 'WR', 'TE'])]
    .groupby(['opponent', 'season', 'week'])
    .agg(_team_fumbles_forced=('fumbles_lost_total', 'sum'))
    .reset_index()
    .rename(columns={'opponent': 'defteam'})
)

print(f"  Team composites: {len(team_pass):,} pass rows, {len(team_rush):,} rush rows, "
      f"{len(team_fumbles):,} fumble rows")

# %%
# --- Step 14b: Roll allowed stats over prior games ---
# Sort each frame by (defteam, season, week) and shift(1).rolling(w).

print("  Rolling allowed stats over prior games...")

def roll_def_frame(df, key_col, rate_cols, windows):
    """Sort by (defteam, season, week), roll rate_cols over prior games."""
    df = df.sort_values(['defteam', 'season', 'week']).reset_index(drop=True)
    out_cols = {}
    for col in rate_cols:
        if col not in df.columns:
            continue
        for w in windows:
            out_col = f'{key_col}_{col.lstrip("_")}_L{w}'
            df[out_col] = (
                df.groupby('defteam')[col]
                .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            )
            out_cols[out_col] = True
    return df, list(out_cols.keys())

def ewm_def_frame(df, key_col, rate_cols, spans):
    """Compute EWM versions of rate_cols over prior games (shift(1), grouped by defteam)."""
    df = df.sort_values(['defteam', 'season', 'week']).reset_index(drop=True)
    out_cols = {}
    for col in rate_cols:
        if col not in df.columns:
            continue
        for s in spans:
            out_col = f'{key_col}_{col.lstrip("_")}_ewm{s}'
            df[out_col] = (
                df.groupby('defteam')[col]
                .transform(lambda x, s=s: x.shift(1).ewm(span=s, min_periods=1).mean())
            )
            out_cols[out_col] = True
    return df, list(out_cols.keys())

DEF_EWM_SPANS = [5, 10, 20]

# QB defense rolling
qb_rate_cols = ['_qb_epa_per_attempt', '_qb_yards_per_attempt', '_qb_td_rate', '_qb_int_rate', '_qb_cpoe']
qb_game, qb_roll_cols = roll_def_frame(
    qb_game, 'opp_def_qb', qb_rate_cols, DEF_WINDOWS
)
qb_game, qb_ewm_cols = ewm_def_frame(
    qb_game, 'opp_def_qb', ['_qb_int_rate'], DEF_EWM_SPANS
)
print(f"  QB defense rolled: {len(qb_roll_cols)} cols + {len(qb_ewm_cols)} EWM cols")

# WR defense rolling
wr_rate_cols = ['_wr_epa_per_target', '_wr_yards_per_target', '_wr_td_rate', '_wr_catch_rate']
wr_game, wr_roll_cols = roll_def_frame(
    wr_game, 'opp_def_wr', wr_rate_cols, DEF_WINDOWS
)
wr_ewm_cols = []
print(f"  WR defense rolled: {len(wr_roll_cols)} cols")

# RB defense rolling
rb_rate_cols = [
    '_rb_rush_epa_per_carry', '_rb_rush_yards_per_carry', '_rb_rush_td_rate',
    '_rb_rec_epa_per_target', '_rb_rec_yards_per_target',
]
rb_game, rb_roll_cols = roll_def_frame(
    rb_game, 'opp_def_rb', rb_rate_cols, DEF_WINDOWS
)
rb_ewm_cols = []
print(f"  RB defense rolled: {len(rb_roll_cols)} cols")

# TE defense rolling
te_rate_cols = ['_te_epa_per_target', '_te_yards_per_target', '_te_td_rate', '_te_catch_rate']
te_game, te_roll_cols = roll_def_frame(
    te_game, 'opp_def_te', te_rate_cols, DEF_WINDOWS
)
te_ewm_cols = []
print(f"  TE defense rolled: {len(te_roll_cols)} cols")

# Team composites rolling
team_pass, pass_roll_cols = roll_def_frame(
    team_pass, 'opp_def_team', ['_team_epa_per_pass'], DEF_WINDOWS
)
team_rush, rush_roll_cols = roll_def_frame(
    team_rush, 'opp_def_team', ['_team_epa_per_rush'], DEF_WINDOWS
)
team_fumbles, fumble_roll_cols = roll_def_frame(
    team_fumbles, 'opp_def_team', ['_team_fumbles_forced'], DEF_WINDOWS
)
team_fumbles, fumble_ewm_cols = ewm_def_frame(
    team_fumbles, 'opp_def_team', ['_team_fumbles_forced'], DEF_EWM_SPANS
)
n_rolled = len(pass_roll_cols) + len(rush_roll_cols) + len(fumble_roll_cols)
n_ewm = len(qb_ewm_cols) + len(fumble_ewm_cols)
print(f"  Team composites rolled: {n_rolled} cols + {n_ewm} EWM cols")

# %%
# --- Step 14c: Join all allowed rolling stats to master ---
# Join key: master['opponent'] == defteam, on (season, week)
# All players on a team get the same opponent defense profile.

print("  Joining allowed stats to master on (opponent, season, week)...")

all_def_roll_cols = (
    qb_roll_cols + qb_ewm_cols +
    wr_roll_cols + rb_roll_cols + te_roll_cols +
    pass_roll_cols + rush_roll_cols +
    fumble_roll_cols + fumble_ewm_cols
)

# Drop any stale columns from a prior run
master.drop(columns=[c for c in all_def_roll_cols if c in master.columns],
            errors='ignore', inplace=True)

for frame, roll_cols in [
    (qb_game,      qb_roll_cols + qb_ewm_cols),
    (wr_game,      wr_roll_cols),
    (rb_game,      rb_roll_cols),
    (te_game,      te_roll_cols),
    (team_pass,    pass_roll_cols),
    (team_rush,    rush_roll_cols),
    (team_fumbles, fumble_roll_cols + fumble_ewm_cols),
]:
    join_df = frame[['defteam', 'season', 'week'] + roll_cols].rename(
        columns={'defteam': 'opponent'}
    )
    master = master.merge(join_df, on=['opponent', 'season', 'week'], how='left')

print(f"  Total opp_def cols joined: {len(all_def_roll_cols)}")
print(f"  Master shape: {master.shape}")

# %%
# --- Validation ---
sample_def_cols = [
    'opp_def_qb_qb_epa_per_attempt_L5',
    'opp_def_qb_qb_cpoe_L5',
    'opp_def_qb_qb_int_rate_L5',
    'opp_def_wr_wr_epa_per_target_L5',
    'opp_def_wr_wr_catch_rate_L5',
    'opp_def_rb_rb_rush_epa_per_carry_L5',
    'opp_def_te_te_epa_per_target_L5',
    'opp_def_team_team_epa_per_pass_L5',
    'opp_def_team_team_epa_per_rush_L5',
    'opp_def_team_team_fumbles_forced_L5',
]
print(f"\n  Null rates on sample Step 14 cols:")
for col in sample_def_cols:
    if col in master.columns:
        pct = master[col].isna().mean() * 100
        print(f"    {col}: {pct:.1f}%")
    else:
        print(f"    {col}: MISSING")

# Spot-check: Ja'Marr Chase 2024 — should see opponent WR defense profile
chase = master[
    (master['player_display_name'].str.contains("Chase", na=False)) &
    (master['season'] == 2024) &
    (master['position'] == 'WR')
][['player_display_name', 'season', 'week', 'opponent',
   'opp_def_wr_wr_epa_per_target_L5',
   'opp_def_wr_wr_yards_per_target_L5',
   'opp_def_wr_wr_catch_rate_L5']].head(6)
print(f"\n  Spot-check Ja'Marr Chase 2024 (opponent WR defense):")
print(chase.to_string(index=False))

# %%
# --- Drop Step 8 raw opponent totals (superseded by Step 14 rate-based opp_def_* cols) ---
# Step 8 produced rolling raw totals (opp_qb_passing_yards_L5, opp_wr_receiving_yards_L5, etc.)
# Step 14 covers the same information via volume-weighted rates which are better normalised.
step8_opp_cols = [c for c in master.columns
                  if c.startswith('opp_') and not c.startswith('opp_def_')]
master.drop(columns=step8_opp_cols, inplace=True)
print(f"  Dropped {len(step8_opp_cols)} Step 8 raw opp_* cols")
print(f"  Final master shape: {master.shape}")

# %%
# --- Save master after Step 14 ---
save_master(master, step=14)
