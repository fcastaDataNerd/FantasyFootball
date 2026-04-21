# %% [markdown]
# # 2026 QB Projections — Passing Yards (Phase 1)
# Week-by-week predictions for all QB1/QB2 with 95% CIs.
# Sources: nfl_data_py (schedule, depth charts), master dataset (rolling features).
# Expand later: add remaining 5 stat models + fantasy totals.

# %%
import io
import sys
import warnings
from pathlib import Path

import joblib
import nfl_data_py as nfl
import numpy as np
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

MASTER_PATH   = DATA_DIR / "data" / "master" / "nfl_master_dataset.parquet"
ARTIFACTS_DIR = DATA_DIR / "data" / "model_artifacts"
REGISTRY_PATH = ARTIFACTS_DIR / "qb_lgb_models.pkl"
OUTPUT_PATH   = DATA_DIR / "predictions_2026_passing_yards.xlsx"

SEASON        = 2026
DEPTH_RANKS   = [1, 2]   # QB1 and QB2 only
CI_Z          = 1.96     # 95% CI
ABBREV_MAP    = {'SD': 'LAC', 'STL': 'LA', 'OAK': 'LV'}
DOME_ROOFS    = {'dome', 'closed'}

# %%
# =============================================================================
# STEP 0 — Load master dataset and model artifacts
# =============================================================================
print("Loading master dataset...")
master = pd.read_parquet(MASTER_PATH)
master['game_date'] = pd.to_datetime(master['game_date'], errors='coerce')
master['_game_month'] = master['game_date'].dt.month
print(f"  Shape: {master.shape}")

print("Loading model artifacts...")
reg          = joblib.load(REGISTRY_PATH)
oof_store    = reg["oof_store"]

# --- Passing yards model ---
model_py     = reg["lgb_models"]["passing_yards"]
feature_cols = reg["feature_cols"]   # QB_FEATURES saved in registry

oof_actual = np.array(oof_store["passing_yards"]["oof_actual"])
oof_pred   = np.array(oof_store["passing_yards"]["oof_pred"])
RMSE_GAME  = float(np.sqrt(np.mean((oof_actual - oof_pred) ** 2)))
print(f"  OOF RMSE (passing yards, game-level): {RMSE_GAME:.2f} yards")
print(f"  Features: {len(feature_cols)}")

# --- Passing TDs model ---
model_td        = reg["lgb_models"]["passing_tds"]
TD_FEATURE_COLS = reg["td_feature_cols"]

oof_actual_td = np.array(oof_store["passing_tds"]["oof_actual"])
oof_pred_td   = np.array(oof_store["passing_tds"]["oof_pred"])
RMSE_GAME_TD  = float(np.sqrt(np.mean((oof_actual_td - oof_pred_td) ** 2)))
print(f"  OOF RMSE (passing tds,   game-level): {RMSE_GAME_TD:.3f} TDs")
print(f"  TD features: {len(TD_FEATURE_COLS)}")

# --- Rushing yards model ---
model_ry        = reg["lgb_models"]["rushing_yards"]
RY_FEATURE_COLS = reg["ry_feature_cols"]

oof_actual_ry = np.array(oof_store["rushing_yards"]["oof_actual"])
oof_pred_ry   = np.array(oof_store["rushing_yards"]["oof_pred"])
RMSE_GAME_RY  = float(np.sqrt(np.mean((oof_actual_ry - oof_pred_ry) ** 2)))
print(f"  OOF RMSE (rushing yards, game-level): {RMSE_GAME_RY:.2f} yards")
print(f"  RY features: {len(RY_FEATURE_COLS)}")

# %%
# =============================================================================
# STEP 1 — 2026 schedule (hardcoded from NFL.com opponent announcements)
# Source: https://www.nfl.com/news/2026-nfl-season-team-by-team-opponents-for-every-game
# NFC West schedules inferred from cross-references in the 28 published team lists.
# Weeks assigned sequentially 1-17 per team (actual week order TBD when NFL publishes).
# rest_days defaults to 7 (standard); game_month estimated from week number.
# =============================================================================
print("\nSTEP 1: Building 2026 schedule from hardcoded opponents...")

# Teams whose home stadium is a dome (affects is_dome and weather features)
DOME_HOME_TEAMS = {'HOU', 'IND', 'LV', 'LAC', 'LA', 'ARI', 'DAL', 'DET', 'MIN', 'ATL', 'NO'}

# Week -> approximate calendar month (NFL season Sept-Jan)
def _week_to_month(w):
    if w <= 4:  return 9   # September
    if w <= 9:  return 10  # October
    if w <= 13: return 11  # November
    return 12              # December

# Opponent lists per team: home games listed first, then away
# game_location: 1 = home, -1 = away
SCHEDULE_RAW = {
    # AFC East
    'NE':  {'home': ['BUF','MIA','NYJ','DEN','GB','LV','MIN','PIT'],
             'away': ['BUF','MIA','NYJ','CHI','DET','JAX','KC','LAC','SEA']},
    'BUF': {'home': ['MIA','NE','NYJ','BAL','CHI','DET','KC','LAC'],
             'away': ['MIA','NE','NYJ','DEN','GB','HOU','LV','LA','MIN']},
    'MIA': {'home': ['BUF','NE','NYJ','CHI','CIN','DET','KC','LAC'],
             'away': ['BUF','NE','NYJ','DEN','IND','GB','LV','MIN','SF']},
    'NYJ': {'home': ['BUF','MIA','NE','CLE','DEN','GB','LV','MIN'],
             'away': ['BUF','MIA','NE','ARI','CHI','DET','KC','LAC','TEN']},
    # AFC North
    'PIT': {'home': ['BAL','CIN','CLE','ATL','CAR','DEN','HOU','IND'],
             'away': ['BAL','CIN','CLE','JAX','NE','NO','PHI','TB','TEN']},
    'BAL': {'home': ['CIN','CLE','PIT','JAX','LAC','NO','TB','TEN'],
             'away': ['CIN','CLE','PIT','ATL','BUF','CAR','DAL','HOU','IND']},
    'CIN': {'home': ['BAL','CLE','PIT','JAX','KC','NO','TB','TEN'],
             'away': ['BAL','CLE','PIT','ATL','CAR','HOU','IND','MIA','WAS']},
    'CLE': {'home': ['BAL','CIN','PIT','ATL','CAR','HOU','IND','LV'],
             'away': ['BAL','CIN','PIT','JAX','NO','TB','NYG','NYJ','TEN']},
    # AFC South
    'JAX': {'home': ['HOU','IND','TEN','CLE','NE','PHI','PIT','WAS'],
             'away': ['HOU','IND','TEN','BAL','CHI','CIN','DAL','DEN','NYG']},
    'HOU': {'home': ['IND','JAX','TEN','BAL','BUF','CIN','DAL','NYG'],
             'away': ['IND','JAX','TEN','CLE','GB','LAC','PHI','PIT','WAS']},
    'IND': {'home': ['HOU','JAX','TEN','BAL','CIN','DAL','MIA','NYG'],
             'away': ['HOU','JAX','TEN','CLE','KC','MIN','PHI','PIT','WAS']},
    'TEN': {'home': ['HOU','IND','JAX','CLE','NYJ','PHI','PIT','WAS'],
             'away': ['HOU','IND','JAX','BAL','CIN','DAL','DET','LV','NYG']},
    # AFC West
    'DEN': {'home': ['KC','LV','LAC','BUF','JAX','LA','MIA','SEA'],
             'away': ['KC','LV','LAC','ARI','CAR','NE','NYJ','PIT','SF']},
    'LAC': {'home': ['DEN','KC','LV','ARI','HOU','NE','NYJ','SF'],
             'away': ['DEN','KC','LV','BAL','BUF','LA','MIA','SEA','TB']},
    'KC':  {'home': ['DEN','LV','LAC','ARI','IND','NE','NYJ','SF'],
             'away': ['DEN','LV','LAC','ATL','BUF','CIN','LA','MIA','SEA']},
    'LV':  {'home': ['DEN','KC','LAC','BUF','LA','MIA','SEA','TEN'],
             'away': ['DEN','KC','LAC','ARI','CLE','NE','NO','NYJ','SF']},
    # NFC East
    'PHI': {'home': ['DAL','NYG','WAS','CAR','HOU','IND','LA','PIT','SEA'],
             'away': ['DAL','NYG','WAS','ARI','CHI','JAX','SF','TEN']},
    'DAL': {'home': ['NYG','PHI','WAS','ARI','BAL','JAX','SF','TB','TEN'],
             'away': ['NYG','PHI','WAS','GB','HOU','IND','LA','SEA']},
    'WAS': {'home': ['DAL','NYG','PHI','ATL','CIN','HOU','IND','LA','SEA'],
             'away': ['DAL','NYG','PHI','ARI','JAX','MIN','SF','TEN']},
    'NYG': {'home': ['DAL','PHI','WAS','ARI','CLE','JAX','NO','SF','TEN'],
             'away': ['DAL','PHI','WAS','DET','HOU','IND','LA','SEA']},
    # NFC North
    'CHI': {'home': ['DET','GB','MIN','JAX','NE','NO','NYJ','PHI','TB'],
             'away': ['DET','GB','MIN','ATL','BUF','CAR','MIA','SEA']},
    'GB':  {'home': ['CHI','DET','MIN','ATL','BUF','CAR','DAL','HOU','MIA'],
             'away': ['CHI','DET','MIN','LA','NE','NO','NYJ','TB']},
    'MIN': {'home': ['CHI','DET','GB','ATL','BUF','CAR','IND','MIA','WAS'],
             'away': ['CHI','DET','GB','NE','NO','NYJ','SF','TB']},
    'DET': {'home': ['CHI','GB','MIN','NE','NO','NYG','NYJ','TB','TEN'],
             'away': ['CHI','GB','MIN','ARI','ATL','BUF','CAR','MIA']},
    # NFC South
    'CAR': {'home': ['ATL','NO','TB','BAL','CHI','CIN','DEN','DET','SEA'],
             'away': ['ATL','NO','TB','CLE','GB','MIN','PHI','PIT']},
    'TB':  {'home': ['ATL','CAR','NO','CLE','GB','LAC','LA','MIN','PIT'],
             'away': ['ATL','CAR','NO','BAL','CHI','CIN','DAL','DET']},
    'ATL': {'home': ['CAR','NO','TB','BAL','CHI','CIN','DET','KC','SF'],
             'away': ['CAR','NO','TB','CLE','GB','MIN','PIT','WAS']},
    'NO':  {'home': ['ATL','CAR','TB','ARI','CLE','GB','LV','MIN','PIT'],
             'away': ['ATL','CAR','TB','BAL','CHI','CIN','DET','NYG']},
    # NFC West — inferred from cross-references in the 28 published schedules
    'ARI': {'home': ['NYJ','DEN','LV','PHI','WAS','DET','LA','SF','SEA'],
             'away': ['LAC','KC','DAL','NYG','NO','LA','SF','SEA']},
    'LA':  {'home': ['GB','DAL','LAC','KC','NYG','BUF','ARI','SF','SEA'],
             'away': ['LV','WAS','PHI','TB','DEN','ARI','SF','SEA']},
    'SF':  {'home': ['LV','DEN','MIA','PHI','WAS','MIN','ARI','LA','SEA'],
             'away': ['LAC','KC','DAL','NYG','ATL','ARI','LA','SEA']},
    'SEA': {'home': ['CHI','KC','LAC','DAL','NYG','NE','ARI','LA','SF'],
             'away': ['DEN','LV','WAS','PHI','CAR','ARI','LA','SF']},
}

# Expand to one row per (team, game)
_rows = []
for team, sched in SCHEDULE_RAW.items():
    week = 1
    for opp in sched['home']:
        home_team = team
        _rows.append({
            'team': team, 'opponent': opp, 'game_location': 1, 'week': week,
            'is_dome': int(home_team in DOME_HOME_TEAMS),
            'game_month': _week_to_month(week),
            'rest_days': 7, 'temp': np.nan, 'wind': np.nan,
        })
        week += 1
    for opp in sched['away']:
        home_team = opp
        _rows.append({
            'team': team, 'opponent': opp, 'game_location': -1, 'week': week,
            'is_dome': int(home_team in DOME_HOME_TEAMS),
            'game_month': _week_to_month(week),
            'rest_days': 7, 'temp': np.nan, 'wind': np.nan,
        })
        week += 1

team_sched = pd.DataFrame(_rows).sort_values(['team', 'week']).reset_index(drop=True)
print(f"  {len(team_sched)} team-game rows, {team_sched['team'].nunique()} teams")
print(f"  Games per team: {team_sched.groupby('team').size().describe()[['min','max','mean']]}")

# %%
# =============================================================================
# STEP 2 — QB depth charts
# Try nfl_data_py 2026 first; fall back to end-of-2025 master data.
# For pre-season predictions, end-of-2025 depth charts are a reliable proxy.
# =============================================================================
print("\nSTEP 2: Resolving QB depth charts...")

dc_mapped = pd.DataFrame()  # will be populated below

try:
    dc_raw = nfl.import_depth_charts(years=[SEASON])
    pos_col = next((c for c in ['pos_abb', 'position'] if c in dc_raw.columns), None)
    if pos_col and len(dc_raw) > 0:
        dc_qb = dc_raw[dc_raw[pos_col].str.upper() == 'QB'].copy()
        dc_qb['dt_parsed'] = pd.to_datetime(dc_qb['dt'], utc=True).dt.tz_localize(None)
        # Synthetic week-end dates: week 1 = Sept 13 2026, then +7 days per week
        _week1_end = pd.Timestamp('2026-09-13')
        _week_ends = pd.DataFrame({
            'week': range(1, 19),
            'dt_parsed': [_week1_end + pd.Timedelta(weeks=w-1) for w in range(1, 19)],
        })
        dc_qb = dc_qb.sort_values('dt_parsed')
        _dc = pd.merge_asof(
            dc_qb[['gsis_id', 'pos_rank', 'dt_parsed']],
            _week_ends,
            on='dt_parsed', direction='forward',
        ).dropna(subset=['week'])
        _dc['week'] = _dc['week'].astype(int)
        _dc = (_dc.groupby(['gsis_id', 'week'])['pos_rank']
               .min().reset_index()
               .rename(columns={'gsis_id': 'player_id', 'pos_rank': 'depth_chart_rank'}))
        dc_mapped = _dc[_dc['depth_chart_rank'].isin(DEPTH_RANKS)].copy()
        print(f"  nfl_data_py 2026 depth charts: {len(dc_mapped)} QB entries")
except Exception as e:
    print(f"  nfl_data_py 2026 depth charts unavailable ({e})")

if len(dc_mapped) == 0:
    print("  Falling back to end-of-2025 depth charts from master dataset...")
    dc_mapped = (
        master[(master['position'] == 'QB') & (master['season'] == 2025)]
        .sort_values('week')
        .groupby('player_id')
        .last()
        [['depth_chart_rank']]
        .reset_index()
    )
    dc_mapped = dc_mapped[dc_mapped['depth_chart_rank'].isin(DEPTH_RANKS)].copy()
    # No 'week' column — STEP 4 will cross-join with team_sched by team
    print(f"  Fallback depth chart: {len(dc_mapped)} QB entries (end-of-2025)")

# %%
# =============================================================================
# STEP 3 — QB identity: names, teams (current 2026), ages
# Priority: nfl_data_py 2026 rosters > ESPN API > stale 2025 master fallback
# =============================================================================
print("\nSTEP 3: Resolving player identities + current 2026 teams...")

import requests
import unicodedata

# Base QB history from 2025 master (names, ages, rolling-feature anchor)
qb_2025 = (
    master[(master['position'] == 'QB') & (master['season'] == 2025)]
    .sort_values('week')
    .groupby('player_id')
    .last()
    [['player_display_name', 'team', 'age']]
    .reset_index()
)
qb_2025['age_2026'] = qb_2025['age'] + 1.0

# Static physical attributes (combine data — don't change year to year)
_static = (
    master[master['position'] == 'QB']
    .sort_values(['player_id', 'season', 'week'])
    .groupby('player_id')
    .last()
    [['forty_yard_dash', 'speed_score']]
    .reset_index()
)
qb_2025 = qb_2025.merge(_static, on='player_id', how='left')

# ------------------------------------------------------------------
# Helper: normalize names for fuzzy matching
# ------------------------------------------------------------------
def _norm(s):
    s = unicodedata.normalize('NFD', str(s))
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.lower().strip()

# Build name -> player_id lookup from master
_name_to_pid = {_norm(r['player_display_name']): r['player_id']
                for _, r in qb_2025.iterrows()}

# ------------------------------------------------------------------
# Stage 1: nfl_data_py 2026 rosters
# ------------------------------------------------------------------
current_team_map = {}  # player_id -> current_team_abbrev

try:
    rosters_2026 = nfl.import_rosters(years=[2026])
    if len(rosters_2026) > 0:
        qb_ros = rosters_2026[rosters_2026['position'] == 'QB'].copy()
        for _, row in qb_ros.iterrows():
            pid = row.get('player_id') or row.get('gsis_id')
            tm  = row.get('team')
            if pd.notna(pid) and pd.notna(tm):
                current_team_map[pid] = str(tm).upper()
        print(f"  nfl_data_py 2026 rosters: {len(current_team_map)} QBs")
    else:
        print("  nfl_data_py 2026 rosters: 0 rows returned, trying ESPN...")
except Exception as e:
    print(f"  nfl_data_py 2026 rosters unavailable ({e}), trying ESPN...")

# ------------------------------------------------------------------
# Stage 2: ESPN unofficial roster API (fallback)
# ------------------------------------------------------------------
# ESPN uses slightly different abbreviations for a few teams
ESPN_ABBREV = {
    'ARI':'ari','ATL':'atl','BAL':'bal','BUF':'buf',
    'CAR':'car','CHI':'chi','CIN':'cin','CLE':'cle',
    'DAL':'dal','DEN':'den','DET':'det','GB':'gb',
    'HOU':'hou','IND':'ind','JAX':'jac','KC':'kc',
    'LA':'lar','LAC':'lac','LV':'lv','MIA':'mia',
    'MIN':'min','NE':'ne','NO':'no','NYG':'nyg',
    'NYJ':'nyj','PHI':'phi','PIT':'pit','SEA':'sea',
    'SF':'sf','TB':'tb','TEN':'ten','WAS':'was',
}

if not current_team_map:
    print("  Scraping ESPN API for current QB rosters...")
    espn_rows = []
    for nfl_abbrev, espn_abbrev in ESPN_ABBREV.items():
        url = (f"https://site.api.espn.com/apis/site/v2/sports/football"
               f"/nfl/teams/{espn_abbrev}/roster")
        try:
            resp = requests.get(url, timeout=10,
                                headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code != 200:
                continue
            data = resp.json()
            for group in data.get('athletes', []):
                for athlete in group.get('items', []):
                    pos = (athlete.get('position') or {}).get('abbreviation', '')
                    if pos.upper() == 'QB':
                        espn_rows.append({
                            'full_name': athlete.get('fullName', ''),
                            'team': nfl_abbrev,
                        })
        except Exception:
            continue

    print(f"  ESPN: found {len(espn_rows)} QB entries across all teams")

    # Match ESPN names -> player_ids via normalized name lookup
    matched = unmatched = 0
    for row in espn_rows:
        key = _norm(row['full_name'])
        pid = _name_to_pid.get(key)
        if pid:
            current_team_map[pid] = row['team']
            matched += 1
        else:
            # Try last-name only as a loose fallback
            last = key.split()[-1] if key.split() else key
            candidates = [p for k, p in _name_to_pid.items() if k.endswith(last)]
            if len(candidates) == 1:
                current_team_map[candidates[0]] = row['team']
                matched += 1
            else:
                unmatched += 1

    print(f"  ESPN name match: {matched} matched, {unmatched} unmatched")

# ------------------------------------------------------------------
# Apply current team overrides to qb_2025
# ------------------------------------------------------------------
if current_team_map:
    overrides = qb_2025['player_id'].map(current_team_map)
    changed = overrides.notna() & (overrides != qb_2025['team'])
    if changed.sum() > 0:
        print(f"  Team overrides applied: {changed.sum()} QBs moved teams")
        for _, r in qb_2025[changed].iterrows():
            new_tm = current_team_map[r['player_id']]
            print(f"    {r['player_display_name']}: {r['team']} -> {new_tm}")
    qb_2025['team'] = overrides.combine_first(qb_2025['team'])
else:
    print("  WARNING: No current roster data — using stale 2025 team assignments")

# %%
# =============================================================================
# STEP 4 — Build prediction skeleton (QB x week)
# =============================================================================
print("\nSTEP 4: Building prediction skeleton...")

# Join depth chart -> player identity
skeleton = dc_mapped.merge(
    qb_2025[['player_id', 'player_display_name', 'team', 'age_2026',
             'forty_yard_dash', 'speed_score']],
    on='player_id', how='left'
)

n_missing = skeleton['player_display_name'].isna().sum()
if n_missing > 0:
    print(f"  WARNING: {n_missing} QB entries have no 2025 history (rookies/new)")

# Always cross-join on team only: each QB gets all 17 games in their team's schedule.
# Drop any 'week' column that came from the depth chart source to avoid merge conflicts.
if 'week' in skeleton.columns:
    skeleton = skeleton.drop(columns=['week'])
skeleton = skeleton.merge(team_sched, on='team', how='left')

# games_played_current_season: 0 for first game of season, cumcount by player
skeleton = skeleton.sort_values(['player_id', 'week']).reset_index(drop=True)
skeleton['games_played_current_season'] = skeleton.groupby('player_id').cumcount()
skeleton['season'] = SEASON
skeleton['age'] = skeleton['age_2026']

print(f"  Skeleton: {len(skeleton)} rows, {skeleton['player_id'].nunique()} QBs")
print(f"  Teams: {skeleton['team'].nunique()}, weeks {skeleton['week'].min()}-{skeleton['week'].max()}")

# Duplicate QB1 diagnostic — each team should have exactly one QB1
_qb1_per_team = (
    skeleton[skeleton['depth_chart_rank'] == 1]
    .groupby('team')['player_display_name']
    .nunique()
)
_dup_teams = _qb1_per_team[_qb1_per_team > 1]
if len(_dup_teams) > 0:
    print(f"  WARNING: {len(_dup_teams)} team(s) have duplicate QB1 entries:")
    for _tm in _dup_teams.index:
        _names = (skeleton[(skeleton['team'] == _tm) & (skeleton['depth_chart_rank'] == 1)]
                  ['player_display_name'].unique())
        print(f"    {_tm}: {list(_names)}")
else:
    n_qb1 = skeleton[skeleton['depth_chart_rank'] == 1]['team'].nunique()
    print(f"  QB1 count: {n_qb1} (one per team — OK)")

# %%
# =============================================================================
# STEP 5 — QB rolling features (include last 2025 game via append trick)
# The shift(1) in rolling computation means the last 2025 game is NOT
# included in any existing master row. Appending a dummy 2026 row and
# recomputing ensures the last game is captured.
# =============================================================================
print("\nSTEP 5: Computing QB rolling features...")

QB_RAW_STATS = [
    # Passing yards model
    'passing_yards', 'yards_per_attempt', 'epa_per_dropback',
    'qb_air_yards_per_attempt', 'epa_per_opportunity', 'rushing_yards',
    'ngs_avg_time_to_throw', 'ngs_avg_intended_air_yards',
    'passing_tds', 'rushing_tds', 'interceptions', 'fumbles_lost_total',
    # Additional for passing TDs model
    'td_rate', 'int_rate', 'completion_pct', 'passing_epa',
    'ngs_completion_pct_above_exp',
    # Additional for rushing yards model
    'carries', 'rushing_epa', 'qb_scramble_rate', 'qb_pressure_rate',
]
available_raw = [s for s in QB_RAW_STATS if s in master.columns]

# Extract QB game history
qb_hist = (
    master[(master['position'] == 'QB') & (master['season'] >= 2006)]
    [['player_id', 'season', 'week'] + available_raw]
    .sort_values(['player_id', 'season', 'week'])
    .reset_index(drop=True)
    .copy()
)

# Dummy rows: one per QB, season=2026 week=1, all stats NaN
dummy = skeleton[['player_id']].drop_duplicates().copy()
dummy['season'] = SEASON
dummy['week']   = 1
for s in available_raw:
    dummy[s] = np.nan

qb_ext = pd.concat([qb_hist, dummy], ignore_index=True)
qb_ext = qb_ext.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

# --- L-window rolling (same formula as build_nfl_dataset.py Step 7) ---
ROLL_STATS = [
    # Passing yards model
    'passing_yards', 'rushing_yards', 'yards_per_attempt',
    'ngs_avg_time_to_throw', 'ngs_avg_intended_air_yards',
    # Passing TDs model extras
    'passing_tds', 'td_rate', 'passing_epa', 'int_rate',
    'ngs_completion_pct_above_exp', 'rushing_tds',
    # Rushing yards model extras
    'carries', 'rushing_epa', 'qb_pressure_rate',
]
ROLL_WINDOWS = [5, 10, 20]

for stat in ROLL_STATS:
    if stat not in qb_ext.columns:
        continue
    for w in ROLL_WINDOWS:
        qb_ext[f'{stat}_L{w}'] = (
            qb_ext.groupby('player_id')[stat]
            .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
        )

# --- EWM (same formula as QB.py Phase 3.1b) ---
EWM_STATS = {
    # Passing yards model
    'passing_yards':            [5, 10, 20],
    'yards_per_attempt':        [5, 10, 20],
    'epa_per_dropback':         [10, 20],
    'qb_air_yards_per_attempt': [10, 20],
    'epa_per_opportunity':      [5, 10, 20],
    # Passing TDs model extras
    'completion_pct':           [20],
    'int_rate':                 [5, 10, 20],
    # Rushing yards model extras (rushing_yards needs span 10 too)
    'rushing_yards':            [10, 20],
    'carries':                  [10, 20],
    'rushing_epa':              [20],
    'qb_scramble_rate':         [20],
    'qb_pressure_rate':         [20],
}
for stat, spans in EWM_STATS.items():
    if stat not in qb_ext.columns:
        continue
    for span in spans:
        qb_ext[f'{stat}_ewm{span}'] = (
            qb_ext.groupby('player_id')[stat]
            .transform(lambda x, s=span: x.ewm(span=s, min_periods=2).mean().shift(1))
        )

# --- Fantasy points EWM + career (same as QB.py Phase 3.1b) ---
_fpts = (
    0.04 * qb_ext['passing_yards'].fillna(0)
    + 4   * qb_ext['passing_tds'].fillna(0)
    + 0.1 * qb_ext['rushing_yards'].fillna(0)
    + 6   * qb_ext['rushing_tds'].fillna(0)
    - 2   * (qb_ext['interceptions'].fillna(0) + qb_ext['fumbles_lost_total'].fillna(0))
)
qb_ext['_fpts_raw'] = _fpts
for span in [10, 20]:
    qb_ext[f'fantasy_pts_ewm{span}'] = (
        qb_ext.groupby('player_id')['_fpts_raw']
        .transform(lambda x, s=span: x.ewm(span=s, min_periods=2).mean().shift(1))
    )
qb_ext['fantasy_pts_per_game_career'] = (
    qb_ext.groupby('player_id')['_fpts_raw']
    .transform(lambda x: x.expanding(min_periods=20).mean().shift(1))
)
qb_ext.drop(columns=['_fpts_raw'], inplace=True)

# Career per-game stats (same expanding mean pattern as QB.py Phase 3.1b)
_CAREER_STATS_PRED = [
    ('passing_yards',      'passing_yards_per_game_career'),
    ('passing_tds',        'passing_tds_per_game_career'),
    ('rushing_yards',      'rushing_yards_per_game_career'),
    ('rushing_tds',        'rushing_tds_per_game_career'),
    ('carries',            'carries_per_game_career'),
    ('rushing_epa',        'rushing_epa_per_game_career'),
    ('interceptions',      'interceptions_per_game_career'),
    ('fumbles_lost_total', 'fumbles_lost_per_game_career'),
]
for _raw, _col in _CAREER_STATS_PRED:
    if _raw in qb_ext.columns:
        qb_ext[_col] = (
            qb_ext.groupby('player_id')[_raw]
            .transform(lambda x: x.expanding(min_periods=20).mean().shift(1))
        )

# Regression-to-mean gap features (same as QB.py)
if 'rushing_yards_per_game_career' in qb_ext.columns and 'rushing_yards_ewm20' in qb_ext.columns:
    qb_ext['rushing_yards_career_vs_recent'] = (
        qb_ext['rushing_yards_per_game_career'] - qb_ext['rushing_yards_ewm20']
    )
if 'carries_per_game_career' in qb_ext.columns and 'carries_ewm20' in qb_ext.columns:
    qb_ext['carries_career_vs_recent'] = (
        qb_ext['carries_per_game_career'] - qb_ext['carries_ewm20']
    )
if 'rushing_epa_per_game_career' in qb_ext.columns and 'rushing_epa_ewm20' in qb_ext.columns:
    qb_ext['rushing_epa_career_vs_recent'] = (
        qb_ext['rushing_epa_per_game_career'] - qb_ext['rushing_epa_ewm20']
    )
if 'passing_yards_per_game_career' in qb_ext.columns and 'passing_yards_ewm20' in qb_ext.columns:
    qb_ext['passing_yards_career_vs_recent'] = (
        qb_ext['passing_yards_per_game_career'] - qb_ext['passing_yards_ewm20']
    )
if 'passing_tds_per_game_career' in qb_ext.columns and 'passing_tds_L20' in qb_ext.columns:
    qb_ext['passing_tds_career_vs_recent'] = (
        qb_ext['passing_tds_per_game_career'] - qb_ext['passing_tds_L20']
    )

# Extract the dummy 2026 rows — these are the frozen week-1 features
qb_week1 = qb_ext[qb_ext['season'] == SEASON].drop(
    columns=['season', 'week'] + available_raw, errors='ignore'
).copy()

# Broadcast to all 17 weeks in skeleton (pre-season: features are frozen at
# end-of-2025 until real 2026 game data arrives)
skeleton = skeleton.merge(qb_week1, on='player_id', how='left')
print(f"  QB rolling features joined.")
print(f"  passing_yards_L5 NaN rate: {skeleton['passing_yards_L5'].isna().mean()*100:.1f}%")

# %%
# =============================================================================
# STEP 6 — Team offense features (frozen at end of 2025)
# =============================================================================
print("\nSTEP 6: Joining team offense features...")

OFF_COLS = [c for c in master.columns
            if c.startswith('off_') and
            any(c.endswith(f'_L{w}') for w in [5, 10, 20])]

team_off = (
    master[master['season'] == 2025]
    .sort_values('week')
    .groupby('team')
    .last()
    [OFF_COLS]
    .reset_index()
)

skeleton = skeleton.merge(team_off, on='team', how='left')
print(f"  Joined {len(OFF_COLS)} off_ columns.")

# %%
# =============================================================================
# STEP 7 — Opponent defense features (frozen at end of 2025)
# Each opp_def_* column in master represents the defending team's rolling
# allowed stats. Group by 'opponent' (defteam) to get their latest profile.
# =============================================================================
print("\nSTEP 7: Joining opponent defense features...")

OPP_DEF_COLS = [c for c in master.columns
                if c.startswith('opp_def_') and
                any(c.endswith(f'_L{w}') for w in [5, 10, 20])]

opp_def = (
    master[master['season'] == 2025]
    .sort_values('week')
    .groupby('opponent')
    .last()
    [OPP_DEF_COLS]
    .reset_index()
    .rename(columns={'opponent': 'opponent'})
)

skeleton = skeleton.merge(opp_def, on='opponent', how='left')
print(f"  Joined {len(OPP_DEF_COLS)} opp_def_ columns.")

# %%
# =============================================================================
# STEP 8 — Weather (historical stadium-month averages from master dataset)
# Outdoor: average temp/wind/precip by home_team + month from master history.
# Dome: fixed values (72F temp, 0 wind, 0 precip).
# =============================================================================
print("\nSTEP 8: Computing weather features...")

# Build stadium-month averages from ALL outdoor games in master.
# Derive actual home team from game_location string ('home'/'away').
_wx = master[master['game_temp'].notna() & (master['is_dome'] == 0)].copy()
_wx['_actual_home'] = np.where(
    _wx['game_location'] == 'home', _wx['team'], _wx['opponent']
)

stadium_wx = (
    _wx.groupby(['_actual_home', '_game_month'])
    [['game_temp', 'game_wind', 'game_precip_mm']]
    .mean()
    .reset_index()
    .rename(columns={'_actual_home': '_home_team',
                     '_game_month': 'game_month'})
)

# Identify home team for each skeleton game (skeleton uses 1/−1 integers)
skeleton['_home_team'] = np.where(
    skeleton['game_location'] == 1, skeleton['team'], skeleton['opponent']
)

skeleton = skeleton.merge(stadium_wx, on=['_home_team', 'game_month'], how='left')

# Dome games: controlled environment — 72F, no wind, no precipitation
skeleton.loc[skeleton['is_dome'] == 1, 'game_temp']      = 72.0
skeleton.loc[skeleton['is_dome'] == 1, 'game_wind']      = 0.0
skeleton.loc[skeleton['is_dome'] == 1, 'game_precip_mm'] = 0.0

skeleton.drop(columns=['_home_team'], inplace=True, errors='ignore')

non_dome = skeleton['is_dome'] == 0
print(f"  Stadium-month lookup: {len(stadium_wx)} team-month combos from {len(_wx):,} outdoor games")
print(f"  game_temp NaN rate (outdoor games): "
      f"{skeleton.loc[non_dome, 'game_temp'].isna().mean()*100:.1f}%")
print(f"  Avg outdoor temp: {skeleton.loc[non_dome, 'game_temp'].mean():.1f}F  "
      f"wind: {skeleton.loc[non_dome, 'game_wind'].mean():.1f}mph")

# %%
# =============================================================================
# STEP 8.5 — Injury risk: projected games missed per QB1 (2026)
# Source: DraftSharks Injury Predictor (https://www.draftsharks.com/injury-predictor/qb)
# DraftSharks requires a paid subscription; the table below must be filled manually
# OR provide your session cookie in DS_COOKIE to enable scraping.
#
# Logic:
#   QB1 adj season yards = per_game_avg * (17 - games_missed_QB1)
#   QB2 adj season yards = per_game_avg * games_missed_QB1
#   (QB2 only gets games when QB1 is out; if no QB2 in dataset, QB1 still gets reduced)
# =============================================================================
print("\nSTEP 8.5: Loading injury risk projections...")

# --- DraftSharks Injury Predictor data (2026 season) ---
# Source: https://www.draftsharks.com/injury-predictor/qb
# Keyed by team abbreviation (our system): value = QB1's projected games missed.
# QB2 on the same team automatically inherits those games (see STEP 11 logic).
# DraftSharks uses LAR/LVR; we use LA/LV — already translated below.
QB1_GAMES_MISSED = {
    'JAX': 2.9,   # Trevor Lawrence      Low Risk
    'NYG': 2.5,   # Jaxon Dart           Low Risk
    'MIN': 2.3,   # J.J. McCarthy        Medium Risk
    'PHI': 2.2,   # Jalen Hurts          Medium Risk
    'WAS': 2.2,   # Jayden Daniels       Very High Risk
    'BAL': 2.1,   # Lamar Jackson        Medium Risk
    'NYJ': 2.1,   # Geno Smith           High Risk
    'NE':  2.0,   # Drake Maye           High Risk
    'HOU': 1.6,   # C.J. Stroud          Low Risk
    'LA':  1.5,   # Matthew Stafford     Low Risk
    'GB':  1.5,   # Jordan Love          Medium Risk
    'DET': 1.4,   # Jared Goff           Low Risk
    'PIT': 1.3,   # Aaron Rodgers        Low Risk
    'IND': 1.2,   # Daniel Jones         Low Risk
    'LAC': 1.1,   # Justin Herbert       Low Risk
    'DAL': 1.0,   # Dak Prescott         Low Risk
    'TEN': 0.9,   # Cam Ward             Low Risk
    'ATL': 0.9,   # Tua Tagovailoa       Low Risk
    'CLE': 0.9,   # Shedeur Sanders      Very Low Risk
    'LV':  0.9,   # Kirk Cousins         Very Low Risk
    'SF':  0.7,   # Brock Purdy          Low Risk
    'BUF': 0.6,   # Josh Allen           Very Low Risk
    'CHI': 0.6,   # Caleb Williams       Low Risk
    'CIN': 0.5,   # Joe Burrow           Low Risk
    'TB':  0.4,   # Baker Mayfield       Low Risk
    'SEA': 0.4,   # Sam Darnold          Low Risk
    'CAR': 0.4,   # Bryce Young          Very Low Risk
    'KC':  0.3,   # Patrick Mahomes      Very Low Risk
    'MIA': 0.2,   # Malik Willis         (no risk listed)
    'ARI': 0.2,   # Jacoby Brissett      Very Low Risk
    'DEN': 0.1,   # Bo Nix               Very Low Risk
    'NO':  0.1,   # Taylor Shough        (no risk listed)
}

# --- Optional: scrape DraftSharks with your session cookie ---
# Set DS_COOKIE to your logged-in 'remember_web_*' cookie value, or leave as ''
DS_COOKIE = ''

if DS_COOKIE:
    print("  Attempting DraftSharks scrape with session cookie...")
    try:
        from bs4 import BeautifulSoup
        _headers = {
            'User-Agent': 'Mozilla/5.0',
            'Cookie': DS_COOKIE,
        }
        _resp = requests.get(
            'https://www.draftsharks.com/injury-predictor/qb',
            headers=_headers, timeout=15
        )
        _soup = BeautifulSoup(_resp.text, 'html.parser')
        _rows = _soup.select('table tbody tr')
        _scraped = {}
        for _row in _rows:
            _cells = [td.get_text(strip=True) for td in _row.find_all('td')]
            if len(_cells) >= 3:
                _name  = _cells[0]
                _missed = next((c for c in _cells[1:] if c.replace('.','',1).isdigit()), None)
                if _missed:
                    _scraped[_name] = float(_missed)
        if _scraped:
            INJURY_GAMES_MISSED.update(_scraped)
            print(f"  DraftSharks scraped: {len(_scraped)} QBs updated")
        else:
            print("  DraftSharks scrape returned no rows (check cookie or page structure)")
    except Exception as _e:
        print(f"  DraftSharks scrape failed ({_e}) — using manual dict")
else:
    print("  DS_COOKIE not set — using manual INJURY_GAMES_MISSED dict")

print(f"  Injury entries loaded: {len(QB1_GAMES_MISSED)} teams")

# %%
# =============================================================================
# STEP 9 — Assemble feature matrix
# =============================================================================
print("\nSTEP 9: Assembling feature matrix...")

# Any feature_cols not yet in skeleton: fill NaN (LightGBM handles missing)
missing_feats = [f for f in feature_cols if f not in skeleton.columns]
if missing_feats:
    print(f"  {len(missing_feats)} features not found, will be NaN: {missing_feats}")
    for f in missing_feats:
        skeleton[f] = np.nan

X_2026 = skeleton[feature_cols].copy()
overall_nan = X_2026.isna().mean().mean() * 100
print(f"  Feature matrix: {X_2026.shape}  |  Overall NaN rate: {overall_nan:.1f}%")

# --- NA audit: separate systematic (rookies) from non-systematic ---
# Rookies = QBs with no rolling history (passing_yards_L5 is NaN for all their rows)
_has_history = X_2026['passing_yards_L5'].notna() if 'passing_yards_L5' in X_2026.columns else pd.Series(True, index=X_2026.index)
_experienced = X_2026[_has_history]
_rookies     = X_2026[~_has_history]

print(f"\n  Experienced QBs ({len(_experienced)} rows) — non-systematic NaN rates:")
_bad_feats = []
for f in feature_cols:
    r = _experienced[f].isna().mean() * 100
    if r > 5.0:   # flag anything above 5%
        _bad_feats.append((f, r))
if _bad_feats:
    for f, r in sorted(_bad_feats, key=lambda x: -x[1]):
        print(f"    {f}: {r:.1f}%  <-- NON-SYSTEMATIC, needs fix")
else:
    print("    All features < 5% NaN for experienced QBs -- OK")

if len(_rookies) > 0:
    _rookie_qbs = skeleton.loc[~_has_history, 'player_display_name'].unique()
    print(f"\n  Systematic NaN QBs (no 2025 history): {list(_rookie_qbs)}")

# %%
# =============================================================================
# STEP 10 — Predict passing yards + CIs
# =============================================================================
print("\nSTEP 10: Predicting passing yards...")

preds = np.clip(model_py.predict(X_2026), 0, None)
skeleton['pred_passing_yards'] = preds

# Game-level CI: pred +/- z * RMSE_game
skeleton['ci_lower_game'] = np.clip(preds - CI_Z * RMSE_GAME, 0, None)
skeleton['ci_upper_game'] = preds + CI_Z * RMSE_GAME

print(f"  Prediction range: {preds.min():.1f} - {preds.max():.1f} yards/game")
print(f"  Mean: {preds.mean():.1f}  |  Std: {preds.std():.1f}")

# %%
# =============================================================================
# STEP 10b — Predict passing TDs + CIs
# =============================================================================
print("\nSTEP 10b: Predicting passing TDs...")

missing_td_feats = [f for f in TD_FEATURE_COLS if f not in skeleton.columns]
if missing_td_feats:
    print(f"  {len(missing_td_feats)} TD features not found, will be NaN: {missing_td_feats}")
    for f in missing_td_feats:
        skeleton[f] = np.nan

X_td = skeleton[TD_FEATURE_COLS].copy()
td_preds = np.clip(model_td.predict(X_td), 0, None)
skeleton['pred_passing_tds'] = td_preds
skeleton['ci_lower_td_game'] = np.clip(td_preds - CI_Z * RMSE_GAME_TD, 0, None)
skeleton['ci_upper_td_game'] = td_preds + CI_Z * RMSE_GAME_TD

print(f"  TD range: {td_preds.min():.2f} - {td_preds.max():.2f}/game")
print(f"  Mean: {td_preds.mean():.2f}  |  Std: {td_preds.std():.2f}")

# TD NA audit (non-systematic only)
_experienced_td = X_td[X_td['passing_yards_L20'].notna()] if 'passing_yards_L20' in X_td.columns else X_td
_bad_td = [(f, _experienced_td[f].isna().mean()*100) for f in TD_FEATURE_COLS
           if _experienced_td[f].isna().mean() > 0.05]
if _bad_td:
    print("  Non-systematic NaN in TD features:")
    for f, r in sorted(_bad_td, key=lambda x: -x[1]):
        print(f"    {f}: {r:.1f}%")

# %%
# =============================================================================
# STEP 10c — Predict rushing yards + CIs
# =============================================================================
print("\nSTEP 10c: Predicting rushing yards...")

missing_ry_feats = [f for f in RY_FEATURE_COLS if f not in skeleton.columns]
if missing_ry_feats:
    print(f"  {len(missing_ry_feats)} RY features not found, will be NaN: {missing_ry_feats}")
    for f in missing_ry_feats:
        skeleton[f] = np.nan

X_ry = skeleton[RY_FEATURE_COLS].copy()
ry_preds = np.clip(model_ry.predict(X_ry), 0, None)
skeleton['pred_rushing_yards'] = ry_preds
skeleton['ci_lower_ry_game']   = np.clip(ry_preds - CI_Z * RMSE_GAME_RY, 0, None)
skeleton['ci_upper_ry_game']   = ry_preds + CI_Z * RMSE_GAME_RY

print(f"  RY range: {ry_preds.min():.1f} - {ry_preds.max():.1f} yards/game")
print(f"  Mean: {ry_preds.mean():.1f}  |  Std: {ry_preds.std():.1f}")

_experienced_ry = X_ry[X_ry['rushing_yards_L20'].notna()] if 'rushing_yards_L20' in X_ry.columns else X_ry
_bad_ry = [(f, _experienced_ry[f].isna().mean()*100) for f in RY_FEATURE_COLS
           if _experienced_ry[f].isna().mean() > 0.05]
if _bad_ry:
    print("  Non-systematic NaN in RY features:")
    for f, r in sorted(_bad_ry, key=lambda x: -x[1]):
        print(f"    {f}: {r:.1f}%")

# %%
# =============================================================================
# STEP 11 — Season totals + CI propagation
# Season variance = sum of per-game variances (independence assumption).
# CI_season = z * sqrt(n_games) * RMSE_game
# =============================================================================
print("\nSTEP 11: Computing season totals + injury adjustments...")

season_totals = (
    skeleton
    .groupby(['player_id', 'player_display_name', 'team', 'depth_chart_rank'])
    .agg(games=('pred_passing_yards', 'count'),
         pred_season_yards=('pred_passing_yards', 'sum'),
         pred_season_tds=('pred_passing_tds', 'sum'),
         pred_season_rush_yards=('pred_rushing_yards', 'sum'))
    .reset_index()
)

# --- Injury adjustment ---
# QB1_GAMES_MISSED keyed by team: QB1 plays (17 - missed), QB2 plays missed games.
def _injury_scale(row):
    missed = QB1_GAMES_MISSED.get(row['team'], 0.0)
    if row['depth_chart_rank'] == 1:
        return (17.0 - missed) / 17.0
    elif row['depth_chart_rank'] == 2:
        return missed / 17.0
    return 1.0

_scale = season_totals.apply(_injury_scale, axis=1)
season_totals['inj_games_missed_qb1'] = season_totals['team'].map(QB1_GAMES_MISSED).fillna(0.0)
season_totals['pred_season_yards_adj']      = season_totals['pred_season_yards']      * _scale
season_totals['pred_season_tds_adj']        = season_totals['pred_season_tds']        * _scale
season_totals['pred_season_rush_yards_adj'] = season_totals['pred_season_rush_yards'] * _scale

season_totals = season_totals.sort_values('pred_season_yards_adj', ascending=False).reset_index(drop=True)

# CIs use adjusted games played
_adj_games = np.where(
    season_totals['depth_chart_rank'] == 1,
    17.0 - season_totals['inj_games_missed_qb1'],
    season_totals['inj_games_missed_qb1'],
).clip(0.5)   # floor to avoid sqrt(0)

season_margin = CI_Z * np.sqrt(_adj_games) * RMSE_GAME
season_totals['ci_lower_season'] = np.clip(
    season_totals['pred_season_yards_adj'] - season_margin, 0, None
)
season_totals['ci_upper_season'] = season_totals['pred_season_yards_adj'] + season_margin

season_margin_td = CI_Z * np.sqrt(_adj_games) * RMSE_GAME_TD
season_totals['ci_lower_season_td'] = np.clip(
    season_totals['pred_season_tds_adj'] - season_margin_td, 0, None
)
season_totals['ci_upper_season_td'] = season_totals['pred_season_tds_adj'] + season_margin_td

season_margin_ry = CI_Z * np.sqrt(_adj_games) * RMSE_GAME_RY
season_totals['ci_lower_season_ry'] = np.clip(
    season_totals['pred_season_rush_yards_adj'] - season_margin_ry, 0, None
)
season_totals['ci_upper_season_ry'] = season_totals['pred_season_rush_yards_adj'] + season_margin_ry

print(f"\n  Top 15 QBs by injury-adjusted projected yards (2026 season):")
print(f"  {'QB':<26} {'Tm':>4} {'Rk':>3} {'Mis':>4} {'AdjYds':>7} {'AdjTDs':>7} {'AdjRuYds':>9}")
print(f"  {'-'*68}")
for _, r in season_totals.head(15).iterrows():
    print(f"  {str(r['player_display_name']):<26} {str(r['team']):>4} "
          f"{int(r['depth_chart_rank']):>3} {r['inj_games_missed_qb1']:>4.1f} "
          f"{r['pred_season_yards_adj']:>7.0f} {r['pred_season_tds_adj']:>7.1f} "
          f"{r['pred_season_rush_yards_adj']:>9.0f}")

# %%
# =============================================================================
# STEP 12 — Output to Excel
# =============================================================================
print(f"\nSTEP 12: Writing {OUTPUT_PATH}...")

game_level = (
    skeleton[[
        'player_display_name', 'team', 'depth_chart_rank', 'week', 'opponent',
        'game_location', 'is_dome', 'rest_days', 'game_temp', 'game_wind',
        'game_precip_mm',
        'pred_passing_yards', 'ci_lower_game',    'ci_upper_game',
        'pred_passing_tds',   'ci_lower_td_game', 'ci_upper_td_game',
        'pred_rushing_yards', 'ci_lower_ry_game', 'ci_upper_ry_game',
    ]]
    .sort_values(['player_display_name', 'week'])
    .rename(columns={
        'player_display_name': 'QB',
        'depth_chart_rank':    'depth_rank',
        'game_location':       'location',
        'pred_passing_yards':  'pred_pass_yards',
        'ci_lower_game':       'ci_lower_pass_yards',
        'ci_upper_game':       'ci_upper_pass_yards',
        'pred_passing_tds':    'pred_pass_tds',
        'ci_lower_td_game':    'ci_lower_pass_tds',
        'ci_upper_td_game':    'ci_upper_pass_tds',
        'pred_rushing_yards':  'pred_rush_yards',
        'ci_lower_ry_game':    'ci_lower_rush_yards',
        'ci_upper_ry_game':    'ci_upper_rush_yards',
    })
)

season_out = season_totals.rename(columns={
    'player_display_name':          'QB',
    'depth_chart_rank':             'depth_rank',
    'inj_games_missed_qb1':         'inj_games_missed',
    'pred_season_yards':            'pred_pass_yards_raw',
    'pred_season_yards_adj':        'pred_pass_yards',
    'ci_lower_season':              'ci_lower_pass_yards',
    'ci_upper_season':              'ci_upper_pass_yards',
    'pred_season_tds':              'pred_pass_tds_raw',
    'pred_season_tds_adj':          'pred_pass_tds',
    'ci_lower_season_td':           'ci_lower_pass_tds',
    'ci_upper_season_td':           'ci_upper_pass_tds',
    'pred_season_rush_yards':       'pred_rush_yards_raw',
    'pred_season_rush_yards_adj':   'pred_rush_yards',
    'ci_lower_season_ry':           'ci_lower_rush_yards',
    'ci_upper_season_ry':           'ci_upper_rush_yards',
})

with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
    game_level.to_excel(writer, sheet_name='game_level', index=False)
    season_out.to_excel(writer, sheet_name='season_totals', index=False)

print(f"  Saved: {OUTPUT_PATH}")
print(f"\nDone. {len(skeleton)} game predictions for "
      f"{skeleton['player_display_name'].nunique()} QBs "
      f"across {skeleton['week'].nunique()} weeks.")
