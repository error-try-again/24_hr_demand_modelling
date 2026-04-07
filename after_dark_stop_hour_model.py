#!/usr/bin/env python3
"""
After Dark, On Tap — Stop-Hour Footfall + Conversion Model v11
===============================================================

Full 24-hour, multi-channel demand model.

  - Full 24-hour trading with per-segment conversion profiles.
  - Stop-level boardings/alightings from ACT Open Data.
  - Weather, events, operations, callouts optional inputs.
  - Free responder cup economics.
  - Ambient pedestrian channel now uses absolute daily estimate (aadt_ped_daily)
    instead of a ratio multiplied against LR passersby. Eliminates the semantic
    mismatch where AADT-calibrated values and hardcoded defaults meant different
    things at different scales.
  - Bus passenger channel added. Calibration already produced bus_daily_frontage
    and bus_conv_mean/sd but the simulation never consumed them. Now it does.
  - Validation check: modelled LR passersby (after board_weight × alight_weight ×
    frontage_share) are compared against calibration lr_daily. Warns if they
    diverge by >2×, indicating the weight/frontage parameters are wrong.
  - Weekend ambient factor now consumed from calibration (aadt_weekend_factor)
    instead of hardcoded 0.70.


Intended ACT data inputs (export these as CSVs from the official sources):
  - Boardings By Stop By Quarter Hr (dataset id 7yh9-wwyp)
  - Alightings By Stop By Quarter Hr (dataset id pwii-q63j)
  - Light Rail Stops (dataset id 28a2-f2xq) [optional; not required by the core model]
  - Canberra weather observations (BoM) [optional]
  - Events schedule / precinct calendar [optional]
  - Operations / delay summary [optional]

Usage:
    python3 after_dark_stop_hour_model.py \
        --boardings boardings_by_stop_qh.csv \
        --alightings alightings_by_stop_qh.csv \
        --weather weather.csv \
        --events events.csv \
        --ops ops.csv \
        --outdir ./output

Notes:
  - The ACT stop-level datasets may be exported in different shapes depending on the portal
    view used. This script auto-detects a common long format and a common wide format.
  - If no optional inputs are provided, the model still runs using stop-level boardings /
    alightings plus priors for venue, event, and walk-in demand.
  - This is still a pre-pilot model. Once POS + frontage counts exist, replace the fixed
    conversion priors with fitted parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import guard
    sys.exit(f"matplotlib required: {exc}")

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python <3.9 fallback unlikely here
    ZoneInfo = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_SIM = 100_000
BREAKEVEN = 49
RECENT_FROM = "2023-01-01"
LOCAL_TZ = "Australia/Sydney"

# Free coffee economics: responder cups cost materials but generate no revenue.
# COGS_PER_CUP is the ingredient/consumable cost per cup.
# REVENUE_PER_CUP is the average revenue per paid cup.
# The effective break-even is: BREAKEVEN + ceil(free_cups * COGS_PER_CUP / REVENUE_PER_CUP)
COGS_PER_CUP = 1.20   # AUD: beans, milk, cup, lid, stirrer
REVENUE_PER_CUP = 4.50  # AUD: average paid cup revenue

C = {
    "blue": "#1B3A5C",
    "gold": "#C8A558",
    "teal": "#2A7B88",
    "red": "#C45B4A",
    "green": "#5B8C5A",
    "grey": "#E0DDD5",
    "charcoal": "#333333",
}


def beta_params_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
    """Return alpha/beta parameters for a Beta distribution.

    Falls back to a mildly informative prior if the provided mean/sd pair is invalid.
    """
    mean = float(np.clip(mean, 1e-4, 1 - 1e-4))
    sd = float(max(sd, 1e-4))
    var = sd ** 2
    limit = mean * (1 - mean)
    if var >= limit:
        concentration = 12.0
        return mean * concentration, (1 - mean) * concentration
    k = limit / var - 1
    alpha = mean * k
    beta = (1 - mean) * k
    if alpha <= 0 or beta <= 0:
        concentration = 12.0
        return mean * concentration, (1 - mean) * concentration
    return alpha, beta


@dataclass(frozen=True)
class Scenario:
    day_type: str = "all"      # all | weekday | weekend
    weather: str = "all"       # all | wet | dry
    ops: str = "all"           # all | normal | disrupted
    season: str = "all"        # all | cold | mild | warm | hot


# ---------------------------------------------------------------------------
# Time-segment definitions
# ---------------------------------------------------------------------------
# Each segment has its own conversion prior, reflecting different customer
# intent: morning commuters actively want coffee, daytime is mixed, evening
# is destination/social, and late-night is very low-volume but the machine
# is still running.

TIME_SEGMENTS = {
    "morning":  {"hours": list(range(6, 10)),  "label": "Morning commute 06–10",
                 "ambient_share": 0.25},  # commuter rush + office arrivals
    "daytime":  {"hours": list(range(10, 17)), "label": "Daytime 10–17",
                 "ambient_share": 0.50},  # lunch, shopping, office movement
    "evening":  {"hours": list(range(17, 23)), "label": "Evening 17–23",
                 "ambient_share": 0.22},  # evening economy, dining, departures
    "latenight": {"hours": list(range(23, 24)) + list(range(0, 6)), "label": "Late night 23–06",
                  "ambient_share": 0.03},  # near-zero foot traffic overnight
}

SITE_CONFIG: Dict[str, Dict[str, object]] = {
    "Alinga Street": {
        "stop_patterns": ["alinga"],
        "stop_is_lr": True,
        "trading_hours": list(range(0, 24)),  # 24/7
        "board_weight": 0.70,  # v12 fix 11: symmetric until site position confirmed
        "alight_weight": 0.70,
        "frontage_share": 0.62,
        # v12 fix 24: suburb/keyword matching for events scraped from events.canberra.com.au
        "event_keywords": ["canberra city", "civic", "braddon", "acton", "northbourne",
                           "london circuit", "city walk", "garema", "petrie plaza",
                           "glebe park", "commonwealth park"],
        # Time-segment conversion priors (mean, sd) — replaces single lr_conv
        "lr_conv_morning":  (0.085, 0.025),   # Commuters actively want coffee
        "lr_conv_daytime":  (0.045, 0.015),   # Mixed intent, lower conversion
        "lr_conv_evening":  (0.030, 0.010),   # Original evening rate
        "lr_conv_latenight": (0.012, 0.006),  # Very low volume, some shift workers
        # Ambient pedestrian multiplier: non-LR foot traffic as a ratio of LR
        # Alinga is CBD — substantial pedestrian flow from offices, bus interchange
        # v11: ambient_ped_multiplier retained as legacy fallback only.
        # Prefer aadt_ped_daily (absolute daily estimate from calibration).
        "ambient_ped_multiplier": 2.8,
        "ambient_ped_sd": 0.6,
        # v11: absolute ambient pedestrian estimate from AADT corridor analysis.
        # When > 0, the simulation uses this directly instead of the ratio.
        "aadt_ped_daily": 0,
        "aadt_ped_sd": 0,
        "aadt_weekend_factor": 0.47,
        # Ambient pedestrians have lower conversion (they're not captive at a stop)
        "ambient_conv_mean": 0.018,
        "ambient_conv_sd": 0.006,
        # v11: Bus frontage passengers — calibration produces these but v10 ignored them
        "bus_daily_frontage": 0,
        "bus_conv_mean": 0.022,
        "bus_conv_sd": 0.006,
        "bus_weekend_ratio": 0.28,  # v12 fix 15: from ACT data
        # Habitual/regular customers: residents, venue staff, shift workers
        # who come regardless of whether they're "passing by"
        "habitual_daily": 8,
        "habitual_daily_sd": 3,
        "habitual_items_mean": 1.3,
        "habitual_items_sd": 0.3,
        # Venue, events, walk-in (carried forward from v9)
        "venue": "Canberra City Uniting Church / Early Morning Centre",
        "venue_daily": 77,
        "venue_daily_sd": 23,
        "venue_conv_mean": 0.25,
        "venue_conv_sd": 0.07,
        # Walk-in: destination visitors NOT captured by any other channel.
        # v12 fix 10: reduced from 12 — pre-pilot walk-in to an unknown café is
        # speculative, and some would overlap with ambient/bus passersby who convert.
        "walk_daily": 6,
        "walk_daily_sd": 3,
        "walk_conv_mean": 0.60,
        "walk_conv_sd": 0.14,
        "fallback_event_attend": 3.0,
        "event_conv_mean": 0.50,
        "event_conv_sd": 0.15,
        # Multi-item factor: some customers buy 2+ items (coffee + snack)
        "multi_item_mean": 1.15,
        "multi_item_sd": 0.08,
    },
    "Dickson": {
        "stop_patterns": ["dickson"],
        # v12 fix: these stop patterns match bus stops (Cowper St, Antill St), not
        # the LR Dickson Interchange. Until lr_patronage_15min data is ingested,
        # Channel 1 should apply bus conversion rates, not LR rates.
        "stop_is_lr": False,
        "trading_hours": list(range(0, 24)),
        "board_weight": 0.70,
        "alight_weight": 0.70,
        "frontage_share": 0.58,
        "event_keywords": ["dickson", "lyneham", "downer", "watson", "hackett"],
        "lr_conv_morning":  (0.078, 0.022),
        "lr_conv_daytime":  (0.040, 0.014),
        "lr_conv_evening":  (0.026, 0.010),
        "lr_conv_latenight": (0.010, 0.005),
        # Dickson is a dining/shopping precinct with high ambient foot traffic
        "ambient_ped_multiplier": 3.5,
        "ambient_ped_sd": 0.8,
        "aadt_ped_daily": 0,
        "aadt_ped_sd": 0,
        "aadt_weekend_factor": 0.58,
        "ambient_conv_mean": 0.015,
        "ambient_conv_sd": 0.005,
        "bus_daily_frontage": 0,
        "bus_conv_mean": 0.022,
        "bus_conv_sd": 0.006,
        "bus_weekend_ratio": 0.23,  # v12 fix 15: from ACT data (Dickson bus stops)
        "habitual_daily": 10,
        "habitual_daily_sd": 4,
        "habitual_items_mean": 1.25,
        "habitual_items_sd": 0.25,
        "venue": "Northside Community Service",
        "venue_daily": 125,
        "venue_daily_sd": 40,
        "venue_conv_mean": 0.25,
        "venue_conv_sd": 0.07,
        "walk_daily": 5,
        "walk_daily_sd": 3,
        "walk_conv_mean": 0.62,
        "walk_conv_sd": 0.14,
        "fallback_event_attend": 3.0,
        "event_conv_mean": 0.50,
        "event_conv_sd": 0.15,
        "multi_item_mean": 1.12,
        "multi_item_sd": 0.07,
    },
    "Gungahlin Place": {
        "stop_patterns": ["gozzard st gungahlin", "temp gungahlin platform",
                          "gungahlin plt 1", "gungahlin plt 2"],
        "stop_is_lr": True,
        "trading_hours": list(range(0, 24)),
        "board_weight": 0.70,
        "alight_weight": 0.70,
        "frontage_share": 0.54,
        "event_keywords": ["gungahlin", "harrison", "franklin", "mitchell",
                           "throsby", "casey", "ngunnawal", "amaroo"],
        "lr_conv_morning":  (0.072, 0.020),
        "lr_conv_daytime":  (0.035, 0.012),
        "lr_conv_evening":  (0.023, 0.010),
        "lr_conv_latenight": (0.008, 0.004),
        # Gungahlin is a town centre with major shopping centre adjacent
        "ambient_ped_multiplier": 4.2,
        "ambient_ped_sd": 1.0,
        "aadt_ped_daily": 0,
        "aadt_ped_sd": 0,
        "aadt_weekend_factor": 0.58,
        "ambient_conv_mean": 0.012,
        "ambient_conv_sd": 0.004,
        "bus_daily_frontage": 0,
        "bus_conv_mean": 0.022,
        "bus_conv_sd": 0.006,
        "bus_weekend_ratio": 0.25,  # v12 fix 15: system average
        "habitual_daily": 7,
        "habitual_daily_sd": 3,
        "habitual_items_mean": 1.20,
        "habitual_items_sd": 0.25,
        "venue": "Gungahlin Uniting Church",
        "venue_daily": 60,
        "venue_daily_sd": 25,
        "venue_conv_mean": 0.25,
        "venue_conv_sd": 0.07,
        "walk_daily": 5,
        "walk_daily_sd": 3,
        "walk_conv_mean": 0.60,
        "walk_conv_sd": 0.14,
        "fallback_event_attend": 3.0,
        "event_conv_mean": 0.50,
        "event_conv_sd": 0.15,
        "multi_item_mean": 1.10,
        "multi_item_sd": 0.06,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(C["grey"])
    ax.tick_params(colors="#888")
    ax.grid(axis="y", alpha=0.15)


def _norm_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = {_norm_text(c): c for c in columns}
    for cand in candidates:
        key = _norm_text(cand)
        if key in cols:
            return cols[key]
    return None


TIME_RANGE_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})$")
DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}")
WIDE_TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}$")


def parse_interval_start(value: object) -> Optional[pd.Timestamp]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    m = TIME_RANGE_RE.match(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(year=2000, month=1, day=1, hour=hh, minute=mm)
    if DATETIME_RE.search(s):
        try:
            return pd.to_datetime(s)
        except Exception:
            return None
    try:
        ts = pd.to_datetime(s, errors="raise")
        return ts
    except Exception:
        return None


DATE_CANDIDATES = [
    "date",
    "service date",
    "service_date",
    "day",
    "trip date",
]

STOP_CANDIDATES = [
    "stop",
    "stop name",
    "stop_name",
    "station",
    "station name",
    "station_name",
    "light rail stop",
    "light_rail_stop",
    "name",
]

TIME_CANDIDATES = [
    "quarter hour",
    "quarter_hour",
    "time",
    "time period",
    "time_period",
    "interval",
    "interval start",
    "interval_start",
    "start time",
    "start_time",
    "timestamp",
    "datetime",
    "date time",
    "date_time",
    "hour",
]

COUNT_CANDIDATES = [
    "boardings",
    "alightings",
    "count",
    "value",
    "passengers",
    "patronage",
    "total",
]





WEATHER_TS_CANDIDATES = [
    "datetime",
    "timestamp",
    "ts",
    "time",
    "observation_time",
    "observation time",
]
WEATHER_TEMP_CANDIDATES = [
    "air_temp",
    "temp",
    "temperature",
    "temp_c",
    "air temperature",
    "air temp",
]
WEATHER_RAIN_CANDIDATES = [
    "rainfall",
    "rain_mm",
    "precip",
    "precip_mm",
    "rain mm",
]
WEATHER_WIND_CANDIDATES = [
    "wind_spd_kmh",
    "wind_kmh",
    "wind_speed",
    "wind speed",
]
def read_csv_any(path: str) -> pd.DataFrame:
    """Read CSV with stable type handling for ACT wide matrix exports."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    """Return a likely date column if one exists."""
    for cand in DATE_CANDIDATES:
        col = _first_present(df.columns, [cand])
        if col is not None:
            return col
    # fallback heuristic
    for col in df.columns:
        cl = _norm_text(col)
        if "date" in cl or cl in {"day", "service day"}:
            return col
    return None


def parse_slot(value: object) -> Tuple[str, int, int]:
    """Parse either:
    - 'Monday, 07:15-07:29'
    - '07:15-07:29'
    - timestamp-like strings
    Returns: (dow_abbrev, hour, minute)
    """
    s = str(value).strip()
    if not s:
        return ("unk", 0, 0)

    # ACT wide format: "Monday, 07:15-07:29"
    m = re.match(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})$",
        s,
        flags=re.I,
    )
    if m:
        dow = m.group(1)[:3].lower()
        hour = int(m.group(2))
        minute = int(m.group(3))
        return (dow, hour, minute)

    # Plain interval format: "07:15-07:29"
    m = re.match(r"^(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})$", s)
    if m:
        return ("unk", int(m.group(1)), int(m.group(2)))

    # Datetime-like
    try:
        ts = pd.to_datetime(s, errors="raise")
        dow = ts.day_name()[:3].lower() if not pd.isna(ts) else "unk"
        return (dow, int(ts.hour), int(ts.minute))
    except Exception:
        return ("unk", 0, 0)


def load_stop_activity(path: str, metric: str = "boardings") -> pd.DataFrame:
    df = read_csv_any(path)

    stop_candidates = ["stop_name", "stop", "station", "light rail stop", "stop name"]
    stop_col = None
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in stop_candidates:
        if cand in norm:
            stop_col = norm[cand]
            break
    if stop_col is None:
        for c in df.columns:
            cl = c.lower()
            if "stop" in cl or "station" in cl:
                stop_col = c
                break
    if stop_col is None:
        raise ValueError(f"{path} is missing a recognised stop column.")

    date_col = infer_date_col(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna()].copy()

        qh_candidates = []
        for c in df.columns:
            if c == stop_col or c == date_col:
                continue
            if re.search(r"\b\d{1,2}:\d{2}\b", c) and "-" in c:
                qh_candidates.append(c)

        if qh_candidates:
            melted = df.melt(
                id_vars=[stop_col, date_col],
                value_vars=qh_candidates,
                var_name="slot",
                value_name=metric,
            )
            melted[metric] = pd.to_numeric(melted[metric], errors="coerce").fillna(0.0)
            slot_info = melted["slot"].apply(parse_slot)
            melted["dow"] = slot_info.apply(lambda x: x[0])
            melted["hour"] = slot_info.apply(lambda x: x[1])
            melted["slot_start_minute"] = slot_info.apply(lambda x: x[2])
            melted["date"] = melted[date_col]
            out = (
                melted.groupby([stop_col, "date", "dow", "hour"], as_index=False)[metric]
                .sum()
                .rename(columns={stop_col: "stop_name"})
            )
            return out

        val_col = None
        for cand in [metric, "value", "count", "boardings", "alightings", "patronage"]:
            if cand in norm:
                val_col = norm[cand]
                break
        if val_col is None:
            numeric_cols = [c for c in df.columns if c not in {stop_col, date_col}]
            if len(numeric_cols) == 1:
                val_col = numeric_cols[0]
        if val_col is None:
            raise ValueError(f"{path} has a date column but no recognised {metric} values.")

        df[val_col] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0)

        hour_col = None
        for cand in ["hour", "hr"]:
            if cand in norm:
                hour_col = norm[cand]
                break
        if hour_col is not None:
            df[hour_col] = pd.to_numeric(df[hour_col], errors="coerce")
        else:
            df["__hour"] = 0
            hour_col = "__hour"

        df["dow"] = df[date_col].dt.day_name().str.lower().str[:3]
        out = (
            df.groupby([stop_col, date_col, "dow", hour_col], as_index=False)[val_col]
            .sum()
            .rename(columns={stop_col: "stop_name", date_col: "date", hour_col: "hour", val_col: metric})
        )
        return out

    qh_cols = [
        c for c in df.columns
        if re.search(
            r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}$",
            c,
            flags=re.I,
        )
    ]
    if not qh_cols:
        raise ValueError(
            f"{path} is missing both a recognised date column and ACT quarter-hour weekday columns."
        )

    melted = df.melt(id_vars=[stop_col], value_vars=qh_cols, var_name="slot", value_name=metric)
    melted[metric] = pd.to_numeric(melted[metric], errors="coerce").fillna(0.0)
    slot_info = melted["slot"].apply(parse_slot)
    melted["dow"] = slot_info.apply(lambda x: x[0])
    melted["hour"] = slot_info.apply(lambda x: x[1])
    melted["slot_start_minute"] = slot_info.apply(lambda x: x[2])

    out = (
        melted.groupby([stop_col, "dow", "hour"], as_index=False)[metric]
        .sum()
        .rename(columns={stop_col: "stop_name"})
    )
    out["date"] = pd.NaT
    return out

def load_weather(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    ts_col = _first_present(df.columns, WEATHER_TS_CANDIDATES)
    if ts_col is None:
        return None

    temp_col = _first_present(df.columns, WEATHER_TEMP_CANDIDATES)
    rain_col = _first_present(df.columns, WEATHER_RAIN_CANDIDATES)
    wind_col = _first_present(df.columns, WEATHER_WIND_CANDIDATES)

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    if temp_col:
        out["temp_c"] = pd.to_numeric(df.loc[out.index, temp_col], errors="coerce")
    else:
        out["temp_c"] = np.nan
    if rain_col:
        out["rain_mm"] = pd.to_numeric(df.loc[out.index, rain_col], errors="coerce").fillna(0)
    else:
        out["rain_mm"] = 0.0
    if wind_col:
        out["wind_kmh"] = pd.to_numeric(df.loc[out.index, wind_col], errors="coerce")
    else:
        out["wind_kmh"] = np.nan

    out["date"] = out["ts"].dt.normalize()
    out["hour"] = out["ts"].dt.hour.astype(int)

    # BoM daily observations often arrive as one row per date stamped to a representative
    # hour (for example 18:00). In that case, expand each date across all hours so the
    # daily weather signal survives the merge into the hourly site panel.
    obs_per_date = out.groupby("date")["hour"].nunique()
    daily_like = (not obs_per_date.empty) and (obs_per_date.quantile(0.9) <= 2)

    if daily_like:
        daily = out.groupby("date", as_index=False).agg(
            temp_c=("temp_c", "mean"),
            rain_mm=("rain_mm", "sum"),
            wind_kmh=("wind_kmh", "mean"),
        )
        hours = pd.DataFrame({"hour": list(range(24))})
        daily["__key"] = 1
        hours["__key"] = 1
        expanded = daily.merge(hours, on="__key", how="outer").drop(columns="__key")
        expanded["wet_hour"] = expanded["rain_mm"] > 0.0
        expanded["weather_granularity"] = "daily_expanded"
        return expanded

    agg = out.groupby(["date", "hour"], as_index=False).agg(
        temp_c=("temp_c", "mean"),
        rain_mm=("rain_mm", "sum"),
        wind_kmh=("wind_kmh", "mean"),
    )
    agg["wet_hour"] = agg["rain_mm"] > 0.0
    agg["weather_granularity"] = "hourly"
    return agg


EVENT_DATE_CANDIDATES = ["date", "event date", "event_date"]
EVENT_START_CANDIDATES = ["start", "start datetime", "start_datetime", "start time", "start_time"]
EVENT_END_CANDIDATES = ["end", "end datetime", "end_datetime", "end time", "end_time"]
EVENT_SITE_CANDIDATES = ["site", "stop", "stop_name", "location", "venue"]
EVENT_ATTEND_CANDIDATES = ["attendees", "attendance", "expected attendees", "expected_attendees", "crowd", "value"]
EVENT_MULT_CANDIDATES = ["multiplier", "intensity", "lift"]


def load_events(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    site_col = _first_present(df.columns, EVENT_SITE_CANDIDATES)
    attend_col = _first_present(df.columns, EVENT_ATTEND_CANDIDATES)
    mult_col = _first_present(df.columns, EVENT_MULT_CANDIDATES)
    date_col = _first_present(df.columns, EVENT_DATE_CANDIDATES)
    start_col = _first_present(df.columns, EVENT_START_CANDIDATES)
    end_col = _first_present(df.columns, EVENT_END_CANDIDATES)

    if site_col is None or attend_col is None:
        return None

    out = pd.DataFrame()
    out["site_raw"] = df[site_col].astype(str)
    out["attendees"] = pd.to_numeric(df[attend_col], errors="coerce").fillna(0)
    out["multiplier"] = pd.to_numeric(df[mult_col], errors="coerce").fillna(1.0) if mult_col else 1.0

    if start_col:
        out["start_ts"] = pd.to_datetime(df[start_col], errors="coerce")
        out["date"] = out["start_ts"].dt.normalize()
        out["start_hour"] = out["start_ts"].dt.hour.fillna(0).astype(int)
    elif date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        out["start_hour"] = 18
    else:
        return None

    if end_col:
        out["end_ts"] = pd.to_datetime(df[end_col], errors="coerce")
        out["end_hour"] = out["end_ts"].dt.hour.fillna(out["start_hour"] + 2).astype(int)
    else:
        out["end_hour"] = out["start_hour"] + 2

    out["site_key"] = out["site_raw"].map(_norm_text)
    out = out.dropna(subset=["date"])
    return out[["date", "site_raw", "site_key", "start_hour", "end_hour", "attendees", "multiplier"]]


OPS_DATE_CANDIDATES = ["date", "service date", "service_date"]
OPS_HOUR_CANDIDATES = ["hour", "time", "time period", "time_period"]
OPS_RELIAB_CANDIDATES = ["service reliability", "service_reliability", "reliability", "on time ratio", "on_time_ratio"]
OPS_DELAY_CANDIDATES = ["avg delay min", "avg_delay_min", "delay", "delay_min", "average delay"]
OPS_DISRUPT_CANDIDATES = ["disrupted", "is_disrupted", "service disruption", "service_disruption"]


def load_ops(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    date_col = _first_present(df.columns, OPS_DATE_CANDIDATES)
    hour_col = _first_present(df.columns, OPS_HOUR_CANDIDATES)
    rel_col = _first_present(df.columns, OPS_RELIAB_CANDIDATES)
    delay_col = _first_present(df.columns, OPS_DELAY_CANDIDATES)
    dis_col = _first_present(df.columns, OPS_DISRUPT_CANDIDATES)

    if date_col is None or hour_col is None:
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["hour"] = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
    out["service_reliability"] = pd.to_numeric(df[rel_col], errors="coerce") if rel_col else np.nan
    out["avg_delay_min"] = pd.to_numeric(df[delay_col], errors="coerce") if delay_col else np.nan
    if dis_col:
        raw = df[dis_col]
        out["is_disrupted"] = raw.astype(str).str.lower().isin(["1", "true", "yes", "y"])
    else:
        out["is_disrupted"] = False
        if rel_col:
            out["is_disrupted"] = out["is_disrupted"] | (out["service_reliability"] < 0.90)
        if delay_col:
            out["is_disrupted"] = out["is_disrupted"] | (out["avg_delay_min"] >= 6)
    out = out.dropna(subset=["date"])
    return out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]]


CALLOUT_TS_CANDIDATES = ["start_datetime", "datetime", "timestamp", "time"]
CALLOUT_SITE_CANDIDATES = ["site", "stop", "location"]
CALLOUT_UPLIFT_CANDIDATES = ["responder_uplift_cups", "free_cups", "uplift_cups", "cups"]
CALLOUT_SERVICE_CANDIDATES = ["service_type", "service", "agency"]
CALLOUT_SEVERITY_CANDIDATES = ["severity", "priority"]


def load_callouts(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load ESA callouts CSV and aggregate to daily free cups per site.

    The builder's normalize-callouts command produces per-incident rows with a
    ``responder_uplift_cups`` column.  We aggregate to daily totals per site so
    the demand model can sample from an empirical distribution of daily free-cup
    load.
    """
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    ts_col = _first_present(df.columns, CALLOUT_TS_CANDIDATES)
    site_col = _first_present(df.columns, CALLOUT_SITE_CANDIDATES)
    uplift_col = _first_present(df.columns, CALLOUT_UPLIFT_CANDIDATES)
    service_col = _first_present(df.columns, CALLOUT_SERVICE_CANDIDATES)
    severity_col = _first_present(df.columns, CALLOUT_SEVERITY_CANDIDATES)

    if ts_col is None:
        return None

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    out["date"] = out["ts"].dt.normalize()
    out["hour"] = out["ts"].dt.hour.astype(int)

    if site_col:
        out["site"] = df[site_col].astype(str).str.strip()
    else:
        out["site"] = ""

    if uplift_col:
        out["free_cups"] = pd.to_numeric(df[uplift_col], errors="coerce").fillna(1.0)
    else:
        out["free_cups"] = 1.0

    if service_col:
        out["service_type"] = df[service_col].astype(str).str.lower().str.strip()
    else:
        out["service_type"] = "other"

    if severity_col:
        out["severity"] = df[severity_col].astype(str).str.lower().str.strip()
    else:
        out["severity"] = "low"

    out = out.dropna(subset=["date"]).copy()

    # Aggregate to daily free cups per site
    daily = out.groupby(["site", "date"], as_index=False).agg(
        free_cups_daily=("free_cups", "sum"),
        callout_count=("free_cups", "count"),
    )
    # Also keep per-service breakdown for reporting
    service_daily = out.groupby(["site", "date", "service_type"], as_index=False)["free_cups"].sum()

    return {"daily": daily, "incidents": out, "service_daily": service_daily}


# ---------------------------------------------------------------------------
# LR 15-minute patronage (v12 fix 18)
# ---------------------------------------------------------------------------

LR_15_STOP_MAP = {
    "Alinga Street": ["alinga"],
    "Dickson": ["dickson"],
    "Gungahlin Place": ["gungahlin"],
}

def load_lr_15min(path: Optional[str]) -> Optional[pd.DataFrame]:
    """Load LR patronage 15-min interval CSV (Socrata xvid-q4du).

    Returns a DataFrame with columns: date, hour, stop_name, boardings, alightings
    This provides actual LR stop-level volumes including Dickson Interchange,
    which is absent from the bus/LR stop activity dataset.
    """
    if not path:
        return None
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return None
    df.columns = [str(c).strip() for c in df.columns]

    # Detect common column patterns in ACT LR 15-min exports
    stop_col = _first_present(df.columns, ["stop", "stop_name", "station", "light_rail_stop", "stop name"])
    date_col = _first_present(df.columns, ["date", "service_date", "trip_date"])

    # Look for 15-min interval columns (e.g. _06_00, _06_15 or 06:00-06:14)
    interval_cols = [c for c in df.columns if re.search(r"_?\d{2}[_:]\d{2}", c)
                     and c != (stop_col or "") and c != (date_col or "")]

    if stop_col is None:
        # v12 fix 22: system-wide data (no per-stop breakdown).
        # Parse wide-format interval columns into hourly aggregates and return
        # with a synthetic "lr_system" stop key so downstream can use the
        # hourly distribution even without stop-level granularity.
        if not interval_cols:
            print("  lr-15min: no stop column and no interval columns — skipping")
            return None
        print(f"  lr-15min: no stop column — using system-wide data ({len(interval_cols)} interval cols)")
        id_vars = [date_col] if date_col else []
        melted = df.melt(id_vars=id_vars, value_vars=interval_cols,
                         var_name="interval", value_name="count")
        melted["count"] = pd.to_numeric(melted["count"], errors="coerce").fillna(0)
        hour_match = melted["interval"].str.extract(r"(\d{2})[_:](\d{2})")
        melted["hour"] = pd.to_numeric(hour_match[0], errors="coerce").fillna(0).astype(int)
        melted["stop_key"] = "lr_system"
        melted["stop_name"] = "LR System-wide"
        if date_col:
            melted["date"] = pd.to_datetime(melted[date_col], errors="coerce").dt.normalize()
        else:
            melted["date"] = pd.NaT
        out = melted.groupby(["stop_key", "date", "hour"], as_index=False)["count"].sum()
        out["stop_name"] = "LR System-wide"
        out = out.dropna(subset=["hour"])
        print(f"  lr-15min: loaded {len(out):,} system-wide rows")
        return out

    out = pd.DataFrame()
    out["stop_name"] = df[stop_col].astype(str).str.strip()
    out["stop_key"] = out["stop_name"].map(_norm_text)

    if date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    else:
        out["date"] = pd.NaT

    # Look for 15-min interval columns (already detected above)
    if interval_cols:
        # Melt wide format
        melted = df.melt(id_vars=[stop_col] + ([date_col] if date_col else []),
                         value_vars=interval_cols, var_name="interval", value_name="count")
        melted["count"] = pd.to_numeric(melted["count"], errors="coerce").fillna(0)
        # Extract hour from interval name
        hour_match = melted["interval"].str.extract(r"(\d{2})[_:](\d{2})")
        melted["hour"] = pd.to_numeric(hour_match[0], errors="coerce").fillna(0).astype(int)
        melted["stop_key"] = melted[stop_col].astype(str).str.strip().map(_norm_text)
        if date_col:
            melted["date"] = pd.to_datetime(melted[date_col], errors="coerce").dt.normalize()
        else:
            melted["date"] = pd.NaT
        out = melted.groupby(["stop_key", "date", "hour"], as_index=False)["count"].sum()
        out["stop_name"] = melted.groupby(["stop_key", "date", "hour"])[stop_col].first().values
    else:
        # Try long format with hour/count columns
        hour_col = _first_present(df.columns, ["hour", "interval_hour"])
        count_col = _first_present(df.columns, ["boardings", "alightings", "count", "patronage", "total"])
        if hour_col and count_col:
            out["hour"] = pd.to_numeric(df[hour_col], errors="coerce").fillna(0).astype(int)
            out["count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
        else:
            print("  lr-15min: cannot parse columns — skipping")
            return None

    out = out.dropna(subset=["hour"])
    print(f"  lr-15min: loaded {len(out):,} rows, {out['stop_key'].nunique()} stops")
    return out


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------



def _recent_dates_for_dow(dow: str, weeks: int = 4) -> List[pd.Timestamp]:
    day_map = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    dow_key = str(dow).strip().lower()[:3]
    if dow_key not in day_map:
        return []
    today = pd.Timestamp.today().normalize()
    dates: List[pd.Timestamp] = []
    cursor = today
    while len(dates) < weeks:
        if cursor.dayofweek == day_map[dow_key]:
            dates.append(cursor)
        cursor -= pd.Timedelta(days=1)
    return dates


def _normalize_activity_input(df: pd.DataFrame, default_metric: str) -> pd.DataFrame:
    work = df.copy()

    if "stop" not in work.columns:
        if "stop_name" in work.columns:
            work = work.rename(columns={"stop_name": "stop"})
        else:
            stop_col = _first_present(work.columns, ["stop", "stop_name", "station", "stop name"])
            if stop_col is None:
                raise ValueError("Activity input is missing a recognised stop column.")
            work = work.rename(columns={stop_col: "stop"})

    if "metric" not in work.columns:
        work["metric"] = default_metric

    if "count" not in work.columns:
        value_col = None
        for cand in [default_metric, "count", "value", "boardings", "alightings", "patronage"]:
            if cand in work.columns:
                value_col = cand
                break
        if value_col is None:
            raise ValueError(f"Activity input for {default_metric} is missing a recognised count column.")
        work["count"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    else:
        work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0.0)

    if "hour" not in work.columns:
        if "quarter_hour" in work.columns:
            starts = work["quarter_hour"].apply(parse_interval_start)
            work["hour"] = starts.apply(lambda x: int(x.hour) if x is not None else 0)
            work["quarter"] = starts.apply(lambda x: int(x.minute // 15) if x is not None else 0)
        else:
            work["hour"] = 0

    work["hour"] = pd.to_numeric(work["hour"], errors="coerce").fillna(0).astype(int)

    if "quarter" not in work.columns:
        if "slot_start_minute" in work.columns:
            work["quarter"] = (pd.to_numeric(work["slot_start_minute"], errors="coerce").fillna(0).astype(int) // 15)
        elif "quarter_hour" in work.columns:
            starts = work["quarter_hour"].apply(parse_interval_start)
            work["quarter"] = starts.apply(lambda x: int(x.minute // 15) if x is not None else 0)
        else:
            work["quarter"] = 0

    if "date" not in work.columns:
        work["date"] = pd.NaT
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    if work["date"].isna().all() and "dow" in work.columns:
        expanded_parts: List[pd.DataFrame] = []
        for dow, chunk in work.groupby(work["dow"].astype(str).str.lower().str[:3], dropna=False):
            dates = _recent_dates_for_dow(dow, weeks=4)
            if not dates:
                continue
            for dt in dates:
                tmp = chunk.copy()
                tmp["date"] = dt
                expanded_parts.append(tmp)
        if expanded_parts:
            work = pd.concat(expanded_parts, ignore_index=True)
    elif "dow" in work.columns:
        missing = work["date"].isna()
        if missing.any():
            expanded_parts: List[pd.DataFrame] = [work.loc[~missing].copy()]
            for dow, chunk in work.loc[missing].groupby(work.loc[missing, "dow"].astype(str).str.lower().str[:3], dropna=False):
                dates = _recent_dates_for_dow(dow, weeks=4)
                if not dates:
                    continue
                for dt in dates:
                    tmp = chunk.copy()
                    tmp["date"] = dt
                    expanded_parts.append(tmp)
            work = pd.concat(expanded_parts, ignore_index=True)

    work = work.dropna(subset=["date"]).copy()
    work["date"] = work["date"].dt.normalize()
    return work[["date", "stop", "hour", "quarter", "metric", "count"]]


def build_activity_panel(boardings: pd.DataFrame, alightings: pd.DataFrame) -> pd.DataFrame:
    b = _normalize_activity_input(boardings, "boardings")
    a = _normalize_activity_input(alightings, "alightings")
    df = pd.concat([b, a], ignore_index=True)
    df["stop_key"] = df["stop"].map(_norm_text)
    df = df.groupby(["date", "stop", "stop_key", "hour", "metric"], as_index=False)["count"].sum()
    return df


def _match_site_rows(stop_keys: pd.Series, patterns: Iterable[str]) -> pd.Series:
    out = pd.Series(False, index=stop_keys.index)
    for pattern in patterns:
        p = _norm_text(pattern)
        out = out | stop_keys.str.contains(p, regex=False)
    return out


def build_site_daily_panel(
    activity: pd.DataFrame,
    weather: Optional[pd.DataFrame],
    events: Optional[pd.DataFrame],
    ops: Optional[pd.DataFrame],
    callouts: Optional[Dict],
    recent_from: str,
    lr_15min: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    recent_ts = pd.Timestamp(recent_from)
    # v12 fix 19: when data has real dates, auto-advance recent_from to keep
    # at most 365 days of history. Avoids using stale pre-COVID data.
    data_max = activity["date"].max()
    if pd.notna(data_max):
        auto_from = data_max - pd.Timedelta(days=365)
        if auto_from > recent_ts:
            print(f"  Auto-tuned recent_from: {recent_ts.date()} → {auto_from.date()} (data max {data_max.date()})")
            recent_ts = auto_from
    activity = activity.loc[activity["date"] >= recent_ts].copy()
    if activity.empty:
        raise ValueError(f"No activity rows on or after {recent_from}")

    # v12 fix 18: when lr_15min data is available, inject LR stop volumes into
    # activity panel for sites where stop_is_lr=True but the bus/LR stop data
    # doesn't include the actual LR stop (e.g. Dickson Interchange).
    if lr_15min is not None and not lr_15min.empty:
        for site, cfg in SITE_CONFIG.items():
            if not bool(cfg.get("stop_is_lr", True)):
                # This site matches bus stops — check if lr_15min has LR data
                lr_site_pats = LR_15_STOP_MAP.get(site, [])
                if not lr_site_pats:
                    continue
                lr_mask = pd.Series(False, index=lr_15min.index)
                for p in lr_site_pats:
                    lr_mask |= lr_15min["stop_key"].str.contains(_norm_text(p), regex=False)
                lr_rows = lr_15min.loc[lr_mask].copy()
                if lr_rows.empty:
                    continue
                # Inject as activity rows — these are genuine LR stop volumes
                injected = pd.DataFrame({
                    "date": lr_rows["date"],
                    "stop": lr_rows.get("stop_name", "LR_15min_" + site),
                    "stop_key": lr_rows["stop_key"],
                    "hour": lr_rows["hour"].astype(int),
                    "metric": "boardings",  # count = total patronage
                    "count": lr_rows["count"],
                    "quarter": 0,
                })
                # Mark these as LR-sourced so the simulation can distinguish
                injected["_lr_15min"] = True
                activity = pd.concat([activity, injected], ignore_index=True)
                print(f"  lr-15min: injected {len(lr_rows):,} rows for {site}")

    site_rows: List[pd.DataFrame] = []
    matched_stops: Dict[str, List[str]] = {}

    for site, cfg in SITE_CONFIG.items():
        mask = _match_site_rows(activity["stop_key"], cfg["stop_patterns"])  # type: ignore[arg-type]
        site_df = activity.loc[mask].copy()
        if site_df.empty:
            matched_stops[site] = []
            continue
        matched_stops[site] = sorted(site_df["stop"].drop_duplicates().tolist())
        site_df["site"] = site
        site_df["trading"] = site_df["hour"].isin(cfg["trading_hours"])  # type: ignore[arg-type]
        site_df = site_df.loc[site_df["trading"]].copy()
        if site_df.empty:
            continue
        site_df["board_component"] = np.where(site_df["metric"].eq("boardings"), site_df["count"] * float(cfg["board_weight"]), 0.0)
        site_df["alight_component"] = np.where(site_df["metric"].eq("alightings"), site_df["count"] * float(cfg["alight_weight"]), 0.0)
        grouped = site_df.groupby(["site", "date", "hour"], as_index=False).agg(
            boardings=("count", lambda s: s[site_df.loc[s.index, "metric"].eq("boardings")].sum()),
            alightings=("count", lambda s: s[site_df.loc[s.index, "metric"].eq("alightings")].sum()),
            board_component=("board_component", "sum"),
            alight_component=("alight_component", "sum"),
        )
        grouped["frontage_share"] = float(cfg["frontage_share"])
        grouped["lr_passersby_hour"] = (grouped["board_component"] + grouped["alight_component"]) * grouped["frontage_share"]
        site_rows.append(grouped)

    if not site_rows:
        raise ValueError("No configured sites could be matched to the stop-level activity data.")

    hourly = pd.concat(site_rows, ignore_index=True)

    # v12 fix 20: ensure all 24 hours present per site-date. Hours with zero
    # activity (esp. late-night 23–05) would otherwise be missing, biasing
    # hours_open downward and making latenight segment empty.
    full_idx = pd.MultiIndex.from_product(
        [hourly["site"].unique(), hourly["date"].unique(), list(range(24))],
        names=["site", "date", "hour"])
    hourly = hourly.set_index(["site", "date", "hour"]).reindex(full_idx).reset_index()
    for col in ["boardings", "alightings", "board_component", "alight_component",
                "frontage_share", "lr_passersby_hour"]:
        if col in hourly.columns:
            hourly[col] = hourly[col].fillna(0.0)

    if weather is not None and not weather.empty:
        hourly = hourly.merge(weather, on=["date", "hour"], how="left")
    else:
        hourly["temp_c"] = np.nan
        hourly["rain_mm"] = 0.0
        hourly["wind_kmh"] = np.nan
        hourly["wet_hour"] = False

    if ops is not None and not ops.empty:
        hourly = hourly.merge(ops, on=["date", "hour"], how="left")
    else:
        hourly["service_reliability"] = np.nan
        hourly["avg_delay_min"] = np.nan
        hourly["is_disrupted"] = False

    if events is not None and not events.empty:
        event_rows = []
        for site, cfg in SITE_CONFIG.items():
            site_key = _norm_text(site)
            event_kws = [_norm_text(k) for k in cfg.get("event_keywords", [])]
            # v12 fix 24: match events by site name OR suburb/keyword proximity
            site_events_mask = events["site_key"].str.contains(site_key, regex=False)
            for kw in event_kws:
                site_events_mask |= events["site_key"].str.contains(kw, regex=False)
            site_events = events[site_events_mask].copy()
            if site_events.empty:
                continue
            for _, row in site_events.iterrows():
                hours = list(range(int(row["start_hour"]), int(row["end_hour"]) + 1))
                for hour in hours:
                    event_rows.append({
                        "site": site,
                        "date": row["date"],
                        "hour": hour,
                        "event_attendees_hour": float(row["attendees"]) / max(len(hours), 1),
                        "event_multiplier": float(row["multiplier"]),
                    })
        if event_rows:
            event_hourly = pd.DataFrame(event_rows).groupby(["site", "date", "hour"], as_index=False).agg(
                event_attendees_hour=("event_attendees_hour", "sum"),
                event_multiplier=("event_multiplier", "max"),
            )
            hourly = hourly.merge(event_hourly, on=["site", "date", "hour"], how="left")
        else:
            hourly["event_attendees_hour"] = 0.0
            hourly["event_multiplier"] = 1.0
    else:
        hourly["event_attendees_hour"] = 0.0
        hourly["event_multiplier"] = 1.0

    hourly["event_attendees_hour"] = hourly["event_attendees_hour"].fillna(0.0)
    hourly["event_multiplier"] = hourly["event_multiplier"].fillna(1.0)
    hourly["wet_hour"] = hourly["wet_hour"].fillna(False)
    hourly["is_disrupted"] = hourly["is_disrupted"].fillna(False)

    daily = hourly.groupby(["site", "date"], as_index=False).agg(
        lr_passersby=("lr_passersby_hour", "sum"),
        boardings=("boardings", "sum"),
        alightings=("alightings", "sum"),
        hours_open=("hour", "nunique"),
        rain_mm=("rain_mm", "sum"),
        wet_hours=("wet_hour", "sum"),
        mean_temp_c=("temp_c", "mean"),
        mean_wind_kmh=("wind_kmh", "mean"),
        disrupted_hours=("is_disrupted", "sum"),
        avg_service_reliability=("service_reliability", "mean"),
        avg_delay_min=("avg_delay_min", "mean"),
        event_attendees=("event_attendees_hour", "sum"),
        event_multiplier=("event_multiplier", "max"),
    )

    # Add per-segment passersby columns for the v10 multi-segment model
    for seg_name, seg_def in TIME_SEGMENTS.items():
        seg_hours = seg_def["hours"]
        seg_hourly = hourly.loc[hourly["hour"].isin(seg_hours)]
        if seg_hourly.empty:
            daily[f"lr_passersby_{seg_name}"] = 0.0
        else:
            seg_daily = seg_hourly.groupby(["site", "date"], as_index=False).agg(
                **{f"lr_passersby_{seg_name}": ("lr_passersby_hour", "sum")}
            )
            daily = daily.merge(seg_daily, on=["site", "date"], how="left")
            daily[f"lr_passersby_{seg_name}"] = daily[f"lr_passersby_{seg_name}"].fillna(0.0)

    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = daily["dow"] >= 5
    daily["day_type"] = np.where(daily["is_weekend"], "weekend", "weekday")
    daily["weather_bucket"] = np.where(daily["wet_hours"] > 0, "wet", "dry")
    daily["ops_bucket"] = np.where(daily["disrupted_hours"] > 0, "disrupted", "normal")
    # v12 fix 7: season from mean daily temperature (Canberra climate)
    # Canberra daily means: winter ~5-10°C, autumn/spring ~10-20°C, summer ~20-30°C
    # Thresholds set so all four buckets are populated across a typical year.
    temp = daily["mean_temp_c"].fillna(15.0)
    daily["season_bucket"] = np.select(
        [temp < 10, temp < 18, temp < 25],
        ["cold", "mild", "warm"],
        default="hot"
    )

    # v12 fix: detect synthetic panel (weekly-average matrix → identical same-dow rows)
    # and inject multiplicative lognormal noise to represent real day-to-day variation.
    # v13 fix: use observed CV from lr_patronage_daily rather than fixed 0.12, and
    # apply monthly seasonal index so winter months see lower passersby than summer.
    _lr_daily_path = None
    for _cand in ["lr_patronage_daily.csv",
                   "output/calibration_data/lr_patronage_daily.csv",
                   "calibration_data/lr_patronage_daily.csv"]:
        if os.path.exists(_cand):
            _lr_daily_path = _cand
            break

    DAILY_CV = 0.165  # fallback: observed LR weekday CV from calibration audit
    _seasonal_index = {}  # month -> multiplier (1.0 = annual mean)
    if _lr_daily_path:
        try:
            _lrd = pd.read_csv(_lr_daily_path)
            _lrd.columns = [str(c).strip() for c in _lrd.columns]
            _date_col = None
            for _c in ["date", "Date"]:
                if _c in _lrd.columns:
                    _date_col = _c; break
            _val_col = None
            for _c in ["total", "myway", "Total"]:
                if _c in _lrd.columns:
                    _val_col = _c; break
            if _date_col and _val_col:
                _lrd["_dt"] = pd.to_datetime(_lrd[_date_col], errors="coerce")
                _lrd["_val"] = pd.to_numeric(_lrd[_val_col], errors="coerce")
                _lrd = _lrd.dropna(subset=["_dt", "_val"])
                _lrd["_dow"] = _lrd["_dt"].dt.dayofweek
                _wd = _lrd[_lrd["_dow"] < 5]["_val"]
                if len(_wd) > 30:
                    DAILY_CV = float(_wd.std() / _wd.mean())
                    print(f"  Noise CV from lr_patronage_daily: {DAILY_CV:.3f} (n={len(_wd)})")
                # Seasonal index: monthly mean / annual mean
                _lrd["_month"] = _lrd["_dt"].dt.month
                _ann_mean = _lrd["_val"].mean()
                if _ann_mean > 0:
                    _monthly = _lrd.groupby("_month")["_val"].mean()
                    _seasonal_index = (_monthly / _ann_mean).to_dict()
        except Exception as _e:
            print(f"  Could not read lr_patronage_daily for noise calibration: {_e}")

    passersby_cols = ["lr_passersby"] + [f"lr_passersby_{s}" for s in TIME_SEGMENTS]
    for site in daily["site"].unique():
        site_mask = daily["site"] == site
        site_data = daily.loc[site_mask]
        # Check if same-dow rows are identical (hallmark of weekly-average input)
        dups = site_data.groupby("dow")["lr_passersby"].apply(
            lambda s: s.nunique() == 1 if len(s) > 1 else True)
        if dups.all() and len(site_data) > 7:
            rng_noise = np.random.default_rng(hash(site) & 0xFFFFFFFF)
            n_rows = int(site_mask.sum())
            noise = rng_noise.lognormal(mean=0, sigma=DAILY_CV, size=n_rows)
            # Apply seasonal modulation if available
            if _seasonal_index:
                months = daily.loc[site_mask, "date"].dt.month
                seasonal_mult = months.map(lambda m: _seasonal_index.get(m, 1.0)).to_numpy()
                noise *= seasonal_mult
            for col in passersby_cols:
                if col in daily.columns:
                    daily.loc[site_mask, col] = daily.loc[site_mask, col] * noise
            daily.loc[site_mask, "boardings"] = daily.loc[site_mask, "boardings"] * noise
            daily.loc[site_mask, "alightings"] = daily.loc[site_mask, "alightings"] * noise

    # Callouts: compute a daily free-cup rate per site from whatever callouts
    # data is available.  ESA feeds are typically live snapshots covering a
    # single day, so we calculate cups/site/day and apply that rate uniformly
    # across the panel rather than trying to match on exact dates.
    if callouts is not None and "daily" in callouts:
        callouts_daily = callouts["daily"].copy()
        n_callout_days = max(int(callouts_daily["date"].nunique()), 1)
        site_rates = (
            callouts_daily.groupby("site", as_index=False)
            .agg(total_free_cups=("free_cups_daily", "sum"),
                 total_callouts=("callout_count", "sum"))
        )
        site_rates["free_cups_rate"] = site_rates["total_free_cups"] / n_callout_days
        site_rates["callout_rate"] = site_rates["total_callouts"] / n_callout_days
        daily = daily.merge(
            site_rates[["site", "free_cups_rate", "callout_rate"]],
            on="site",
            how="left",
        )
    else:
        daily["free_cups_rate"] = 0.0
        daily["callout_rate"] = 0.0
    daily["free_cups_rate"] = daily["free_cups_rate"].fillna(0.0)
    daily["callout_rate"] = daily["callout_rate"].fillna(0.0)

    return daily, matched_stops


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _filter_pool(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    MIN_POOL_DAYS = 5  # minimum unique days to trust bootstrap sampling
    pool = df.copy()
    if scenario.day_type in {"weekday", "weekend"}:
        pool = pool.loc[pool["day_type"].eq(scenario.day_type)]
    for attr, col, values in [
        ("weather", "weather_bucket", {"wet", "dry"}),
        ("ops", "ops_bucket", {"normal", "disrupted"}),
        ("season", "season_bucket", {"cold", "mild", "warm", "hot"}),
    ]:
        scenario_val = getattr(scenario, attr)
        if scenario_val in values and col in pool:
            tmp = pool.loc[pool[col].eq(scenario_val)]
            if tmp.empty:
                print(f"  ⚠ No {scenario_val} days in pool — using all {attr} conditions")
            elif tmp["date"].nunique() < MIN_POOL_DAYS:
                print(f"  ⚠ Only {tmp['date'].nunique()} {scenario_val} day(s) in pool "
                      f"(need ≥{MIN_POOL_DAYS}) — using full pool with {attr} modifier")
            else:
                pool = tmp
    return pool


def _get_segment_for_hour(hour: int) -> str:
    """Return the time segment name for a given hour."""
    for seg_name, seg_def in TIME_SEGMENTS.items():
        if hour in seg_def["hours"]:
            return seg_name
    return "latenight"  # fallback


def simulate(daily_panel: pd.DataFrame, scenario: Scenario, n_sim: int = N_SIM, seed: int = SEED, habitual_factor: float = 1.0) -> Dict[str, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    results: Dict[str, Dict[str, object]] = {}

    for site, cfg in SITE_CONFIG.items():
        hist = daily_panel.loc[daily_panel["site"].eq(site)].copy()
        if hist.empty:
            results[site] = {
                "status": "missing_history",
                "message": "No matched stop history for this configured site.",
            }
            continue
        pool = _filter_pool(hist, scenario)
        if pool.empty:
            pool = hist.copy()
        idx = rng.integers(0, len(pool), size=n_sim)
        draws = pool.iloc[idx].reset_index(drop=True)

        wet = draws["weather_bucket"].to_numpy() == "wet"
        # v12 fix 6: proportional wet impact based on hours of rain, not binary flag.
        # 1 wet hour barely affects daily totals; 8+ wet hours approaches full reduction.
        wet_hours = np.nan_to_num(draws["wet_hours"].to_numpy(), nan=0.0) if "wet_hours" in draws.columns else np.zeros(n_sim)
        wet_frac = np.clip(wet_hours / 24.0 * 3.0, 0.0, 1.0)  # ×3: wet hours disproportionately impactful
        cold = np.nan_to_num(draws["mean_temp_c"].to_numpy(), nan=18.0) < 15.0
        windy = np.nan_to_num(draws["mean_wind_kmh"].to_numpy(), nan=12.0) >= 30.0
        comfort_seek = wet | cold

        # v12 fix 21: disruption penalty — service delays/cancellations reduce
        # LR and bus passersby (fewer riders) but slightly increase conversion
        # (stranded passengers seek shelter/coffee).  Applied when scenario is
        # explicitly disrupted OR when individual draws land on disrupted days.
        if "ops_bucket" in draws.columns:
            disrupted = draws["ops_bucket"].to_numpy() == "disrupted"
        else:
            disrupted = np.zeros(n_sim, dtype=bool)
        # Disruption reduces PT passersby by 15-25% (service unreliability)
        disruption_atten = np.where(disrupted, rng.uniform(0.75, 0.85, n_sim), 1.0)
        # Stranded passengers more likely to buy coffee (comfort-seeking uplift)
        comfort_seek = comfort_seek | disrupted

        # ---------------------------------------------------------------
        # Channel 1: Stop passersby with per-segment conversion rates
        # ---------------------------------------------------------------
        # The daily panel now contains per-segment passersby columns.
        # Each segment converts at a different rate reflecting customer intent.
        # v12 fix: when stop_is_lr is False (matched stops are bus stops),
        # use bus conversion rates instead of LR rates.
        stop_is_lr = bool(cfg.get("stop_is_lr", True))
        lr_cups_total = np.zeros(n_sim)
        lr_passersby_total = np.zeros(n_sim)

        # v12 fix 23: bias correction — when the history window systematically
        # undershoots (or overshoots) calibration daily volumes, apply a
        # correction factor to passersby before conversion.  This compensates
        # for seasonal dips, atypical weeks, or stop-matching gaps.
        lr_daily_cal = float(cfg.get("lr_daily", 0))
        pool_lr_mean = float(pool["lr_passersby"].mean()) if "lr_passersby" in pool.columns and lr_daily_cal > 0 else 0.0
        if pool_lr_mean > 0 and lr_daily_cal > 0:
            raw_ratio = pool_lr_mean / lr_daily_cal
            # Only correct if ratio is between 0.50 and 0.90 (mild undershoot)
            # or 1.15 and 2.0 (mild overshoot). Outside 0.5-2.0 the data is
            # suspect and we leave the warning to flag it instead.
            if 0.50 <= raw_ratio < 0.90:
                # Partial correction: close 60% of the gap (conservative)
                bias_corr = 1.0 + 0.60 * (1.0 / raw_ratio - 1.0)
            elif 1.15 < raw_ratio <= 2.0:
                bias_corr = 1.0 - 0.60 * (1.0 - 1.0 / raw_ratio)
            else:
                bias_corr = 1.0
        else:
            bias_corr = 1.0

        for seg_name in TIME_SEGMENTS:
            col = f"lr_passersby_{seg_name}"
            if col not in draws.columns:
                continue
            seg_pass = np.maximum(draws[col].to_numpy(), 0)
            seg_pass *= (1.0 - 0.03 * wet_frac)
            seg_pass *= np.where(windy, 0.96, 1.0)
            seg_pass *= disruption_atten  # v12 fix 21
            seg_pass *= bias_corr         # v12 fix 23

            if stop_is_lr:
                conv_key = f"lr_conv_{seg_name}"
                if conv_key in cfg:
                    c_mean, c_sd = cfg[conv_key]
                else:
                    c_mean, c_sd = 0.030, 0.010
            else:
                # Bus stops: use flat bus conversion rate for all segments
                c_mean = float(cfg.get("bus_conv_mean", 0.022))
                c_sd = float(cfg.get("bus_conv_sd", 0.006))

            seg_alpha, seg_beta = beta_params_from_mean_sd(float(c_mean), float(c_sd))
            seg_conv = rng.beta(seg_alpha, seg_beta, n_sim)
            seg_conv *= np.where(comfort_seek, 1.12, 1.0)
            seg_conv = np.clip(seg_conv, 0.001, 0.95)

            seg_cups = rng.binomial(np.rint(seg_pass).astype(int), seg_conv)
            lr_cups_total += seg_cups
            lr_passersby_total += seg_pass

        # ---------------------------------------------------------------
        # Validation: compare modelled LR passersby against calibration
        # ---------------------------------------------------------------
        lr_daily_cal = float(cfg.get("lr_daily", 0))
        modelled_lr_mean = float(np.mean(lr_passersby_total))
        stop_label = "LR" if stop_is_lr else "bus"
        # v12 fix 12: with symmetric weights, effective pass-through =
        # board_weight × frontage_share (since bw == aw now)
        eff_pass_through = float(cfg.get("board_weight", 0.70)) * float(cfg.get("frontage_share", 0.50))
        if lr_daily_cal > 0 and modelled_lr_mean > 0:
            lr_ratio = modelled_lr_mean / lr_daily_cal
            bias_note = f" bias_corr={bias_corr:.2f}" if abs(bias_corr - 1.0) > 0.01 else ""
            if lr_ratio < 0.5 or lr_ratio > 2.0:
                print(f"  ⚠ VALIDATION WARNING: {site}: modelled {stop_label} passersby "
                      f"({modelled_lr_mean:.0f}) vs calibration lr_daily "
                      f"({lr_daily_cal:.0f}) ratio={lr_ratio:.2f}.{bias_note} "
                      f"Check board_weight/alight_weight/frontage_share.")
            else:
                print(f"  ✓ {site}: {stop_label} validation ratio={lr_ratio:.2f} "
                      f"(modelled {modelled_lr_mean:.0f} vs cal {lr_daily_cal:.0f}){bias_note}")

        # ---------------------------------------------------------------
        # Channel 2: Ambient (non-LR, non-bus) pedestrian traffic
        # ---------------------------------------------------------------
        # v11: uses absolute daily pedestrian estimate from AADT when available.
        # v12 fix: the AADT corridor_ped_amplifier was set to include bus interchange
        # activity, so aadt_ped_daily already contains bus-generated pedestrians.
        # Subtract bus_daily_frontage to avoid double-counting with Channel 2b.
        bus_frontage = float(cfg.get("bus_daily_frontage", 0))
        aadt_ped = float(cfg.get("aadt_ped_daily", 0))
        aadt_ped_sd_val = float(cfg.get("aadt_ped_sd", 0))
        aadt_wknd = float(cfg.get("aadt_weekend_factor", 0.70))

        if aadt_ped > 0:
            aadt_ped_excl_bus = max(aadt_ped - bus_frontage, 0)
            aadt_sd_excl_bus = max(aadt_ped_sd_val - bus_frontage * 0.15, aadt_ped_excl_bus * 0.10)
            ambient_pass = np.maximum(
                rng.normal(aadt_ped_excl_bus, max(aadt_sd_excl_bus, 1.0), n_sim), 0)
            ambient_pass *= np.where(draws["day_type"].to_numpy() == "weekend", aadt_wknd, 1.0)
        else:
            # Legacy fallback: ratio of LR passersby
            ambient_mult = np.maximum(
                rng.normal(float(cfg.get("ambient_ped_multiplier", 1.0)),
                           float(cfg.get("ambient_ped_sd", 0.3)), n_sim), 0.5)
            ambient_pass = lr_passersby_total * ambient_mult
            ambient_pass *= np.where(draws["day_type"].to_numpy() == "weekend", aadt_wknd, 1.0)

        ambient_pass *= (1.0 - 0.15 * wet_frac)
        ambient_pass *= np.where(windy, 0.90, 1.0)

        # v12 fix 5: distribute ambient daily total across time segments using
        # the ambient_share profile, then convert each segment separately.
        # Morning ambient walkers near a coffee machine convert higher than
        # evening strollers. Latenight gets near-zero foot traffic.
        AMBIENT_CONV_SCALE = {
            "morning": 1.25,    # commuters actively want coffee
            "daytime": 1.00,    # baseline
            "evening": 0.75,    # less coffee demand
            "latenight": 0.40,  # very few people, very low intent
        }
        ambient_cups = np.zeros(n_sim, dtype=int)
        amb_conv_mean_base = float(cfg.get("ambient_conv_mean", 0.015))
        amb_conv_sd_base = float(cfg.get("ambient_conv_sd", 0.005))
        for seg_name, seg_def in TIME_SEGMENTS.items():
            share = seg_def.get("ambient_share", 0.25)
            seg_amb_pass = np.rint(ambient_pass * share).astype(int)
            scale = AMBIENT_CONV_SCALE.get(seg_name, 1.0)
            seg_amb_mean = min(amb_conv_mean_base * scale, 0.49)
            seg_amb_alpha, seg_amb_beta = beta_params_from_mean_sd(seg_amb_mean, amb_conv_sd_base)
            seg_amb_conv = rng.beta(seg_amb_alpha, seg_amb_beta, n_sim)
            seg_amb_conv *= np.where(comfort_seek, 1.15, 1.0)
            seg_amb_conv = np.clip(seg_amb_conv, 0.001, 0.50)
            ambient_cups += rng.binomial(seg_amb_pass, seg_amb_conv)

        # ---------------------------------------------------------------
        # Channel 2b: Bus frontage passengers (new in v11)
        # ---------------------------------------------------------------
        # bus_frontage already read above (used in Channel 2 deduction)
        # v12 fix: skip when stop_is_lr=False — those bus passengers are
        # already counted in Channel 1 via the stop matching.
        if bus_frontage > 0 and stop_is_lr:
            bus_pass = np.maximum(
                rng.normal(bus_frontage, bus_frontage * 0.15, n_sim), 0)
            bus_pass *= np.where(draws["day_type"].to_numpy() == "weekend",
                                 float(cfg.get("bus_weekend_ratio", 0.28)), 1.0)
            bus_pass *= (1.0 - 0.05 * wet_frac)
            bus_pass *= np.where(windy, 0.97, 1.0)
            bus_pass *= disruption_atten  # v12 fix 21
            bus_alpha, bus_beta = beta_params_from_mean_sd(
                float(cfg.get("bus_conv_mean", 0.022)),
                float(cfg.get("bus_conv_sd", 0.006)))
            bus_conv = rng.beta(bus_alpha, bus_beta, n_sim)
            bus_conv *= np.where(comfort_seek, 1.10, 1.0)
            bus_conv = np.clip(bus_conv, 0.001, 0.50)
            bus_cups = rng.binomial(np.rint(bus_pass).astype(int), bus_conv)
        else:
            bus_pass = np.zeros(n_sim)
            bus_cups = np.zeros(n_sim, dtype=int)

        # ---------------------------------------------------------------
        # Channel 3: Venue community (v13: split into staff/volunteer and service user)
        # ---------------------------------------------------------------
        # Staff/volunteers visit regularly, know the machine, higher conversion.
        # Service users (including people in crisis) have lower purchasing power
        # and different intent. The split avoids instrumentalising vulnerable
        # populations through a single blended conversion rate.
        vv = np.maximum(rng.normal(float(cfg["venue_daily"]), float(cfg["venue_daily_sd"]), n_sim), 0)
        vv = vv * np.where(draws["day_type"].to_numpy() == "weekend", 0.90, 1.0)
        vv *= (1.0 - 0.04 * wet_frac)
        vv *= np.where(windy, 0.97, 1.0)

        venue_staff_share = float(cfg.get("venue_staff_share", 0.20))
        venue_staff_conv_mean = float(cfg.get("venue_staff_conv_mean", 0.40))
        venue_staff_conv_sd = float(cfg.get("venue_staff_conv_sd", 0.10))
        venue_user_conv_mean = float(cfg.get("venue_user_conv_mean", 0.08))
        venue_user_conv_sd = float(cfg.get("venue_user_conv_sd", 0.03))

        # Staff/volunteer sub-population
        vv_staff = np.rint(vv * venue_staff_share).astype(int)
        staff_a, staff_b = beta_params_from_mean_sd(venue_staff_conv_mean, venue_staff_conv_sd)
        staff_conv = np.clip(rng.beta(staff_a, staff_b, n_sim), 0.001, 0.95)
        venue_cups_staff = rng.binomial(vv_staff, staff_conv)

        # Service user sub-population
        vv_users = np.rint(vv * (1 - venue_staff_share)).astype(int)
        user_a, user_b = beta_params_from_mean_sd(venue_user_conv_mean, venue_user_conv_sd)
        user_conv = np.clip(rng.beta(user_a, user_b, n_sim), 0.001, 0.95)
        venue_cups_users = rng.binomial(vv_users, user_conv)

        venue_cups = venue_cups_staff + venue_cups_users

        # ---------------------------------------------------------------
        # Channel 4: Events (unchanged from v9)
        # ---------------------------------------------------------------
        event_attendees = draws["event_attendees"].to_numpy()
        if np.allclose(event_attendees, 0):
            event_attendees = np.full(n_sim, float(cfg["fallback_event_attend"]))
        event_attendees = np.maximum(event_attendees * draws["event_multiplier"].to_numpy(), 0)
        event_attendees *= (1.0 - 0.12 * wet_frac)
        event_attendees *= np.where(windy, 0.94, 1.0)
        event_alpha, event_beta = beta_params_from_mean_sd(float(cfg["event_conv_mean"]), float(cfg["event_conv_sd"]))
        event_conv = rng.beta(event_alpha, event_beta, n_sim)
        # v12 fix 14: no comfort_seek uplift — event attendees already present
        event_conv = np.clip(event_conv, 0.001, 0.95)
        event_cups = rng.binomial(np.rint(event_attendees).astype(int), event_conv)

        # ---------------------------------------------------------------
        # Channel 5: Walk-in (unchanged from v9)
        # ---------------------------------------------------------------
        walk_mean = np.full(n_sim, float(cfg["walk_daily"]))
        walk_mean *= np.where(draws["day_type"].to_numpy() == "weekend", 1.10, 1.0)
        walk_mean *= (1.0 - 0.22 * wet_frac)
        walk_mean *= np.where(windy, 0.92, 1.0)
        walk_visitors = np.maximum(rng.normal(walk_mean, float(cfg["walk_daily_sd"]), n_sim), 0)
        walk_alpha, walk_beta = beta_params_from_mean_sd(float(cfg["walk_conv_mean"]), float(cfg["walk_conv_sd"]))
        walk_conv = rng.beta(walk_alpha, walk_beta, n_sim)
        walk_conv *= np.where(comfort_seek, 1.10, 1.0)
        walk_conv = np.clip(walk_conv, 0.001, 0.98)
        walk_cups = rng.binomial(np.rint(walk_visitors).astype(int), walk_conv)

        # ---------------------------------------------------------------
        # Channel 6: Habitual / regular customers
        # ---------------------------------------------------------------
        # v12 fix 9: scale by habitual_factor (0.0 for pre-pilot day 1,
        # 0.5 for midpoint of 12-week ramp, 1.0 for steady-state)
        hab_visitors = np.maximum(
            rng.normal(float(cfg.get("habitual_daily", 0)) * habitual_factor,
                       float(cfg.get("habitual_daily_sd", 0)) * habitual_factor, n_sim), 0)
        hab_items_per = np.maximum(
            rng.normal(float(cfg.get("habitual_items_mean", 1.0)),
                       float(cfg.get("habitual_items_sd", 0.2)), n_sim), 1.0)
        habitual_cups = np.rint(hab_visitors * hab_items_per).astype(int)

        # ---------------------------------------------------------------
        # Multi-item factor: applied to commercial channels only
        # ---------------------------------------------------------------
        # v12 fix 13: venue community and event attendees don't multi-item
        # at an automated machine — they get one drink. Multi-item applies
        # to LR, ambient, bus, walk-in (coffee + snack combos).
        multi_item = np.maximum(
            rng.normal(float(cfg.get("multi_item_mean", 1.0)),
                       float(cfg.get("multi_item_sd", 0.05)), n_sim), 1.0)

        commercial_cups = lr_cups_total + ambient_cups + bus_cups + walk_cups
        paid_cups = np.rint(commercial_cups * multi_item).astype(int) + venue_cups + event_cups + habitual_cups

        # ---------------------------------------------------------------
        # Channel 7: Free responder cups (emergency shift workers)
        # ---------------------------------------------------------------
        # The daily panel carries a free_cups_rate per site, derived from
        # the callouts feed.  We sample each simulation day from a Poisson
        # at that rate.  If no callouts data was provided, fall back to a
        # territory-wide prior.
        rate_col = "free_cups_rate" if "free_cups_rate" in draws.columns else None
        observed_rate = float(draws[rate_col].mean()) if rate_col and not np.allclose(draws[rate_col].to_numpy(), 0) else 0.0

        if observed_rate > 0:
            free_cups = rng.poisson(observed_rate, n_sim)
        else:
            # v12 fix 16: site-configurable prior, overridable via calibration
            # when ESA callouts data is processed through normalize-callouts.
            prior_free_mean = float(cfg.get("free_cups_prior", 3.5))
            free_cups = rng.poisson(prior_free_mean, n_sim)

        free_cups = free_cups.astype(int)

        # Adjusted break-even: paid cups must cover fixed costs PLUS the
        # material cost of giving away free cups.
        # adj_be[i] = BREAKEVEN + ceil(free_cups[i] * COGS / REVENUE)
        free_cup_cost_in_paid_units = np.ceil(free_cups * COGS_PER_CUP / REVENUE_PER_CUP).astype(int)
        adj_breakeven = BREAKEVEN + free_cup_cost_in_paid_units

        total = paid_cups + free_cups  # Total cups dispensed (paid + free)

        wet_hist = hist.loc[hist["weather_bucket"].eq("wet")]
        dry_hist = hist.loc[hist["weather_bucket"].eq("dry")]

        results[site] = {
            "status": "ok",
            "pool_days": int(len(pool)),
            "history_days": int(len(hist)),
            "mean_trading_passersby": float(hist["lr_passersby"].mean()),
            "mean_total_passersby": float(lr_passersby_total.mean()),
            "mean_ambient_passersby": float(ambient_pass.mean()),
            "mean_bus_passersby": float(bus_pass.mean()),
            "wet_days": int((hist["weather_bucket"] == "wet").sum()),
            "dry_days": int((hist["weather_bucket"] == "dry").sum()),
            "wet_mean_passersby": float(wet_hist["lr_passersby"].mean()) if not wet_hist.empty else float("nan"),
            "dry_mean_passersby": float(dry_hist["lr_passersby"].mean()) if not dry_hist.empty else float("nan"),
            "total_mean": float(np.mean(total)),
            "paid_mean": float(np.mean(paid_cups)),
            "free_mean": float(np.mean(free_cups)),
            "ci_lo": float(np.percentile(paid_cups, 2.5)),
            "ci_hi": float(np.percentile(paid_cups, 97.5)),
            "p_above_be": float(np.mean(paid_cups >= adj_breakeven)),
            "p_above_be_unadj": float(np.mean(paid_cups >= BREAKEVEN)),
            "mean_adj_breakeven": float(np.mean(adj_breakeven)),
            "lr": float(np.mean(lr_cups_total)),
            "ambient": float(np.mean(ambient_cups)),
            "bus": float(np.mean(bus_cups)),
            "venue": float(np.mean(venue_cups)),
            "activation": float(np.mean(event_cups)),
            "walkin": float(np.mean(walk_cups)),
            "habitual": float(np.mean(habitual_cups)),
            "free_responder": float(np.mean(free_cups)),
            # v11 validation
            "lr_daily_calibration": lr_daily_cal,
            "lr_modelled_mean": modelled_lr_mean,
            "lr_validation_ratio": float(modelled_lr_mean / lr_daily_cal) if lr_daily_cal > 0 else float("nan"),
            "ambient_mode": "absolute_aadt" if aadt_ped > 0 else "ratio_of_lr",
            "raw": total,
            "raw_paid": paid_cups,
        }
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plot_stacked(results: Dict[str, Dict[str, object]], subtitle: str, outdir: str) -> str:
    sites = [s for s, r in results.items() if r.get("status") == "ok"]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(sites))
    w = 0.54

    lr = [results[s]["lr"] for s in sites]
    am = [results[s].get("ambient", 0) for s in sites]
    bu = [results[s].get("bus", 0) for s in sites]
    ve = [results[s]["venue"] for s in sites]
    ev = [results[s]["activation"] for s in sites]
    wk = [results[s]["walkin"] for s in sites]
    hb = [results[s].get("habitual", 0) for s in sites]
    fr = [results[s].get("free_responder", 0) for s in sites]
    paid = [results[s].get("paid_mean", results[s]["total_mean"]) for s in sites]
    clo = [results[s].get("paid_mean", results[s]["total_mean"]) - results[s]["ci_lo"] for s in sites]
    chi = [results[s]["ci_hi"] - results[s].get("paid_mean", results[s]["total_mean"]) for s in sites]

    bot1 = lr
    bot2 = [a + b for a, b in zip(lr, am)]
    bot2b = [a + b for a, b in zip(bot2, bu)]
    bot3 = [a + b for a, b in zip(bot2b, ve)]
    bot4 = [a + b for a, b in zip(bot3, ev)]
    bot5 = [a + b for a, b in zip(bot4, wk)]
    bot6 = [a + b for a, b in zip(bot5, hb)]

    ax.bar(x, lr, w, color=C["blue"], alpha=0.85, label="LR passersby")
    ax.bar(x, am, w, bottom=bot1, color="#6B8E9B", alpha=0.85, label="Ambient pedestrians")
    ax.bar(x, bu, w, bottom=bot2, color="#4A7B6F", alpha=0.85, label="Bus frontage")
    ax.bar(x, ve, w, bottom=bot2b, color=C["teal"], alpha=0.85, label="Venue community")
    ax.bar(x, ev, w, bottom=bot3, color=C["gold"], alpha=0.85, label="Events")
    ax.bar(x, wk, w, bottom=bot4, color=C["green"], alpha=0.85, label="Walk-in")
    ax.bar(x, hb, w, bottom=bot5, color="#8B6B99", alpha=0.85, label="Habitual regulars")
    ax.bar(x, fr, w, bottom=bot6, color=C["red"], alpha=0.50, label="Free (responders)", hatch="//", edgecolor=C["red"], linewidth=0.5)

    ax.errorbar(x, paid, yerr=[clo, chi], fmt="none", capsize=6, color=C["charcoal"], lw=1.4)

    mean_adj_be = np.mean([results[s].get("mean_adj_breakeven", BREAKEVEN) for s in sites])
    ax.axhline(BREAKEVEN, color=C["red"], ls="--", lw=1.8, alpha=0.8)
    ax.text(len(sites) - 0.1, BREAKEVEN + 1.2, f"Base BE: {BREAKEVEN}", ha="right", va="bottom", fontsize=8.2, color=C["red"])
    if mean_adj_be > BREAKEVEN + 0.5:
        ax.axhline(mean_adj_be, color=C["red"], ls=":", lw=1.2, alpha=0.5)
        ax.text(len(sites) - 0.1, mean_adj_be + 1.2, f"Adj BE (incl free): {mean_adj_be:.0f}", ha="right", va="bottom", fontsize=8.2, color=C["red"], alpha=0.7)

    for i, site in enumerate(sites):
        r = results[site]
        paid_v = float(r.get('paid_mean', r['total_mean']))
        free_v = float(r.get('free_responder', 0))
        pbe = float(r['p_above_be']) * 100
        label = f"{paid_v:.0f} paid + {free_v:.0f} free\n({pbe:.0f}% > adj BE)"
        ax.text(
            i,
            float(r["ci_hi"]) + 2.0,
            label,
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color=C["charcoal"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in sites], fontsize=9)
    ax.set_ylabel("Estimated daily cups", fontsize=10, color="#555")
    ax.set_title(f"Stop-Hour Demand Estimate by Site\n{subtitle}", fontsize=10.2, fontweight="bold", color=C["blue"])
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_ylim(0, max(float(results[s]["ci_hi"]) for s in sites) * 1.38)
    _style(ax)
    plt.tight_layout()
    out = os.path.join(outdir, "fig_stop_hour_demand.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def plot_passersby_history(daily_panel: pd.DataFrame, outdir: str) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    sites = list(SITE_CONFIG.keys())
    data = [daily_panel.loc[daily_panel["site"].eq(site), "lr_passersby"].to_numpy() for site in sites]
    bp = ax.boxplot(data, tick_labels=[s.replace(" ", "\n") for s in sites], patch_artist=True)
    fills = [C["blue"], C["teal"], C["gold"]]
    for patch, fill in zip(bp["boxes"], fills):
        patch.set_facecolor(fill)
        patch.set_alpha(0.55)
    ax.set_ylabel("Observed trading-window passersby", fontsize=10, color="#555")
    ax.set_title("Historical stop-level passersby by site", fontsize=10.2, fontweight="bold", color=C["blue"])
    _style(ax)
    plt.tight_layout()
    out = os.path.join(outdir, "fig_passersby_history.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    return out


def write_summary(
    results: Dict[str, Dict[str, object]],
    matched_stops: Dict[str, List[str]],
    daily_panel: pd.DataFrame,
    outdir: str,
    scenario: Scenario,
    source_meta: Dict[str, object],
) -> str:
    lines: List[str] = []
    lines += [
        "# After Dark — Stop-Hour Footfall Model",
        "",
        "This is the upgraded model built from the original multi-source Monte Carlo. The light-rail piece now uses stop-level boardings and alightings during the trading window rather than a single system-wide daily total.",
        "",
        "## Scenario",
        "",
        f"- Day type filter: **{scenario.day_type}**",
        f"- Weather filter: **{scenario.weather}**",
        f"- Operations filter: **{scenario.ops}**",
        f"- Break-even: **{BREAKEVEN} cups/day**",
        f"- Simulations per site: **{N_SIM:,}**",
        "",
        "## Inputs used",
        "",
        f"- Boardings rows loaded: **{source_meta['boardings_rows']:,}**",
        f"- Alightings rows loaded: **{source_meta['alightings_rows']:,}**",
        f"- Recent history window: **{source_meta['date_min']} to {source_meta['date_max']}**",
        f"- Weather file used: **{source_meta['weather_used']}**",
        f"- Weather rows loaded: **{source_meta['weather_rows']:,}**",
        f"- Weather granularity: **{source_meta['weather_granularity']}**",
        f"- Site-days with wet weather: **{source_meta['wet_days']} / {source_meta['panel_days']}**",
        f"- Mean rain across site-days: **{source_meta['mean_rain_mm']:.2f} mm**",
        f"- Events file used: **{source_meta['events_used']}**",
        f"- Operations file used: **{source_meta['ops_used']}**",
        "",
        "## Matched stops",
        "",
    ]
    for site, stops in matched_stops.items():
        if stops:
            lines.append(f"- **{site}**: {', '.join(stops)}")
        else:
            lines.append(f"- **{site}**: no stop names matched the configured patterns")

    lines += ["", "## Weather diagnostics", "", "| Site | Wet days | Dry days | Mean passersby on wet days | Mean passersby on dry days |", "|---|---:|---:|---:|---:|"]
    for site, r in results.items():
        if r.get("status") != "ok":
            lines.append(f"| {site} | 0 | 0 | n/a | n/a |")
            continue
        wet_pass = "n/a" if math.isnan(float(r["wet_mean_passersby"])) else f"{float(r['wet_mean_passersby']):.0f}"
        dry_pass = "n/a" if math.isnan(float(r["dry_mean_passersby"])) else f"{float(r['dry_mean_passersby']):.0f}"
        lines.append(f"| {site} | {int(r['wet_days'])} | {int(r['dry_days'])} | {wet_pass} | {dry_pass} |")

    lines += ["", "## Results", "", "| Site | Pool | Passersby | Paid | Free | 95% CI | P(>adjBE) | LR | Ambient | Bus | Venue | Evt | Walk | Hab | Free |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for site, r in results.items():
        if r.get("status") != "ok":
            lines.append(f"| {site} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        paid_v = float(r.get('paid_mean', r['total_mean']))
        free_v = float(r.get('free_responder', 0))
        lines.append(
            f"| {site} | {int(r['pool_days'])} | {float(r.get('mean_total_passersby', r['mean_trading_passersby'])):.0f} "
            f"| **{paid_v:.0f}** | {free_v:.0f} | [{float(r['ci_lo']):.0f}–{float(r['ci_hi']):.0f}] "
            f"| **{float(r['p_above_be']) * 100:.0f}%** "
            f"| {float(r['lr']):.0f} | {float(r.get('ambient', 0)):.0f} | {float(r.get('bus', 0)):.0f} "
            f"| {float(r['venue']):.0f} | {float(r['activation']):.0f} | {float(r['walkin']):.0f} "
            f"| {float(r.get('habitual', 0)):.0f} | {free_v:.0f} |"
        )

    lines += [
        "",
        "## Validation checks (v11)",
        "",
        "| Site | LR cal daily | LR modelled | Ratio | Ambient mode |",
        "|---|---:|---:|---:|---|",
    ]
    for site, r in results.items():
        if r.get("status") != "ok":
            continue
        lr_cal = r.get("lr_daily_calibration", 0)
        lr_mod = r.get("lr_modelled_mean", 0)
        ratio = r.get("lr_validation_ratio", float("nan"))
        ratio_str = f"{ratio:.2f}" if not math.isnan(ratio) else "n/a"
        flag = " ⚠" if (not math.isnan(ratio) and (ratio < 0.5 or ratio > 2.0)) else ""
        amb_mode = r.get("ambient_mode", "unknown")
        lines.append(f"| {site} | {lr_cal:.0f} | {lr_mod:.0f} | {ratio_str}{flag} | {amb_mode} |")

    lines += [
        "",
        "## Free coffee economics",
        "",
        f"- Material cost per free cup: **${COGS_PER_CUP:.2f}**",
        f"- Average revenue per paid cup: **${REVENUE_PER_CUP:.2f}**",
        f"- Each free cup costs **{COGS_PER_CUP / REVENUE_PER_CUP:.2f}** paid-cup-equivalents in lost margin",
        f"- Adjusted break-even = base ({BREAKEVEN}) + ceil(free_cups × {COGS_PER_CUP:.2f} / {REVENUE_PER_CUP:.2f})",
        "",
    ]

    lines += [
        "",
        "## What is improved (v11 over v10)",
        "",
        "1. **Absolute ambient pedestrian estimate**: when AADT calibration data is available, the ambient channel uses `aadt_ped_daily` directly instead of multiplying a ratio against LR passersby. Eliminates the semantic mismatch where the multiplier meant different things at different scales.",
        "2. **Bus frontage channel added**: calibration produces `bus_daily_frontage` and `bus_conv_mean/sd` — v10 ignored these. v11 simulates bus passengers as a separate demand channel with weather/weekend adjustments.",
        "3. **LR validation check**: modelled LR passersby (after board_weight × alight_weight × frontage_share) are compared against calibration `lr_daily`. Warns at console and in summary if they diverge by >2×.",
        "4. **Weekend ambient factor from calibration**: uses `aadt_weekend_factor` per site (0.47 for CBD, 0.58 for suburban) instead of hardcoded 0.70.",
        "",
        "Carried forward from v10:",
        "- Full 24-hour trading with per-segment conversion profiles.",
        "- Ambient pedestrian traffic as separate channel.",
        "- Habitual/regular customer baseline.",
        "- Multi-item purchase factor.",
        "- Stop-level rail demand.",
        "- Events, weather, operations, callouts optional inputs.",
        "",
        "## Remaining gaps",
        "",
        "- **Ambient pedestrian multipliers are priors** based on precinct type, not observed frontage counts. Replace with calibrated values after 2–4 weeks of manual counting.",
        "- **Morning conversion rates** are literature-informed estimates. A pilot with POS data will produce fitted segment-level parameters.",
        "- The **habitual customer baseline** is a placeholder. Once operating, track unique repeat customers via loyalty/app data.",
        "- GTFS-realtime is best converted into a simple hourly operations file (reliability / delay / disrupted flag) before feeding it into this script.",
        "",
        "## Next calibration loop",
        "",
        "- Collect 2–4 weeks of hourly cups sold, manual passersby counts (all-hours, not just evening), and on-site notes about queues / events / weather.",
        "- Fit separate conversion layers per time segment for commuter, ambient, venue, and event traffic.",
        "- Re-estimate ambient_ped_multiplier per site from observed counts vs LR data.",
        "- Track habitual customer counts from app/loyalty data.",
        "",
    ]

    out = os.path.join(outdir, "stop_hour_model_summary.md")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Sample data generator (optional convenience)
# ---------------------------------------------------------------------------


def generate_sample_inputs(outdir: str, seed: int = SEED) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2025-11-01", "2026-01-29", freq="D")
    stops = [
        "Alinga Street",
        "Dickson",
        "Gungahlin Place",
        "Elouera Street",
    ]
    qhours = [f"{h:02d}:{m:02d}-{(h + ((m + 15) // 60)) % 24:02d}:{(m + 15) % 60:02d}" for h in range(24) for m in [0, 15, 30, 45]]

    records_b: List[Dict[str, object]] = []
    records_a: List[Dict[str, object]] = []
    for date in dates:
        dow = date.dayofweek
        weekend = dow >= 5
        rain = rng.random() < (0.28 if weekend else 0.18)
        event_site = rng.choice(["Alinga Street", "Dickson", "Gungahlin Place", None], p=[0.08, 0.07, 0.06, 0.79])
        event_lift = 1.0 + (rng.uniform(0.2, 0.8) if event_site else 0.0)
        for stop in stops:
            base = {
                "Alinga Street": 26,
                "Dickson": 20,
                "Gungahlin Place": 22,
                "Elouera Street": 14,
            }[stop]
            for h in range(24):
                for m in [0, 15, 30, 45]:
                    peak_morning = math.exp(-((h + m / 60 - 8.0) ** 2) / 3.5)
                    peak_evening = math.exp(-((h + m / 60 - 17.8) ** 2) / 5.5)
                    nightlife = math.exp(-((h + m / 60 - 20.0) ** 2) / 6.2)
                    stop_bias = 1.0
                    if stop == "Alinga Street":
                        stop_bias = 1.35
                    elif stop == "Gungahlin Place":
                        stop_bias = 1.20
                    elif stop == "Dickson":
                        stop_bias = 1.05
                    weekend_mult = 0.86 if weekend else 1.0
                    rain_mult = 0.92 if rain else 1.0
                    event_mult = event_lift if stop == event_site and 17 <= h < 22 else 1.0

                    b_mean = max(base * stop_bias * weekend_mult * rain_mult * event_mult * (0.45 + 1.6 * peak_morning + 0.85 * peak_evening), 0.5)
                    a_mean = max(base * stop_bias * weekend_mult * rain_mult * event_mult * (0.40 + 0.85 * peak_morning + 1.55 * peak_evening + 0.35 * nightlife), 0.5)
                    records_b.append({
                        "date": date.date().isoformat(),
                        "stop_name": stop,
                        "quarter_hour": f"{h:02d}:{m:02d}-{(h + ((m + 15) // 60)) % 24:02d}:{(m + 15) % 60:02d}",
                        "boardings": rng.poisson(b_mean),
                    })
                    records_a.append({
                        "date": date.date().isoformat(),
                        "stop_name": stop,
                        "quarter_hour": f"{h:02d}:{m:02d}-{(h + ((m + 15) // 60)) % 24:02d}:{(m + 15) % 60:02d}",
                        "alightings": rng.poisson(a_mean),
                    })

    boardings_path = outdir_p / "sample_boardings_by_stop_qh.csv"
    alightings_path = outdir_p / "sample_alightings_by_stop_qh.csv"
    pd.DataFrame(records_b).to_csv(boardings_path, index=False)
    pd.DataFrame(records_a).to_csv(alightings_path, index=False)

    weather_rows: List[Dict[str, object]] = []
    for date in dates:
        wet_day = rng.random() < 0.22
        for hour in range(24):
            ts = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            rain = rng.gamma(1.2, 0.7) if wet_day and 16 <= hour <= 22 and rng.random() < 0.5 else 0.0
            weather_rows.append({
                "datetime": ts.isoformat(),
                "air_temp": 9 + 12 * math.sin((hour / 24) * math.pi) + rng.normal(0, 1.2),
                "rainfall": round(rain, 2),
                "wind_spd_kmh": max(0, rng.normal(12, 4)),
            })
    weather_path = outdir_p / "sample_weather.csv"
    pd.DataFrame(weather_rows).to_csv(weather_path, index=False)

    event_rows: List[Dict[str, object]] = []
    for date in pd.date_range("2025-11-01", "2026-01-29", freq="7D"):
        site = rng.choice(["Alinga Street", "Dickson", "Gungahlin Place"])
        start_hour = int(rng.choice([17, 18, 19]))
        attendees = int(rng.integers(25, 70))
        event_rows.append({
            "site": site,
            "start_datetime": (pd.Timestamp(date) + pd.Timedelta(hours=start_hour)).isoformat(),
            "end_datetime": (pd.Timestamp(date) + pd.Timedelta(hours=start_hour + 2)).isoformat(),
            "attendees": attendees,
            "multiplier": round(rng.uniform(1.0, 1.3), 2),
        })
    events_path = outdir_p / "sample_events.csv"
    pd.DataFrame(event_rows).to_csv(events_path, index=False)

    ops_rows: List[Dict[str, object]] = []
    for date in dates:
        for hour in range(24):
            disrupted = rng.random() < (0.08 if 17 <= hour < 22 else 0.03)
            ops_rows.append({
                "date": date.date().isoformat(),
                "hour": hour,
                "service_reliability": round(0.84 + rng.random() * 0.16, 3),
                "avg_delay_min": round(rng.uniform(0, 9 if disrupted else 4), 2),
                "is_disrupted": disrupted,
            })
    ops_path = outdir_p / "sample_ops.csv"
    pd.DataFrame(ops_rows).to_csv(ops_path, index=False)

    return {
        "boardings": str(boardings_path),
        "alightings": str(alightings_path),
        "weather": str(weather_path),
        "events": str(events_path),
        "ops": str(ops_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _apply_calibration(json_path: str) -> None:
    """Read calibration_params.json and override SITE_CONFIG values."""
    with open(json_path, "r", encoding="utf-8") as f:
        cal = json.load(f)

    overridable = {
        "bus_stop_patterns",
        "bus_frontage_share",
        "bus_daily_frontage",
        "bus_conv_mean",
        "bus_conv_sd",
        "bike_daily",
        "bike_daily_sd",
        "bike_conv_mean",
        "bike_conv_sd",
        "bike_weekend_ratio",
        "bus_weekend_ratio",
        "ambient_ped_multiplier",
        "ambient_ped_sd",
        "ambient_conv_mean",
        "ambient_conv_sd",
        "aadt_ped_daily",
        "aadt_ped_sd",
        "aadt_weekend_factor",
        "lr_conv_morning",
        "lr_conv_daytime",
        "lr_conv_evening",
        "lr_conv_latenight",
        "lr_daily",
        "free_cups_prior",
    }

    # calibrate_model.py historically emitted different key names — map them
    key_aliases = {
        "residual_ambient_mult": "ambient_ped_multiplier",
        "residual_ambient_sd": "ambient_ped_sd",
        "residual_ambient_conv_mean": "ambient_conv_mean",
        "residual_ambient_conv_sd": "ambient_conv_sd",
    }

    applied = 0
    for site, params in cal.items():
        if site not in SITE_CONFIG:
            continue
        cfg = SITE_CONFIG[site]
        # Direct key matches
        for key in overridable:
            if key in params:
                cfg[key] = params[key]
                applied += 1
        # Aliased keys (only if the native key wasn't already set)
        for alias_from, alias_to in key_aliases.items():
            if alias_from in params and alias_to not in params:
                cfg[alias_to] = params[alias_from]
                applied += 1

    print(f"Calibration: applied {applied} overrides from {json_path}")


def run_sensitivity_sweep(daily_panel, scenario, n_sim, seed, outdir):
    """Run simulation at multiple conversion rate scales and report breakeven probability."""
    import copy
    scales = [0.50, 0.75, 1.00, 1.25, 1.50]
    conv_keys = ["lr_conv_morning", "lr_conv_daytime", "lr_conv_evening", "lr_conv_latenight",
                 "ambient_conv_mean", "bus_conv_mean"]
    # Save originals
    originals = {site: {k: copy.deepcopy(cfg.get(k)) for k in conv_keys if k in cfg}
                 for site, cfg in SITE_CONFIG.items()}
    lines = ["# Conversion Rate Sensitivity Sweep", "",
             "| Scale | Site | Paid mean | 95% CI | P(>adjBE) |",
             "|---:|---|---:|---:|---:|"]
    print("\n" + "=" * 72)
    print("SENSITIVITY SWEEP — conversion rates at 50%/75%/100%/125%/150%")
    print("=" * 72)
    print(f"{'Scale':>6} {'Site':<20} {'Paid':>6} {'95% CI':>16} {'P(>adjBE)':>10}")
    print("-" * 62)
    for scale in scales:
        for site, cfg in SITE_CONFIG.items():
            for k in conv_keys:
                orig = originals[site].get(k)
                if orig is None:
                    continue
                if isinstance(orig, (list, tuple)):
                    cfg[k] = (orig[0] * scale, orig[1])
                else:
                    cfg[k] = orig * scale
        res = simulate(daily_panel, scenario=scenario, n_sim=n_sim, seed=seed)
        for site, r in res.items():
            if r.get("status") != "ok":
                continue
            paid = float(r.get("paid_mean", r["total_mean"]))
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            pbe = float(r["p_above_be"]) * 100
            print(f"{scale:>5.0%} {site:<20} {paid:>6.0f} [{lo:>4.0f}-{hi:>4.0f}] {pbe:>9.0f}%")
            lines.append(f"| {scale:.0%} | {site} | {paid:.0f} | [{lo:.0f}–{hi:.0f}] | {pbe:.0f}% |")
    # Restore originals
    for site, cfg in SITE_CONFIG.items():
        for k, v in originals[site].items():
            cfg[k] = v

    # --- Sweep 2: Ambient pedestrian volume (epistemic uncertainty in AADT chain) ---
    amb_keys = ["aadt_ped_daily", "aadt_ped_sd", "ambient_ped_multiplier", "ambient_ped_sd"]
    amb_originals = {site: {k: copy.deepcopy(cfg.get(k)) for k in amb_keys if k in cfg}
                     for site, cfg in SITE_CONFIG.items()}
    lines += ["", "## Ambient Pedestrian Volume Sensitivity", "",
              "Tests epistemic uncertainty in the AADT-to-pedestrian chain.",
              "At 50%, the corridor amplifier and pedestrian estimate are halved.",
              "",
              "| Scale | Site | Paid mean | 95% CI | P(>adjBE) |",
              "|---:|---|---:|---:|---:|"]
    print("\n" + "=" * 72)
    print("SENSITIVITY SWEEP — ambient pedestrian volume at 50%/75%/100%/125%/150%")
    print("=" * 72)
    print(f"{'Scale':>6} {'Site':<20} {'Paid':>6} {'95% CI':>16} {'P(>adjBE)':>10}")
    print("-" * 62)
    for scale in scales:
        for site, cfg in SITE_CONFIG.items():
            for k in amb_keys:
                orig = amb_originals[site].get(k)
                if orig is None:
                    continue
                cfg[k] = orig * scale
        res = simulate(daily_panel, scenario=scenario, n_sim=n_sim, seed=seed)
        for site, r in res.items():
            if r.get("status") != "ok":
                continue
            paid = float(r.get("paid_mean", r["total_mean"]))
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            pbe = float(r["p_above_be"]) * 100
            print(f"{scale:>5.0%} {site:<20} {paid:>6.0f} [{lo:>4.0f}-{hi:>4.0f}] {pbe:>9.0f}%")
            lines.append(f"| {scale:.0%} | {site} | {paid:.0f} | [{lo:.0f}–{hi:.0f}] | {pbe:.0f}% |")
    for site, cfg in SITE_CONFIG.items():
        for k, v in amb_originals[site].items():
            cfg[k] = v

    out = os.path.join(outdir, "sensitivity_sweep.md")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print("-" * 62)
    print(f"Saved to {out}")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="After Dark stop-hour demand model")
    ap.add_argument("--boardings", help="CSV export of Boardings By Stop By Quarter Hr")
    ap.add_argument("--alightings", help="CSV export of Alightings By Stop By Quarter Hr")
    ap.add_argument("--weather", help="Optional weather observations CSV")
    ap.add_argument("--events", help="Optional events CSV")
    ap.add_argument("--ops", help="Optional operations / reliability CSV")
    ap.add_argument("--callouts", help="Optional ESA callouts CSV (from normalize-callouts)")
    ap.add_argument("--addinsight", help="Optional Addinsight corridor summary CSV")
    ap.add_argument("--lr-15min", help="Optional LR patronage 15-min CSV (Socrata xvid-q4du) — provides actual LR stop volumes")
    ap.add_argument("--recent-from", default=RECENT_FROM, help=f"Lower bound for history window (default {RECENT_FROM})")
    ap.add_argument("--n-sim", type=int, default=N_SIM, help=f"Simulation count per site (default {N_SIM:,})")
    ap.add_argument("--day-type", choices=["all", "weekday", "weekend"], default="all")
    ap.add_argument("--weather-scenario", choices=["all", "wet", "dry"], default="all")
    ap.add_argument("--ops-scenario", choices=["all", "normal", "disrupted"], default="all")
    ap.add_argument("--season", choices=["all", "cold", "mild", "warm", "hot"], default="all")
    ap.add_argument("--ramp-weeks", type=int, default=0,
                    help="Habitual customer ramp period in weeks (0=steady-state, 12=3-month pilot ramp → avg factor 0.5)")
    ap.add_argument("--calibration", help="calibration_params.json from calibrate_model.py")
    ap.add_argument("--sensitivity", action="store_true", help="Run conversion rate sensitivity sweep at 50/75/100/125/150%%")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--generate-sample-data", action="store_true", help="Generate synthetic sample inputs in outdir/sample_inputs and run the model on them")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.calibration:
        _apply_calibration(args.calibration)

    if args.generate_sample_data:
        sample_paths = generate_sample_inputs(str(outdir / "sample_inputs"))
        args.boardings = args.boardings or sample_paths["boardings"]
        args.alightings = args.alightings or sample_paths["alightings"]
        args.weather = args.weather or sample_paths["weather"]
        args.events = args.events or sample_paths["events"]
        args.ops = args.ops or sample_paths["ops"]

    if not args.boardings or not args.alightings:
        raise SystemExit("Provide --boardings and --alightings, or use --generate-sample-data.")

    boardings = load_stop_activity(args.boardings, metric="boardings")
    alightings = load_stop_activity(args.alightings, metric="alightings")
    weather = load_weather(args.weather)
    events = load_events(args.events)
    ops = load_ops(args.ops)
    callouts = load_callouts(args.callouts)
    lr_15min = load_lr_15min(getattr(args, 'lr_15min', None))

    # Addinsight corridor disruption → supplement ops panel
    if getattr(args, 'addinsight', None):
        try:
            ai_df = pd.read_csv(args.addinsight)
            ai_disrupted = set(ai_df.loc[ai_df["is_disrupted"] == True, "site"].tolist())
            if ai_disrupted:
                print(f"Addinsight:       corridor disruption at {', '.join(ai_disrupted)}")
            else:
                print(f"Addinsight:       no corridor disruptions ({len(ai_df)} sites checked)")
            # Attach per-site corridor speed for ambient amplifier adjustment
            _ai_speed = dict(zip(ai_df["site"], ai_df["avg_speed_kmh"]))
            _ai_disrupt = dict(zip(ai_df["site"], ai_df["is_disrupted"]))
        except Exception as e:
            print(f"Addinsight:       load error — {e}")
            _ai_speed, _ai_disrupt = {}, {}
    else:
        _ai_speed, _ai_disrupt = {}, {}

    activity = build_activity_panel(boardings, alightings)
    daily_panel, matched_stops = build_site_daily_panel(activity, weather, events, ops, callouts, recent_from=args.recent_from, lr_15min=lr_15min)

    # Overlay Addinsight corridor disruption onto daily panel
    if _ai_disrupt:
        for site, disrupted in _ai_disrupt.items():
            if disrupted and site in daily_panel["site"].values:
                mask = daily_panel["site"] == site
                before = daily_panel.loc[mask, "disrupted_hours"].sum()
                # Mark all hours as disrupted for this site when corridor is disrupted
                daily_panel.loc[mask, "disrupted_hours"] = daily_panel.loc[mask, "disrupted_hours"].clip(lower=1)
                daily_panel.loc[mask, "ops_bucket"] = "disrupted"
                after = daily_panel.loc[mask, "disrupted_hours"].sum()
                if after > before:
                    print(f"  Addinsight: {site} — {int(after - before)} additional disrupted site-days from corridor")

    scenario = Scenario(day_type=args.day_type, weather=args.weather_scenario, ops=args.ops_scenario, season=args.season)
    # v12 fix 9: habitual ramp — average fraction during ramp period
    hab_factor = 0.5 if args.ramp_weeks > 0 else 1.0  # midpoint of linear ramp
    if args.ramp_weeks > 0:
        print(f"Habitual ramp:    {args.ramp_weeks} weeks → avg factor {hab_factor:.2f}")
    results = simulate(daily_panel, scenario=scenario, n_sim=args.n_sim, seed=SEED, habitual_factor=hab_factor)

    source_meta = {
        "boardings_rows": int(len(boardings)),
        "alightings_rows": int(len(alightings)),
        "date_min": str(daily_panel["date"].min().date()),
        "date_max": str(daily_panel["date"].max().date()),
        "weather_used": bool(weather is not None and not weather.empty),
        "weather_rows": int(len(weather)) if weather is not None else 0,
        "weather_granularity": str(weather["weather_granularity"].dropna().iloc[0]) if weather is not None and not weather.empty and "weather_granularity" in weather.columns else "none",
        "panel_days": int(len(daily_panel)),
        "wet_days": int((daily_panel["weather_bucket"] == "wet").sum()) if "weather_bucket" in daily_panel.columns else 0,
        "mean_rain_mm": float(daily_panel["rain_mm"].mean()) if "rain_mm" in daily_panel.columns else 0.0,
        "events_used": bool(events is not None and not events.empty),
        "ops_used": bool(ops is not None and not ops.empty),
        "callouts_used": bool(callouts is not None),
        "callout_incidents": int(len(callouts["incidents"])) if callouts is not None else 0,
        "callout_sites": int(callouts["daily"]["site"].nunique()) if callouts is not None else 0,
    }

    subtitle = (
        f"{len(daily_panel):,} site-days | scenario day={scenario.day_type}, weather={scenario.weather}, ops={scenario.ops}, season={scenario.season}"
    )
    fig1 = plot_stacked(results, subtitle, str(outdir))
    fig2 = plot_passersby_history(daily_panel, str(outdir))
    summary_path = write_summary(results, matched_stops, daily_panel, str(outdir), scenario, source_meta)

    print("=" * 72)
    print("After Dark, On Tap — Stop-Hour Footfall + Conversion Model")
    print("=" * 72)
    print(f"Boardings rows:   {len(boardings):,}")
    print(f"Alightings rows:  {len(alightings):,}")
    print(f"History window:   {source_meta['date_min']} to {source_meta['date_max']}")
    print(f"Scenario:         day={scenario.day_type}, weather={scenario.weather}, ops={scenario.ops}, season={scenario.season}")
    print(f"Weather input:    used={source_meta['weather_used']} rows={source_meta['weather_rows']:,} granularity={source_meta['weather_granularity']} wet site-days={source_meta['wet_days']}/{source_meta['panel_days']}")
    print(f"Callouts input:   used={source_meta['callouts_used']} incidents={source_meta['callout_incidents']} sites={source_meta['callout_sites']}")
    print(f"Free cup cost:    COGS=${COGS_PER_CUP:.2f} / revenue=${REVENUE_PER_CUP:.2f} per cup")
    print("-" * 72)
    print(f"{'Site':<20} {'Passersby':>10} {'Paid':>6} {'Free':>6} {'95% CI':>16} {'P(>adjBE)':>10}")
    print("-" * 72)
    for site, r in results.items():
        if r.get("status") != "ok":
            print(f"{site:<20} {'n/a':>10} {'n/a':>6} {'n/a':>6} {'n/a':>16} {'n/a':>10}")
            continue
        print(
            f"{site:<20} {float(r['mean_total_passersby']):>10.0f} {float(r.get('paid_mean', r['total_mean'])):>6.0f} "
            f"{float(r.get('free_responder', 0)):>6.0f} "
            f"[{float(r['ci_lo']):>4.0f}-{float(r['ci_hi']):>4.0f}] {float(r['p_above_be']) * 100:>9.0f}%"
        )
        print(
            f"  {'':>18} LR={float(r['lr']):.0f} Amb={float(r.get('ambient',0)):.0f} "
            f"Bus={float(r.get('bus',0)):.0f} "
            f"Ven={float(r['venue']):.0f} Evt={float(r['activation']):.0f} "
            f"Walk={float(r['walkin']):.0f} Hab={float(r.get('habitual',0)):.0f} "
            f"Free={float(r.get('free_responder',0)):.0f} adjBE={float(r.get('mean_adj_breakeven', BREAKEVEN)):.0f}"
        )
        # v11: validation + ambient mode
        amb_mode = r.get("ambient_mode", "unknown")
        lr_val = r.get("lr_validation_ratio", float("nan"))
        lr_val_str = f"{lr_val:.2f}" if not math.isnan(lr_val) else "n/a"
        print(f"  {'':>18} ambient_mode={amb_mode} lr_validation={lr_val_str}")
    print("-" * 72)
    print(f"Outputs:\n  {fig1}\n  {fig2}\n  {summary_path}")
    if args.sensitivity:
        sens_path = run_sensitivity_sweep(daily_panel, scenario, args.n_sim, SEED, str(outdir))
        print(f"  {sens_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
