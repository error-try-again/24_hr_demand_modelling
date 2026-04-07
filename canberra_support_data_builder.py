#!/usr/bin/env python3
"""
canberra_support_data_builder.py

Build canonical support files for after_dark_stop_hour_model.py:
- weather.csv
- events.csv
- ops.csv
- Accept messy CSV/XLSX/JSON/HTML exports or raw downloaded pages.
- Normalize to the exact columns your demand model loader accepts.
- Stay useful even when public source formats change.

Requires:
    pip install pandas requests lxml openpyxl beautifulsoup4


Useage:
    python canberra_support_data_builder.py templates --outdir ./support_templates

    python canberra_support_data_builder.py fetch \
      --url https://www.bom.gov.au/act/observations/canberra.shtml \
      --output raw_weather.html

    python canberra_support_data_builder.py normalize-weather \
      --input raw_weather.html \
      --output weather.csv

    python canberra_support_data_builder.py fetch-weather-history \
      --output weather.csv \
      --station-code IDCJDW2801 \
      --months 14

    python canberra_support_data_builder.py normalize-events \
      --input raw_events.csv \
      --output events.csv \
      --site-map site_map.csv \
      --event-overrides event_overrides.csv

    python canberra_support_data_builder.py normalize-ops \
      --input raw_ops.csv \
      --output ops.csv \
      --include-regions "Central Canberra,Gungahlin" \
      --include-keywords "light rail,detour"

    python canberra_support_data_builder.py normalize-callouts \
      --input esa_current_incidents.xml \
      --output callouts.csv \
      --site-map site_map.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


SITE_NAMES = ["Alinga Street", "Dickson", "Gungahlin Place"]

DEFAULT_BOM_URL = "https://www.bom.gov.au/act/observations/canberra.shtml"
DEFAULT_EVENTS_URL = "https://events.canberra.com.au/whats-on"
DEFAULT_ALERTS_URL = "https://www.transport.act.gov.au/service-alerts"

WEATHER_TS_CANDIDATES = [
    "datetime", "timestamp", "ts", "time", "observation time", "observation_time"
]
WEATHER_DATE_CANDIDATES = [
    "date", "local date", "observation date", "day", "date/time"
]
WEATHER_TIME_CANDIDATES = [
    "time", "local time", "observation time"
]
WEATHER_TEMP_CANDIDATES = [
    "air temp", "air temperature", "temp", "temperature", "temp c", "temp_c"
]
WEATHER_MAX_TEMP_CANDIDATES = [
    "maximum temperature", "max temp", "max temperature"
]
WEATHER_MIN_TEMP_CANDIDATES = [
    "minimum temperature", "min temp", "min temperature"
]
WEATHER_RAIN_CANDIDATES = [
    "rainfall", "rain mm", "precip", "precip mm", "rain since 9am", "rainfall amount"
]
WEATHER_WIND_CANDIDATES = [
    "wind spd kmh", "wind kmh", "wind speed", "wind speed km/h", "wind speed kmh"
]

EVENT_SITE_CANDIDATES = ["site", "stop", "stop name", "stop_name", "location", "venue"]
EVENT_TITLE_CANDIDATES = ["title", "event", "event name", "event_name"]
EVENT_ATTEND_CANDIDATES = [
    "attendees", "attendance", "expected attendees", "expected_attendees", "crowd", "value"
]
EVENT_MULT_CANDIDATES = ["multiplier", "intensity", "lift"]
EVENT_DATE_CANDIDATES = ["date", "event date", "event_date"]
EVENT_START_CANDIDATES = ["start datetime", "start_datetime", "start", "start time", "start_time"]
EVENT_END_CANDIDATES = ["end datetime", "end_datetime", "end", "end time", "end_time"]

OPS_DATE_CANDIDATES = ["date", "service date", "service_date"]
OPS_HOUR_CANDIDATES = ["hour", "time", "time period", "time_period"]
OPS_RELIAB_CANDIDATES = [
    "service reliability", "service_reliability", "reliability", "on time ratio", "on_time_ratio"
]
OPS_DELAY_CANDIDATES = ["avg delay min", "avg_delay_min", "delay", "delay min", "average delay"]
OPS_DISRUPT_CANDIDATES = ["disrupted", "is_disrupted", "service disruption", "service_disruption"]
OPS_START_CANDIDATES = ["start datetime", "start_datetime", "start", "from", "begin"]
OPS_END_CANDIDATES = ["end datetime", "end_datetime", "end", "to", "finish"]


def norm_text(value: object) -> str:
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\u2013\u2014]", "-", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {norm_text(c): c for c in columns}
    for cand in candidates:
        hit = normalized.get(norm_text(cand))
        if hit is not None:
            return hit
    return None


def maybe_int_hour(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{1,2}", s):
        hour = int(s)
        return hour if 0 <= hour <= 23 else None
    m = re.search(r"(\d{1,2})", s)
    if m:
        hour = int(m.group(1))
        if 0 <= hour <= 23:
            return hour
    return None


def read_any_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            # try common wrappers
            for key in ["results", "data", "items", "events", "alerts"]:
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            return pd.json_normalize(payload)
    if suffix in {".html", ".htm"}:
        tables = pd.read_html(path)
        if tables:
            # choose the widest non-empty table
            tables = [t for t in tables if not t.empty]
            if tables:
                tables.sort(key=lambda df: (df.shape[1], df.shape[0]), reverse=True)
                return tables[0]
        raise ValueError(f"No HTML tables found in {path}")
    raise ValueError(f"Unsupported input type: {path}")


def write_csv(df: pd.DataFrame, output: str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def fetch_url(url: str, output: str, timeout: int = 30) -> None:
    if requests is None:
        raise RuntimeError("requests is required for fetch. Install with: pip install requests")
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    Path(output).write_bytes(resp.content)


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", str(value).strip(), flags=re.I))


def materialize_input_path(input_path: str, default_url: str, suffix: str) -> tuple[str, Optional[str]]:
    """Return a readable local path for normalize-* commands.

    - If input_path is an existing local file, use it directly.
    - If input_path is a URL, fetch it to a temp file.
    - If input_path is missing, fetch the command's default public source instead.

    Returns (path_to_use, temp_path_to_cleanup).
    """
    raw = str(input_path).strip()
    candidate = Path(raw)
    if raw and candidate.exists():
        return raw, None

    url = raw if is_url(raw) else default_url
    if requests is None:
        if raw and not candidate.exists():
            raise FileNotFoundError(
                f"Input file not found: {raw}. Also cannot auto-fetch {url} because requests is not installed."
            )
        raise RuntimeError("requests is required to fetch remote inputs. Install with: pip install requests")

    tmp = tempfile.NamedTemporaryFile(prefix="canberra_support_", suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()
    fetch_url(url, tmp_path)
    return tmp_path, tmp_path


def build_datetime_series(df: pd.DataFrame, daily_hour: int = 18) -> pd.Series:
    ts_col = first_present(df.columns, WEATHER_TS_CANDIDATES)
    if ts_col:
        return pd.to_datetime(df[ts_col], errors="coerce")

    date_col = first_present(df.columns, WEATHER_DATE_CANDIDATES)
    time_col = first_present(df.columns, WEATHER_TIME_CANDIDATES)
    if date_col and time_col and date_col != time_col:
        combined = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        return pd.to_datetime(combined, errors="coerce")

    if date_col:
        ts = pd.to_datetime(df[date_col], errors="coerce")
        return ts + pd.to_timedelta(daily_hour, unit="h")

    return pd.Series([pd.NaT] * len(df))



def parse_bom_observations_html(input_path: str, station: str = "Canberra") -> pd.DataFrame:
    """
    Parse BoM 'Latest Weather Observations for the Canberra Area' HTML snapshot.
    Returns at most one row for the chosen station because the source page is a live snapshot,
    not a historical series.
    """
    html = Path(input_path).read_text(encoding="utf-8", errors="ignore")

    # Issued timestamp, e.g.:
    # Issued at 12:31 pm EDT Thursday 2 April 2026
    issued_match = re.search(
        r"Issued at\s+(\d{1,2}:\d{2}\s*[ap]m)\s+[A-Z]{2,4}\s+\w+\s+(\d{1,2}\s+\w+\s+\d{4})",
        html,
        flags=re.I,
    )
    issued_ts = None
    if issued_match:
        issued_ts = pd.to_datetime(
            f"{issued_match.group(2)} {issued_match.group(1)}",
            errors="coerce",
            dayfirst=True,
        )

    # Open the visible text through pandas read_html first; it's usually easier to locate station rows there.
    tables = pd.read_html(input_path)
    station_norm = norm_text(station)

    record = None
    for table in tables:
        if table.empty:
            continue
        # Flatten multiindex columns if needed
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in table.columns]
        else:
            table.columns = [str(c).strip() for c in table.columns]
        # Find a candidate station/name column
        cols = [str(c) for c in table.columns]
        station_col = first_present(cols, ["station", "station name", "name", cols[0] if cols else ""])
        if station_col is None:
            station_col = cols[0] if cols else None
        if station_col is None:
            continue
        tmp = table.copy()
        tmp[station_col] = tmp[station_col].astype(str)
        hits = tmp[tmp[station_col].map(norm_text).str.contains(station_norm, na=False)]
        if hits.empty:
            continue
        row = hits.iloc[0]

        # Try to locate common weather columns. The observations page often has flattened columns with units.
        def pick(cands):
            return first_present(cols, cands)

        dt_col = pick(["date/time", "date time", "datetime", "time"])
        temp_col = pick(["temp °c", "temp c", "temp", "air temp", "air temperature"])
        rain_col = pick(["rain since 9am mm", "rainfall", "rain mm", "precip"])
        wind_col = pick(["dir spd km/h", "spd km/h", "wind speed", "wind kmh", "wind spd kmh"])

        obs_dt = None
        if dt_col:
            raw_dt = str(row[dt_col]).strip()
            # row style: 02/12:30pm  -> day/time only, month/year from issued timestamp
            m = re.match(r"(\d{1,2})/(\d{1,2}:\d{2}\s*[ap]m|\d{1,2}:\d{2}[ap]m)", raw_dt, flags=re.I)
            if m and issued_ts is not None:
                obs_dt = pd.to_datetime(
                    f"{issued_ts.year}-{issued_ts.month:02d}-{int(m.group(1)):02d} {m.group(2)}",
                    errors="coerce",
                )
            else:
                obs_dt = pd.to_datetime(raw_dt, errors="coerce")

        if obs_dt is None or pd.isna(obs_dt):
            obs_dt = issued_ts

        wind_val = None
        if wind_col:
            wind_text = str(row[wind_col])
            m = re.search(r"(\d+(?:\.\d+)?)", wind_text)
            if m:
                wind_val = float(m.group(1))

        record = {
            "datetime": obs_dt,
            "air_temp": pd.to_numeric(row[temp_col], errors="coerce") if temp_col else pd.NA,
            "rainfall": pd.to_numeric(row[rain_col], errors="coerce") if rain_col else 0.0,
            "wind_spd_kmh": wind_val,
        }
        break

    if record is None:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    out = pd.DataFrame([record]).dropna(subset=["datetime"]).copy()
    return out






def _find_col_contains(columns: Iterable[str], include: Iterable[str], exclude: Iterable[str] = ()) -> Optional[str]:
    """
    Return the first column whose normalized text contains all include tokens
    and none of the exclude tokens.
    """
    include_n = [norm_text(x) for x in include]
    exclude_n = [norm_text(x) for x in exclude]
    for col in columns:
        c = norm_text(col)
        if all(tok in c for tok in include_n) and not any(tok in c for tok in exclude_n):
            return col
    return None


def _flatten_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for tup in out.columns:
            parts = [str(x).strip() for x in tup if str(x).strip() and str(x).strip().lower() != "nan"]
            cols.append(" ".join(parts))
        out.columns = cols
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out
def parse_bom_daily_weather_html(input_path: str, daily_hour: int = 18) -> pd.DataFrame:
    """
    Parse BoM 'Daily Weather Observations' monthly HTML pages, e.g.
    https://www.bom.gov.au/climate/dwo/IDCJDW2801.latest.shtml

    Output columns:
      datetime, air_temp, rainfall, wind_spd_kmh
    """
    html = Path(input_path).read_text(encoding="utf-8", errors="ignore")

    month_year_match = re.search(
        r"([A-Z][a-z]+\s+\d{4})\s+Daily Weather Observations",
        html,
        flags=re.I,
    )
    if month_year_match is None:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    month_anchor = pd.to_datetime("1 " + month_year_match.group(1), errors="coerce", dayfirst=True)
    if pd.isna(month_anchor):
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    tables = pd.read_html(input_path)
    if not tables:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    def score_table(df: pd.DataFrame) -> int:
        flat = _flatten_bom_columns(df)
        cols = " | ".join([norm_text(c) for c in flat.columns])
        score = 0
        for token in ["date", "temps", "min", "max", "rain"]:
            if token in cols:
                score += 2
        if "max wind gust" in cols:
            score += 2
        if len(flat) >= 2:
            score += 1
        return score

    candidates = [_flatten_bom_columns(t) for t in tables if not t.empty]
    candidates.sort(key=score_table, reverse=True)
    if not candidates or score_table(candidates[0]) < 6:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    df = candidates[0].copy()
    cols = list(df.columns)

    date_col = _find_col_contains(cols, ["date"]) or (cols[0] if cols else None)
    min_col = _find_col_contains(cols, ["temps", "min"])
    max_col = _find_col_contains(cols, ["temps", "max"])
    rain_col = _find_col_contains(cols, ["rain"])
    # Prefer max gust speed, then 3 pm speed, then 9 am speed
    wind_col = (
        _find_col_contains(cols, ["max wind gust", "spd"])
        or _find_col_contains(cols, ["3 pm", "spd"])
        or _find_col_contains(cols, ["9 am", "spd"])
    )
    # Useful fallbacks for incomplete current-day rows
    temp_9_col = _find_col_contains(cols, ["9 am", "temp"])
    temp_3_col = _find_col_contains(cols, ["3 pm", "temp"])

    if date_col is None:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    date_str = df[date_col].astype(str).str.strip()
    day_num = pd.to_numeric(date_str.str.extract(r"^(\d{1,2})")[0], errors="coerce")
    keep = day_num.notna()
    df = df.loc[keep].copy()
    day_num = day_num.loc[keep].astype(int)

    if df.empty:
        return pd.DataFrame(columns=["datetime", "air_temp", "rainfall", "wind_spd_kmh"])

    dt = pd.to_datetime(
        {
            "year": int(month_anchor.year),
            "month": int(month_anchor.month),
            "day": day_num.astype(int),
        },
        errors="coerce",
    ) + pd.to_timedelta(int(daily_hour), unit="h")

    out = pd.DataFrame({"datetime": dt})

    mn = pd.to_numeric(df[min_col], errors="coerce") if min_col else pd.Series([pd.NA] * len(df), index=df.index)
    mx = pd.to_numeric(df[max_col], errors="coerce") if max_col else pd.Series([pd.NA] * len(df), index=df.index)
    t9 = pd.to_numeric(df[temp_9_col], errors="coerce") if temp_9_col else pd.Series([pd.NA] * len(df), index=df.index)
    t3 = pd.to_numeric(df[temp_3_col], errors="coerce") if temp_3_col else pd.Series([pd.NA] * len(df), index=df.index)

    air_temp = pd.Series(index=df.index, dtype="float64")
    # Best: min/max midpoint
    air_temp = ((mn + mx) / 2.0)
    # Fallback: mean of 9am and 3pm observed temps
    fallback_mean = pd.concat([t9, t3], axis=1).mean(axis=1)
    air_temp = air_temp.fillna(fallback_mean)
    # Final fallback: whichever observed temp is available
    air_temp = air_temp.fillna(t9).fillna(t3)
    out["air_temp"] = air_temp.values

    if rain_col:
        out["rainfall"] = pd.to_numeric(df[rain_col], errors="coerce").fillna(0.0).values
    else:
        out["rainfall"] = 0.0

    # Wind fallback: prefer max gust speed when present, otherwise use mean of 9am/3pm speeds.
    spd_9_col = _find_col_contains(cols, ["9 am", "spd"])
    spd_3_col = _find_col_contains(cols, ["3 pm", "spd"])
    s9 = pd.to_numeric(df[spd_9_col], errors="coerce") if spd_9_col else pd.Series([pd.NA] * len(df), index=df.index)
    s3 = pd.to_numeric(df[spd_3_col], errors="coerce") if spd_3_col else pd.Series([pd.NA] * len(df), index=df.index)
    fallback_wind = pd.concat([s9, s3], axis=1).mean(axis=1)

    if wind_col:
        gust = pd.to_numeric(df[wind_col], errors="coerce")
        out["wind_spd_kmh"] = gust.fillna(fallback_wind).values
    else:
        out["wind_spd_kmh"] = fallback_wind.values

    out = out.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates()
    return out

def normalize_weather(input_path: str, output_path: str, daily_hour: int = 18, station: str = "Canberra") -> pd.DataFrame:
    input_lower = str(input_path).lower()

    if input_lower.endswith((".html", ".htm")):
        try:
            html_text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            html_text = ""

        # Special-case BoM monthly daily observations pages.
        if "Daily Weather Observations" in html_text:
            out = parse_bom_daily_weather_html(input_path, daily_hour=daily_hour)
            write_csv(out, output_path)
            return out

        # Special-case the BoM live observations snapshot page.
        if "Latest Weather Observations for the Canberra Area" in html_text:
            out = parse_bom_observations_html(input_path, station=station)
            write_csv(out, output_path)
            return out

    df = read_any_table(input_path)
    if df.empty:
        raise ValueError("Weather input is empty.")
    df.columns = [str(c).strip() for c in df.columns]

    ts = build_datetime_series(df, daily_hour=daily_hour)
    out = pd.DataFrame({"datetime": ts})
    out = out.dropna(subset=["datetime"]).copy()

    temp_col = first_present(df.columns, WEATHER_TEMP_CANDIDATES)
    max_temp_col = first_present(df.columns, WEATHER_MAX_TEMP_CANDIDATES)
    min_temp_col = first_present(df.columns, WEATHER_MIN_TEMP_CANDIDATES)
    rain_col = first_present(df.columns, WEATHER_RAIN_CANDIDATES)
    wind_col = first_present(df.columns, WEATHER_WIND_CANDIDATES)

    if temp_col:
        out["air_temp"] = pd.to_numeric(df.loc[out.index, temp_col], errors="coerce")
    elif max_temp_col and min_temp_col:
        mx = pd.to_numeric(df.loc[out.index, max_temp_col], errors="coerce")
        mn = pd.to_numeric(df.loc[out.index, min_temp_col], errors="coerce")
        out["air_temp"] = (mx + mn) / 2.0
    else:
        out["air_temp"] = pd.NA

    if rain_col:
        out["rainfall"] = pd.to_numeric(df.loc[out.index, rain_col], errors="coerce").fillna(0)
    else:
        out["rainfall"] = 0.0

    if wind_col:
        out["wind_spd_kmh"] = pd.to_numeric(df.loc[out.index, wind_col], errors="coerce")
    else:
        out["wind_spd_kmh"] = pd.NA

    out = out.sort_values("datetime").drop_duplicates()
    write_csv(out, output_path)
    return out

def load_site_map(path: Optional[str]) -> list[tuple[str, str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    df = read_any_table(path)
    df.columns = [str(c).strip() for c in df.columns]
    patt_col = first_present(df.columns, ["pattern", "match", "contains", "regex"])
    site_col = first_present(df.columns, ["site", "target site", "target_site"])
    mode_col = first_present(df.columns, ["mode", "match mode", "match_mode"])
    if patt_col is None or site_col is None:
        raise ValueError("site_map must contain at least pattern and site columns")
    out: list[tuple[str, str, str]] = []
    for _, row in df.iterrows():
        pattern = str(row[patt_col]).strip()
        site = str(row[site_col]).strip()
        mode = str(row[mode_col]).strip().lower() if mode_col else "contains"
        if pattern and site:
            out.append((pattern, site, mode))
    return out


def apply_site_map(raw_site: str, mappings: list[tuple[str, str, str]]) -> str:
    text = str(raw_site)
    text_n = norm_text(text)
    for pattern, site, mode in mappings:
        if mode == "regex":
            if re.search(pattern, text, flags=re.I):
                return site
        elif mode == "exact":
            if norm_text(pattern) == text_n:
                return site
        else:
            if norm_text(pattern) in text_n:
                return site
    return text


def parse_event_html(input_path: str) -> pd.DataFrame:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is required for HTML event parsing. pip install beautifulsoup4")
    html = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict] = []

    # Best-effort parsing for Events Canberra / VisitCanberra cards.
    # Intentionally conservative: only extract obvious title/time/place text blocks.
    for card in soup.find_all(["article", "div", "li"]):
        text = " ".join(card.stripped_strings)
        if not text or len(text) < 20:
            continue

        title_tag = card.find(["h2", "h3", "h4"])
        title = title_tag.get_text(" ", strip=True) if title_tag else None

        date_match = re.search(
            r"(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})", text
        )
        time_match = re.search(
            r"from\s+(\d{1,2}:\d{2}(?:am|pm)?)\s+to\s+(\d{1,2}:\d{2}(?:am|pm)?)",
            text,
            flags=re.I,
        )

        # crude venue guess: first short non-title line with comma or suburb-like pattern
        location = None
        lines = [ln.strip() for ln in card.stripped_strings if ln.strip()]
        for ln in lines:
            if title and ln == title:
                continue
            if len(ln) <= 80 and ("," in ln or re.search(r"\b(Canberra|City|Braddon|Dickson|Gungahlin|Lyneham|Parkes|Kingston)\b", ln, flags=re.I)):
                location = ln
                break

        if title and (date_match or time_match or location):
            rec = {
                "title": title,
                "location": location,
                "date": date_match.group(1) if date_match else None,
                "start_time": time_match.group(1) if time_match else None,
                "end_time": time_match.group(2) if time_match else None,
            }
            records.append(rec)

    if not records:
        raise ValueError(f"Could not parse event cards from HTML: {input_path}")
    return pd.DataFrame(records).drop_duplicates()


def normalize_events(
    input_path: str,
    output_path: str,
    site_map_path: Optional[str] = None,
    default_attendees: int = 100,
    default_multiplier: float = 1.0,
    default_start_hour: int = 18,
    default_duration_hours: int = 2,
) -> pd.DataFrame:
    if Path(input_path).suffix.lower() in {".html", ".htm"}:
        df = parse_event_html(input_path)
    else:
        df = read_any_table(input_path)

    if df.empty:
        raise ValueError("Event input is empty.")
    df.columns = [str(c).strip() for c in df.columns]

    site_col = first_present(df.columns, EVENT_SITE_CANDIDATES)
    title_col = first_present(df.columns, EVENT_TITLE_CANDIDATES)
    attend_col = first_present(df.columns, EVENT_ATTEND_CANDIDATES)
    mult_col = first_present(df.columns, EVENT_MULT_CANDIDATES)
    date_col = first_present(df.columns, EVENT_DATE_CANDIDATES)
    start_col = first_present(df.columns, EVENT_START_CANDIDATES)
    end_col = first_present(df.columns, EVENT_END_CANDIDATES)

    mappings = load_site_map(site_map_path)

    out = pd.DataFrame()
    if site_col:
        out["site"] = df[site_col].astype(str)
    elif title_col:
        out["site"] = df[title_col].astype(str)
    else:
        raise ValueError("Events input needs a site/location/venue column or at least a title column.")

    out["site"] = out["site"].map(lambda s: apply_site_map(s, mappings))
    out["attendees"] = pd.to_numeric(df[attend_col], errors="coerce").fillna(default_attendees) if attend_col else default_attendees
    out["multiplier"] = pd.to_numeric(df[mult_col], errors="coerce").fillna(default_multiplier) if mult_col else default_multiplier

    if start_col:
        out["start_datetime"] = pd.to_datetime(df[start_col], errors="coerce")
    elif date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        out["start_datetime"] = dates + pd.to_timedelta(default_start_hour, unit="h")
    else:
        raise ValueError("Events input needs start_datetime or date.")

    if end_col:
        out["end_datetime"] = pd.to_datetime(df[end_col], errors="coerce")
    else:
        out["end_datetime"] = out["start_datetime"] + pd.to_timedelta(default_duration_hours, unit="h")

    if title_col and "title" not in out.columns:
        out["title"] = df[title_col].astype(str)

    out = out.dropna(subset=["start_datetime"]).copy()
    out = out.sort_values(["start_datetime", "site"]).drop_duplicates()
    write_csv(out[["site", "start_datetime", "end_datetime", "attendees", "multiplier"]], output_path)
    return out[["site", "start_datetime", "end_datetime", "attendees", "multiplier"]]


def normalize_ops(
    input_path: str,
    output_path: str,
    default_reliability: float = 0.85,
    default_delay_min: float = 8.0,
    default_start_hour: int = 17,
    default_duration_hours: int = 4,
) -> pd.DataFrame:
    df = read_any_table(input_path)
    if df.empty:
        raise ValueError("Ops input is empty.")
    df.columns = [str(c).strip() for c in df.columns]

    date_col = first_present(df.columns, OPS_DATE_CANDIDATES)
    hour_col = first_present(df.columns, OPS_HOUR_CANDIDATES)
    rel_col = first_present(df.columns, OPS_RELIAB_CANDIDATES)
    delay_col = first_present(df.columns, OPS_DELAY_CANDIDATES)
    dis_col = first_present(df.columns, OPS_DISRUPT_CANDIDATES)

    # Case 1: already hourly
    if date_col and hour_col:
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        out["hour"] = df[hour_col].map(maybe_int_hour)
        out["service_reliability"] = pd.to_numeric(df[rel_col], errors="coerce") if rel_col else default_reliability
        out["avg_delay_min"] = pd.to_numeric(df[delay_col], errors="coerce") if delay_col else default_delay_min
        if dis_col:
            raw = df[dis_col].astype(str).str.lower()
            out["is_disrupted"] = raw.isin(["1", "true", "yes", "y"])
        else:
            out["is_disrupted"] = (out["avg_delay_min"] >= 6) | (out["service_reliability"] < 0.90)
        out = out.dropna(subset=["date", "hour"]).copy()
        out["hour"] = out["hour"].astype(int)
        out = out.sort_values(["date", "hour"]).drop_duplicates()
        write_csv(out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]], output_path)
        return out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]]

    # Case 2: alert intervals; expand to hours
    start_col = first_present(df.columns, OPS_START_CANDIDATES)
    end_col = first_present(df.columns, OPS_END_CANDIDATES)

    if not start_col and date_col:
        starts = pd.to_datetime(df[date_col], errors="coerce") + pd.to_timedelta(default_start_hour, unit="h")
        ends = starts + pd.to_timedelta(default_duration_hours, unit="h")
    elif start_col:
        starts = pd.to_datetime(df[start_col], errors="coerce")
        if end_col:
            ends = pd.to_datetime(df[end_col], errors="coerce")
        else:
            ends = starts + pd.to_timedelta(default_duration_hours, unit="h")
    else:
        raise ValueError("Ops input needs either date+hour columns or start/end datetime columns.")

    rel_series = pd.to_numeric(df[rel_col], errors="coerce").fillna(default_reliability) if rel_col else pd.Series([default_reliability] * len(df))
    delay_series = pd.to_numeric(df[delay_col], errors="coerce").fillna(default_delay_min) if delay_col else pd.Series([default_delay_min] * len(df))
    if dis_col:
        raw = df[dis_col].astype(str).str.lower()
        dis_series = raw.isin(["1", "true", "yes", "y"])
    else:
        dis_series = pd.Series([True] * len(df))

    rows: list[dict] = []
    for start, end, rel, delay, disrupted in zip(starts, ends, rel_series, delay_series, dis_series):
        if pd.isna(start):
            continue
        if pd.isna(end) or end <= start:
            end = start + pd.Timedelta(hours=default_duration_hours)
        current = pd.Timestamp(start).floor("h")
        stop = pd.Timestamp(end).ceil("h")
        while current < stop:
            rows.append({
                "date": current.normalize(),
                "hour": int(current.hour),
                "service_reliability": float(rel),
                "avg_delay_min": float(delay),
                "is_disrupted": bool(disrupted),
            })
            current += pd.Timedelta(hours=1)

    out = pd.DataFrame(rows).drop_duplicates()
    out = out.sort_values(["date", "hour"])
    write_csv(out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]], output_path)
    return out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]]


def write_templates(outdir: str) -> None:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)

    weather_rows = [
        ["datetime", "air_temp", "rainfall", "wind_spd_kmh"],
        ["2026-03-01T17:00:00", "18.2", "0.0", "14"],
        ["2026-03-01T18:00:00", "16.9", "0.4", "17"],
    ]
    events_rows = [
        ["site", "start_datetime", "end_datetime", "attendees", "multiplier"],
        ["Alinga Street", "2026-03-14T18:00:00", "2026-03-14T20:00:00", "300", "1.2"],
        ["Dickson", "2026-03-20T17:30:00", "2026-03-20T19:30:00", "120", "1.0"],
    ]
    ops_rows = [
        ["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"],
        ["2026-03-01", "17", "0.96", "1.8", "false"],
        ["2026-03-01", "18", "0.82", "7.5", "true"],
    ]
    site_map_rows = [
        ["pattern", "site", "mode"],
        ["civic", "Alinga Street", "contains"],
        ["canberra city", "Alinga Street", "contains"],
        ["dickson", "Dickson", "contains"],
        ["gungahlin", "Gungahlin Place", "contains"],
    ]

    callouts_rows = [
        ["start_datetime", "service_type", "incident_type", "severity", "suburb", "lat", "lon", "duration_min", "public_access_impact", "transport_impact", "site", "responder_uplift_cups"],
        ["2026-04-02T18:05:00", "ambulance", "AMBULANCE RESPONSE - DICKSON", "low", "Dickson", "", "", "30", "0", "0", "Dickson", "1.0"],
        ["2026-04-02T19:10:00", "fire", "STRUCTURE FIRE - CITY", "high", "City", "", "", "90", "1", "1", "Alinga Street", "2.5"],
    ]
    for filename, rows in [
        ("weather_template.csv", weather_rows),
        ("events_template.csv", events_rows),
        ("ops_template.csv", ops_rows),
        ("site_map_template.csv", site_map_rows),
        ("callouts_template.csv", callouts_rows),
    ]:
        with open(path / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)




def month_iter(latest: pd.Timestamp, months: int) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    current = latest.replace(day=1)
    for _ in range(months):
        out.append(current)
        current = (current - pd.offsets.MonthBegin(1)).replace(day=1)
    return out


def bom_monthly_html_url(station_code: str, month_ts: pd.Timestamp) -> str:
    yyyymm = month_ts.strftime("%Y%m")
    return f"https://www.bom.gov.au/climate/dwo/{yyyymm}/html/{station_code}.{yyyymm}.shtml"


def fetch_weather_history(
    output_path: str,
    station_code: str = "IDCJDW2801",
    months: int = 14,
    daily_hour: int = 18,
    latest_month: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download the last N monthly BoM daily-observation HTML pages and concatenate
    them into canonical weather.csv format.

    This intentionally uses the monthly HTML archive rather than the 'plain text'
    CSV links, because the HTML pages are consistently accessible and we already
    have a dedicated parser for them.
    """
    if requests is None:
        raise RuntimeError("requests is required for fetch-weather-history. Install with: pip install requests")

    if latest_month:
        latest_ts = pd.to_datetime("1 " + latest_month, errors="coerce", dayfirst=True)
    else:
        latest_ts = pd.Timestamp.now().normalize().replace(day=1)

    if pd.isna(latest_ts):
        raise ValueError("Could not parse latest_month. Use format like 'Apr 2026' or 'April 2026'.")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for month_ts in month_iter(latest_ts, months):
        url = bom_monthly_html_url(station_code, month_ts)
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

            # Parse from in-memory HTML instead of requiring a saved file.
            html_text = resp.text
            tmp_path = Path(output_path).with_suffix(f".{month_ts.strftime('%Y%m')}.tmp.html")
            tmp_path.write_text(html_text, encoding="utf-8")
            try:
                out = parse_bom_daily_weather_html(str(tmp_path), daily_hour=daily_hour)
            finally:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

            if out.empty:
                raise ValueError("parsed zero rows from monthly HTML")
            frames.append(out)
        except Exception:
            failed.append(url)

    if not frames:
        raise RuntimeError(
            "No BoM monthly HTML pages were parsed successfully. "
            "Check network access, station_code, or latest_month."
        )

    final = (
        pd.concat(frames, ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .reset_index(drop=True)
    )
    write_csv(final, output_path)

    if failed:
        sys.stderr.write("Skipped monthly files:\n" + "\n".join(failed[:20]) + "\n")
    return final



# ---------------------------------------------------------------------------
# Callouts parser (ACT ESA GeoRSS / CAP feeds)
# ---------------------------------------------------------------------------

ESA_CURRENT_INCIDENTS_URL = "https://esa.act.gov.au/feeds/currentincidents.xml"

# ── Addinsight Bluetooth detector network (real-time road stats) ──────────
ADDINSIGHT_BASE = "http://data.addinsight.com/ACT"
ADDINSIGHT_LINKS_URL = f"{ADDINSIGHT_BASE}/links_prop_stats_geo.json"
ADDINSIGHT_ROUTES_URL = f"{ADDINSIGHT_BASE}/routes_prop_stats_geo.json"
ADDINSIGHT_SITE_COORDS = {
    "Alinga Street":   {"lat": -35.2784, "lon": 149.1305,
                        "road_kw": ["northbourne", "london circuit", "cooyong", "alinga", "bunda", "east row"]},
    "Dickson":         {"lat": -35.2504, "lon": 149.1413,
                        "road_kw": ["northbourne", "cowper", "antill", "badham", "woolley"]},
    "Gungahlin Place": {"lat": -35.1853, "lon": 149.1330,
                        "road_kw": ["hibberson", "gungahlin place", "flemington", "gozzard"]},
}
ADDINSIGHT_RADIUS_M = 1500
ESA_CAP_INCIDENTS_URL = "https://data.esa.act.gov.au/feeds/esa-cap-incidents.xml"


def infer_callout_service(title: str, event: str = "") -> str:
    txt = f"{title} {event}".lower()
    if "ambulance" in txt:
        return "ambulance"
    if "fire" in txt or "bushfire" in txt or "hazard reduction" in txt:
        return "fire"
    if "ses" in txt or "storm" in txt or "flood" in txt:
        return "ses"
    if "rfs" in txt:
        return "rfs"
    return "other"


def infer_callout_severity(title: str, event: str = "", cap_severity: Optional[str] = None) -> str:
    if cap_severity:
        sev = str(cap_severity).strip().lower()
        if sev in {"extreme", "severe", "minor", "moderate"}:
            return "high" if sev in {"extreme", "severe"} else "medium"
    txt = f"{title} {event}".lower()
    if any(tok in txt for tok in ["house fire", "structure fire", "bushfire", "grass fire", "hazmat", "hazardous", "person trapped", "rescue"]):
        return "high"
    if any(tok in txt for tok in ["motor vehicle incident", "vehicle fire", "storm damage", "road closure", "diversion"]):
        return "medium"
    if "ambulance response" in txt:
        return "low"
    return "low"


def infer_callout_duration_min(service_type: str, severity: str, default_duration_min: int = 30) -> int:
    base = {
        "ambulance": 30,
        "fire": 90,
        "ses": 60,
        "rfs": 90,
        "other": default_duration_min,
    }.get(service_type, default_duration_min)
    mult = {"low": 1.0, "medium": 1.5, "high": 2.0}.get(severity, 1.0)
    return int(round(base * mult))


def infer_public_access_impact(title: str, description: str = "") -> int:
    txt = f"{title} {description}".lower()
    return int(any(tok in txt for tok in [
        "fire", "smoke", "hazmat", "road closure", "lane blocked", "blocked", "diversion", "cordon", "vehicle incident"
    ]))


def infer_transport_impact(title: str, description: str = "") -> int:
    txt = f"{title} {description}".lower()
    return int(any(tok in txt for tok in [
        "northbourne", "light rail", "tram", "commonwealth avenue", "road closure", "lane blocked", "diversion", "bus stop", "interchange", "rail"
    ]))


def infer_responder_uplift_cups(service_type: str, severity: str,
    ambulance_uplift: float, fire_uplift: float,
    ses_uplift: float, other_uplift: float) -> float:
    base = {
        "ambulance": ambulance_uplift,
        "fire": fire_uplift,
        "ses": ses_uplift,
        "rfs": ses_uplift,
        "other": other_uplift,
    }.get(service_type, other_uplift)
    mult = {"low": 1.0, "medium": 1.2, "high": 1.5}.get(severity, 1.0)
    return float(base * mult)


def infer_suburb_from_title(title: str, area_desc: str = "") -> str:
    if area_desc:
        return str(area_desc).strip()
    title = str(title).strip()
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip().title()
    return ""




def parse_esa_datetime(text: str) -> pd.Timestamp:
    s = str(text or "").strip()
    if not s:
        return pd.NaT

    # Handle strings like "2026-04-02 20:38 AEDT" or "2026-04-02 20:39 AEST"
    m = re.match(r"^(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s+[A-Z]{3,4}$", s)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")

    # Handle plain ISO-like timestamps without dayfirst.
    if re.match(r"^\d{4}-\d{2}-\d{2}\b", s):
        return pd.to_datetime(s, errors="coerce")

    # Handle ACT ESA description timestamps, e.g. "02 Apr 2026 19:48:01.957"
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def extract_esa_description_field(description: str, field_name: str) -> str:
    desc = " ".join(str(description or "").replace("\r", " ").replace("\n", " ").split())
    if not desc:
        return ""
    # Match from "Field:" until the next "Label:" or end of string.
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?=\s+[A-Za-z][A-Za-z ]{{1,30}}:\s|$)"
    m = re.search(pattern, desc, flags=re.I)
    return m.group(1).strip() if m else ""


def apply_site_map_or_blank(raw_site: str, mappings: list[tuple[str, str, str]]) -> str:
    if not mappings:
        return str(raw_site).strip()
    text_n = norm_text(raw_site)
    for pattern, site, mode in mappings:
        patt_n = norm_text(pattern)
        if mode == "regex":
            if re.search(pattern, raw_site, flags=re.I):
                return site
        elif mode == "exact":
            if patt_n == text_n:
                return site
        else:
            if patt_n and patt_n in text_n:
                return site
    return ""
def parse_esa_callouts_xml(input_path: str) -> pd.DataFrame:
    import xml.etree.ElementTree as ET

    def local_name(tag: object) -> str:
        s = str(tag)
        if "}" in s:
            s = s.rsplit("}", 1)[-1]
        if ":" in s:
            s = s.rsplit(":", 1)[-1]
        return s.strip().lower()

    def first_desc_text(node, names: list[str]) -> str:
        wanted = {n.lower() for n in names}
        for el in node.iter():
            if local_name(el.tag) in wanted:
                txt = " ".join("".join(el.itertext()).split()).strip()
                if txt:
                    return txt
        return ""

    def parse_point_text(point_text: str) -> tuple[float | None, float | None]:
        if not point_text:
            return None, None
        parts = point_text.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                return None, None
        return None, None

    def parse_html_fallback(path: str) -> pd.DataFrame:
        try:
            lines = extract_visible_lines(path)
        except Exception:
            return pd.DataFrame()

        records: list[dict] = []
        pending_dt = None
        pending_kind = None

        for ln in lines:
            txt = str(ln).strip()
            if not txt:
                continue

            m = re.match(r"^(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}\s+\d{1,2}:\d{2}:\d{2})\s+(.+)$", txt)
            if m:
                pending_dt = parse_esa_datetime(m.group(1))
                pending_kind = m.group(2).strip()
                continue

            if pending_dt is not None and " - " in txt:
                title = txt
                suburb = infer_suburb_from_title(title, "")
                records.append({
                    "start_datetime": pending_dt,
                    "title": title,
                    "description": "",
                    "service_type": infer_callout_service(title, pending_kind or ""),
                    "incident_type": pending_kind or title,
                    "severity": infer_callout_severity(title, pending_kind or "", None),
                    "suburb": suburb,
                    "lat": None,
                    "lon": None,
                    "public_access_impact": infer_public_access_impact(title, ""),
                    "transport_impact": infer_transport_impact(title, ""),
                })
                pending_dt = None
                pending_kind = None

        return pd.DataFrame(records).drop_duplicates()

    text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return pd.DataFrame()

    lower_text = text.lower()
    if "<html" in lower_text or "incident updates are currently unavailable" in lower_text:
        return parse_html_fallback(input_path)

    try:
        tree = ET.parse(input_path)
    except Exception:
        return parse_html_fallback(input_path)

    root = tree.getroot()
    nodes = [el for el in root.iter() if local_name(el.tag) in {"item", "entry"}]
    records: list[dict] = []

    for node in nodes:
        title = first_desc_text(node, ["title", "headline"])
        description = first_desc_text(node, ["description", "summary", "content"])
        pub_date = first_desc_text(node, ["pubdate", "updated", "published", "issued", "sent", "effective", "date"])
        event = first_desc_text(node, ["event", "eventtype", "category", "type"])
        cap_severity = first_desc_text(node, ["severity"])
        area_desc = first_desc_text(node, ["areadesc", "area", "location"])
        point = first_desc_text(node, ["point", "pos", "coordinates"])
        agency = first_desc_text(node, ["agency"])
        status = first_desc_text(node, ["status"])

        lat, lon = parse_point_text(point)

        # Prefer the incident's time-of-call from description, then updated, then pubDate.
        time_of_call = extract_esa_description_field(description, "Time of Call")
        updated_desc = extract_esa_description_field(description, "Updated")
        suburb_desc = extract_esa_description_field(description, "Suburb")
        type_desc = extract_esa_description_field(description, "Type")
        agency_desc = extract_esa_description_field(description, "Agency")

        dt = parse_esa_datetime(time_of_call) if time_of_call else pd.NaT
        if pd.isna(dt) and updated_desc:
            dt = parse_esa_datetime(updated_desc)
        if pd.isna(dt):
            dt = parse_esa_datetime(pub_date)
        if pd.isna(dt):
            continue

        incident_type = type_desc or event or title
        agency_final = agency_desc or agency
        suburb = suburb_desc or infer_suburb_from_title(title, area_desc)
        service_type = infer_callout_service(title, agency_final or incident_type)
        severity = infer_callout_severity(title, incident_type, cap_severity or status)

        records.append({
            "start_datetime": dt,
            "title": title,
            "description": description,
            "service_type": service_type,
            "incident_type": incident_type,
            "severity": severity,
            "suburb": suburb,
            "lat": lat,
            "lon": lon,
            "public_access_impact": infer_public_access_impact(title, description),
            "transport_impact": infer_transport_impact(title, description),
        })

    return pd.DataFrame(records).drop_duplicates()

def normalize_callouts(
    input_path: str,
    output_path: str,
    site_map_path: Optional[str] = None,
    default_duration_min: int = 30,
    ambulance_uplift: float = 1.0,
    fire_uplift: float = 2.5,
    ses_uplift: float = 1.0,
    other_uplift: float = 0.5,
) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, ESA_CURRENT_INCIDENTS_URL, ".xml")
    try:
        suffix = Path(materialized_input).suffix.lower()
        if suffix in {".xml", ".rss", ".atom"}:
            df = parse_esa_callouts_xml(materialized_input)
        else:
            df = read_any_table(materialized_input)
            if df.empty:
                df = pd.DataFrame()
            else:
                df.columns = [str(c).strip() for c in df.columns]

        canonical_cols = [
            "start_datetime",
            "service_type",
            "incident_type",
            "severity",
            "suburb",
            "lat",
            "lon",
            "duration_min",
            "public_access_impact",
            "transport_impact",
            "site",
            "responder_uplift_cups",
        ]
        if df.empty:
            out = pd.DataFrame(columns=canonical_cols)
            write_csv(out, output_path)
            return out

        mappings = load_site_map(site_map_path)

        if "start_datetime" not in df.columns:
            time_col = first_present(df.columns, ["start_datetime", "datetime", "timestamp", "time", "updated", "published"])
            if not time_col:
                raise ValueError("Callouts input needs a datetime/start_datetime field.")
            df["start_datetime"] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)

        if "service_type" not in df.columns:
            title_col = first_present(df.columns, ["title", "incident", "event"])
            df["service_type"] = df[title_col].astype(str).map(lambda s: infer_callout_service(s, "")) if title_col else "other"

        if "incident_type" not in df.columns:
            title_col = first_present(df.columns, ["title", "incident", "event"])
            df["incident_type"] = df[title_col].astype(str) if title_col else df["service_type"].astype(str)

        if "severity" not in df.columns:
            title_col = first_present(df.columns, ["title", "incident", "event"])
            df["severity"] = df[title_col].astype(str).map(lambda s: infer_callout_severity(s, "")) if title_col else "low"

        if "suburb" not in df.columns:
            suburb_col = first_present(df.columns, ["suburb", "area", "location"])
            if suburb_col:
                df["suburb"] = df[suburb_col].astype(str)
            else:
                title_col = first_present(df.columns, ["title", "incident", "event"])
                df["suburb"] = df[title_col].astype(str).map(lambda s: infer_suburb_from_title(s, "")) if title_col else ""

        if "duration_min" not in df.columns:
            df["duration_min"] = [
                infer_callout_duration_min(svc, sev, default_duration_min=default_duration_min)
                for svc, sev in zip(df["service_type"], df["severity"])
            ]

        if "public_access_impact" not in df.columns:
            title_col = first_present(df.columns, ["title", "incident", "event"])
            desc_col = first_present(df.columns, ["description", "summary"])
            df["public_access_impact"] = [
                infer_public_access_impact(
                    df[title_col].iloc[i] if title_col else df["incident_type"].iloc[i],
                    df[desc_col].iloc[i] if desc_col else "",
                )
                for i in range(len(df))
            ]

        if "transport_impact" not in df.columns:
            title_col = first_present(df.columns, ["title", "incident", "event"])
            desc_col = first_present(df.columns, ["description", "summary"])
            df["transport_impact"] = [
                infer_transport_impact(
                    df[title_col].iloc[i] if title_col else df["incident_type"].iloc[i],
                    df[desc_col].iloc[i] if desc_col else "",
                )
                for i in range(len(df))
            ]

        if "responder_uplift_cups" not in df.columns:
            df["responder_uplift_cups"] = [
                infer_responder_uplift_cups(svc, sev, ambulance_uplift, fire_uplift, ses_uplift, other_uplift)
                for svc, sev in zip(df["service_type"], df["severity"])
            ]

        map_source = (
            df["suburb"].fillna("").astype(str).str.strip()
            + " "
            + df["incident_type"].fillna("").astype(str).str.strip()
        ).str.strip()
        if mappings:
            site_series = map_source.map(lambda s: apply_site_map_or_blank(s, mappings))
        else:
            site_series = df["suburb"].astype(str).str.strip()

        out = pd.DataFrame({
            "start_datetime": pd.to_datetime(df["start_datetime"], errors="coerce"),
            "service_type": df["service_type"].astype(str).str.lower(),
            "incident_type": df["incident_type"].astype(str),
            "severity": df["severity"].astype(str).str.lower(),
            "suburb": df["suburb"].astype(str),
            "lat": pd.to_numeric(df["lat"], errors="coerce") if "lat" in df.columns else pd.NA,
            "lon": pd.to_numeric(df["lon"], errors="coerce") if "lon" in df.columns else pd.NA,
            "duration_min": pd.to_numeric(df["duration_min"], errors="coerce").fillna(default_duration_min),
            "public_access_impact": pd.to_numeric(df["public_access_impact"], errors="coerce").fillna(0).astype(int),
            "transport_impact": pd.to_numeric(df["transport_impact"], errors="coerce").fillna(0).astype(int),
            "site": site_series.astype(str).str.strip(),
            "responder_uplift_cups": pd.to_numeric(df["responder_uplift_cups"], errors="coerce").fillna(0.0),
        })
        out = out.dropna(subset=["start_datetime"]).sort_values("start_datetime").drop_duplicates()
        if mappings:
            out = out[out["site"] != ""].copy()
        write_csv(out, output_path)
        return out
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


def fetch_default_events(output_path: str) -> None:
    fetch_url(DEFAULT_EVENTS_URL, output_path)


def fetch_default_ops(output_path: str) -> None:
    fetch_url(DEFAULT_ALERTS_URL, output_path)


def fetch_default_callouts(output_path: str) -> None:
    fetch_url(ESA_CURRENT_INCIDENTS_URL, output_path)


# ── Addinsight real-time road network ─────────────────────────────────────

import math as _math

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    rlat1, rlat2 = _math.radians(lat1), _math.radians(lat2)
    dlat, dlon = _math.radians(lat2 - lat1), _math.radians(lon2 - lon1)
    a = _math.sin(dlat/2)**2 + _math.cos(rlat1)*_math.cos(rlat2)*_math.sin(dlon/2)**2
    return R * 2 * _math.atan2(_math.sqrt(a), _math.sqrt(1 - a))


def _centroid(coords: list) -> tuple:
    if not coords:
        return (0.0, 0.0)
    return (sum(c[1] for c in coords)/len(coords),
            sum(c[0] for c in coords)/len(coords))


def fetch_addinsight(output_dir: str, radius: int = ADDINSIGHT_RADIUS_M) -> None:
    """Fetch Addinsight link + route snapshots and write raw JSON to output_dir."""
    if requests is None:
        raise RuntimeError("requests required")
    od = Path(output_dir); od.mkdir(parents=True, exist_ok=True)
    for label, url, fname in [
        ("links", ADDINSIGHT_LINKS_URL, "raw_addinsight_links.json"),
        ("routes", ADDINSIGHT_ROUTES_URL, "raw_addinsight_routes.json"),
    ]:
        print(f"  Fetching Addinsight {label} ...")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            (od / fname).write_text(resp.text, encoding="utf-8")
            data = resp.json()
            n = len(data.get("features", [])) if isinstance(data, dict) else len(data)
            print(f"    -> {n} features -> {od/fname}")
        except Exception as e:
            print(f"    -> ERROR: {e} (will use corridor priors)")


def normalize_addinsight(
    input_dir: str,
    output_path: str,
    radius: int = ADDINSIGHT_RADIUS_M,
) -> pd.DataFrame:
    """Parse raw Addinsight JSON into per-site corridor summary CSV.

    Produces addinsight_corridor_summary.csv with columns:
        site, n_links, total_corridor_length_m, avg_speed_kmh, avg_delay_s,
        avg_excess_delay_s, max_score, n_congested_links, n_closed_links,
        is_disrupted, pct_enough_data
    """
    import json as _json
    ind = Path(input_dir)
    links_path = ind / "raw_addinsight_links.json"
    if not links_path.exists():
        print(f"  No Addinsight links file at {links_path} — skipping")
        return pd.DataFrame()

    with open(links_path) as f:
        geojson = _json.load(f)

    # Flatten features
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        clat, clon = _centroid(coords) if coords else (None, None)
        rows.append({
            "link_id": props.get("Id"), "name": (props.get("Name") or "").lower(),
            "length_m": props.get("Length", 0), "direction": props.get("Direction", ""),
            "min_tt_s": props.get("MinTT"), "tt_s": props.get("TT"),
            "delay_s": props.get("Delay"), "speed_kmh": props.get("Speed"),
            "excess_delay_s": props.get("ExcessDelay"), "congestion": props.get("Congestion"),
            "score": props.get("Score", 0), "enough_data": props.get("EnoughData", False),
            "closed": props.get("Closed", False), "is_freeway": props.get("IsFreeway", False),
            "interval_start": props.get("IntervalStart"),
            "centroid_lat": clat, "centroid_lon": clon,
        })
    links = pd.DataFrame(rows)
    if links.empty:
        print("  Addinsight links GeoJSON contained no features")
        return pd.DataFrame()
    print(f"  Parsed {len(links)} links from Addinsight")

    # Match links to sites by proximity + keyword
    summaries = []
    for site, cfg in ADDINSIGHT_SITE_COORDS.items():
        slat, slon = cfg["lat"], cfg["lon"]
        dists = links.apply(
            lambda r: _haversine_m(slat, slon, r["centroid_lat"], r["centroid_lon"])
            if pd.notna(r["centroid_lat"]) else float("inf"), axis=1)
        within = dists <= radius
        kw_match = pd.Series(False, index=links.index)
        for kw in cfg["road_kw"]:
            kw_match |= links["name"].str.contains(kw, regex=False, na=False)
        matched = links[within | kw_match]
        if matched.empty:
            print(f"    {site}: 0 links matched")
            summaries.append({"site": site, "n_links": 0, "is_disrupted": False})
            continue

        lengths = matched["length_m"].fillna(0).astype(float)
        total_len = lengths.sum()
        def _wmean(col):
            v = pd.to_numeric(matched[col], errors="coerce")
            ok = v.notna() & (lengths > 0)
            return float((v[ok] * lengths[ok]).sum() / lengths[ok].sum()) if ok.any() else None

        n_closed = int(matched["closed"].sum())
        n_cong = int((matched["score"].fillna(0) >= 3).sum())
        summaries.append({
            "site": site,
            "n_links": len(matched),
            "total_corridor_length_m": int(total_len),
            "avg_speed_kmh": round(_wmean("speed_kmh") or 0, 1),
            "avg_delay_s": round(_wmean("delay_s") or 0, 1),
            "avg_excess_delay_s": round(_wmean("excess_delay_s") or 0, 1),
            "max_score": int(matched["score"].max()) if matched["score"].notna().any() else 0,
            "n_congested_links": n_cong,
            "n_closed_links": n_closed,
            "is_disrupted": n_closed > 0 or n_cong >= 2,
            "pct_enough_data": round(100 * matched["enough_data"].sum() / len(matched), 1),
        })
        print(f"    {site}: {len(matched)} links, "
              f"avg {_wmean('speed_kmh') or 0:.0f} km/h, "
              f"disrupted={n_closed > 0 or n_cong >= 2}")

    out = pd.DataFrame(summaries)
    write_csv(out, output_path)
    return out


def parse_events(
    input_path: Optional[str],
    output_path: str,
) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, DEFAULT_EVENTS_URL, ".html")
    try:
        input_suffix = Path(materialized_input).suffix.lower()
        if input_suffix in {".html", ".htm"}:
            html = Path(materialized_input).read_text(encoding="utf-8", errors="ignore")
            if "What's on in Canberra" in html or "Major Events" in html:
                df = parse_events_canberra_html(materialized_input)
            else:
                df = parse_event_html(materialized_input)
        else:
            df = read_any_table(materialized_input)
        if df.empty:
            raise ValueError("Event input is empty.")
        df.columns = [str(c).strip() for c in df.columns]
        if "title" not in df.columns:
            title_col = first_present(df.columns, EVENT_TITLE_CANDIDATES)
            if title_col:
                df["title"] = df[title_col].astype(str)
        if "location" not in df.columns:
            location_col = first_present(df.columns, ["location", "venue", "site", "stop", "stop name"])
            df["location"] = df[location_col].astype(str) if location_col else ""
        if "start_datetime" not in df.columns:
            start_col = first_present(df.columns, EVENT_START_CANDIDATES)
            date_col = first_present(df.columns, EVENT_DATE_CANDIDATES)
            if start_col:
                df["start_datetime"] = pd.to_datetime(df[start_col], errors="coerce")
            elif date_col:
                base = pd.to_datetime(df[date_col], errors="coerce")
                df["start_datetime"] = base + pd.to_timedelta(18, unit="h")
            else:
                df["start_datetime"] = pd.NaT
        if "end_datetime" not in df.columns:
            end_col = first_present(df.columns, EVENT_END_CANDIDATES)
            if end_col:
                df["end_datetime"] = pd.to_datetime(df[end_col], errors="coerce")
            else:
                df["end_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce") + pd.to_timedelta(2, unit="h")
        out = pd.DataFrame({
            "title": df.get("title", "").astype(str),
            "location": df.get("location", "").astype(str),
            "start_datetime": pd.to_datetime(df.get("start_datetime"), errors="coerce"),
            "end_datetime": pd.to_datetime(df.get("end_datetime"), errors="coerce"),
        })
        out = out.dropna(subset=["start_datetime"]).sort_values(["start_datetime", "title"]).drop_duplicates()
        write_csv(out, output_path)
        return out
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


def parse_ops(
    input_path: Optional[str],
    output_path: str,
    default_start_hour: int = 17,
    default_duration_hours: int = 4,
    further_notice_days: int = 30,
) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, DEFAULT_ALERTS_URL, ".html")
    try:
        input_suffix = Path(materialized_input).suffix.lower()
        if input_suffix in {".html", ".htm"}:
            html = Path(materialized_input).read_text(encoding="utf-8", errors="ignore")
            if "Service alerts and updates" in html:
                df = parse_transport_alerts_html(
                    materialized_input,
                    default_start_hour=default_start_hour,
                    default_duration_hours=default_duration_hours,
                    further_notice_days=further_notice_days,
                )
            else:
                df = read_any_table(materialized_input)
        else:
            df = read_any_table(materialized_input)
        if df.empty:
            raise ValueError("Ops input is empty.")
        write_csv(df, output_path)
        return df
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


def parse_callouts(input_path: Optional[str], output_path: str) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, ESA_CURRENT_INCIDENTS_URL, ".xml")
    try:
        suffix = Path(materialized_input).suffix.lower()
        if suffix in {".xml", ".rss", ".atom", ".html", ".htm"}:
            df = parse_esa_callouts_xml(materialized_input)
        else:
            df = read_any_table(materialized_input)
        if df.empty:
            raise ValueError("Callouts input is empty.")
        write_csv(df, output_path)
        return df
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build canonical weather/events/ops files for the Canberra demand model.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("templates", help="Write CSV templates.")
    p.add_argument("--outdir", default="support_templates")

    p = sub.add_parser("fetch", help="Fetch a raw URL to a local file.")
    p.add_argument("--url", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("fetch-events", help="Fetch the default Canberra events page.")
    p.add_argument("--output", default="raw_events.html")

    p = sub.add_parser("fetch-ops", help="Fetch the default Transport Canberra alerts page.")
    p.add_argument("--output", default="raw_ops.html")

    p = sub.add_parser("fetch-callouts", help="Fetch the default ACT ESA current incidents feed.")
    p.add_argument("--output", default="esa_current_incidents.xml")

    p = sub.add_parser("normalize-weather", help="Normalize raw weather data to canonical weather.csv.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="weather.csv")
    p.add_argument("--daily-hour", type=int, default=18, help="Hour to assign when input is daily-only.")
    p.add_argument("--station", default="Canberra", help="Station name for BoM live observations HTML parsing.")

    p = sub.add_parser("fetch-weather-history", help="Download and concatenate BoM monthly daily-weather CSVs.")
    p.add_argument("--output", default="weather.csv")
    p.add_argument("--station-code", default="IDCJDW2801")
    p.add_argument("--months", type=int, default=14)
    p.add_argument("--daily-hour", type=int, default=18)
    p.add_argument("--latest-month", default=None, help="Optional anchor month like 'Apr 2026'.")

    p = sub.add_parser("parse-events", help="Parse raw Canberra events HTML or exports into a structured intermediate CSV.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="parsed_events.csv")

    p = sub.add_parser("normalize-events", help="Normalize raw or parsed events data to canonical events.csv.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="events.csv")
    p.add_argument("--site-map", default=None, help="Optional CSV with pattern,site,mode to map venues to modeled sites.")
    p.add_argument("--event-overrides", default=None, help="Optional CSV with pattern,attendees,multiplier,mode to override event estimates.")
    p.add_argument("--default-attendees", type=int, default=100)
    p.add_argument("--default-multiplier", type=float, default=1.0)
    p.add_argument("--default-start-hour", type=int, default=18)
    p.add_argument("--default-duration-hours", type=int, default=2)

    p = sub.add_parser("parse-ops", help="Parse raw Transport Canberra alerts into a structured intermediate CSV.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="parsed_ops.csv")
    p.add_argument("--default-start-hour", type=int, default=17)
    p.add_argument("--default-duration-hours", type=int, default=4)
    p.add_argument("--further-notice-days", type=int, default=30)

    p = sub.add_parser("normalize-ops", help="Normalize raw or parsed ops/alerts data to canonical ops.csv.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="ops.csv")
    p.add_argument("--default-reliability", type=float, default=0.85)
    p.add_argument("--default-delay-min", type=float, default=8.0)
    p.add_argument("--default-start-hour", type=int, default=17)
    p.add_argument("--default-duration-hours", type=int, default=4)
    p.add_argument("--further-notice-days", type=int, default=30)
    p.add_argument("--include-regions", default=None, help="Comma-separated region filters for HTML alerts, e.g. 'Central Canberra,Gungahlin'.")
    p.add_argument("--include-keywords", default=None, help="Comma-separated title keyword filters for HTML alerts.")

    p = sub.add_parser("parse-callouts", help="Parse raw ESA XML or HTML into a structured intermediate CSV.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="parsed_callouts.csv")

    p = sub.add_parser("normalize-callouts", help="Normalize raw or parsed ACT ESA callouts data to canonical callouts.csv.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default="callouts.csv")
    p.add_argument("--site-map", default=None, help="Optional CSV with pattern,site,mode to map suburbs/incidents to modeled sites.")
    p.add_argument("--default-duration-min", type=int, default=30)
    p.add_argument("--ambulance-uplift", type=float, default=1.0)
    p.add_argument("--fire-uplift", type=float, default=2.5)
    p.add_argument("--ses-uplift", type=float, default=1.0)
    p.add_argument("--other-uplift", type=float, default=0.5)

    p = sub.add_parser("fetch-addinsight", help="Fetch real-time Addinsight road network data.")
    p.add_argument("--outdir", required=True, help="Directory for raw JSON files")
    p.add_argument("--radius", type=int, default=ADDINSIGHT_RADIUS_M)

    p = sub.add_parser("normalize-addinsight", help="Normalize raw Addinsight JSON to corridor summary CSV.")
    p.add_argument("--input-dir", required=True, help="Directory containing raw_addinsight_links.json")
    p.add_argument("--output", default="addinsight_corridor_summary.csv")
    p.add_argument("--radius", type=int, default=ADDINSIGHT_RADIUS_M)

    return ap

def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    if args.command == "templates":
        write_templates(args.outdir)
        print(f"Wrote templates to {args.outdir}")
        return

    if args.command == "fetch":
        fetch_url(args.url, args.output)
        print(f"Fetched {args.url} -> {args.output}")
        return

    if args.command == "fetch-events":
        fetch_default_events(args.output)
        print(f"Fetched {DEFAULT_EVENTS_URL} -> {args.output}")
        return

    if args.command == "fetch-ops":
        fetch_default_ops(args.output)
        print(f"Fetched {DEFAULT_ALERTS_URL} -> {args.output}")
        return

    if args.command == "fetch-callouts":
        fetch_default_callouts(args.output)
        print(f"Fetched {ESA_CURRENT_INCIDENTS_URL} -> {args.output}")
        return

    if args.command == "fetch-addinsight":
        fetch_addinsight(args.outdir, radius=args.radius)
        return

    if args.command == "normalize-addinsight":
        df = normalize_addinsight(args.input_dir, args.output, radius=args.radius)
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "normalize-weather":
        df = normalize_weather(args.input, args.output, daily_hour=args.daily_hour, station=args.station)
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "fetch-weather-history":
        df = fetch_weather_history(
            args.output,
            station_code=args.station_code,
            months=args.months,
            daily_hour=args.daily_hour,
            latest_month=args.latest_month,
        )
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "parse-events":
        df = parse_events(args.input, args.output)
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "normalize-events":
        df = normalize_events(
            args.input,
            args.output,
            site_map_path=args.site_map,
            default_attendees=args.default_attendees,
            default_multiplier=args.default_multiplier,
            default_start_hour=args.default_start_hour,
            default_duration_hours=args.default_duration_hours,
            overrides_path=args.event_overrides,
        )
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "parse-ops":
        df = parse_ops(
            args.input,
            args.output,
            default_start_hour=args.default_start_hour,
            default_duration_hours=args.default_duration_hours,
            further_notice_days=args.further_notice_days,
        )
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "normalize-ops":
        df = normalize_ops(
            args.input,
            args.output,
            default_reliability=args.default_reliability,
            default_delay_min=args.default_delay_min,
            default_start_hour=args.default_start_hour,
            default_duration_hours=args.default_duration_hours,
            further_notice_days=args.further_notice_days,
            include_regions=args.include_regions,
            include_keywords=args.include_keywords,
        )
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "parse-callouts":
        df = parse_callouts(args.input, args.output)
        print(f"Wrote {len(df):,} rows to {args.output}")
        return

    if args.command == "normalize-callouts":
        df = normalize_callouts(
            args.input,
            args.output,
            site_map_path=args.site_map,
            default_duration_min=args.default_duration_min,
            ambulance_uplift=args.ambulance_uplift,
            fire_uplift=args.fire_uplift,
            ses_uplift=args.ses_uplift,
            other_uplift=args.other_uplift,
        )
        print(f"Wrote {len(df):,} rows to {args.output}")
        return



# ---------------------------------------------------------------------------
# Events / ops parsers tuned for official Canberra pages
# ---------------------------------------------------------------------------

DATE_RANGE_ANY_RE = re.compile(r"^\d{1,2}(?:\s*[–-]\s*\d{1,2})?(?:\s+[A-Z][a-z]+)?(?:\s*[–-]\s*\d{1,2}\s+[A-Z][a-z]+)?(?:\s+\d{4})?$")
TIME_SPAN_RE = re.compile(r"from\s+(\d{1,2}(?::\d{2})?\s*[ap]m)\s+to\s+(\d{1,2}(?::\d{2})?\s*[ap]m)", re.I)
EXPLICIT_NEXT_ON_DATE_RE = re.compile(r"Next on\s+(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})", re.I)
DAY_MONTH_YEAR_RE = re.compile(r"(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})")
DAY_MONTH_RE = re.compile(r"(\d{1,2}\s+[A-Z][a-z]{2,8})(?!\s+\d{4})")
DAY_RANGE_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})\s*[–-]\s*(\d{1,2})\s+([A-Z][a-z]{2,8})\s+(\d{4})$")
FULL_RANGE_RE = re.compile(r"^(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})\s*[–-]\s*(\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4})$")

def extract_visible_lines(input_path: str) -> list[str]:
    html = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        lines = []
        for ln in soup.get_text("\n").splitlines():
            ln = re.sub(r"\s+", " ", ln).strip()
            if ln:
                lines.append(ln)
        return lines
    text = re.sub(r"<[^>]+>", "\n", html)
    return [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]


def parse_date_hint(text: str, default_year: int | None = None) -> pd.Timestamp | None:
    text = str(text).strip().replace("–", "-")
    if not text:
        return None

    # full date first
    m = DAY_MONTH_YEAR_RE.search(text)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce", dayfirst=True)

    # forms like "2-6 Apr 2026"
    m = DAY_RANGE_MONTH_YEAR_RE.match(text)
    if m:
        day = int(m.group(1))
        month = m.group(3)
        year = int(m.group(4))
        return pd.to_datetime(f"{day} {month} {year}", errors="coerce", dayfirst=True)

    # forms like "27 February - 9 March" with year from page context
    if default_year is not None:
        pieces = re.split(r"\s*[–-]\s*", text)
        first_piece = pieces[0].strip()
        m2 = DAY_MONTH_RE.search(first_piece)
        if m2:
            return pd.to_datetime(f"{m2.group(1)} {default_year}", errors="coerce", dayfirst=True)
        # maybe first piece is just a day and second piece has month
        if re.fullmatch(r"\d{1,2}", first_piece) and len(pieces) >= 2:
            m3 = DAY_MONTH_RE.search(pieces[1])
            if m3:
                month = m3.group(1).split(" ", 1)[1]
                return pd.to_datetime(f"{first_piece} {month} {default_year}", errors="coerce", dayfirst=True)

    return None


def apply_event_overrides(df: pd.DataFrame, overrides_path: Optional[str]) -> pd.DataFrame:
    if not overrides_path:
        return df
    p = Path(overrides_path)
    if not p.exists():
        return df
    ov = read_any_table(overrides_path)
    ov.columns = [str(c).strip() for c in ov.columns]
    patt_col = first_present(ov.columns, ["pattern", "match", "title pattern", "title_pattern", "regex"])
    att_col = first_present(ov.columns, ["attendees", "attendance", "expected attendees", "expected_attendees"])
    mult_col = first_present(ov.columns, ["multiplier", "lift", "intensity"])
    mode_col = first_present(ov.columns, ["mode", "match mode", "match_mode"])
    if patt_col is None:
        raise ValueError("event_overrides file must contain a pattern column")
    if "title" not in df.columns:
        df["title"] = ""
    out = df.copy()
    for _, row in ov.iterrows():
        patt = str(row[patt_col]).strip()
        if not patt:
            continue
        mode = str(row[mode_col]).strip().lower() if mode_col else "contains"
        title_norm = out["title"].astype(str).map(norm_text)
        if mode == "regex":
            mask = out["title"].astype(str).str.contains(patt, case=False, regex=True, na=False)
        elif mode == "exact":
            mask = title_norm.eq(norm_text(patt))
        else:
            mask = title_norm.str.contains(norm_text(patt), regex=False, na=False)
        if att_col is not None and not pd.isna(row[att_col]):
            out.loc[mask, "attendees"] = pd.to_numeric(row[att_col], errors="coerce")
        if mult_col is not None and not pd.isna(row[mult_col]):
            out.loc[mask, "multiplier"] = pd.to_numeric(row[mult_col], errors="coerce")
    return out


def parse_events_canberra_html(input_path: str, default_year: int | None = None) -> pd.DataFrame:
    lines = extract_visible_lines(input_path)
    page_year = default_year
    if page_year is None:
        for ln in lines[:40]:
            m = re.search(r"Major Events\s+(\d{4})", ln, re.I)
            if m:
                page_year = int(m.group(1))
                break

    records: list[dict] = []
    skip_titles = {
        "What's on in Canberra", "Filter results", "Loading results", "Page 1 of 21",
        "Major Events 2026", "Produced by the ACT Government", "Other times and other places",
    }

    i = 0
    while i < len(lines):
        ln = lines[i]
        if DATE_RANGE_ANY_RE.match(ln) and i + 1 < len(lines):
            title = lines[i + 1].lstrip("#").strip()
            if title and title not in skip_titles and not title.startswith("Next on") and len(title) >= 3:
                location = None
                next_on = None
                j = i + 2
                while j < min(i + 8, len(lines)):
                    probe = lines[j]
                    if probe.startswith("Next on"):
                        next_on = probe
                        break
                    if location is None and ("," in probe or "Various venues" in probe):
                        location = probe
                    j += 1

                base_date = None
                if next_on:
                    m = EXPLICIT_NEXT_ON_DATE_RE.search(next_on)
                    if m:
                        base_date = pd.to_datetime(m.group(1), errors="coerce", dayfirst=True)
                if base_date is None:
                    base_date = parse_date_hint(ln, default_year=page_year)

                start_dt = pd.NaT
                end_dt = pd.NaT
                if base_date is not None and not pd.isna(base_date):
                    start_dt = pd.Timestamp(base_date).normalize()
                    end_dt = start_dt + pd.Timedelta(hours=2)
                    if next_on:
                        mt = TIME_SPAN_RE.search(next_on)
                        if mt:
                            st = pd.to_datetime(mt.group(1), errors="coerce")
                            et = pd.to_datetime(mt.group(2), errors="coerce")
                            if not pd.isna(st):
                                start_dt = start_dt + pd.Timedelta(hours=int(st.hour), minutes=int(st.minute))
                            else:
                                start_dt = start_dt + pd.Timedelta(hours=18)
                            if not pd.isna(et):
                                end_dt = start_dt.normalize() + pd.Timedelta(hours=int(et.hour), minutes=int(et.minute))
                                if end_dt <= start_dt:
                                    end_dt += pd.Timedelta(days=1)
                            else:
                                end_dt = start_dt + pd.Timedelta(hours=2)
                        else:
                            start_dt = start_dt + pd.Timedelta(hours=18)
                            end_dt = start_dt + pd.Timedelta(hours=2)
                records.append({
                    "title": title,
                    "location": location,
                    "date_hint": ln,
                    "start_datetime": start_dt,
                    "end_datetime": end_dt,
                })
                i = j
                continue
        i += 1

    df = pd.DataFrame(records).drop_duplicates()
    if df.empty:
        raise ValueError(f"Could not parse Canberra events HTML: {input_path}")
    return df


ALERT_POSTED_RE = re.compile(r"^Posted:\s*(.+)$", re.I)
ALERT_REGION_RE = re.compile(r"^Region:\s*(.+)$", re.I)
TITLE_DATE_RE = re.compile(r"(\d{1,2}\s+[A-Z][a-z]{2,8}(?:\s+\d{4})?)", re.I)
TITLE_TIME_RE = re.compile(r"(\d{1,2}(?::\d{2})?\s*[ap]m)", re.I)

def parse_alert_interval_from_title(title: str, posted_date: pd.Timestamp, default_start_hour: int = 17, default_duration_hours: int = 4, further_notice_days: int = 30) -> tuple[pd.Timestamp, pd.Timestamp]:
    title_clean = title.replace("–", "-")
    year = int(posted_date.year)

    start_dt = pd.Timestamp(posted_date).normalize() + pd.Timedelta(hours=default_start_hour)
    end_dt = start_dt + pd.Timedelta(hours=default_duration_hours)

    dates = TITLE_DATE_RE.findall(title_clean)
    times = TITLE_TIME_RE.findall(title_clean)

    parsed_dates = []
    for d in dates:
        txt = d
        if re.search(r"\d{4}", txt) is None:
            txt = f"{txt} {year}"
        ts = pd.to_datetime(txt, errors="coerce", dayfirst=True)
        if not pd.isna(ts):
            parsed_dates.append(ts.normalize())

    parsed_times = []
    for t in times:
        ts = pd.to_datetime(t, errors="coerce")
        if not pd.isna(ts):
            parsed_times.append((int(ts.hour), int(ts.minute)))

    if parsed_dates:
        start_dt = parsed_dates[0] + pd.Timedelta(hours=default_start_hour)
        if parsed_times:
            start_dt = parsed_dates[0] + pd.Timedelta(hours=parsed_times[0][0], minutes=parsed_times[0][1])

    if "until further notice" in title_clean.lower():
        end_dt = start_dt + pd.Timedelta(days=further_notice_days)
    elif len(parsed_dates) >= 2:
        end_dt = parsed_dates[1] + pd.Timedelta(hours=default_start_hour)
        if len(parsed_times) >= 2:
            end_dt = parsed_dates[1] + pd.Timedelta(hours=parsed_times[1][0], minutes=parsed_times[1][1])
        elif len(parsed_times) == 1:
            end_dt = parsed_dates[1] + pd.Timedelta(hours=parsed_times[0][0], minutes=parsed_times[0][1])
    elif len(parsed_dates) == 1:
        end_dt = start_dt + pd.Timedelta(hours=default_duration_hours)

    if end_dt <= start_dt:
        end_dt = start_dt + pd.Timedelta(hours=default_duration_hours)
    return start_dt, end_dt


def parse_transport_alerts_html(input_path: str, default_start_hour: int = 17, default_duration_hours: int = 4, further_notice_days: int = 30) -> pd.DataFrame:
    lines = extract_visible_lines(input_path)
    records: list[dict] = []
    for i, ln in enumerate(lines):
        if ln.startswith("Posted:") and i >= 1:
            title = lines[i - 1].lstrip("#").strip()
            posted_match = ALERT_POSTED_RE.match(ln)
            if not posted_match:
                continue
            posted_date = pd.to_datetime(posted_match.group(1), errors="coerce", dayfirst=True)
            if pd.isna(posted_date):
                continue
            region = None
            if i + 1 < len(lines):
                mreg = ALERT_REGION_RE.match(lines[i + 1])
                if mreg:
                    region = mreg.group(1).strip()
            start_dt, end_dt = parse_alert_interval_from_title(
                title, pd.Timestamp(posted_date), default_start_hour=default_start_hour,
                default_duration_hours=default_duration_hours, further_notice_days=further_notice_days
            )
            records.append({
                "title": title,
                "region": region,
                "posted_date": pd.Timestamp(posted_date).normalize(),
                "start_datetime": start_dt,
                "end_datetime": end_dt,
            })
    df = pd.DataFrame(records).drop_duplicates()
    if df.empty:
        raise ValueError(f"Could not parse Transport Canberra alerts HTML: {input_path}")
    return df


def normalize_events(
    input_path: str,
    output_path: str,
    site_map_path: Optional[str] = None,
    default_attendees: int = 100,
    default_multiplier: float = 1.0,
    default_start_hour: int = 18,
    default_duration_hours: int = 2,
    overrides_path: Optional[str] = None,
) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, DEFAULT_EVENTS_URL, ".html")
    try:
        input_suffix = Path(materialized_input).suffix.lower()
        if input_suffix in {".html", ".htm"}:
            html = Path(materialized_input).read_text(encoding="utf-8", errors="ignore")
            if "What's on in Canberra" in html or "Major Events" in html:
                df = parse_events_canberra_html(materialized_input)
            else:
                df = parse_event_html(materialized_input)
        else:
            df = read_any_table(materialized_input)

        if df.empty:
            raise ValueError("Event input is empty.")
        df.columns = [str(c).strip() for c in df.columns]

        site_col = first_present(df.columns, EVENT_SITE_CANDIDATES)
        location_col = first_present(df.columns, ["location"])
        title_col = first_present(df.columns, EVENT_TITLE_CANDIDATES)
        attend_col = first_present(df.columns, EVENT_ATTEND_CANDIDATES)
        mult_col = first_present(df.columns, EVENT_MULT_CANDIDATES)
        date_col = first_present(df.columns, EVENT_DATE_CANDIDATES)
        start_col = first_present(df.columns, EVENT_START_CANDIDATES)
        end_col = first_present(df.columns, EVENT_END_CANDIDATES)

        mappings = load_site_map(site_map_path)

        out = pd.DataFrame()
        if site_col:
            raw_site = df[site_col].astype(str)
        elif location_col:
            raw_site = df[location_col].astype(str)
        elif title_col:
            raw_site = df[title_col].astype(str)
        else:
            raise ValueError("Events input needs a site/location/venue column or at least a title column.")

        out["site"] = raw_site.map(lambda s: apply_site_map(s, mappings))
        out["title"] = df[title_col].astype(str) if title_col else ""
        out["attendees"] = pd.to_numeric(df[attend_col], errors="coerce").fillna(default_attendees) if attend_col else default_attendees
        out["multiplier"] = pd.to_numeric(df[mult_col], errors="coerce").fillna(default_multiplier) if mult_col else default_multiplier

        if start_col:
            out["start_datetime"] = pd.to_datetime(df[start_col], errors="coerce")
        elif date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            out["start_datetime"] = dates + pd.to_timedelta(default_start_hour, unit="h")
        else:
            out["start_datetime"] = pd.NaT

        if end_col:
            out["end_datetime"] = pd.to_datetime(df[end_col], errors="coerce")
        else:
            out["end_datetime"] = out["start_datetime"] + pd.to_timedelta(default_duration_hours, unit="h")

        if "start_datetime" in df.columns:
            parsed_start = pd.to_datetime(df["start_datetime"], errors="coerce")
            out["start_datetime"] = out["start_datetime"].fillna(parsed_start)
        if "end_datetime" in df.columns:
            parsed_end = pd.to_datetime(df["end_datetime"], errors="coerce")
            out["end_datetime"] = out["end_datetime"].fillna(parsed_end)

        out = apply_event_overrides(out, overrides_path)
        out = out.dropna(subset=["start_datetime"]).copy()
        out = out.sort_values(["start_datetime", "site"]).drop_duplicates(subset=["site", "title", "start_datetime", "end_datetime"])
        write_csv(out[["site", "start_datetime", "end_datetime", "attendees", "multiplier"]], output_path)
        return out[["site", "start_datetime", "end_datetime", "attendees", "multiplier"]]
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


def normalize_ops(
    input_path: str,
    output_path: str,
    default_reliability: float = 0.85,
    default_delay_min: float = 8.0,
    default_start_hour: int = 17,
    default_duration_hours: int = 4,
    further_notice_days: int = 30,
    include_regions: Optional[str] = None,
    include_keywords: Optional[str] = None,
) -> pd.DataFrame:
    materialized_input, cleanup_path = materialize_input_path(input_path, DEFAULT_ALERTS_URL, ".html")
    try:
        input_suffix = Path(materialized_input).suffix.lower()
        if input_suffix in {".html", ".htm"}:
            html = Path(materialized_input).read_text(encoding="utf-8", errors="ignore")
            if "Service alerts and updates" in html:
                df = parse_transport_alerts_html(
                    materialized_input,
                    default_start_hour=default_start_hour,
                    default_duration_hours=default_duration_hours,
                    further_notice_days=further_notice_days,
                )
            else:
                df = read_any_table(materialized_input)
        else:
            df = read_any_table(materialized_input)

        if df.empty:
            raise ValueError("Ops input is empty.")
        df.columns = [str(c).strip() for c in df.columns]

        if include_regions and "region" in df.columns:
            regions = [norm_text(x) for x in str(include_regions).split(",") if x.strip()]
            if regions:
                mask = df["region"].astype(str).map(norm_text).apply(lambda x: any(r in x for r in regions))
                df = df.loc[mask].copy()

        if include_keywords and "title" in df.columns:
            kws = [norm_text(x) for x in str(include_keywords).split(",") if x.strip()]
            if kws:
                mask = df["title"].astype(str).map(norm_text).apply(lambda x: any(k in x for k in kws))
                df = df.loc[mask].copy()

        date_col = first_present(df.columns, OPS_DATE_CANDIDATES)
        hour_col = first_present(df.columns, OPS_HOUR_CANDIDATES)
        rel_col = first_present(df.columns, OPS_RELIAB_CANDIDATES)
        delay_col = first_present(df.columns, OPS_DELAY_CANDIDATES)
        dis_col = first_present(df.columns, OPS_DISRUPT_CANDIDATES)

        if date_col and hour_col:
            out = pd.DataFrame()
            out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
            out["hour"] = df[hour_col].map(maybe_int_hour)
            out["service_reliability"] = pd.to_numeric(df[rel_col], errors="coerce") if rel_col else default_reliability
            out["avg_delay_min"] = pd.to_numeric(df[delay_col], errors="coerce") if delay_col else default_delay_min
            if dis_col:
                raw = df[dis_col].astype(str).str.lower()
                out["is_disrupted"] = raw.isin(["1", "true", "yes", "y"])
            else:
                out["is_disrupted"] = (out["avg_delay_min"] >= 6) | (out["service_reliability"] < 0.90)
            out = out.dropna(subset=["date", "hour"]).copy()
            out["hour"] = out["hour"].astype(int)
            out = out.sort_values(["date", "hour"]).drop_duplicates()
            write_csv(out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]], output_path)
            return out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]]

        start_col = first_present(df.columns, OPS_START_CANDIDATES)
        end_col = first_present(df.columns, OPS_END_CANDIDATES)
        if not start_col:
            raise ValueError("Ops input needs either date+hour columns or start/end datetime columns.")

        starts = pd.to_datetime(df[start_col], errors="coerce")
        ends = pd.to_datetime(df[end_col], errors="coerce") if end_col else starts + pd.to_timedelta(default_duration_hours, unit="h")

        rel_series = pd.to_numeric(df[rel_col], errors="coerce").fillna(default_reliability) if rel_col else pd.Series([default_reliability] * len(df))
        delay_series = pd.to_numeric(df[delay_col], errors="coerce").fillna(default_delay_min) if delay_col else pd.Series([default_delay_min] * len(df))
        if dis_col:
            raw = df[dis_col].astype(str).str.lower()
            dis_series = raw.isin(["1", "true", "yes", "y"])
        else:
            dis_series = pd.Series([True] * len(df))

        rows: list[dict] = []
        for start_ts, end_ts, rel, delay, disrupted in zip(starts, ends, rel_series, delay_series, dis_series):
            if pd.isna(start_ts):
                continue
            if pd.isna(end_ts) or end_ts <= start_ts:
                end_ts = start_ts + pd.Timedelta(hours=default_duration_hours)
            current = pd.Timestamp(start_ts).floor("h")
            stop = pd.Timestamp(end_ts).ceil("h")
            while current < stop:
                rows.append({
                    "date": current.normalize(),
                    "hour": int(current.hour),
                    "service_reliability": float(rel),
                    "avg_delay_min": float(delay),
                    "is_disrupted": bool(disrupted),
                })
                current += pd.Timedelta(hours=1)

        out = pd.DataFrame(rows)
        if out.empty:
            raise ValueError("No valid ops rows generated after parsing/filtering. Check the input format or filters.")
        out = out.drop_duplicates()
        out = out.sort_values(["date", "hour"])
        write_csv(out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]], output_path)
        return out[["date", "hour", "service_reliability", "avg_delay_min", "is_disrupted"]]
    finally:
        if cleanup_path:
            try:
                Path(cleanup_path).unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
