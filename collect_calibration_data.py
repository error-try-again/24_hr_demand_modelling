#!/usr/bin/env python3
"""
collect_calibration_data.py — After Dark calibration data collector
====================================================================

Downloads public datasets from ACT Open Data, ABS, and other sources
to replace model priors with observed values. Run this on your local
machine (not sandboxed) — it needs internet access.

This patched version improves two things:
  1. If local ACT HTS Excel workbooks are present, it uses those instead of
     hitting blocked/403 Socrata endpoints.
  2. Traffic AADT matching is more robust: it searches likely road/route name
     columns only, casts mixed-type columns safely, and falls back to the
     traffic-routes dataset when links data yields no matches.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas openpyxl")

try:
    import requests
except ImportError:
    sys.exit("requests required: pip install requests")


ACT_DATA_BASE = "https://www.data.act.gov.au/resource"

DATASETS = {
    "traffic_links": {
        "id": "jn4p-azhb",
        "description": "Vehicle AADT per road segment (ACT Roads)",
        "limit": 5000,
        "filters": {},
    },
    "wifi_monthly": {
        "id": "sptb-jhn6",
        "description": "CBRfree Wi-Fi monthly client connections by location",
        "limit": 5000,
        "filters": {},
    },
    "hts_method": {
        "id": "efex-weer",
        "description": "ACT HTS 2022 — Method of Travel (walk/cycle/bus/LR shares)",
        "limit": 500,
        "filters": {},
    },
    "hts_time": {
        "id": "cnu8-gvab",
        "description": "ACT HTS 2022 — Time of Travel (hourly distribution)",
        "limit": 500,
        "filters": {},
    },
    "hts_walk_cycle": {
        "id": "mp5u-jfb3",
        "description": "ACT HTS 2022 — Walking and Cycling Travel",
        "limit": 500,
        "filters": {},
    },
    "hts_purpose": {
        "id": "5wi6-bkzs",
        "description": "ACT HTS 2022 — Purpose of Travel",
        "limit": 500,
        "filters": {},
    },
    "bike_barometer": {
        "id": "62sb-92ea",
        "description": "Bike Barometer counts — MacArthur Avenue (Northbourne corridor)",
        "limit": 10000,
        "filters": {},
    },
    "bus_alightings_hourly": {
        "id": "4gsk-t7z5",
        "description": "Bus/LR alightings by stop by hour",
        "limit": 50000,
        "filters": {},
    },
    "lr_patronage_15min": {
        "id": "xvid-q4du",
        "description": "Light Rail Patronage at 15-minute intervals",
        "limit": 50000,
        "filters": {},
    },
    "lr_patronage_daily": {
        "id": "x7dn-77he",
        "description": "Light Rail Patronage daily totals",
        "limit": 5000,
        "filters": {},
    },
    "pt_daily_by_service": {
        "id": "4f52-nub8",
        "description": "Daily PT boardings split by bus/LR service type",
        "limit": 5000,
        "filters": {},
    },
    "traffic_routes": {
        "id": "mgzi-6f8j",
        "description": "Traffic route statistics (vehicle counts by route)",
        "limit": 5000,
        "filters": {},
    },
}

LOCAL_HTS_FILES = {
    "hts_method": {
        "filename": "ACT_HTS_-_01_Method_of_travel.xlsx",
        "preferred_sheet": "Method of travel (%)",
    },
    "hts_purpose": {
        "filename": "ACT_HTS_-_02_Purpose_of_travel.xlsx",
        "preferred_sheet": "Primary purpose (%)",
    },
    "hts_time": {
        "filename": "ACT_HTS_-_03_Time_of_travel.xlsx",
        "preferred_sheet": "Time of travel, by mode",
    },
    "hts_walk_cycle": {
        "filename": "ACT_HTS_-_10_Walking_and_cycling_travel.xlsx",
        "preferred_sheet": "Walk-Bicycle comparison (%)",
    },
}

ABS_QUICKSTATS = {
    "Civic_City":      "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80041",
    "Dickson":         "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80049",
    "Gungahlin":       "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80068",
    "Braddon":         "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80024",
    "Turner":          "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80154",
    "OConnor":         "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80119",
    "Mitchell":        "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80105",
    "Harrison":        "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80071",
    "Franklin":        "https://abs.gov.au/census/find-census-data/quickstats/2021/SAL80060",
}

SITES = {
    "Alinga Street": {
        "road_keywords": ["northbourne", "alinga", "london circuit"],
        "wifi_keywords": ["civic", "northbourne", "city"],
        "bus_stops": ["city bus stn", "city west platform"],
        "suburbs": ["city", "civic", "braddon", "acton", "turner"],
        "lat": -35.2784, "lon": 149.1305,
    },
    "Dickson": {
        "road_keywords": ["cowper", "antill", "northbourne"],
        "wifi_keywords": ["dickson"],
        "bus_stops": ["cowper st dickson", "antill st.*dickson"],
        "suburbs": ["dickson", "lyneham", "oconnor", "downer", "watson"],
        "lat": -35.2504, "lon": 149.1413,
    },
    "Gungahlin Place": {
        "road_keywords": ["hibberson", "flemington", "gungahlin", "gozzard"],
        "wifi_keywords": ["gungahlin"],
        "bus_stops": ["gozzard st gungahlin", "hibberson"],
        "suburbs": ["gungahlin", "harrison", "franklin", "mitchell", "throsby"],
        "lat": -35.1853, "lon": 149.1330,
    },
}


def norm_text(value: object) -> str:
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fetch_socrata(dataset_id: str, limit: int = 5000, filters: dict | None = None) -> pd.DataFrame:
    url = f"{ACT_DATA_BASE}/{dataset_id}.json"
    params = {"$limit": limit}
    if filters:
        params.update(filters)
    print(f"  Fetching {url} (limit={limit})...")
    try:
        resp = requests.get(url, params=params, timeout=60, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"    → {len(df)} rows, {len(df.columns)} columns")
            return df
        print(f"    → Unexpected response type: {type(data)}")
        return pd.DataFrame()
    except Exception as e:
        print(f"    → ERROR: {e}")
        return pd.DataFrame()


def load_local_hts_workbook(path: Path, preferred_sheet: str | None = None) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        print(f"    → ERROR reading local workbook {path.name}: {e}")
        return pd.DataFrame()

    sheet_name: str | None = None
    if preferred_sheet and preferred_sheet in xl.sheet_names:
        sheet_name = preferred_sheet
    else:
        for candidate in xl.sheet_names:
            if candidate.lower() != "about":
                sheet_name = candidate
                break
    if sheet_name is None:
        return pd.DataFrame()

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"    → ERROR parsing sheet {sheet_name!r} from {path.name}: {e}")
        return pd.DataFrame()


def maybe_load_local_hts(dataset_name: str, hts_dir: Path) -> tuple[pd.DataFrame, Optional[Path], Optional[str]]:
    spec = LOCAL_HTS_FILES.get(dataset_name)
    if not spec:
        return pd.DataFrame(), None, None
    path = hts_dir / spec["filename"]
    if not path.exists():
        return pd.DataFrame(), None, None
    df = load_local_hts_workbook(path, spec.get("preferred_sheet"))
    if df.empty:
        return pd.DataFrame(), path, spec.get("preferred_sheet")
    return df, path, spec.get("preferred_sheet")


def candidate_text_columns(df: pd.DataFrame) -> list[str]:
    strong = []
    weak = []
    for col in df.columns:
        nc = norm_text(col)
        if any(tok in nc for tok in ["road", "route", "street", "st", "avenue", "ave", "from", "to", "link", "description", "name"]):
            strong.append(col)
        elif df[col].dtype == "object":
            weak.append(col)
    return strong or weak


def candidate_aadt_columns(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        nc = norm_text(col)
        if any(tok in nc for tok in ["aadt", "annual average daily", "average annual daily", "traffic volume", "volume", "vehicles", "veh"]):
            candidates.append(col)
    if candidates:
        return candidates
    # fallback to numeric-ish columns with traffic-like values
    numericish = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > max(10, len(df) * 0.2) and vals.max(skipna=True) > 100:
            numericish.append(col)
    return numericish


def extract_site_traffic(traffic_df: pd.DataFrame, site_config: dict) -> dict:
    if traffic_df.empty:
        return {"aadt_total": None, "roads_matched": 0}

    # Guard: Addinsight travel-time data contains detector site IDs, not AADT
    addinsight_sig = {"destsiteid", "originsiteid", "mintt", "congestion", "score"}
    if addinsight_sig.issubset({c.lower().strip() for c in traffic_df.columns}):
        return {"aadt_total": None, "roads_matched": 0,
                "note": "Addinsight travel-time schema detected (not an AADT source)"}

    road_cols = candidate_text_columns(traffic_df)
    aadt_cols = candidate_aadt_columns(traffic_df)
    if not road_cols or not aadt_cols:
        return {"aadt_total": None, "roads_matched": 0, "road_columns": road_cols, "aadt_columns": aadt_cols}

    text_blob = pd.Series("", index=traffic_df.index, dtype="object")
    for col in road_cols:
        text_blob = text_blob + " " + traffic_df[col].astype(str).map(norm_text)

    matched_idx = set()
    for kw in site_config["road_keywords"]:
        kw_n = norm_text(kw)
        mask = text_blob.str.contains(kw_n, na=False, regex=False)
        matched_idx.update(traffic_df.index[mask].tolist())

    if not matched_idx:
        return {
            "aadt_total": None,
            "roads_matched": 0,
            "road_columns": road_cols,
            "aadt_columns": aadt_cols,
        }

    matched = traffic_df.loc[sorted(matched_idx)].copy()
    best_col = None
    best_non_null = -1
    best_vals = pd.Series(dtype="float64")
    for aadt_col in aadt_cols:
        vals = pd.to_numeric(matched[aadt_col], errors="coerce")
        nn = int(vals.notna().sum())
        # v12 fix 25: reject columns with negative values — AADT counts can't be negative
        if (vals < 0).any():
            continue
        if nn > best_non_null and vals.sum(skipna=True) > 0:
            best_non_null = nn
            best_col = aadt_col
            best_vals = vals.dropna()

    if best_col is None or best_vals.empty:
        return {
            "aadt_total": None,
            "roads_matched": int(len(matched)),
            "road_columns": road_cols,
            "aadt_columns": aadt_cols,
        }

    # Warn if selected column doesn't look like an AADT field
    nc = norm_text(best_col)
    is_named_aadt = any(tok in nc for tok in ["aadt", "volume", "vehicle", "traffic", "count"])
    result = {
        "aadt_total": float(best_vals.sum()),
        "aadt_mean": float(best_vals.mean()),
        "roads_matched": int(len(matched)),
        "aadt_column": best_col,
        "road_columns": road_cols,
        "values": [float(v) for v in best_vals.tolist()[:50]],
    }
    if not is_named_aadt:
        result["warning"] = f"Column '{best_col}' may not be AADT — name doesn't contain volume/vehicle/traffic keywords"
    return result


def extract_wifi_traffic(wifi_df: pd.DataFrame, site_config: dict) -> dict:
    if wifi_df.empty:
        return {"monthly_connections": None}

    results: Dict[str, Any] = {}
    for kw in site_config["wifi_keywords"]:
        kw_n = norm_text(kw)
        for col in wifi_df.columns:
            if wifi_df[col].dtype != "object":
                continue
            mask = wifi_df[col].astype(str).map(norm_text).str.contains(kw_n, na=False, regex=False)
            if not mask.any():
                continue
            matched = wifi_df[mask]
            for num_col in wifi_df.columns:
                vals = pd.to_numeric(matched[num_col], errors="coerce").dropna()
                if not vals.empty and vals.mean() > 10:
                    results[f"{kw}_{num_col}"] = {
                        "mean": float(vals.mean()),
                        "max": float(vals.max()),
                        "count": int(len(vals)),
                    }
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect calibration data from public sources")
    ap.add_argument("--outdir", default="calibration_data")
    ap.add_argument("--skip-abs", action="store_true", help="Skip ABS downloads (manual step)")
    ap.add_argument("--hts-dir", default=".", help="Directory containing local ACT HTS Excel workbooks")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hts_dir = Path(args.hts_dir)

    print("=" * 70)
    print("After Dark — Calibration Data Collector")
    print("=" * 70)

    print("\n[1/3] Downloading ACT Open Data datasets...")
    downloaded: Dict[str, pd.DataFrame] = {}
    local_sources: Dict[str, str] = {}

    for name, cfg in DATASETS.items():
        print(f"\n  {name}: {cfg['description']}")

        if name in LOCAL_HTS_FILES:
            df, local_path, preferred_sheet = maybe_load_local_hts(name, hts_dir)
            if local_path is not None and not df.empty:
                print(f"    → Using local workbook {local_path.name} (sheet: {preferred_sheet})")
                csv_path = outdir / f"{name}.csv"
                df.to_csv(csv_path, index=False)
                print(f"    Saved extracted HTS sheet to {csv_path}")
                downloaded[name] = df
                local_sources[name] = str(local_path)
                continue
            elif local_path is not None:
                print(f"    → Found local workbook {local_path.name} but could not parse it; falling back to API")

        df = fetch_socrata(cfg["id"], limit=cfg["limit"], filters=cfg.get("filters"))
        if not df.empty:
            csv_path = outdir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            print(f"    Saved to {csv_path}")
        downloaded[name] = df

    print("\n[2/3] Extracting site-level calibration metrics...")
    calibration: Dict[str, Any] = {}
    for site, scfg in SITES.items():
        print(f"\n  --- {site} ---")
        cal: Dict[str, Any] = {"site": site}

        traffic = {"aadt_total": None, "roads_matched": 0}
        if not downloaded.get("traffic_links", pd.DataFrame()).empty:
            traffic = extract_site_traffic(downloaded["traffic_links"], scfg)
            traffic["source"] = "traffic_links"
        if traffic.get("roads_matched", 0) == 0 and not downloaded.get("traffic_routes", pd.DataFrame()).empty:
            route_traffic = extract_site_traffic(downloaded["traffic_routes"], scfg)
            if route_traffic.get("roads_matched", 0) > 0:
                traffic = route_traffic
                traffic["source"] = "traffic_routes"
        cal["vehicle_aadt"] = traffic
        print(f"    Vehicle AADT: {traffic}")
        if traffic.get("warning"):
            print(f"    ⚠ {traffic['warning']}")

        for wifi_key in ["wifi_monthly"]:
            if not downloaded.get(wifi_key, pd.DataFrame()).empty:
                wifi = extract_wifi_traffic(downloaded[wifi_key], scfg)
                cal[f"wifi_{wifi_key}"] = wifi
                if wifi:
                    print(f"    WiFi ({wifi_key}): {len(wifi)} metric(s) found")

        if site == "Alinga Street" and not downloaded.get("bike_barometer", pd.DataFrame()).empty:
            bike = downloaded["bike_barometer"]
            for col in bike.columns:
                vals = pd.to_numeric(bike[col], errors="coerce").dropna()
                if not vals.empty and vals.mean() > 5:
                    cal["bike_barometer"] = {
                        "column": col,
                        "daily_mean": float(vals.mean()),
                        "daily_max": float(vals.max()),
                        "count": int(len(vals)),
                    }
                    print(f"    Bike barometer: mean={vals.mean():.1f}/day")
                    break

        for lr_key in ["lr_patronage_15min", "lr_patronage_daily"]:
            if not downloaded.get(lr_key, pd.DataFrame()).empty:
                lr = downloaded[lr_key]
                for col in lr.columns:
                    vals = pd.to_numeric(lr[col], errors="coerce").dropna()
                    if not vals.empty and vals.sum() > 100:
                        cal[f"lr_{lr_key}"] = {
                            "column": col,
                            "total": float(vals.sum()),
                            "mean": float(vals.mean()),
                            "rows": int(len(vals)),
                        }
                        break

        calibration[site] = cal

    print("\n[3/3] Processing Household Travel Survey data...")
    hts: Dict[str, Any] = {}
    for hts_key in ["hts_method", "hts_time", "hts_walk_cycle", "hts_purpose"]:
        df = downloaded.get(hts_key, pd.DataFrame())
        if not df.empty:
            hts[hts_key] = {
                "columns": list(df.columns),
                "rows": int(len(df)),
                "sample": df.head(3).to_dict(orient="records"),
                "source": local_sources.get(hts_key, "api"),
            }
            print(f"  {hts_key}: {len(df)} rows, source={hts[hts_key]['source']}")

    output = {
        "calibration_by_site": calibration,
        "household_travel_survey": hts,
        "datasets_downloaded": {
            name: {
                "rows": int(len(df)),
                "columns": list(df.columns)[:10],
                "source": local_sources.get(name, "api"),
            }
            for name, df in downloaded.items() if not df.empty
        },
        "abs_quickstats_urls": ABS_QUICKSTATS,
        "notes": {
            "abs_manual_step": (
                "ABS Working Population data requires manual download from TableBuilder or the QuickStats URLs above. "
                "Look for 'Method of Travel to Work' and 'Industry of Employment' by place of work for each SA2/suburb."
            ),
            "google_popular_times": (
                "Google Maps Popular Times data for businesses near each site can be accessed via the Google Places API "
                "(requires API key) or manually by searching Google Maps for businesses at each location and noting the hourly popularity bars."
            ),
        },
    }

    output_path = outdir / "calibration_inputs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote calibration inputs to {output_path}")

    print("\n" + "=" * 70)
    print("DATASETS COLLECTED")
    print("=" * 70)
    for name, df in downloaded.items():
        status = f"{len(df)} rows" if not df.empty else "EMPTY/FAILED"
        source = local_sources.get(name)
        if source:
            status += f" (local: {Path(source).name})"
        print(f"  {name:<30} {status}")

    print("\n" + "=" * 70)
    print("WHAT TO DO WITH THESE")
    print("=" * 70)
    print(
        """
1. TRAFFIC LINKS / ROUTES → Vehicle AADT on Northbourne Ave, Cowper St, Flemington Rd.
   Use vehicle counts × pedestrian mode share (from HTS) to estimate total
   pedestrian volumes near each site. This replaces the ambient multiplier.

2. CBRfree WiFi REPORTS → Monthly unique connections at Civic, Dickson, Gungahlin.
   ~20,000 users/month territory-wide. Site-level counts are a lower bound
   on foot traffic (only ~15-25% of pedestrians connect to public WiFi).

3. HOUSEHOLD TRAVEL SURVEY → Walk/cycle/bus/LR mode shares by district.
   This patched collector prefers your local Excel exports when present.

4. BIKE BAROMETER → Daily cycling counts on the LR corridor.

5. DAILY PT BOARDINGS BY SERVICE TYPE → Bus vs LR split over time.
"""
    )

    print("=" * 70)
    print("NEXT STEP: Run calibrate_model.py, then feed calibration_params.json")
    print("into after_dark_stop_hour_model_v10.py with --calibration.")
    print("=" * 70)


if __name__ == "__main__":
    main()