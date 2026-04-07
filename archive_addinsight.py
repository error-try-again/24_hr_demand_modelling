#!/usr/bin/env python3
"""
archive_addinsight.py — Addinsight Bluetooth detector archiver
===============================================================

Archives Addinsight link statistics snapshots to build a historical
corridor profile. Run via cron every 5 minutes for 2-4 weeks.

After collection, normalize mode produces:
- Hourly speed/congestion profiles per corridor (replaces static AADT)
- Empirical disruption frequency per site
- Time-of-day volume proxy distributions

Usage:
    # Archive mode (every 5 min via cron):
    python3 archive_addinsight.py archive --outdir ./addinsight_archive

    # Normalize mode (after collection):
    python3 archive_addinsight.py normalize \
        --archive-dir ./addinsight_archive \
        --output-dir ./support_data

Crontab entry:
    */5 * * * * cd /path/to/project && python3 archive_addinsight.py archive --outdir ./addinsight_archive >> ./addinsight_archive/cron.log 2>&1

Requires: pip install requests pandas
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError:
    sys.exit("requests required: pip install requests")

LINKS_STATS_URL = "http://data.addinsight.com/ACT/links_stats.json"

SITE_COORDS = {
    "Alinga Street":   (-35.2784, 149.1305,
                        ["northbourne", "london circuit", "cooyong", "alinga"]),
    "Dickson":         (-35.2504, 149.1413,
                        ["northbourne", "cowper", "antill"]),
    "Gungahlin Place": (-35.1853, 149.1330,
                        ["hibberson", "gungahlin place", "flemington"]),
}

# We archive links_stats.json (no geometry, ~50KB) instead of the full
# GeoJSON (~500KB) to keep storage manageable at 5-min intervals.
# 30 days × 288 snapshots/day × 50KB ≈ 430MB uncompressed, ~60MB gzipped.


def archive_snapshot(outdir: str) -> None:
    """Download and save a timestamped links_stats snapshot (gzipped)."""
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"links_stats_{ts}.json.gz"
    path = od / fname

    try:
        resp = requests.get(LINKS_STATS_URL, timeout=15)
        resp.raise_for_status()
        with gzip.open(path, "wb") as f:
            f.write(resp.content)
        data = resp.json()
        n = len(data) if isinstance(data, list) else 0
        print(f"  {ts}: {n} links, {len(resp.content)} bytes -> {fname}")
    except Exception as e:
        print(f"  {ts}: ERROR {e}")


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def normalize_archive(archive_dir: str, output_dir: str) -> None:
    """Process archived snapshots into corridor profiles.

    Outputs:
    - addinsight_hourly_profile.csv: mean speed/delay/score by site, dow, hour
    - addinsight_disruption_rates.csv: empirical disruption frequency per site
    - addinsight_aadt_estimate.csv: volume proxy from speed-flow, per site
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        sys.exit("pandas and numpy required for normalize mode")

    ad = Path(archive_dir)
    files = sorted(ad.glob("links_stats_*.json.gz"))
    if not files:
        # Try uncompressed
        files = sorted(ad.glob("links_stats_*.json"))
    if not files:
        print("No archived snapshots found.")
        return

    print(f"Processing {len(files)} snapshots...")

    # We need link definitions (names, geometry) to match links to sites.
    # Try to load from a full GeoJSON if available, otherwise we'll match by ID
    # against any raw_addinsight_links.json in the parent directory.
    link_meta = {}  # id -> {name, lat, lon}
    for cand in [ad / "link_definitions.json",
                 ad.parent / "raw_addinsight_links.json",
                 ad.parent / "support_data" / "raw_addinsight_links.json"]:
        if cand.exists():
            try:
                with open(cand) as f:
                    geo = json.load(f)
                for feat in geo.get("features", []):
                    p = feat["properties"]
                    coords = feat["geometry"]["coordinates"]
                    clat = sum(c[1] for c in coords) / len(coords)
                    clon = sum(c[0] for c in coords) / len(coords)
                    link_meta[p["Id"]] = {
                        "name": (p.get("Name") or "").lower(),
                        "lat": clat, "lon": clon,
                        "length": p.get("Length", 0),
                        "lanes": p.get("MinNumberOfLanes", 1),
                        "is_freeway": p.get("IsFreeway", False),
                    }
                print(f"  Loaded {len(link_meta)} link definitions from {cand.name}")
                break
            except Exception as e:
                print(f"  Could not load link defs from {cand}: {e}")

    if not link_meta:
        print("  WARNING: No link definitions found. Cannot match links to sites.")
        print("  Place raw_addinsight_links.json in the archive or parent directory.")
        return

    # Assign each link to a site (or None)
    link_site = {}  # link_id -> site_name
    for link_id, meta in link_meta.items():
        for site, (slat, slon, keywords) in SITE_COORDS.items():
            dist = _haversine(slat, slon, meta["lat"], meta["lon"])
            kw_match = any(k in meta["name"] for k in keywords)
            if dist <= 1000 or kw_match:
                link_site[link_id] = site
                break

    print(f"  Matched {len(link_site)} links to sites")

    # Parse all snapshots
    records: List[Dict] = []
    for i, fp in enumerate(files):
        if i % 100 == 0 and i > 0:
            print(f"    ...{i}/{len(files)}")
        try:
            if fp.suffix == ".gz":
                import gzip as _gz
                with _gz.open(fp, "rb") as f:
                    data = json.loads(f.read())
            else:
                with open(fp) as f:
                    data = json.load(f)
            if not isinstance(data, list):
                continue
            for link in data:
                lid = link.get("Id")
                site = link_site.get(lid)
                if site is None:
                    continue
                records.append({
                    "link_id": lid,
                    "site": site,
                    "interval_start": link.get("IntervalStart"),
                    "speed": link.get("Speed"),
                    "tt": link.get("TT"),
                    "delay": link.get("Delay"),
                    "excess_delay": link.get("ExcessDelay"),
                    "congestion": link.get("Congestion"),
                    "score": link.get("Score", 0),
                    "enough_data": link.get("EnoughData", False),
                    "closed": link.get("Closed", False),
                })
        except Exception:
            continue

    if not records:
        print("No site-matched records extracted.")
        return

    df = pd.DataFrame(records)
    df["ts"] = pd.to_datetime(df["interval_start"], errors="coerce", utc=True)
    df["local_ts"] = df["ts"].dt.tz_convert("Australia/Sydney")
    df["date"] = df["local_ts"].dt.normalize()
    df["hour"] = df["local_ts"].dt.hour
    df["dow"] = df["local_ts"].dt.dayofweek
    df = df[df["enough_data"] == True].copy()

    print(f"  {len(df):,} records with sufficient data across {df['date'].nunique()} days")

    od = Path(output_dir)
    od.mkdir(parents=True, exist_ok=True)

    # 1. Hourly profile per site
    profile = df.groupby(["site", "dow", "hour"], as_index=False).agg(
        mean_speed=("speed", "mean"),
        mean_delay=("delay", "mean"),
        mean_excess_delay=("excess_delay", "mean"),
        mean_score=("score", "mean"),
        pct_congested=("score", lambda x: (x >= 3).mean() * 100),
        n_obs=("speed", "count"),
    )
    profile_path = od / "addinsight_hourly_profile.csv"
    profile.to_csv(profile_path, index=False)
    print(f"  Wrote hourly profile -> {profile_path}")

    # 2. Disruption rates per site
    daily_site = df.groupby(["site", "date"], as_index=False).agg(
        n_congested=("score", lambda x: (x >= 3).sum()),
        n_closed=("closed", "sum"),
        mean_speed=("speed", "mean"),
    )
    daily_site["is_disrupted"] = (daily_site["n_closed"] > 0) | (daily_site["n_congested"] >= 2)
    disruption = daily_site.groupby("site", as_index=False).agg(
        total_days=("date", "nunique"),
        disrupted_days=("is_disrupted", "sum"),
        mean_speed=("mean_speed", "mean"),
    )
    disruption["disruption_rate"] = disruption["disrupted_days"] / disruption["total_days"]
    disruption_path = od / "addinsight_disruption_rates.csv"
    disruption.to_csv(disruption_path, index=False)
    print(f"  Wrote disruption rates -> {disruption_path}")

    # 3. Volume proxy estimate per site (from speed-flow)
    # Average across all observations to get representative AADT
    site_summary = []
    for site in df["site"].unique():
        sd = df[df["site"] == site]
        links_used = sd["link_id"].nunique()
        # Use median speed (more robust than mean to outliers)
        med_speed = sd["speed"].median()
        site_summary.append({
            "site": site,
            "n_links": links_used,
            "n_observations": len(sd),
            "n_days": sd["date"].nunique(),
            "median_speed_kmh": round(med_speed, 1),
            "mean_speed_kmh": round(sd["speed"].mean(), 1),
            "mean_score": round(sd["score"].mean(), 2),
            "disruption_pct": round(
                daily_site[daily_site["site"] == site]["is_disrupted"].mean() * 100, 1),
        })
    summary_path = od / "addinsight_corridor_profile.csv"
    pd.DataFrame(site_summary).to_csv(summary_path, index=False)
    print(f"  Wrote corridor profile -> {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="Archive and normalize Addinsight traffic data")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("archive", help="Download a links_stats snapshot (run via cron)")
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("normalize", help="Process archived snapshots into corridor profiles")
    p.add_argument("--archive-dir", required=True)
    p.add_argument("--output-dir", default="support_data")

    args = ap.parse_args()

    if args.command == "archive":
        archive_snapshot(args.outdir)
    elif args.command == "normalize":
        normalize_archive(args.archive_dir, args.output_dir)


if __name__ == "__main__":
    main()
