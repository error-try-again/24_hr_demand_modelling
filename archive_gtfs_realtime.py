#!/usr/bin/env python3
"""
archive_gtfs_realtime.py — Transport Canberra GTFS-realtime archiver
=====================================================================

Archives GTFS-realtime vehicle positions and trip updates from the ACT
public transport feed. Run via cron every 5 minutes to build an empirical
service reliability dataset that replaces the synthetic ops.csv.

The ACT GTFS-realtime feeds are published at:
  https://www.transport.act.gov.au/googletransit/

After 2-4 weeks of archival, run normalize mode to produce a canonical
ops.csv with actual per-hour service reliability and delay metrics.

Usage:
    # Archive mode (run every 5 min via cron):
    python3 archive_gtfs_realtime.py archive --outdir ./gtfs_archive

    # Normalize mode (run once after collection period):
    python3 archive_gtfs_realtime.py normalize \
        --archive-dir ./gtfs_archive \
        --output ops.csv

Crontab entry:
    */5 * * * * cd /path/to/project && python3 archive_gtfs_realtime.py archive --outdir ./gtfs_archive >> ./gtfs_archive/cron.log 2>&1

Requires: pip install requests protobuf gtfs-realtime-bindings pandas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    sys.exit("requests required: pip install requests")

# ACT GTFS-realtime endpoints
GTFS_RT_BASE = "https://www.transport.act.gov.au/googletransit"
TRIP_UPDATES_URL = f"{GTFS_RT_BASE}/ACT_GTFSR_TripUpdates.pb"
VEHICLE_POSITIONS_URL = f"{GTFS_RT_BASE}/ACT_GTFSR_VehiclePositions.pb"

# Light rail route IDs (from GTFS static)
LR_ROUTE_IDS = {"Light_Rail_1", "LightRail1", "CR1"}

# Corridor sites for ops aggregation
SITE_STOPS = {
    "Alinga Street": ["Alinga"],
    "Dickson": ["Dickson"],
    "Gungahlin Place": ["Gungahlin"],
}


def archive_snapshot(outdir: str) -> None:
    """Download and save a timestamped GTFS-RT snapshot."""
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for label, url, ext in [
        ("trip_updates", TRIP_UPDATES_URL, "pb"),
        ("vehicle_positions", VEHICLE_POSITIONS_URL, "pb"),
    ]:
        fname = f"{label}_{ts}.{ext}"
        path = od / fname
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            print(f"  {ts} {label}: {len(resp.content)} bytes -> {path.name}")
        except Exception as e:
            print(f"  {ts} {label}: ERROR {e}")

    # Also try JSON feed as fallback (some agencies provide both)
    for label, url in [
        ("trip_updates_json", TRIP_UPDATES_URL.replace(".pb", ".json")),
        ("vehicle_positions_json", VEHICLE_POSITIONS_URL.replace(".pb", ".json")),
    ]:
        fname = f"{label}_{ts}.json"
        path = od / fname
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/json"):
                path.write_bytes(resp.content)
                print(f"  {ts} {label}: {len(resp.content)} bytes -> {path.name}")
        except Exception:
            pass  # JSON endpoint may not exist


def normalize_archive(archive_dir: str, output_path: str) -> None:
    """Process archived GTFS-RT snapshots into canonical ops.csv.

    Reads all archived trip update files, extracts delay information per
    trip/stop, and aggregates to hourly service reliability metrics.
    """
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas required for normalize mode")

    ad = Path(archive_dir)

    # Try JSON files first (easier to parse without protobuf dependency)
    json_files = sorted(ad.glob("trip_updates_json_*.json"))
    pb_files = sorted(ad.glob("trip_updates_*.pb"))

    records: List[Dict] = []

    if json_files:
        print(f"Processing {len(json_files)} JSON trip update files...")
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                ts_str = jf.stem.split("_")[-1]  # e.g. 20260410T120000Z
                entities = data.get("entity", [])
                for ent in entities:
                    tu = ent.get("tripUpdate", ent.get("trip_update", {}))
                    if not tu:
                        continue
                    trip = tu.get("trip", {})
                    route_id = trip.get("routeId", trip.get("route_id", ""))
                    for stu in tu.get("stopTimeUpdate", tu.get("stop_time_update", [])):
                        delay = 0
                        arr = stu.get("arrival", {})
                        dep = stu.get("departure", {})
                        if arr.get("delay") is not None:
                            delay = arr["delay"]
                        elif dep.get("delay") is not None:
                            delay = dep["delay"]
                        stop_id = stu.get("stopId", stu.get("stop_id", ""))
                        records.append({
                            "snapshot_ts": ts_str,
                            "route_id": route_id,
                            "trip_id": trip.get("tripId", trip.get("trip_id", "")),
                            "stop_id": stop_id,
                            "delay_s": delay,
                            "is_lr": route_id in LR_ROUTE_IDS,
                        })
            except Exception as e:
                print(f"  Skip {jf.name}: {e}")

    elif pb_files:
        # Try protobuf parsing
        try:
            from google.transit import gtfs_realtime_pb2
        except ImportError:
            sys.exit("Install gtfs-realtime-bindings: pip install gtfs-realtime-bindings")

        print(f"Processing {len(pb_files)} protobuf trip update files...")
        for pf in pb_files:
            try:
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(pf.read_bytes())
                ts_str = pf.stem.split("_")[-1]
                for ent in feed.entity:
                    tu = ent.trip_update
                    route_id = tu.trip.route_id
                    for stu in tu.stop_time_update:
                        delay = stu.arrival.delay if stu.HasField("arrival") else (
                            stu.departure.delay if stu.HasField("departure") else 0)
                        records.append({
                            "snapshot_ts": ts_str,
                            "route_id": route_id,
                            "trip_id": tu.trip.trip_id,
                            "stop_id": stu.stop_id,
                            "delay_s": delay,
                            "is_lr": route_id in LR_ROUTE_IDS,
                        })
            except Exception as e:
                print(f"  Skip {pf.name}: {e}")
    else:
        print("No archived trip update files found.")
        return

    if not records:
        print("No trip update records extracted.")
        return

    df = pd.DataFrame(records)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    df["date"] = df["snapshot_ts"].dt.tz_convert("Australia/Sydney").dt.normalize()
    df["hour"] = df["snapshot_ts"].dt.tz_convert("Australia/Sydney").dt.hour
    df["delay_min"] = df["delay_s"] / 60.0
    df["is_disrupted"] = df["delay_s"].abs() > 300  # >5 min delay = disrupted

    print(f"Extracted {len(df):,} stop-time records across {df['date'].nunique()} days")

    # Aggregate to hourly ops metrics
    hourly = df.groupby(["date", "hour"], as_index=False).agg(
        service_reliability=("is_disrupted", lambda x: 1.0 - x.mean()),
        avg_delay_min=("delay_min", "mean"),
        is_disrupted=("is_disrupted", lambda x: x.mean() > 0.15),
        n_records=("delay_s", "count"),
    )
    hourly["date"] = hourly["date"].dt.strftime("%Y-%m-%d")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False)
    print(f"Wrote {len(hourly):,} hourly ops rows to {out_path}")
    print(f"  Date range: {hourly['date'].min()} to {hourly['date'].max()}")
    print(f"  Mean reliability: {hourly['service_reliability'].mean():.3f}")
    print(f"  Mean delay: {hourly['avg_delay_min'].mean():.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Archive and normalize ACT GTFS-realtime data")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("archive", help="Download a GTFS-RT snapshot (run via cron)")
    p.add_argument("--outdir", required=True)

    p = sub.add_parser("normalize", help="Process archived snapshots into ops.csv")
    p.add_argument("--archive-dir", required=True)
    p.add_argument("--output", default="ops.csv")

    args = ap.parse_args()

    if args.command == "archive":
        archive_snapshot(args.outdir)
    elif args.command == "normalize":
        normalize_archive(args.archive_dir, args.output)


if __name__ == "__main__":
    main()
