#!/usr/bin/env python3
"""
calibrate_model.py — After Dark demand model parameter calibration
===================================================================

Pipeline position:
  1. collect_calibration_data.py   → downloads CSVs from ACT Open Data
  2. (manual)                      → download HTS xlsx from data.act.gov.au
  3. THIS SCRIPT                   → processes all into calibration_params.json
  4. after_dark_stop_hour_model_v10.py --calibration calibration_params.json

Inputs (all optional — script derives what it can from whatever is provided):
  --boardings         Boardings by Stop by Quarter Hour CSV
  --alightings        Alightings by Stop by Quarter Hour CSV
  --hts-method        ACT HTS 2022 Method of Travel Excel (xlsx)
  --bike-barometer    ACT Bike Barometer CSV (MacArthur Ave)
  --lr-daily          LR Patronage Daily CSV
  --lr-15min          LR Patronage 15-min interval CSV
  --pt-service        PT Daily by Service Type CSV
  --wifi-monthly      CBRfree WiFi Monthly CSV

Outputs:
  calibration_params.json   — parameter overrides, keyed by site name
  calibration_audit.txt     — full derivation log with working shown

Requires: pandas, numpy, openpyxl
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
try:
    import pandas as pd, numpy as np
except ImportError:
    sys.exit("pip install pandas numpy openpyxl")

SITE_DEFS = {
    "Alinga Street": {
        "lr_stop_patterns": ["alinga"],
        "bus_stop_patterns": ["city bus stn", "city west platform"],
        "bus_frontage_share": 0.18,
        "hts_region": "North Canberra",
        "hts_colocation_factor": 0.08,
        "abs_workers": 20000, "abs_attendance": 0.70,
        "bike_corridor_share": 0.80, "bike_frontage_share": 0.05,
        "aadt_road_keywords": ["northbourne", "london circuit", "cooyong"],
    },
    "Dickson": {
        "lr_stop_patterns": ["dickson"],
        "bus_stop_patterns": ["cowper st dickson", "antill st.*dickson"],
        "bus_frontage_share": 0.35,
        "hts_region": "North Canberra",
        "hts_colocation_factor": 0.04,
        "abs_workers": 3500, "abs_attendance": 0.70,
        "bike_corridor_share": 0.90, "bike_frontage_share": 0.05,
        "aadt_road_keywords": ["northbourne", "cowper", "antill"],
    },
    "Gungahlin Place": {
        "lr_stop_patterns": ["gungahlin"],
        "bus_stop_patterns": ["gozzard st gungahlin", "hibberson"],
        "bus_frontage_share": 0.30,
        "hts_region": "Gungahlin",
        "hts_colocation_factor": 0.12,
        "abs_workers": 4100, "abs_attendance": 0.70,
        "bike_corridor_share": 0.0, "bike_frontage_share": 0.0,
        "aadt_road_keywords": ["hibberson", "gungahlin place", "flemington"],
    },
}
PRIORS = {
    "lr_conv_morning": (0.085, 0.025), "lr_conv_daytime": (0.045, 0.015),
    "lr_conv_evening": (0.030, 0.010), "lr_conv_latenight": (0.012, 0.006),
    "bus_conv": (0.022, 0.006), "ambient_conv": (0.016, 0.006),
    "bike_conv": (0.008, 0.003), "venue_conv": (0.25, 0.07),
    "wifi_penetration_rate": 0.20,
}
DUAL_SD = 0.80
_audit: List[str] = []
def log(m=""): _audit.append(m); print(m)

def _match(df, pats):
    mask = pd.Series(False, index=df.index)
    col = "Stop Name" if "Stop Name" in df.columns else "stop_name"
    for p in pats: mask |= df[col].astype(str).str.lower().str.contains(p, regex=True, na=False)
    return df[mask]

def compute_stops(bp, ap):
    log("\n" + "="*70 + "\n1. BUS + LR STOP VOLUMES\n" + "="*70)
    b = pd.read_csv(bp, low_memory=False)
    a = pd.read_csv(ap, low_memory=False)
    b["Total"] = pd.to_numeric(b["Total"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
    a["Total"] = pd.to_numeric(a["Total"].astype(str).str.replace(",",""), errors="coerce").fillna(0)
    tc = [c for c in b.columns if c not in ("Stop ID","Stop Name","Total")]
    def seg(df, hrs):
        t = 0.0
        for _, r in df.iterrows():
            for c in tc:
                m = re.search(r"(\d{1,2}):\d{2}", c)
                if m and int(m.group(1)) in hrs:
                    v = pd.to_numeric(r.get(c, 0), errors="coerce")
                    if pd.notna(v): t += v
        return t/7
    out = {}
    for site, sd in SITE_DEFS.items():
        log(f"\n  {site}")
        lb, la = _match(b, sd["lr_stop_patterns"]), _match(a, sd["lr_stop_patterns"])
        segs = {}
        for nm, hr in [("morning",range(6,10)),("daytime",range(10,17)),
                       ("evening",range(17,23)),("latenight",list(range(23,24))+list(range(0,6)))]:
            segs[nm] = seg(lb,hr)+seg(la,hr)
        st = sum(segs.values()) or 1
        # v12 fix 17: derive daily total from QH sums for consistency with segment shares
        ld = st
        ld_from_total = (lb["Total"].sum()+la["Total"].sum())/7
        if abs(ld - ld_from_total) > 1:
            log(f"    NOTE: QH-sum daily={ld:.0f} vs Total-col daily={ld_from_total:.0f} (using QH-sum)")
        log(f"    LR: {list(lb['Stop Name'].values)}  daily={ld:.0f}")
        for s,v in segs.items(): log(f"      {s}: {v:.0f} ({v/st*100:.1f}%)")
        bb, ba = _match(b, sd["bus_stop_patterns"]), _match(a, sd["bus_stop_patterns"])
        # v12 fix 17: bus daily from QH sums too
        bus_segs = {}
        for nm, hr in [("morning",range(6,10)),("daytime",range(10,17)),
                       ("evening",range(17,23)),("latenight",list(range(23,24))+list(range(0,6)))]:
            bus_segs[nm] = seg(bb,hr)+seg(ba,hr)
        br = sum(bus_segs.values()) or 0
        bf = br*sd["bus_frontage_share"]
        log(f"    Bus: {list(bb['Stop Name'].values)}  raw={br:.0f} → front={bf:.0f}")
        out[site] = dict(lr_daily=ld, lr_segments=segs,
                         lr_segment_shares={k:v/st for k,v in segs.items()},
                         bus_daily_raw=br, bus_frontage_share=sd["bus_frontage_share"], bus_daily_frontage=bf)
    return out

def _parse_hts(df, targets):
    modes = {"Vehicle driver","Vehicle passenger","Public transport","Walking","Bicycle","Other"}
    regs, done = {}, set()
    cr, ct = None, None
    for i in range(len(df)):
        row = df.iloc[i]
        vs = [str(row.iloc[j]).strip() if pd.notna(row.iloc[j]) else "" for j in range(min(15,len(row)))]
        rtxt = vs[2] if len(vs)>2 and vs[2] and vs[2]!="nan" else vs[0]
        has_r = bool(rtxt) and rtxt!="nan"
        has_d = len(vs)>3 and vs[3].strip()=="Daily"
        if has_r and has_d:
            mt = None
            for t in targets:
                if t in rtxt: mt=t; break
            if mt and mt not in done:
                cr, ct = mt, "Daily"
                if cr not in regs: regs[cr]={}
            else:
                if cr and len(regs.get(cr,{}))>=3: done.add(cr)
                cr, ct = None, None
        if len(vs)>3:
            tp = vs[3].strip().lower()
            if any(k in tp for k in ("peak","interpeak","off-peak")):
                if cr and len(regs.get(cr,{}))>=3: done.add(cr)
                cr, ct = None, None
        if cr and ct=="Daily" and len(vs)>11:
            m = vs[4] if len(vs)>4 else ""
            if m in modes:
                v = row.iloc[11]
                if pd.notna(v) and isinstance(v,(int,float)):
                    regs[cr][m] = float(v)
                    if len(regs[cr])>=len(modes): done.add(cr)
    return regs

def compute_hts(path):
    log("\n" + "="*70 + "\n2. HTS MODE SHARES\n" + "="*70)
    tgt = {"All ACT"} | {sd["hts_region"] for sd in SITE_DEFS.values()}
    pct = _parse_hts(pd.read_excel(path, sheet_name="Method of travel (%)", header=None), tgt)
    ab = _parse_hts(pd.read_excel(path, sheet_name="Method of travel", header=None), tgt)
    out = {}
    for r in sorted(pct):
        p = pct[r]; a = ab.get(r, {})
        pt,wk,bk = p.get("Public transport",0), p.get("Walking",0), p.get("Bicycle",0)
        tt = sum(a.values()); wt = a.get("Walking",0); bt = a.get("Bicycle",0)
        log(f"\n  {r}: PT={pt:.1f}% Walk={wk:.1f}% Bike={bk:.1f}%"
            f" Walk:PT={wk/pt:.1f}:1" if pt>0 else f"\n  {r}: PT=0")
        log(f"    Trips: {tt:,.0f}  Walk={wt:,.0f}  Bike={bt:,.0f}")
        out[r] = dict(pt_pct=pt, walk_pct=wk, bike_pct=bk,
                      walk_pt_ratio=wk/pt if pt>0 else 0,
                      total_daily_trips=tt, walk_daily_trips=wt, bike_daily_trips=bt)
    return out

def compute_bike(path):
    log("\n" + "="*70 + "\n3. BIKE BAROMETER\n" + "="*70)
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    d = df.groupby(df["date_time"].dt.date)["count"].sum()
    dt = pd.to_datetime(d.index)
    wd, we = d[dt.dayofweek<5], d[dt.dayofweek>=5]
    r = dict(weekday_mean=float(wd.mean()), weekday_sd=float(wd.std()),
             weekend_mean=float(we.mean()), weekend_ratio=float(we.mean()/wd.mean()))
    log(f"  Weekday: {r['weekday_mean']:.0f}±{r['weekday_sd']:.0f}  Weekend: {r['weekend_mean']:.0f}  ratio={r['weekend_ratio']:.2f}")
    return r

def compute_lr_daily(path):
    log("\n" + "="*70 + "\n4. LR PATRONAGE DAILY\n" + "="*70)
    df = pd.read_csv(path); df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df["total"]=pd.to_numeric(df["total"],errors="coerce"); df=df.dropna(subset=["date","total"])
    rc = df[df["date"]>="2024-01-01"]; rc = rc if len(rc) else df.tail(365)
    wd = rc[rc["date"].dt.dayofweek<5]; we = rc[rc["date"].dt.dayofweek>=5]
    r = dict(weekday_mean=float(wd["total"].mean()), weekday_sd=float(wd["total"].std()),
             weekday_cv=float(wd["total"].std()/wd["total"].mean()),
             weekend_ratio=float(we["total"].mean()/wd["total"].mean()))
    log(f"  Weekday: {r['weekday_mean']:,.0f}±{r['weekday_sd']:,.0f} CV={r['weekday_cv']:.3f}  WE ratio={r['weekend_ratio']:.2f}")
    return r

def compute_lr_15(path):
    log("\n" + "="*70 + "\n5. LR 15-MIN SEGMENTS\n" + "="*70)
    df = pd.read_csv(path)
    hs = {}
    for c in [c for c in df.columns if c.startswith("_")]:
        try: h=int(c.strip("_").split("_")[0]); hs[h]=hs.get(h,0)+pd.to_numeric(df[c],errors="coerce").sum()
        except: pass
    t = sum(hs.values()) or 1
    sg = {"morning":0,"daytime":0,"evening":0,"latenight":0}
    for h,v in hs.items():
        if 6<=h<10: sg["morning"]+=v
        elif 10<=h<17: sg["daytime"]+=v
        elif 17<=h<23: sg["evening"]+=v
        else: sg["latenight"]+=v
    sh = {k:v/t for k,v in sg.items()}
    for s,p in sh.items(): log(f"  {s}: {p:.1%}")
    return dict(segment_shares=sh)

def compute_pt(path):
    log("\n" + "="*70 + "\n6. PT SERVICE RATIO\n" + "="*70)
    df = pd.read_csv(path); df["date"]=pd.to_datetime(df["date"],errors="coerce")
    for c in ["local_route","light_rail","peak_service","rapid_route"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["bus"]=df[["local_route","peak_service","rapid_route"]].sum(axis=1)
    rc = df[df["date"]>="2024-01-01"]; rc = rc if len(rc) else df.tail(365)
    wd = rc[rc["date"].dt.dayofweek<5]
    lr,bs = wd["light_rail"].mean(), wd["bus"].mean()
    r = dict(lr_mean=float(lr), bus_mean=float(bs), ratio=float(bs/lr) if lr>0 else 0)
    log(f"  LR={r['lr_mean']:,.0f} Bus={r['bus_mean']:,.0f} Ratio={r['ratio']:.2f}:1")
    return r

def compute_wifi(path):
    log("\n" + "="*70 + "\n7. WIFI MONTHLY\n" + "="*70)
    df = pd.read_csv(path); c="number_of_unique_clients"
    if c not in df.columns: log("  No data"); return dict(civic_daily_floor=0)
    m = float(pd.to_numeric(df[c],errors="coerce").dropna().tail(12).mean())
    civic = m/30/PRIORS["wifi_penetration_rate"]*0.35
    log(f"  Monthly={m:,.0f} → Civic floor={civic:,.0f}/day")
    return dict(monthly_clients=m, civic_daily_floor=civic)

def compute_aadt(path):
    """Parse aadt_corridors.csv → per-site ambient pedestrian estimate from vehicle AADT.

    CSV columns:
        site, road_name, segment_description, aadt_vehicles, aadt_year,
        road_class, lanes, ped_vehicle_ratio, frontage_exposure,
        weekend_factor, source, notes

    For each site, aggregates: Σ(aadt_vehicles × ped_vehicle_ratio × frontage_exposure)
    then multiplies by corridor_ped_amplifier to account for non-vehicle pedestrian
    sources (office workers, bus transferees, residents, shoppers arriving on foot).

    The amplifier converts "vehicle-generated pedestrians" into "total ambient
    pedestrians" — it is the key tuneable parameter expressing precinct type:
        CBD (Alinga):      5–8×  (major interchange, offices, university)
        Group centre:      3–6×  (dining, shopping, residential mix)
        Suburban centre:   2–4×  (car-dominated, fewer walk-only trips)
    """
    log("\n" + "="*70 + "\n8. AADT CORRIDOR ANALYSIS\n" + "="*70)
    df = pd.read_csv(path)
    required = {"site", "road_name", "aadt_vehicles", "ped_vehicle_ratio", "frontage_exposure"}
    missing = required - set(df.columns)
    if missing:
        log(f"  WARNING: missing columns {missing} — skipping AADT")
        return None
    df["aadt_vehicles"] = pd.to_numeric(df["aadt_vehicles"], errors="coerce").fillna(0)
    df["ped_vehicle_ratio"] = pd.to_numeric(df["ped_vehicle_ratio"], errors="coerce").fillna(0)
    df["frontage_exposure"] = pd.to_numeric(df["frontage_exposure"], errors="coerce").fillna(0)
    df["weekend_factor"] = pd.to_numeric(df.get("weekend_factor", pd.Series(0.55)), errors="coerce").fillna(0.55)
    if "corridor_ped_amplifier" in df.columns:
        df["corridor_ped_amplifier"] = pd.to_numeric(df["corridor_ped_amplifier"], errors="coerce").fillna(1.0)
    else:
        df["corridor_ped_amplifier"] = 1.0

    out = {}
    for site in SITE_DEFS:
        rows = df[df["site"].str.strip() == site]
        if rows.empty:
            log(f"\n  {site}: no AADT rows matched")
            continue
        # Per-road contribution: vehicles × ratio that generates pedestrians × frontage share
        rows = rows.copy()
        rows["ped_contribution"] = rows["aadt_vehicles"] * rows["ped_vehicle_ratio"] * rows["frontage_exposure"]
        veh_ped = float(rows["ped_contribution"].sum())
        amp = float(rows["corridor_ped_amplifier"].iloc[0])  # per-site amplifier
        total_ped = veh_ped * amp
        total_aadt = float(rows["aadt_vehicles"].sum())
        wknd = float(rows["weekend_factor"].mean())
        # Uncertainty: ±30% on the estimate (compound of AADT uncertainty + ratio uncertainty)
        ped_sd = total_ped * 0.30

        log(f"\n  {site}:")
        for _, r in rows.iterrows():
            log(f"    {r['road_name']}: AADT={int(r['aadt_vehicles']):,} "
                f"× pvr={r['ped_vehicle_ratio']:.2f} × fe={r['frontage_exposure']:.2f} "
                f"= {r['ped_contribution']:.0f} peds/day")
        log(f"    TOTAL: {total_aadt:,.0f} vehicles → {veh_ped:,.0f} veh-peds × {amp:.1f}× amplifier = {total_ped:,.0f} ± {ped_sd:,.0f} ambient peds/day")
        log(f"    Weekend factor: {wknd:.2f}")

        out[site] = dict(
            aadt_total_vehicles=total_aadt,
            aadt_ped_daily=total_ped,
            aadt_ped_sd=ped_sd,
            aadt_veh_ped_raw=veh_ped,
            aadt_corridor_amplifier=amp,
            aadt_weekend_factor=wknd,
            road_count=len(rows),
            roads=[r["road_name"] for _, r in rows.iterrows()],
        )
    return out if out else None

# Speed-to-pedestrian amplifier: slower corridors → more foot traffic
_SPEED_AMP = [(20, 1.20), (30, 1.10), (40, 1.00), (50, 0.95), (60, 0.90)]

# BPR speed-flow parameters for urban arterials (Bureau of Public Roads function)
# v_free = posted speed, capacity per lane ≈ 800 veh/hr for signalised, 1800 for freeway
_CAPACITY_PER_LANE = {"arterial": 800, "signalised": 800, "freeway": 1800, "collector": 600}
_BPR_ALPHA = 0.15
_BPR_BETA = 4.0

def _estimate_hourly_volume(speed_kmh, min_tt_s, tt_s, length_m, n_lanes, is_freeway):
    """Estimate hourly vehicle volume from speed-flow relationship (BPR inverse)."""
    if min_tt_s <= 0 or tt_s <= 0 or length_m <= 0 or n_lanes <= 0:
        return None
    free_speed = length_m / min_tt_s * 3.6  # km/h
    if free_speed <= 0:
        return None
    tt_ratio = tt_s / min_tt_s
    cap_per_lane = 1800 if is_freeway else 800
    capacity = cap_per_lane * max(n_lanes, 1)
    # Inverse BPR: tt_ratio = 1 + alpha * (V/C)^beta → V/C = ((tt_ratio - 1)/alpha)^(1/beta)
    if tt_ratio <= 1.0:
        vc_ratio = 0.3  # below free-flow, assume light traffic
    else:
        vc_ratio = min(((tt_ratio - 1.0) / _BPR_ALPHA) ** (1.0 / _BPR_BETA), 1.2)
    hourly_vol = vc_ratio * capacity
    return hourly_vol

def compute_addinsight(path, links_json_path=None):
    """Load addinsight_corridor_summary.csv → per-site corridor context.

    If links_json_path is provided, also estimate current vehicle AADT from
    speed-flow relationships on nearby links, which can replace stale 2016 AADT.
    """
    if not path:
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    log("\n" + "="*70 + "\n8b. ADDINSIGHT CORRIDOR\n" + "="*70)

    # Optionally estimate AADT from raw links
    _link_aadt = {}
    if links_json_path:
        import json as _json, math as _math
        try:
            with open(links_json_path) as _f:
                _geo = _json.load(_f)
            _feats = _geo.get("features", [])
            _site_coords = {
                "Alinga Street": (-35.2784, 149.1305),
                "Dickson": (-35.2504, 149.1413),
                "Gungahlin Place": (-35.1853, 149.1330),
            }
            _site_kw = {
                "Alinga Street": ["northbourne", "london circuit", "cooyong", "alinga"],
                "Dickson": ["northbourne", "cowper", "antill"],
                "Gungahlin Place": ["hibberson", "gungahlin place", "flemington"],
            }
            for _site, (_slat, _slon) in _site_coords.items():
                _matched_vols = []
                for _feat in _feats:
                    _p = _feat["properties"]
                    if not _p.get("EnoughData"):
                        continue
                    _coords = _feat["geometry"]["coordinates"]
                    _clat = sum(c[1] for c in _coords)/len(_coords)
                    _clon = sum(c[0] for c in _coords)/len(_coords)
                    _dlat = _math.radians(_clat - _slat)
                    _dlon = _math.radians(_clon - _slon)
                    _a = _math.sin(_dlat/2)**2 + _math.cos(_math.radians(_slat))*_math.cos(_math.radians(_clat))*_math.sin(_dlon/2)**2
                    _dist = 6371000 * 2 * _math.atan2(_math.sqrt(_a), _math.sqrt(1-_a))
                    _name = (_p.get("Name") or "").lower()
                    _kw_match = any(k in _name for k in _site_kw.get(_site, []))
                    if _dist > 1000 and not _kw_match:
                        continue
                    _vol = _estimate_hourly_volume(
                        _p.get("Speed", 0), _p.get("MinTT", 0), _p.get("TT", 0),
                        _p.get("Length", 0), _p.get("MinNumberOfLanes", 1),
                        _p.get("IsFreeway", False))
                    if _vol and _vol > 10:
                        _matched_vols.append((_p.get("Name",""), _vol, _p.get("Length",0)))
                if _matched_vols:
                    # Sum hourly volumes across matched links, scale to daily (×14 for urban)
                    _total_hourly = sum(v for _, v, _ in _matched_vols)
                    _daily_est = _total_hourly * 14  # ~14 effective traffic hours
                    _link_aadt[_site] = {
                        "aadt_estimate": round(_daily_est),
                        "n_links_used": len(_matched_vols),
                        "total_hourly_vol": round(_total_hourly),
                        "method": "BPR_speed_flow",
                    }
                    log(f"  {_site}: Addinsight AADT estimate={_daily_est:,.0f} "
                        f"({len(_matched_vols)} links, hourly={_total_hourly:,.0f})")
        except Exception as _e:
            log(f"  Addinsight links JSON parse error: {_e}")

    out = {}
    for _, row in df.iterrows():
        site = row["site"]
        speed = float(row.get("avg_speed_kmh", 40))
        amp = 0.85
        for thresh, a in _SPEED_AMP:
            if speed <= thresh:
                amp = a; break
        out[site] = {
            "avg_speed_kmh": round(speed, 1),
            "ped_amplifier": amp,
            "n_links": int(row.get("n_links", 0)),
            "is_disrupted": bool(row.get("is_disrupted", False)),
            "n_congested": int(row.get("n_congested_links", 0)),
            "n_closed": int(row.get("n_closed_links", 0)),
            "pct_enough_data": float(row.get("pct_enough_data", 0)),
        }
        if site in _link_aadt:
            out[site]["aadt_estimate"] = _link_aadt[site]
        log(f"  {site}: {speed:.0f} km/h → amp={amp} "
            f"({out[site]['n_links']} links, disrupted={out[site]['is_disrupted']})")
    return out

def derive(stops, hts, bike, lr_d, lr_15, pt, wifi, aadt=None, addinsight=None):
    log("\n" + "="*70 + "\n9. DERIVATION\n" + "="*70)
    sf = DUAL_SD if lr_15 else 1.0
    log(f"  Dual confirm: {'Y' if lr_15 else 'N'} → SD×{sf}")
    out = {}
    for site, sd in SITE_DEFS.items():
        log(f"\n  {site}")
        sv = stops.get(site, {}); lr = sv.get("lr_daily",0)
        br = sv.get("bus_daily_raw",0); bf = sv.get("bus_daily_frontage",0)
        p: Dict[str,Any] = {}
        p["bus_stop_patterns"]=sd["bus_stop_patterns"]; p["bus_frontage_share"]=sd["bus_frontage_share"]
        p["bus_conv_mean"]=PRIORS["bus_conv"][0]; p["bus_conv_sd"]=round(PRIORS["bus_conv"][1]*sf,4)
        p["bus_daily_raw"]=round(br); p["bus_daily_frontage"]=round(bf)
        bd = bike["weekday_mean"]*sd["bike_corridor_share"]*sd["bike_frontage_share"] if bike and sd["bike_corridor_share"]>0 else 0
        bs = bike["weekday_sd"]*sd["bike_corridor_share"]*sd["bike_frontage_share"] if bike and sd["bike_corridor_share"]>0 else 0
        p["bike_daily"]=round(bd); p["bike_daily_sd"]=round(bs)
        p["bike_conv_mean"]=PRIORS["bike_conv"][0]; p["bike_conv_sd"]=round(PRIORS["bike_conv"][1]*sf,4)
        p["bike_weekend_ratio"]=bike["weekend_ratio"] if bike else 0.37
        log(f"    Bus={br:.0f}→{bf:.0f}  Bike={bd:.0f}")
        if aadt and site in aadt:
            aa = aadt[site]
            # If Addinsight provides a current-year AADT estimate with good coverage,
            # use it to scale the existing ped estimate proportionally
            if (addinsight and site in addinsight
                    and "aadt_estimate" in addinsight[site]
                    and addinsight[site].get("pct_enough_data", 0) >= 40):
                ai_aadt = addinsight[site]["aadt_estimate"]["aadt_estimate"]
                old_aadt = aa["aadt_total_vehicles"]
                if old_aadt > 0 and ai_aadt > 0:
                    scale = ai_aadt / old_aadt
                    aa = dict(aa)  # don't mutate original
                    aa["aadt_ped_daily"] = aa["aadt_ped_daily"] * scale
                    aa["aadt_ped_sd"] = aa["aadt_ped_sd"] * scale
                    aa["aadt_total_vehicles"] = ai_aadt
                    log(f"    Addinsight AADT override: {old_aadt:,.0f} → {ai_aadt:,.0f} "
                        f"(scale={scale:.2f}, peds {aa['aadt_ped_daily']:,.0f})")
            # AADT-derived: direct pedestrian estimate from vehicle corridor data
            mult = aa["aadt_ped_daily"]/lr if lr>0 else 1.0
            msd = aa["aadt_ped_sd"]/lr if lr>0 else mult*0.30
            src = (f"AADT: {aa['aadt_total_vehicles']:,.0f} veh → "
                   f"{aa['aadt_ped_daily']:,.0f} peds ({aa['road_count']} roads)")
            # Cross-check against HTS if available
            if hts and sd["hts_region"] in hts:
                h = hts[sd["hts_region"]]
                hts_est = (h["walk_daily_trips"]*sd["hts_colocation_factor"]
                           +h["bike_daily_trips"]*sd["hts_colocation_factor"])
                hts_mult = hts_est/lr if lr>0 else 1.0
                log(f"    AADT×HTS cross-check: AADT={mult:.2f}× vs HTS={hts_mult:.2f}×")
                # If they diverge >2×, widen the SD to reflect uncertainty
                if max(mult,hts_mult)/max(min(mult,hts_mult),0.01) > 2.0:
                    msd = max(msd, abs(mult-hts_mult)*0.5)
                    log(f"    Divergence >2×: widened SD to {msd:.2f}")
        elif hts and sd["hts_region"] in hts:
            h = hts[sd["hts_region"]]
            wn = h["walk_daily_trips"]*sd["hts_colocation_factor"]
            bn = h["bike_daily_trips"]*sd["hts_colocation_factor"]
            mult = (wn+bn)/lr if lr>0 else 1.0; msd = mult*0.25
            src = f"HTS: {h['walk_daily_trips']:,.0f}×{sd['hts_colocation_factor']:.0%}={wn:,.0f}"
        else:
            wt = sd["abs_workers"]*sd["abs_attendance"]*0.10
            mult = wt/lr if lr>0 else 1.0; msd = mult*0.35; src = "ABS fallback"
        # Output both key variants: calibrate_model names + model SITE_CONFIG names
        p["residual_ambient_mult"]=round(mult,2); p["residual_ambient_sd"]=round(msd,2)
        p["ambient_ped_multiplier"]=round(mult,2); p["ambient_ped_sd"]=round(msd,2)
        p["residual_ambient_conv_mean"]=PRIORS["ambient_conv"][0]
        p["residual_ambient_conv_sd"]=round(PRIORS["ambient_conv"][1]*sf,4)
        p["ambient_conv_mean"]=PRIORS["ambient_conv"][0]
        p["ambient_conv_sd"]=round(PRIORS["ambient_conv"][1]*sf,4)
        if wifi and site=="Alinga Street" and lr>0:
            fl = wifi["civic_daily_floor"]; obs = lr+bf+bd
            imp = (fl-obs)/lr if fl>obs else 0
            if imp>mult:
                p["residual_ambient_mult"]=round(imp,2); p["ambient_ped_multiplier"]=round(imp,2)
                log(f"    WiFi floor binding: {imp:.2f}×>{mult:.2f}×")
        log(f"    Ambient: {p['residual_ambient_mult']}×±{p['residual_ambient_sd']}  ({src})")
        p["conversion_sd_factor"]=sf
        p["lr_conv_morning"]=[PRIORS["lr_conv_morning"][0],round(PRIORS["lr_conv_morning"][1]*sf,4)]
        p["lr_conv_daytime"]=[PRIORS["lr_conv_daytime"][0],round(PRIORS["lr_conv_daytime"][1]*sf,4)]
        p["lr_conv_evening"]=[PRIORS["lr_conv_evening"][0],round(PRIORS["lr_conv_evening"][1]*sf,4)]
        p["lr_conv_latenight"]=list(PRIORS["lr_conv_latenight"])
        if lr_d: p["lr_weekday_cv"]=round(lr_d["weekday_cv"],4); p["lr_weekend_ratio"]=round(lr_d["weekend_ratio"],2)
        if pt: p["bus_lr_system_ratio"]=round(pt["ratio"],2); p["bus_lr_site_ratio"]=round(br/lr,1) if lr>0 else 0
        p["lr_daily"]=round(lr); p["lr_segment_shares"]=sv.get("lr_segment_shares",{})
        # Append AADT corridor data for audit / downstream use
        if aadt and site in aadt:
            aa = aadt[site]
            p["aadt_total_vehicles"]=round(aa["aadt_total_vehicles"])
            p["aadt_ped_daily"]=round(aa["aadt_ped_daily"])
            p["aadt_ped_sd"]=round(aa["aadt_ped_sd"])
            p["aadt_weekend_factor"]=round(aa["aadt_weekend_factor"],2)
            p["aadt_roads"]=aa["roads"]
            p["ambient_source"]="AADT"
        elif hts and sd["hts_region"] in hts:
            p["ambient_source"]="HTS"
        else:
            p["ambient_source"]="ABS_fallback"
        # Addinsight corridor speed → blend into ambient pedestrian amplifier
        if addinsight and site in addinsight:
            ai = addinsight[site]
            if ai["pct_enough_data"] >= 40:
                old_mult = p["residual_ambient_mult"]
                # Blend: 70% existing (AADT/HTS), 30% real-time Addinsight speed signal
                blended = round(0.7 * old_mult + 0.3 * (old_mult * ai["ped_amplifier"]), 2)
                p["residual_ambient_mult"] = blended
                p["ambient_ped_multiplier"] = blended
                log(f"    Addinsight blend: {old_mult}× → {blended}× "
                    f"(speed={ai['avg_speed_kmh']} km/h, amp={ai['ped_amplifier']})")
            else:
                log(f"    Addinsight: skipped (only {ai['pct_enough_data']:.0f}% data coverage)")
            p["addinsight_speed_kmh"] = ai["avg_speed_kmh"]
            p["addinsight_disrupted"] = ai["is_disrupted"]
            p["addinsight_n_links"] = ai["n_links"]
        out[site] = p
    return out

def main():
    ap = argparse.ArgumentParser(description="Calibrate After Dark model")
    ap.add_argument("--boardings"); ap.add_argument("--alightings")
    ap.add_argument("--hts-method"); ap.add_argument("--bike-barometer")
    ap.add_argument("--lr-daily"); ap.add_argument("--lr-15min")
    ap.add_argument("--pt-service"); ap.add_argument("--wifi-monthly")
    ap.add_argument("--aadt-csv", help="AADT corridors CSV (aadt_corridors.csv)")
    ap.add_argument("--addinsight", help="Addinsight corridor summary CSV (addinsight_corridor_summary.csv)")
    ap.add_argument("--addinsight-links", help="Raw Addinsight links GeoJSON for AADT estimation")
    ap.add_argument("--outdir", default="calibration_output")
    args = ap.parse_args()
    od = Path(args.outdir); od.mkdir(parents=True, exist_ok=True)
    log("="*70+"\nAfter Dark — Model Calibration\n"+"="*70)
    st = compute_stops(args.boardings, args.alightings) if args.boardings and args.alightings else {}
    ht = compute_hts(args.hts_method) if args.hts_method else None
    bk = compute_bike(args.bike_barometer) if args.bike_barometer else None
    ld = compute_lr_daily(args.lr_daily) if args.lr_daily else None
    l5 = compute_lr_15(args.lr_15min) if args.lr_15min else None
    pt = compute_pt(args.pt_service) if args.pt_service else None
    wf = compute_wifi(args.wifi_monthly) if args.wifi_monthly else None
    aa = compute_aadt(args.aadt_csv) if args.aadt_csv else None
    ai = compute_addinsight(args.addinsight, getattr(args, 'addinsight_links', None)) if args.addinsight else None
    params = derive(st, ht, bk, ld, l5, pt, wf, aadt=aa, addinsight=ai)
    with open(od/"calibration_params.json","w") as f: json.dump(params,f,indent=2,default=str)
    with open(od/"calibration_audit.txt","w") as f: f.write("\n".join(_audit))
    log(f"\n→ {od/'calibration_params.json'}\n→ {od/'calibration_audit.txt'}")
    log("\n"+"="*70+"\nSUMMARY\n"+"="*70)
    for s,sp in params.items():
        log(f"  {s}: LR={sp['lr_daily']} Bus={sp['bus_daily_raw']} Bike={sp['bike_daily']} "
            f"Ambient={sp['residual_ambient_mult']}×±{sp['residual_ambient_sd']}")
    log("\nSOURCES:")
    for l,v in [("Boardings",args.boardings),("Alightings",args.alightings),("HTS",args.hts_method),
                ("Bike",args.bike_barometer),("LR Daily",args.lr_daily),("LR 15min",args.lr_15min),
                ("PT Service",args.pt_service),("WiFi",args.wifi_monthly),("AADT",args.aadt_csv),
                ("Addinsight",args.addinsight)]:
        log(f"  {'✓' if v else '✗'} {l}: {v or 'N/A'}")
    with open(od/"calibration_audit.txt","w") as f: f.write("\n".join(_audit))

if __name__=="__main__": main()