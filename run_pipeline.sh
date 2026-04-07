#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — After Dark, On Tap full pipeline runner
# ============================================================================
#
# Runs the complete data pipeline from raw data acquisition through
# calibration, demand modelling, scenario analysis, and sensitivity sweep.
#
# Prerequisites:
#   pip install pandas numpy openpyxl requests lxml beautifulsoup4 matplotlib
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                    # full run (all stages)
#   ./run_pipeline.sh --skip-fetch       # skip data download (use cached)
#   ./run_pipeline.sh --skip-calibrate   # skip calibration (use existing params)
#   ./run_pipeline.sh --sim-only         # skip fetch + calibrate, run model only
#   ./run_pipeline.sh --quick            # 10k sims instead of 100k
#
# Output:
#   output/
#     calibration_data/       raw downloads from ACT Open Data + BoM
#     calibration_output/     calibration_params.json + audit
#     support_data/           weather.csv, events.csv, ops.csv, callouts.csv
#     model_output/           charts, summaries, sensitivity sweep
#     scenarios/              per-scenario results
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"
CAL_DATA="${OUTPUT_DIR}/calibration_data"
CAL_OUT="${OUTPUT_DIR}/calibration_output"
SUPPORT="${OUTPUT_DIR}/support_data"
MODEL_OUT="${OUTPUT_DIR}/model_output"
SCENARIO_OUT="${OUTPUT_DIR}/scenarios"

# Pipeline scripts (expected in same directory)
COLLECTOR="${SCRIPT_DIR}/collect_calibration_data.py"
BUILDER="${SCRIPT_DIR}/canberra_support_data_builder_v14.py"
CALIBRATOR="${SCRIPT_DIR}/calibrate_model.py"
MODEL="${SCRIPT_DIR}/after_dark_stop_hour_model_v12.py"

# Data files — set these if your boardings/alightings CSVs are elsewhere
BOARDINGS="${SCRIPT_DIR}/boardings_by_stop_qh.csv"
ALIGHTINGS="${SCRIPT_DIR}/alightings_by_stop_qh.csv"
AADT_CSV="${SCRIPT_DIR}/aadt_corridors.csv"
HTS_DIR="${SCRIPT_DIR}"  # directory containing ACT_HTS_-_01_*.xlsx etc.

N_SIM=100000
SKIP_FETCH=false
SKIP_CALIBRATE=false
SIM_ONLY=false

# ── Parse arguments ────────────────────────────────────────────────────────

for arg in "$@"; do
  case "$arg" in
    --skip-fetch)     SKIP_FETCH=true ;;
    --skip-calibrate) SKIP_CALIBRATE=true ;;
    --sim-only)       SKIP_FETCH=true; SKIP_CALIBRATE=true ;;
    --quick)          N_SIM=10000 ;;
    --help|-h)
      head -28 "$0" | tail -22
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (try --help)"
      exit 1
      ;;
  esac
done

# ── Preflight checks ──────────────────────────────────────────────────────

echo "========================================================================"
echo "  After Dark, On Tap — Full Pipeline Run"
echo "========================================================================"
echo ""

missing=()
for script in "$COLLECTOR" "$BUILDER" "$CALIBRATOR" "$MODEL"; do
  [ -f "$script" ] || missing+=("$(basename "$script")")
done
for data in "$BOARDINGS" "$ALIGHTINGS" "$AADT_CSV"; do
  [ -f "$data" ] || missing+=("$(basename "$data")")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing files: ${missing[*]}"
  echo "Place all pipeline scripts and data files in: $SCRIPT_DIR"
  exit 1
fi

python3 -c "import pandas, numpy, matplotlib" 2>/dev/null || {
  echo "ERROR: Missing Python dependencies."
  echo "  pip install pandas numpy openpyxl requests lxml beautifulsoup4 matplotlib"
  exit 1
}

mkdir -p "$OUTPUT_DIR" "$CAL_DATA" "$CAL_OUT" "$SUPPORT" "$MODEL_OUT" "$SCENARIO_OUT"

echo "  Scripts:     $(basename "$COLLECTOR"), $(basename "$CALIBRATOR"), $(basename "$MODEL")"
echo "  Boardings:   $BOARDINGS"
echo "  Alightings:  $ALIGHTINGS"
echo "  AADT:        $AADT_CSV"
echo "  Simulations: ${N_SIM}"
echo "  Skip fetch:  ${SKIP_FETCH}"
echo "  Skip calib:  ${SKIP_CALIBRATE}"
echo ""

# ── Stage 1: Acquire ──────────────────────────────────────────────────────

if [ "$SKIP_FETCH" = false ]; then
  echo "========================================================================"
  echo "  STAGE 1: Acquire — downloading public datasets"
  echo "========================================================================"
  echo ""

  # 1a. ACT Open Data (Socrata API)
  echo "── 1a. ACT Open Data (via collect_calibration_data.py) ──"
  python3 "$COLLECTOR" \
    --outdir "$CAL_DATA" \
    --hts-dir "$HTS_DIR" \
    --skip-abs \
    2>&1 | tee "${OUTPUT_DIR}/01_collect.log"
  echo ""

  # 1b. BoM weather history (14 months)
  echo "── 1b. BoM weather history (14 months) ──"
  python3 "$BUILDER" fetch-weather-history \
    --output "${SUPPORT}/weather.csv" \
    --station-code IDCJDW2801 \
    --months 14 \
    2>&1 | tee "${OUTPUT_DIR}/02_weather.log"
  echo ""

  # 1c. Events (Canberra events page → normalize)
  echo "── 1c. Events ──"
  python3 "$BUILDER" fetch-events \
    --output "${SUPPORT}/raw_events.html" \
    2>&1 || echo "  (fetch-events failed — will use fallback priors)"

  if [ -f "${SUPPORT}/raw_events.html" ]; then
    python3 "$BUILDER" normalize-events \
      --input "${SUPPORT}/raw_events.html" \
      --output "${SUPPORT}/events.csv" \
      2>&1 | tee -a "${OUTPUT_DIR}/03_events.log"
  fi
  echo ""

  # 1d. Operations (Transport Canberra alerts → normalize)
  echo "── 1d. Operations ──"
  python3 "$BUILDER" fetch-ops \
    --output "${SUPPORT}/raw_ops.html" \
    2>&1 || echo "  (fetch-ops failed — will use fallback priors)"

  if [ -f "${SUPPORT}/raw_ops.html" ]; then
    python3 "$BUILDER" normalize-ops \
      --input "${SUPPORT}/raw_ops.html" \
      --output "${SUPPORT}/ops.csv" \
      --include-regions "Central Canberra,Gungahlin" \
      --include-keywords "light rail,detour,diversion,closure,relocation,disruption,road closure" \
      2>&1 | tee -a "${OUTPUT_DIR}/04_ops.log"
  fi
  echo ""

  # 1e. ESA callouts (emergency services → normalize)
  echo "── 1e. ESA callouts ──"
  python3 "$BUILDER" fetch-callouts \
    --output "${SUPPORT}/raw_callouts.xml" \
    2>&1 || echo "  (fetch-callouts failed — will use Poisson prior)"

  if [ -f "${SUPPORT}/raw_callouts.xml" ]; then
    python3 "$BUILDER" normalize-callouts \
      --input "${SUPPORT}/raw_callouts.xml" \
      --output "${SUPPORT}/callouts.csv" \
      2>&1 | tee -a "${OUTPUT_DIR}/05_callouts.log"
  fi
  echo ""

  # 1f. Addinsight real-time road network (Bluetooth detectors)
  echo "── 1f. Addinsight corridor traffic ──"
  python3 "$BUILDER" fetch-addinsight \
    --outdir "${SUPPORT}" \
    2>&1 || echo "  (fetch-addinsight failed — will use calibration priors)"

  if [ -f "${SUPPORT}/raw_addinsight_links.json" ]; then
    python3 "$BUILDER" normalize-addinsight \
      --input-dir "${SUPPORT}" \
      --output "${SUPPORT}/addinsight_corridor_summary.csv" \
      2>&1 | tee "${OUTPUT_DIR}/06_addinsight.log"
  fi
  echo ""

  echo "Stage 1 complete."
  echo ""
else
  echo "(Skipping Stage 1 — using cached data)"
  echo ""
fi

# ── Stage 2: Calibrate ────────────────────────────────────────────────────

if [ "$SKIP_CALIBRATE" = false ]; then
  echo "========================================================================"
  echo "  STAGE 2: Calibrate — deriving site-level parameters"
  echo "========================================================================"
  echo ""

  # Build calibrate_model.py argument list from available data
  CAL_ARGS=(
    --boardings "$BOARDINGS"
    --alightings "$ALIGHTINGS"
    --aadt-csv "$AADT_CSV"
    --outdir "$CAL_OUT"
  )

  # Optional inputs — add only if files exist
  [ -f "$HTS_DIR/ACT_HTS_-_01_Method_of_travel.xlsx" ] && \
    CAL_ARGS+=(--hts-method "$HTS_DIR/ACT_HTS_-_01_Method_of_travel.xlsx")

  [ -f "$CAL_DATA/bike_barometer.csv" ] && \
    CAL_ARGS+=(--bike-barometer "$CAL_DATA/bike_barometer.csv")

  [ -f "$CAL_DATA/lr_patronage_daily.csv" ] && \
    CAL_ARGS+=(--lr-daily "$CAL_DATA/lr_patronage_daily.csv")

  [ -f "$CAL_DATA/lr_patronage_15min.csv" ] && \
    CAL_ARGS+=(--lr-15min "$CAL_DATA/lr_patronage_15min.csv")

  [ -f "$CAL_DATA/pt_daily_by_service.csv" ] && \
    CAL_ARGS+=(--pt-service "$CAL_DATA/pt_daily_by_service.csv")

  [ -f "$CAL_DATA/wifi_monthly.csv" ] && \
    CAL_ARGS+=(--wifi-monthly "$CAL_DATA/wifi_monthly.csv")

  [ -f "${SUPPORT}/addinsight_corridor_summary.csv" ] && \
    CAL_ARGS+=(--addinsight "${SUPPORT}/addinsight_corridor_summary.csv")

  [ -f "${SUPPORT}/raw_addinsight_links.json" ] && \
    CAL_ARGS+=(--addinsight-links "${SUPPORT}/raw_addinsight_links.json")

  echo "  Arguments: ${CAL_ARGS[*]}"
  echo ""

  python3 "$CALIBRATOR" "${CAL_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/06_calibrate.log"
  echo ""

  echo "Stage 2 complete → ${CAL_OUT}/calibration_params.json"
  echo ""
else
  echo "(Skipping Stage 2 — using existing calibration_params.json)"
  echo ""
fi

# ── Stage 3: Model — baseline run ─────────────────────────────────────────

echo "========================================================================"
echo "  STAGE 3: Model — demand simulation"
echo "========================================================================"
echo ""

# Build model argument list from available data
MODEL_ARGS=(
  --boardings "$BOARDINGS"
  --alightings "$ALIGHTINGS"
  --calibration "${CAL_OUT}/calibration_params.json"
  --n-sim "$N_SIM"
  --sensitivity
)

# Optional support data
[ -f "${SUPPORT}/weather.csv" ]   && MODEL_ARGS+=(--weather "${SUPPORT}/weather.csv")
[ -f "${SUPPORT}/events.csv" ]    && MODEL_ARGS+=(--events "${SUPPORT}/events.csv")
[ -f "${SUPPORT}/ops.csv" ]       && MODEL_ARGS+=(--ops "${SUPPORT}/ops.csv")
[ -f "${SUPPORT}/callouts.csv" ]  && MODEL_ARGS+=(--callouts "${SUPPORT}/callouts.csv")
[ -f "${SUPPORT}/addinsight_corridor_summary.csv" ] && MODEL_ARGS+=(--addinsight "${SUPPORT}/addinsight_corridor_summary.csv")
[ -f "$CAL_DATA/lr_patronage_15min.csv" ] && MODEL_ARGS+=(--lr-15min "$CAL_DATA/lr_patronage_15min.csv")

echo "── 3a. Baseline (all scenarios, steady-state) ──"
python3 "$MODEL" "${MODEL_ARGS[@]}" \
  --outdir "${MODEL_OUT}" \
  2>&1 | tee "${OUTPUT_DIR}/07_model_baseline.log"
echo ""

# ── Stage 4: Scenario matrix ──────────────────────────────────────────────

echo "========================================================================"
echo "  STAGE 4: Scenario matrix"
echo "========================================================================"
echo ""

# Core scenario arguments (reuse MODEL_ARGS minus --sensitivity for speed)
SCEN_ARGS=()
for a in "${MODEL_ARGS[@]}"; do
  [ "$a" = "--sensitivity" ] || SCEN_ARGS+=("$a")
done

run_scenario() {
  local label="$1"; shift
  local outdir="${SCENARIO_OUT}/${label}"
  mkdir -p "$outdir"
  echo -n "  ${label}... "
  python3 "$MODEL" "${SCEN_ARGS[@]}" "$@" --outdir "$outdir" \
    > "${outdir}/run.log" 2>&1
  # Extract headline numbers
  grep -E "^Alinga|^Dickson|^Gungahlin" "${outdir}/run.log" | \
    while IFS= read -r line; do
      site=$(echo "$line" | awk '{print $1}')
      nums=( $(echo "$line" | grep -oE '[0-9.]+') )
      printf "%s: paid=%s P(>BE)=%s  " "$site" "${nums[1]}" "${nums[5]}"
    done
  echo ""
}

# Day types
run_scenario "weekday"        --day-type weekday
run_scenario "weekend"        --day-type weekend

# Weather
run_scenario "weather_wet"    --weather-scenario wet
run_scenario "weather_dry"    --weather-scenario dry

# Seasons
run_scenario "season_cold"    --season cold
run_scenario "season_mild"    --season mild
run_scenario "season_warm"    --season warm
run_scenario "season_hot"     --season hot

# Operations
run_scenario "ops_normal"     --ops-scenario normal
run_scenario "ops_disrupted"  --ops-scenario disrupted

# Pre-pilot (ramp)
run_scenario "ramp_12wk"      --ramp-weeks 12
run_scenario "ramp_weekend"   --ramp-weeks 12 --day-type weekend

# Worst-case combos
run_scenario "worst_case"     --ramp-weeks 12 --day-type weekend --season cold
run_scenario "best_case"      --day-type weekday --season mild

echo ""

# ── Stage 5: Summary ──────────────────────────────────────────────────────

echo "========================================================================"
echo "  STAGE 5: Summary"
echo "========================================================================"
echo ""

# Aggregate all scenario results into a single table
{
  echo "# After Dark — Pipeline Run Summary"
  echo ""
  echo "| Scenario | Alinga Paid | Dickson Paid | Gungahlin Paid | A P(>BE) | D P(>BE) | G P(>BE) |"
  echo "|---|---:|---:|---:|---:|---:|---:|"

  for dir in "$MODEL_OUT" "$SCENARIO_OUT"/*/; do
    label=$(basename "$dir")
    [ "$label" = "model_output" ] && label="baseline"
    logfile="${dir}/run.log"
    [ "$label" = "baseline" ] && logfile="${OUTPUT_DIR}/07_model_baseline.log"
    [ -f "$logfile" ] || continue

    a_paid="—"; d_paid="—"; g_paid="—"
    a_pbe="—"; d_pbe="—"; g_pbe="—"

    while IFS= read -r line; do
      nums=( $(echo "$line" | grep -oE '[0-9.]+') )
      if echo "$line" | grep -q "^Alinga"; then
        a_paid="${nums[1]:-—}"; a_pbe="${nums[5]:-—}"
      elif echo "$line" | grep -q "^Dickson"; then
        d_paid="${nums[1]:-—}"; d_pbe="${nums[5]:-—}"
      elif echo "$line" | grep -q "^Gungahlin"; then
        g_paid="${nums[1]:-—}"; g_pbe="${nums[5]:-—}"
      fi
    done < <(grep -E "^Alinga|^Dickson|^Gungahlin" "$logfile" 2>/dev/null)

    echo "| ${label} | ${a_paid} | ${d_paid} | ${g_paid} | ${a_pbe}% | ${d_pbe}% | ${g_pbe}% |"
  done
} > "${OUTPUT_DIR}/pipeline_summary.md"

cat "${OUTPUT_DIR}/pipeline_summary.md"

echo ""
echo "========================================================================"
echo "  PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "  Outputs:"
echo "    ${CAL_OUT}/calibration_params.json    — site parameters"
echo "    ${CAL_OUT}/calibration_audit.txt      — derivation log"
echo "    ${MODEL_OUT}/fig_stop_hour_demand.png — stacked bar chart"
echo "    ${MODEL_OUT}/stop_hour_model_summary.md"
echo "    ${MODEL_OUT}/sensitivity_sweep.md     — conversion rate sweep"
echo "    ${SCENARIO_OUT}/*/                    — per-scenario results"
echo "    ${OUTPUT_DIR}/pipeline_summary.md     — scenario comparison table"
echo ""
echo "  Logs:"
for log in "${OUTPUT_DIR}"/*.log; do
  [ -f "$log" ] && echo "    $log"
done
echo ""
echo "========================================================================"