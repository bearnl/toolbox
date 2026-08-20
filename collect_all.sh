#!/bin/bash
# One-shot readout of every job queued in this session: waves 11, 13, 14, 15 and
# the prediction-saving probe. Safe to run before everything finishes -- each
# section reports what is present and says so when an arm is still missing.
cd ~/toolbox || exit 1
source ~/venvs/venv311/bin/activate
R=$SCRATCH/hiride2

echo "=================== 1. job states (anything not COMPLETED is a problem) ==================="
sacct -u "$USER" -S now-24hours -X --format=JobID%16,JobName%16,State%12,Elapsed,MaxRSS \
  | grep -E "hiride|JobID" | tail -40
echo
echo "--- failures, with the reason ---"
BAD=$(sacct -u "$USER" -S now-24hours -X -n -P --format=JobID,State,JobName \
      | grep -Ev "COMPLETED|RUNNING|PENDING" | head -20)
if [ -z "$BAD" ]; then echo "  none"; else
  echo "$BAD"
  # Match the FAILED task exactly. Globbing on the array's parent id picked
  # whichever log was newest -- often a task that succeeded -- so the failure
  # printed no error at all.
  for jid in $(echo "$BAD" | cut -d'|' -f1 | head -6); do
    f=$(ls -t logs/*"$jid".out 2>/dev/null | head -1)
    [ -z "$f" ] && f=$(ls -t logs/*"${jid/_/_*}"*.out 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      echo "  --- $jid -> $f ---"
      grep -E "Error|Traceback|error:|Killed|OOM|SystemExit|Exception" "$f" | tail -6
      tail -3 "$f"
    else
      echo "  --- $jid: no log found ---"
    fi
  done
fi

echo; echo "=================== 2. ladder + best-recipe cells, ORIGINAL prep ==================="
python hiride_collate.py --runs "$R/runs" --floor "$R/results"

echo; echo "=================== 3. boundary-corrected shards (waves 11, 13) ==================="
python hiride_collate.py --runs "$R/runs_edges" --floor "$R/results"

echo; echo "=================== 4. DOES THE BOUNDARY FIX DO ANYTHING? (wave 13 vs 14) ==========="
python hiride_compare.py --runs "$R/runs" "$R/runs_edges" --axis prep

echo; echo "=================== 5. IS THE R4 DEFICIT A FRAMING SHIFT? (wave 15 vs 14) =========="
python hiride_compare.py --runs "$R/runs" --axis eligibility

echo; echo "=================== 6. head and augmentation, restated as paired deltas ============"
python hiride_compare.py --runs "$R/runs" --axis head
python hiride_compare.py --runs "$R/runs" --axis augment

echo; echo "=================== 7. tracklet fusion (window, not frame, is the unit) ============"
python - <<'PY'
import json, glob, os
rows = []
for f in glob.glob(os.path.expandvars("$SCRATCH/hiride2/runs*/results_*_tf*.json")):
    r = json.load(open(f))
    if r.get("fused_acc") is not None:
        rows.append((r["policy"], r["modality"], r["condition"], r.get("head", "gap"),
                     r.get("eligibility", "cues"), r["seed"],
                     100 * r["frame_acc"], 100 * r["fused_acc"], r["fused_n"],
                     os.path.basename(os.path.dirname(f))))
if not rows:
    print("  (none yet)")
for p, m, c, h, e, s, fa, fu, n, src in sorted(rows):
    print(f"  {p:<20s}{m:<6s}{c:<15s}{h:<8s}{e:<10s}s{s} {src:<11s} "
          f"frame {fa:6.2f}%  tracklet {fu:6.2f}%  ({n} windows)")
PY

echo; echo "=================== 8. within-bin truncation contrast (the clean test) ============="
P=$(ls -t "$R"/results/signal_preds_*.npz 2>/dev/null | head -1)
if [ -z "$P" ]; then echo "  (hiride-preds has not written signal_preds_*.npz yet)"; else
  echo "  using $P"
  python hiride_range_profile.py --prep "$R/prep" --policy R4_cross_session \
    --signal "$(ls -t "$R"/results/signal_diagnostic*.json | head -1)" --preds "$P"
fi

echo; echo "=================== 9. subject-cluster CIs on the headline cells ==================="
python hiride_stats.py --runs "$R/runs" --json "$R/results/stats_final.json"

echo; echo "=================== done ==================="
echo "Read in this order: (1) nothing failed -> (4) and (5) are the two new questions ->"
echo "(8) is the only test that separates truncation from distance -> (9) is what the"
echo "paper can actually claim, because seeds are not subjects."
