#!/bin/bash
# One-shot readout of the overnight batch.
cd ~/toolbox
source ~/venvs/venv311/bin/activate
echo "=================== job states ==================="
sacct -u $USER -S now-14hours -X -P --format=JobID%16,JobName%22,State,Elapsed | grep -E "hiride" | tail -30
echo; echo "=================== boundary-fix prep ==================="
grep -hE "^\[done\]|straddled|Error|Traceback" logs/hiride-edges_*.out 2>/dev/null | tail -5
echo; echo "=================== waves 10 + 12 (existing prep) ==================="
python hiride_collate.py --runs $SCRATCH/hiride2/runs --floor $SCRATCH/hiride2/results \
  | grep -E "arch |stripe|flatten|aug|tf10|depth_sil|interior"
echo; echo "=================== wave 11 (boundary-fixed shards) ==================="
python hiride_collate.py --runs $SCRATCH/hiride2/runs_edges --floor $SCRATCH/hiride2/results 2>/dev/null | head -20
echo; echo "=================== tracklet-fused cells ==================="
python - <<'PY'
import json,glob,os
rows=[]
for f in glob.glob(os.path.expandvars("$SCRATCH/hiride2/runs*/results_*_tf*.json")):
    r=json.load(open(f))
    if "fused_acc" in r:
        rows.append((r["policy"],r["modality"],r["condition"],r.get("head","gap"),
                     r["seed"],100*r["frame_acc"],100*r["fused_acc"],r["fused_n"]))
if not rows: print("  (none yet)")
for p,m,c,h,s,fa,fu,n in sorted(rows):
    print(f"  {p:<20s}{m:<6s}{c:<16s}{h:<8s}s{s}  frame {fa:6.2f}%  tracklet {fu:6.2f}%  ({n} windows)")
PY
echo; echo "=================== probe on rebuilt shards ==================="
grep -hE "^depth|^rgb|representation" logs/hiride-signal-edges_*.out 2>/dev/null | head -20
