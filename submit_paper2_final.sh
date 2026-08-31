#!/bin/bash
# THE FINAL PRE-MANUSCRIPT BATCH -- everything outstanding, one submission.
#
#     cd ~/toolbox && git pull && bash submit_paper2_final.sh
#
# Run on a LOGIN node (the dataset download and weight caching need internet).
# Next morning:  bash collect_final.sh
#
# What this submits and why (HIRIDE_HANDOFF sections in brackets):
#   A  rgbseg    GPU     RGB-derived person masks, torchvision DeepLabV3 [13.15]
#   B  wave 20   GPU 30  rgb mask conditions with those masks (needs A)
#   C  wave 19   GPU 60  feature-level fusion: --aux metric vs none, best recipe
#                        [13.10 / 12.8 item 4]
#   D  wave 16   GPU 38  idempotent top-up of interior_only/erode cells, in case
#                        array 20165081 (wave 16b) did not fully land [13.11.1]
#   E  tvrid     CPU     second-corpus download+prep (external validity) [14.3]
#   F  wave 21   GPU 46  the TVRID ladder (needs E; R4-named rung = cross-passage)
#   G  latency   GPU     measured efficiency negative result [2.8 / 11.4]
#   H  signal    CPU 6h  hiride-signal-fair re-run past the 2 h wall [12.8 item 3]
#   I  analysis  CPU     retag audit; stats_final.json at --boot 20000 (order-
#                        independent intervals, 13.7); metric-floor CIs [12.8.1];
#                        sequence contrasts incl. the paired fusion-metric CI at
#                        W=25; cohort curve with retrained fusion [13.18]; fusion,
#                        dropcorr, aux/mask-source paired deltas; figures; tables.
#                        Runs afterany:everything, so a failed cell cannot block
#                        the morning report.
#
# GPU budget stated up front (working agreement, HIRIDE_HANDOFF section 10):
# ~170 GPU cells at 2-6 min each plus two ~30 min single jobs = roughly
# 10-14 GPU-hours on 1g.10gb MIG slices, %8-16 concurrency -> done overnight.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PREP="${PREP:-$SCRATCH/hiride2/prep}"
RUNS="${RUNS:-$SCRATCH/hiride2/runs}"
RES="${RES:-$SCRATCH/hiride2/results}"
IPREP="${IPREP:-$SCRATCH/hiride2/prep_tvrid}"
IRUNS="${IRUNS:-$SCRATCH/hiride2/runs_tvrid}"
IDATA="${IDATA:-$SCRATCH/datasets/tvrid}"
ACC_CPU=def-czarnuch_cpu
ACC_GPU=def-czarnuch_gpu
GRES="gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"

echo "== 0. sanity =="
[ -f "$PREP/prep_meta.json" ] || { echo "no certified prep at $PREP"; exit 1; }
source ~/venvs/venv311/bin/activate

echo "== 1. torch venv + segmentation weights (login node, cached) =="
if [ ! -d ~/venvs/venv-torch ]; then
    module load StdEnv/2023 python/3.11 2>/dev/null || true
    python -m venv ~/venvs/venv-torch
    # --no-index: Compute Canada wheelhouse only, so nothing can clobber a
    # CC-patched wheel the way `pip install matplot` nearly did (13.5).
    ~/venvs/venv-torch/bin/pip install --no-index torch torchvision scipy numpy
fi
export TORCH_HOME="$HOME/.torch"
~/venvs/venv-torch/bin/python - <<'PY'
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
print("[weights] deeplabv3_resnet50 cached")
PY

echo "== 2. second corpus download: TVRID (Zenodo 10.5281/zenodo.20070280, CC-BY-4.0) =="
# Verified 2026-08-31: open access, no registration, sizes+md5 from the record.
# original.zip holds train/<person>/<cam>/<passage> AND test_public/<hash>
# (de-anonymised by TVRID_labels/test_secret_map.csv) -- 88 identities, raw
# 640x480 16-bit depth PNGs. ~15 GiB; wget -c makes re-runs resumable.
mkdir -p "$IDATA"
cd "$IDATA"
wget -c -O TVRID_labels.zip "https://zenodo.org/records/20070280/files/TVRID_labels.zip?download=1"
wget -c -O original.zip     "https://zenodo.org/records/20070280/files/original.zip?download=1"
md5sum -c - <<'MD5' || { echo "TVRID md5 MISMATCH -- delete the bad file and rerun"; exit 1; }
0bb96a896b2f1059b4433c73370acfcf  TVRID_labels.zip
d5821e51dc3aba7608e84156265e8d01  original.zip
MD5
unzip -q -o TVRID_labels.zip
cd "$OLDPWD"

echo "== 3. runs files =="
python make_runs.py --wave 19 > runs19.txt
python make_runs.py --wave 20 > runs20.txt
python make_runs.py --wave 16 > runs16.txt
python make_runs.py --wave 21 > runs21.txt
wc -l runs19.txt runs20.txt runs16.txt runs21.txt

echo "== 4. embedded job scripts =="
cat > .final_rgbseg.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-rgbseg
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11 cuda cudnn
export TORCH_HOME="$HOME/.torch"
~/venvs/venv-torch/bin/python -u hiride_rgbseg.py --prep "$PREP"
SH

cat > .final_tvridprep.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-tvridprep
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
D="$SLURM_TMPDIR/tvrid"; mkdir -p "$D"
echo "[$(date)] staging original.zip to node-local disk"
cp "$IDATA/original.zip" "$D/"
( cd "$D" && { unzip -q original.zip || python -m zipfile -e original.zip .; } \
  && rm -f original.zip )
cp -r "$IDATA/TVRID_labels" "$D/"
echo "  train tracklet dirs: $(find "$D/train" -mindepth 3 -maxdepth 3 -type d 2>/dev/null | wc -l)"
echo "  test tracklet dirs:  $(find "$D/test_public" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
python -u hiride_tvrid_prep.py --root "$D" --labels "$D/TVRID_labels" --out "$IPREP"
SH

cat > .final_latency.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-latency
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11 cuda cudnn
source ~/venvs/venv311/bin/activate
python -u hiride_latency.py --out "$RES"
SH

cat > .final_signal.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-signal-fair
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
python -u hiride_signal.py --prep "$PREP" --per-class 400 --out "$RES"
SH

cat > .final_analysis.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-analysis
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out
set -uo pipefail   # deliberately no -e: every step reports; none may silence the rest
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
R="$RES"; mkdir -p "$R"
step() { echo; echo "===================== $* ====================="; }

step "retag audit (must report NOTHING to rename)"
python hiride_retag.py --runs "$RUNS"

step "subject-cluster CIs, order-independent bootstrap (stats_final.json)"
python hiride_stats.py --runs "$RUNS" --boot 20000 --json "$R/stats_final.json"

step "metric floor with CIs -- full-body test gate (the 28.06 / 32.06 rows)"
python hiride_metric_floor.py --prep "$PREP" --test-eligibility full_body \
    --out "$R" && mv "$R/metric_floor.json" "$R/metric_floor_fbtest.json"
step "metric floor with CIs -- headline (19.04) configuration (runs LAST so metric_floor.json is the headline file)"
python hiride_metric_floor.py --prep "$PREP" --out "$R"

step "sequence: operating point, gated, best recipe (paired fusion-metric CI at W=25)"
python hiride_sequence.py --prep "$PREP" --runs "$RUNS" \
    --arch alexnet/stripe/aug8/tf10 --condition scale_removed --full-body \
    --out "$R" --out-name sequence_gated_best.json
step "sequence: same, ungated (the no-gate comparison row)"
python hiride_sequence.py --prep "$PREP" --runs "$RUNS" \
    --arch alexnet/stripe/aug8/tf10 --condition scale_removed \
    --out "$R" --out-name sequence_ungated_best.json

step "cohort curve with retrained-CNN fusion column"
python hiride_cohort.py --prep "$PREP" --runs "$RUNS" \
    --arch alexnet/stripe/aug8/tf10 --condition scale_removed --full-body --out "$R"

step "score-level fusion table (regenerated)"
python hiride_fuse.py --prep "$PREP" --runs "$RUNS" --out "$R"

step "wave 19 readout: does feature-level fusion move the number?"
python hiride_compare.py --runs "$RUNS" --axis aux
step "wave 20 readout: rgb with its own segmenter vs the userMap"
python hiride_compare.py --runs "$RUNS" --axis mask_source
[ -f "$PREP/rgbseg_meta.json" ] && python -c "import json;print(json.dumps(json.load(open('$PREP/rgbseg_meta.json'))['sequences'],indent=1))"

step "per-identity R0->R1 drop vs near-duplicate ratio"
python hiride_dropcorr.py --runs "$SCRATCH/hiride2/runs_inhouse" \
    --probe "$R/inhouse_probe.json" --biwi-runs "$RUNS" --out "$R"

step "second corpus (TVRID): the ladder -- R4-named rung is CROSS-PASSAGE here"
python hiride_collate.py --runs "$IRUNS" --floor "$R" 2>/dev/null \
    || echo "(no tvrid runs yet)"
python hiride_stats.py --runs "$IRUNS" --boot 20000 \
    --json "$R/stats_tvrid.json" 2>/dev/null || true

step "figures + tables regenerated from the artifacts above"
python hiride_figures.py --stats "$R/stats_final.json" \
    --range "$R/range_profile.json" --prep "$PREP" --out "$R/figs"
python hiride_report.py --runs "$RUNS" --floor "$R" --latex > "$R/tables.tex"
python hiride_report.py --runs "$RUNS" --floor "$R" > "$R/report.md"
echo; echo "analysis complete -> $R"
SH

echo "== 5. submit =="
EXP="ALL,PREP=$PREP,RUNS=$RUNS,RES=$RES,IPREP=$IPREP,IRUNS=$IRUNS,IDATA=$IDATA"
# wave 21 runs against the SECOND corpus: its own PREP, stated once (never rely
# on duplicate keys in --export)
EXP21="ALL,PREP=$IPREP,RUNS=$RUNS,RES=$RES,IPREP=$IPREP,IRUNS=$IRUNS,IDATA=$IDATA"
JSEG=$(sbatch --parsable --account=$ACC_GPU --gres=$GRES --export="$EXP" .final_rgbseg.sh)
JW20=$(sbatch --parsable --account=$ACC_GPU --dependency=afterok:$JSEG \
       --array=1-$(wc -l < runs20.txt)%8 \
       --export="$EXP,RUNS_FILE=$PWD/runs20.txt,OUT=$RUNS" run_hiride.slurm)
JW19=$(sbatch --parsable --account=$ACC_GPU \
       --array=1-$(wc -l < runs19.txt)%8 \
       --export="$EXP,RUNS_FILE=$PWD/runs19.txt,OUT=$RUNS" run_hiride.slurm)
JW16=$(sbatch --parsable --account=$ACC_GPU \
       --array=1-$(wc -l < runs16.txt)%8 \
       --export="$EXP,RUNS_FILE=$PWD/runs16.txt,OUT=$RUNS" run_hiride.slurm)
JIPR=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .final_tvridprep.sh)
JW21=$(sbatch --parsable --account=$ACC_GPU --dependency=afterok:$JIPR \
       --array=1-$(wc -l < runs21.txt)%8 \
       --export="$EXP21,RUNS_FILE=$PWD/runs21.txt,OUT=$IRUNS" run_hiride.slurm)
JLAT=$(sbatch --parsable --account=$ACC_GPU --gres=$GRES --export="$EXP" .final_latency.sh)
JSIG=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .final_signal.sh)
JANA=$(sbatch --parsable --account=$ACC_CPU \
       --dependency=afterany:$JSEG:$JW20:$JW19:$JW16:$JIPR:$JW21:$JLAT:$JSIG \
       --export="$EXP" .final_analysis.sh)

cat <<EOT

submitted:
  $JSEG  rgbseg          (GPU, ~30-60 min)
  $JW20  wave 20         (GPU array $(wc -l < runs20.txt), after rgbseg)
  $JW19  wave 19         (GPU array $(wc -l < runs19.txt))
  $JW16  wave 16 top-up  (GPU array $(wc -l < runs16.txt), idempotent)
  $JIPR  tvrid prep      (CPU, stages + shards the second corpus)
  $JW21  wave 21         (GPU array $(wc -l < runs21.txt), after prep -- TVRID ladder)
  $JLAT  latency         (GPU, ~15 min)
  $JSIG  signal-fair     (CPU, 6 h wall)
  $JANA  ANALYSIS        (CPU, afterany: all of the above)

tomorrow:  bash collect_final.sh
EOT

cat > collect_final.sh <<'SH'
#!/bin/bash
cd "$(dirname "$0")"
R="${RES:-$SCRATCH/hiride2/results}"
echo "=================== job states (anything not COMPLETED is a problem) ==================="
sacct -u "$USER" -S now-24hours -X --format=JobID%16,JobName%18,State%12,Elapsed \
  | grep -E "hiride|JobID" | tail -40
echo
echo "--- failures, exact task logs ---"
BAD=$(sacct -u "$USER" -S now-24hours -X -n -P --format=JobID,State,JobName \
      | grep -Ev "COMPLETED|RUNNING|PENDING" | head -20)
if [ -z "$BAD" ]; then echo "  none"; else
  echo "$BAD"
  for jid in $(echo "$BAD" | cut -d'|' -f1 | head -8); do
    f=$(ls -t logs/*"$jid".out 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "  --- $jid -> $f"; grep -E "Error|error:|Traceback|Killed|OOM" "$f" | tail -4; tail -2 "$f"; }
  done
fi
echo
echo "=================== the analysis job's full output ==================="
f=$(ls -t logs/hiride-analysis_*.out 2>/dev/null | head -1)
[ -n "$f" ] && grep -v "oneDNN\|cuInit\|TF-TRT\|cpu_feature\|^2026-" "$f" || echo "(not finished yet)"
SH
chmod +x collect_final.sh
