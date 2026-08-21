#!/bin/bash
# Re-extract metric 3D features, including the crown-anchored shape block.
#
#   sbatch run_hiride_metric.slurm
#
# hiride_metric.py needs the RAW BIWI tree (--root), not the prep shards: it
# unprojects each frame with the Kinect intrinsics and reads the per-frame
# ground plane, neither of which survives into the shards. The tree is staged
# from the .rar archives into node-local $SLURM_TMPDIR, exactly as
# run_hiride_night.slurm does -- two earlier attempts at this as an sbatch
# --wrap one-liner died in argparse for want of --root.
#
# BACKS UP metric_features.npz FIRST. hiride_metric.py writes it in place, and
# that file is the basis of 19.04 % at R4, 67.97 % at R1 and the 18.83 % the
# fusion work quotes. A half-written file would take those with it.
#SBATCH --job-name=hiride-metric
#SBATCH --account=def-czarnuch_cpu
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
mkdir -p logs
PREP="${PREP:-$SCRATCH/hiride2/prep}"

if [ -f "$PREP/metric_features.npz" ] && [ ! -f "$PREP/metric_features_12col.npz.bak" ]; then
    cp "$PREP/metric_features.npz" "$PREP/metric_features_12col.npz.bak"
    echo "[backup] metric_features_12col.npz.bak written"
fi

module purge; module load StdEnv/2023 python/3.11
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"

D="$SLURM_TMPDIR/biwi"; mkdir -p "$D"; cd "$D"
echo "[$(date)] staging archives"
cp "$SCRATCH/datasets/Training.rar" . && unrar x -y Training.rar >/dev/null && rm -f Training.rar
cp "$SCRATCH/datasets/Testing.rar"  . && unrar x -y Testing.rar  >/dev/null && rm -f Testing.rar
echo "  depth=$(find "$D" -name '*_depth.pgm' | wc -l)" \
     "ground=$(find "$D" -name '*_groundCoeff.txt' | wc -l)"

cd "$SLURM_SUBMIT_DIR"
python -u hiride_metric.py --prep "$PREP" --root "$D"
echo "[$(date)] done"
python - <<'PY'
import os, numpy as np
p = os.path.join(os.environ.get("PREP", os.environ["SCRATCH"] + "/hiride2/prep"),
                 "metric_features.npz")
z = np.load(p, allow_pickle=False)
names = [str(n) for n in z["names"]]
new = [n for n in names if n.startswith(("hw_", "ht_", "hc_", "ha_", "r_"))]
print(f"[verify] {z['feats'].shape[0]} frames x {len(names)} columns, "
      f"{len(new)} crown-anchored")
missing = {"stature_mm", "w_45", "surface_area_m2"} - set(names)
assert not missing, f"published columns lost: {missing}"
assert len(new) == 35, f"expected 35 shape columns, got {len(new)}"
PY
