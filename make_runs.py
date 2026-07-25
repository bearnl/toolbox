"""Emit the runs file consumed by run_hiride.slurm's array mode.

    python make_runs.py > runs.txt            # wave 2 (49 lines); --wave 3 -> runs3.txt (100 lines)
    wc -l runs.txt                    # 49 = 40 cells + 9 null controls
    sbatch --account=def-czarnuch_gpu --array=1-$(wc -l < runs.txt)%8 \
           --export=ALL,RUNS_FILE=$PWD/runs.txt run_hiride.slurm

Line numbers (= array task IDs): 1-5 R0 depth, 6-10 R0 rgb, 11-15 R1 depth,
16-20 R1 rgb, 21-25 R3 depth, 26-30 R3 rgb, 31-35 R4 depth, 36-40 R4 rgb,
41-43 R1 perm, 44-46 R3 perm, 47-49 R4 perm.

R3_cross_recording IS included, on the strength of a floor result: it trains on
only 1,821 frames (~65 per subject, 4.3x fewer than R4's 7,894) yet scores 8.55%
against R4's 5.35% on a BYTE-IDENTICAL test set. Less data, better accuracy --
the only difference being whether training came from the same session. That
contrast is worth 10 cells. Its starvation must still be stated whenever its
absolute value is quoted.
"""
import argparse

WAVE2 = dict(
    # policy, extra args
    policies=[("R0_frame_random", ""),           # the retraction figure
              ("R1_block", "--guard 150"),       # within-session reference
              ("R3_cross_recording", ""),        # same session, different recording
              ("R4_cross_session", "")],         # PRIMARY: different day + clothes
    modalities=["depth", "rgb"],
    archs=["alexnet"],
    seeds=[0, 1, 2, 3, 4],
)

# Null controls: if a permuted-label cell lands materially above chance, that
# rung leaks and nothing from it is reportable. Every rung whose CNN number is
# quoted gets one; R3 was added once its depth/RGB arms turned out to be a
# reportable result (33.9 % / 81.1 %, array 18539954) rather than a footnote.
CONTROLS = [("R1_block", "--guard 150"), ("R3_cross_recording", ""),
            ("R4_cross_session", "")]

# Wave 3 -- mechanism suite, part 1: the three single-edit conditions the
# trainer already implements, at every rung of the ladder, both modalities
# (silhouette is modality-free, so once). 100 cells, ~6.5 GPU-h at 2-6 min each.
#   person     background -> constant   : does the collapse at R4 survive when
#                                         the ROOM is removed? (adjacency shows
#                                         Training->Walking differs by 1.3 m
#                                         over the whole frame, i.e. the scene
#                                         changed, not only the clothes)
#   bg_hole    person -> constant       : how much is recording/scene nuisance
#                                         (+ the silhouette-shaped hole)
#   silhouette binary mask only         : outline without metric depth
WAVE3 = dict(
    policies=WAVE2["policies"],
    cells=[("person", "depth"), ("person", "rgb"),
           ("bg_hole", "depth"), ("bg_hole", "rgb"),
           ("silhouette", "depth")],
    seeds=[0, 1, 2, 3, 4],
)


# Wave 4 -- backbone/initialisation control at the rungs that matter. The
# reviewer's first objection to "RGB collapses at R4" is that a from-scratch
# AlexNet on ~8k frames cannot generalise; an ImageNet-initialised ConvNeXt-Tiny
# is the answer either way (identical optimiser, normalisation, splits, seeds).
# 30 cells; needs the weights pre-cached in ~/.keras/models and a 2g.20gb slice:
#   sbatch --account=def-czarnuch_gpu --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 \
#          --array=1-30%6 --export=ALL,RUNS_FILE=$PWD/runs4.txt run_hiride.slurm
WAVE4 = dict(
    policies=[("R1_block", "--guard 150"), ("R3_cross_recording", ""), ("R4_cross_session", "")],
    modalities=["depth", "rgb"],
    seeds=[0, 1, 2, 3, 4],
    init="imagenet",            # --wave 4 --scratch for the from-scratch ConvNeXt arm
)

# Wave 5 -- mechanism suite, part 2: SCALE-REMOVED. Wave 3 showed the depth
# CNN's within-session number is mostly scene (bg_hole ~ full) and that across
# sessions the binary silhouette (10.1 %) beats full depth (6.7 %). What is
# left when apparent size, image position and standing distance are normalised
# away is body shape (+ internal depth structure). 60 AlexNet cells, ~4 GPU-h.
WAVE5 = dict(
    policies=WAVE2["policies"],
    cells=[("scale_removed", "depth"), ("scale_removed", "rgb"), ("sil_scaled", "depth")],
    seeds=[0, 1, 2, 3, 4],
)

# Wave 6 -- (a) POSITION vs SIZE. scale_removed both re-centres and rescales, and
# it beat person-only at every rung (R1 47.6 vs 23.3, R4 12.5 vs 7.9 depth), so
# the two effects must be separated: person_centred is pure integer translation,
# no resampling. (b) The Z-PRECISION AXIS, one of the paper's two novelty claims:
# accuracy vs depth quantisation at fixed global range, run on scale_removed
# because that is the condition carrying the cross-session signal. 90 cells.
WAVE6 = dict(
    policies=WAVE2["policies"],
    cells=[("person_centred", "depth"), ("person_centred", "rgb")],
    seeds=[0, 1, 2, 3, 4],
    bits=[8, 4, 3, 2, 1],
    bits_policies=[("R1_block", "--guard 150"), ("R4_cross_session", "")],
)

# Wave 7 -- the exact-complement scene control (novelty claim 1). bg_hole leaves
# a silhouette-shaped hole, so its R3 result (35.8, ABOVE full's 34.8) cannot
# distinguish "reads the scene" from "reads the hole". bg_plate replaces the
# person with the recording's own background plate, so no person-shaped
# boundary survives anywhere: whatever it scores is scene/recording nuisance
# with the silhouette provably gone. Needs hiride_plates.py first. 40 cells.
#
# SWAP (person A on background B) is deliberately NOT run: BIWI Training is one
# room with a fixed camera and the plates differ between recordings by a median
# of 55 mm (HIRIDE_HANDOFF section 5.3), so a swap changes almost nothing and the
# condition is null by construction here. Say that in the paper rather than
# reporting an uninformative null as a result.
WAVE7 = dict(
    policies=WAVE2["policies"],
    cells=[("bg_plate", "depth"), ("bg_plate", "rgb")],
    seeds=[0, 1, 2, 3, 4],
)

# Wave 8 -- the IN-HOUSE ladder, a sensor-generation replication. Same two rungs
# as BIWI's within-session pair, both modalities, 5 seeds, plus permutation
# nulls: 22 cells, ~1-2 GPU-h. Needs --prep $SCRATCH/hiride2/prep_inhouse and
# --eligibility all (no cues); --condition full only (no masks exist).
#
# Interpretation, fixed in advance: the ABSOLUTE numbers are inflated by the
# room/identity confound -- each recording directory holds a disjoint set of
# people, so "identity" and "room" are not separable here, and unlike BIWI we
# cannot run bg_plate to decompose it. The R0 -> R1 DROP is still attributable,
# because the confound is identical on both rungs. Quote the drop, not the level,
# and quote it against the 34.3 % majority-class rate.
WAVE8 = dict(
    policies=[("R0_frame_random", ""), ("R1_block", "--guard 150")],
    modalities=["depth", "rgb"],
    seeds=[0, 1, 2, 3, 4],
    extra="--eligibility all",
)

# Wave 9 -- BEST-EFFORT DEPTH, on the levers the diagnostics point at. Reported
# as a clearly separate arm from the ladder: the ladder's one-shared-policy
# control is what makes the protocol result trustworthy, and this arm
# deliberately breaks it by giving depth preprocessing RGB cannot have.
#
# Why these and not others (all measured, see HIRIDE_HANDOFF 11.1b):
#   * the dynamic-range sweep is NOT here -- it was run and is flat (R4 13.9 ->
#     14.3 -> 11.4 % from slab 6000 to 300 despite 15x more contrast), because
#     BatchNorm already absorbs a linear input rescale. Paper 3's clip effect was
#     foreground SELECTION, a different operation.
#   * --frames N median fusion IS here: quantisation is ~25 mm at the person's
#     2955 mm median and measured sigma is 40-70 mm against 20-80 mm of
#     between-person curvature, so sqrt(N) denoising is the one intervention
#     aimed at the actual bottleneck. Windows never leave their own split, so no
#     adjacency leak is introduced (see temporal_windows).
#   * interior_only IS here: it is the cell the mechanism suite never had, and it
#     also drops the rim that prep's AREA resize contaminated by up to ~440 mm.
#   * normals ARE here, but only stacked on fusion -- derivatives amplify noise,
#     and a 1-px lateral step at 3 m spans ~5 mm against 40-70 mm of sigma.
# 60 cells, ~4 GPU-h. RGB gets the same fusion treatment so the comparison at
# the best operating point stays honest.
WAVE9 = dict(
    policies=[("R1_block", "--guard 150"), ("R4_cross_session", "")],
    cells=[("interior_only", "depth", ""),
           ("scale_removed", "depth", "--frames 10"),
           ("interior_only", "depth", "--frames 10"),
           ("sil_scaled",    "depth", "--frames 10"),
           ("scale_removed", "depth", "--frames 10 --depth-encoding normals"),
           ("scale_removed", "rgb",   "--frames 10")],
    seeds=[0, 1, 2, 3, 4],
)

# Wave 10 -- THE HEAD. On aligned input a linear nearest-class-mean probe over 96
# PCA components scores 60.5 % at R1 on interior_only while the GAP CNN scores
# 40.9 %, trained on MORE frames (signal_diagnostic.json vs wave 9). A template
# matcher should not beat a 58M-parameter network -- unless the network is
# discarding what the alignment established. scale_removed/interior_only fix the
# person's size and position, so pixel (i,j) means a body location; then
# GlobalAveragePooling averages exactly that away. stripe (average over width,
# keep rows) and flatten (keep the map) preserve it. 60 cells, ~4 GPU-h.
WAVE10 = dict(
    policies=[("R1_block", "--guard 150"), ("R4_cross_session", "")],
    cells=[("scale_removed", "depth", "stripe"), ("scale_removed", "depth", "flatten"),
           ("interior_only", "depth", "stripe"), ("interior_only", "depth", "flatten"),
           ("sil_scaled", "depth", "stripe"), ("scale_removed", "rgb", "stripe")],
    seeds=[0, 1, 2, 3, 4],
)

# Wave 11 -- THE BOUNDARY FIX, on shards rebuilt by hiride_prep_edges.py. The old
# AREA resize averaged person depth (~2955 mm) against background (~3842 mm)
# across the silhouette before the NEAREST-resized mask was applied: measured on
# an elliptical test silhouette, 3.1 % of person pixels carried errors up to
# 444 mm. Eroding the rim away (interior_only) already lifts the linear probe
# 57.6 -> 60.5 % at R1, which is what says the rim really is corrupt; this
# repairs the cause instead of cutting around it. Same conditions as the
# published ladder cells so the comparison is like-for-like. 50 cells.
#   sbatch ... --export=ALL,RUNS_FILE=$PWD/runs11.txt,PREP=$SCRATCH/hiride2/prep_edges,\
#          OUT=$SCRATCH/hiride2/runs_edges run_hiride.slurm
WAVE11 = dict(
    policies=[("R1_block", "--guard 150"), ("R4_cross_session", "")],
    cells=[("full", "depth"), ("scale_removed", "depth"), ("interior_only", "depth"),
           ("sil_scaled", "depth"), ("scale_removed", "rgb")],
    seeds=[0, 1, 2, 3, 4],
)

# Wave 12 -- SPECULATIVE. Cheap shots that have not been tried and each have a
# stated reason to work. Reported as exploratory; anything that survives gets a
# clean confirmatory run afterwards.
#
#  augment N   random translation + flip at TRAIN time. The mechanism suite says
#              framing is the decisive nuisance; every fix so far normalised it
#              away in preprocessing, and nobody tried making the net invariant
#              to it. Standard practice, never run here (the ladder deliberately
#              had no augmentation to keep the modality comparison clean).
#  test-fuse N TRACKLET evaluation: average predicted probabilities over N
#              consecutive test frames. This is the fusion idea done properly --
#              --frames N fused the INPUT and blurred pose, which is why it lost
#              1-3 pp; averaging predictions cannot blur anything, and it is how
#              a deployed camera actually sees a person. Changes the unit of
#              analysis to the window, so it is reported separately.
#  depth_sil   depth and the binary outline as 2 channels. sil_scaled (outline
#              only) beats scale_removed (outline + interior) at R4, which may
#              just mean one channel cannot carry both.
#  erode sweep the rim is contaminated by up to 444 mm (hiride_prep_edges.py);
#              0/1/2/4/6 px finds where cutting stops helping and starts
#              deleting real outline.
#  cnxt+stripe the two levers that each helped, combined: ImageNet init and a
#              head that respects alignment.
WAVE12 = dict(
    policies=[("R1_block", "--guard 150"), ("R4_cross_session", "")],
    cells=[
        ("scale_removed", "depth", "--augment 8", "alexnet"),
        ("scale_removed", "depth", "--augment 16", "alexnet"),
        ("sil_scaled",    "depth", "--augment 8", "alexnet"),
        ("scale_removed", "rgb",   "--augment 8", "alexnet"),
        ("scale_removed", "depth", "--test-fuse 10", "alexnet"),
        ("sil_scaled",    "depth", "--test-fuse 10", "alexnet"),
        ("scale_removed", "rgb",   "--test-fuse 10", "alexnet"),
        ("scale_removed", "depth", "--depth-encoding depth_sil", "alexnet"),
        ("interior_only", "depth", "--erode 1", "alexnet"),
        ("interior_only", "depth", "--erode 4", "alexnet"),
        ("interior_only", "depth", "--erode 6", "alexnet"),
        ("scale_removed", "depth", "--head stripe --augment 8 --test-fuse 10", "alexnet"),
        ("sil_scaled",    "depth", "--head stripe --augment 8 --test-fuse 10", "alexnet"),
        ("scale_removed", "rgb",   "--head stripe --augment 8 --test-fuse 10", "alexnet"),
    ],
    seeds=[0, 1, 2],
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=2,
                    choices=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    ap.add_argument("--scratch", action="store_true", help="wave 4: ConvNeXt from scratch")
    ap.add_argument("--control-seeds", type=int, default=3)
    args = ap.parse_args()

    lines = []
    if args.wave == 2:
        for policy, extra in WAVE2["policies"]:
            for mod in WAVE2["modalities"]:
                for arch in WAVE2["archs"]:
                    for seed in WAVE2["seeds"]:
                        lines.append(f"--policy {policy} --modality {mod} "
                                     f"--arch {arch} --seed {seed} {extra}".strip())
        for policy, extra in CONTROLS:
            for seed in range(args.control_seeds):
                lines.append(f"--policy {policy} --modality depth --arch alexnet "
                             f"--seed {seed} --permute-labels {extra}".strip())
    elif args.wave == 4:
        init = "scratch" if args.scratch else WAVE4["init"]
        for policy, extra in WAVE4["policies"]:
            for mod in WAVE4["modalities"]:
                for seed in WAVE4["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch convnext_tiny "
                                 f"--init {init} --seed {seed} --track-test {extra}".strip())
    elif args.wave == 6:
        for policy, extra in WAVE6["policies"]:
            for cond, mod in WAVE6["cells"]:
                for seed in WAVE6["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--condition {cond} --seed {seed} {extra}".strip())
        for policy, extra in WAVE6["bits_policies"]:
            for bits in WAVE6["bits"]:
                for seed in WAVE6["seeds"]:
                    lines.append(f"--policy {policy} --modality depth --arch alexnet "
                                 f"--condition scale_removed --bits {bits} "
                                 f"--seed {seed} {extra}".strip())
    elif args.wave == 8:
        for policy, extra in WAVE8["policies"]:
            for mod in WAVE8["modalities"]:
                for seed in WAVE8["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--seed {seed} {WAVE8['extra']} {extra}".strip())
        for policy, extra in WAVE8["policies"]:
            for seed in range(args.control_seeds):
                lines.append(f"--policy {policy} --modality depth --arch alexnet "
                             f"--seed {seed} --permute-labels {WAVE8['extra']} {extra}".strip())
    elif args.wave == 9:
        for policy, extra in WAVE9["policies"]:
            for cond, mod, flags in WAVE9["cells"]:
                for seed in WAVE9["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--condition {cond} --seed {seed} {flags} {extra}".strip())
    elif args.wave == 10:
        for policy, extra in WAVE10["policies"]:
            for cond, mod, head in WAVE10["cells"]:
                for seed in WAVE10["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--condition {cond} --head {head} --seed {seed} "
                                 f"{extra}".strip())
    elif args.wave == 12:
        for policy, extra in WAVE12["policies"]:
            for cond, mod, flags, arch in WAVE12["cells"]:
                for seed in WAVE12["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch {arch} "
                                 f"--condition {cond} --seed {seed} {flags} {extra}".strip())
    elif args.wave == 11:
        for policy, extra in WAVE11["policies"]:
            for cond, mod in WAVE11["cells"]:
                for seed in WAVE11["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--condition {cond} --seed {seed} {extra}".strip())
    else:
        wave = {3: WAVE3, 5: WAVE5, 7: WAVE7}[args.wave]
        for policy, extra in wave["policies"]:
            for cond, mod in wave["cells"]:
                for seed in wave["seeds"]:
                    lines.append(f"--policy {policy} --modality {mod} --arch alexnet "
                                 f"--condition {cond} --seed {seed} {extra}".strip())
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
