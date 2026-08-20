"""Figures for the paper-2 re-run, from artifacts already on disk.

    python hiride_figures.py --stats $SCRATCH/hiride2/results/stats_final.json \
        --range $SCRATCH/hiride2/results/range_profile.json \
        --out $SCRATCH/hiride2/results/figs
    python hiride_figures.py --stats ... --prep $SCRATCH/hiride2/prep --out ...   # adds Fig 6

Nothing is recomputed here. Every number comes from stats_final.json (written by
hiride_stats.py, which is the only place subject-cluster CIs exist) or from
range_profile.json. Figure 6 is the exception: it renders example frames through
hiride_train.apply_mask_condition, the SAME function the trainer feeds the
network, so the panel cannot drift from what was actually trained on.

matplotlib only -- venv311 has no seaborn.
"""
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The ladder, in the order the argument is made. R2 was never run.
RUNGS = ["R0_frame_random", "R1_block", "R3_cross_recording", "R4_cross_session"]
RUNG_LABEL = {"R0_frame_random": "R0\nframe-random",
              "R1_block": "R1\nblock+guard",
              "R3_cross_recording": "R3\ncross-recording",
              "R4_cross_session": "R4\ncross-session"}
DEPTH_C, RGB_C = "#1f77b4", "#d62728"


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  [written] {os.path.join(out, name)}.pdf/.png")


def cells_by(stats, **match):
    """Every cell record matching the given fields exactly."""
    return [c for c in stats["cells"]
            if all(c.get(k) == v for k, v in match.items())]


# --------------------------------------------------------------------------
def fig_collapse(stats, out):
    """Figure 1 -- the RGB advantage collapses at the cross-session rung.

    This is the paper's central claim in one panel. Each point is one
    (arch, condition) cell's mean rgb-minus-depth difference, paired by seed on
    identical test frames. The spread within a rung is across model variants,
    which is the point: the collapse is not one lucky configuration.
    """
    per = {}
    for r in stats["paired"]:
        key = (r["policy"], r.get("arch", "?"), r["condition"])
        per.setdefault(key, []).append(r["diff"] * 100)
    pts = {}
    for (policy, arch, cond), ds in per.items():
        pts.setdefault(policy, []).append((np.mean(ds), f"{arch}/{cond}"))

    have = [p for p in RUNGS if pts.get(p)]
    if not have:
        print("  (fig1 skipped: no paired records)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rng = np.random.default_rng(0)
    for i, policy in enumerate(have):
        vals = [v for v, _ in pts[policy]]
        jitter = rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=34, alpha=0.75,
                   color="#444444", zorder=3, edgecolor="none")
        ax.hlines(np.mean(vals), i - 0.3, i + 0.3, color="#000000", lw=2.2, zorder=4)
        ax.annotate(f"{np.mean(vals):+.1f}", (i, np.mean(vals)),
                    textcoords="offset points", xytext=(24, -4), fontsize=9)
    ax.axhline(0, color="#888888", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([RUNG_LABEL[p] for p in have])
    ax.set_ylabel("RGB accuracy − depth accuracy (pp)")
    ax.set_title("The RGB advantage is protocol-dependent, not modality-intrinsic",
                 fontsize=11)
    ax.text(0.01, 0.02, "each point = one (architecture, condition) cell, paired by seed\n"
                        "on identical test frames; bar = mean over cells",
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig1_collapse")


def fig_ladder(stats, out, condition="scale_removed",
               arch="alexnet/stripe/aug8/tf10", fallback_arch="alexnet"):
    """Figure 2 -- accuracy down the ladder, both modalities, with CIs.

    The reference lines matter as much as the curves: chance is 1/K, but the
    honest floor is the majority-class rate, and a cell whose CI lower bound sits
    under it is not an identification result whatever its mean says.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    drawn = False
    for mod, colour in (("depth", DEPTH_C), ("rgb", RGB_C)):
        xs, ys, los, his = [], [], [], []
        for i, policy in enumerate(RUNGS):
            c = (cells_by(stats, policy=policy, modality=mod, arch=arch,
                          condition=condition, permuted=False)
                 or cells_by(stats, policy=policy, modality=mod, arch=fallback_arch,
                             condition=condition, permuted=False))
            if not c:
                continue
            c = max(c, key=lambda r: r["n_seeds"])
            xs.append(i); ys.append(c["frame_acc_mean"] * 100)
            los.append(c["subj_ci_lo_mean"] * 100); his.append(c["subj_ci_hi_mean"] * 100)
        if not xs:
            continue
        drawn = True
        ax.plot(xs, ys, "o-", color=colour, label=mod, lw=2, ms=6, zorder=3)
        ax.fill_between(xs, los, his, color=colour, alpha=0.16, zorder=2)
    if not drawn:
        print("  (fig2 skipped: no cells for the requested arch/condition)")
        plt.close(fig)
        return
    maj = [cells_by(stats, policy=p, permuted=False) for p in RUNGS]
    mv = [np.mean([c["majority"] for c in g if c.get("majority")]) * 100 if g else np.nan
          for g in maj]
    ax.plot(range(len(RUNGS)), mv, ls=":", color="#333333", lw=1.4,
            label="majority-class rate")
    ax.set_xticks(range(len(RUNGS)))
    ax.set_xticklabels([RUNG_LABEL[p] for p in RUNGS])
    ax.set_ylabel("frame accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{condition}, {arch}  —  band = 95 % subject-cluster bootstrap", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig2_ladder")


def fig_mechanism(stats, out, arch="alexnet"):
    """Figure 3 -- the mechanism suite as conditions x rungs, one panel per modality."""
    conds = ["full", "person", "bg_hole", "bg_plate", "silhouette",
             "person_centred", "scale_removed", "sil_scaled", "interior_only"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    any_drawn = False
    for ax, mod in zip(axes, ("depth", "rgb")):
        M = np.full((len(conds), len(RUNGS)), np.nan)
        for i, cond in enumerate(conds):
            for j, policy in enumerate(RUNGS):
                c = cells_by(stats, policy=policy, modality=mod, arch=arch,
                             condition=cond, permuted=False)
                if c:
                    M[i, j] = max(c, key=lambda r: r["n_seeds"])["frame_acc_mean"] * 100
        if np.isfinite(M).any():
            any_drawn = True
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        ax.set_xticks(range(len(RUNGS)))
        ax.set_xticklabels([p.split("_")[0] for p in RUNGS])
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=9)
        ax.set_title(mod, fontsize=11)
        for i in range(len(conds)):
            for j in range(len(RUNGS)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] < 55 else "black")
    if not any_drawn:
        print("  (fig3 skipped: no mechanism cells)")
        plt.close(fig)
        return
    fig.colorbar(im, ax=axes, label="frame accuracy (%)", fraction=0.025)
    fig.suptitle(f"Mechanism suite, {arch}. Blank = not run.", fontsize=10)
    save(fig, out, "fig3_mechanism")


def fig_bits(stats, out):
    """Figure 4 -- accuracy vs depth quantisation, R1 and R4."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    drawn = False
    for policy, colour, marker in (("R1_block", DEPTH_C, "o"),
                                   ("R4_cross_session", "#7f3fbf", "s")):
        pts = []
        for c in stats["cells"]:
            if (c["policy"] != policy or c["modality"] != "depth"
                    or c["permuted"] or c.get("base_condition") != "scale_removed"
                    or c.get("head", "gap") != "gap" or c.get("augment", 0)
                    or c.get("frames", 1) != 1 or c.get("encoding", "raw") != "raw"
                    or c.get("eligibility", "cues") != "cues"):
                continue
            pts.append((c.get("bits", 16), c["frame_acc_mean"] * 100,
                        c["subj_ci_lo_mean"] * 100, c["subj_ci_hi_mean"] * 100))
        if not pts:
            continue
        drawn = True
        pts.sort()
        b, y, lo, hi = map(np.array, zip(*pts))
        ax.plot(b, y, marker + "-", color=colour, label=policy.split("_")[0], lw=2)
        ax.fill_between(b, lo, hi, color=colour, alpha=0.15)
    if not drawn:
        print("  (fig4 skipped: no bit-depth cells)")
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_xlabel("depth quantisation (bits, fixed global scale)")
    ax.set_ylabel("frame accuracy (%)")
    ax.set_title("Z-precision axis  —  band = 95 % subject-cluster bootstrap", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig4_bits")


def fig_range(rp, out):
    """Figure 5 -- accuracy and body-clipping rate against standing distance.

    Plotted together because the two are confounded in BIWI: the near bins are
    also the clipped bins. The figure is drawn to make that confound visible
    rather than to hide it behind a single curve.
    """
    if not rp or not rp.get("probe_acc"):
        print("  (fig5 skipped: range_profile.json has no probe_acc -- "
              "re-run hiride_range_profile.py with --signal)")
        return
    edges = rp["bins"]
    centres = [(edges[k] + min(edges[k + 1], 4500)) / 2 for k in range(len(edges) - 1)]
    labels = [f"{int(edges[k])}–{int(edges[k+1])}" for k in range(len(edges) - 1)]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for label, accs in sorted(rp["probe_acc"].items()):
        if "normals" in label or " f" in label:
            continue                      # keep the panel to the reported arms
        colour = RGB_C if label.startswith("rgb") else DEPTH_C
        ys = [None if a is None else a * 100 for a in accs]
        xs = [c for c, y in zip(centres, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, "o-", color=colour, alpha=0.85, lw=1.8, label=label)
    ax.set_xlabel("person median depth (mm)")
    ax.set_ylabel("linear-probe accuracy (%)")
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=8)
    ax2 = ax.twinx()
    test = rp["splits"].get("test", [])
    train = rp["splits"].get("train", [])
    for split, style, name in ((test, "-", "test"), (train, "--", "train")):
        clip = [None if b.get("bot_touch") is None
                else 100 * max(b["bot_touch"], b.get("top_touch") or 0) for b in split]
        xs = [c for c, v in zip(centres, clip) if v is not None]
        vv = [v for v in clip if v is not None]
        ax2.plot(xs, vv, style, color="#999999", lw=1.4,
                 label=f"{name}: body clipped at frame edge")
    ax2.set_ylabel("frames with body clipped (%)", color="#666666")
    ax2.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("Distance and body-clipping are confounded in BIWI", fontsize=10)
    save(fig, out, "fig5_range")


def fig_conditions(prep, out, subject=None, n_show=7):
    """Figure 6 -- what each condition actually does to one frame.

    Rendered through hiride_train.apply_mask_condition, the same function the
    trainer calls, so the panel cannot drift from what the network was fed.
    Importing hiride_train pulls in TensorFlow; CUDA is masked off first so this
    never competes with a training job for a GPU.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        from hiride_train import open_shards, apply_mask_condition, DEPTH_CLIP_MM
        from hiride_data import load_manifest, eligible_mask
    except Exception as exc:
        print(f"  (fig6 skipped: cannot import the trainer -- {exc})")
        return
    conds = ["full", "person", "bg_hole", "silhouette", "person_centred",
             "scale_removed", "sil_scaled", "interior_only"][:n_show]
    man = load_manifest(os.path.join(prep, "manifest.npz"))
    z = np.load(os.path.join(prep, "cues.npz"), allow_pickle=False)
    feats = [str(f) for f in z["feats"]]
    keep = eligible_mask(z["cues"], feats, full_body=True)     # a clean, whole body
    npx = z["cues"][:, feats.index("n_person_px")]
    pool = np.where(keep)[0]
    if subject is not None:
        pool = pool[man["subject"][pool] == subject]
    if not len(pool):
        print("  (fig6 skipped: no eligible full-body frame found)")
        return
    row = int(pool[np.argmax(npx[pool])])                      # largest clean mask
    shards = open_shards(prep, "depth")
    imgs, masks, where = shards
    tag, pos = where[row]
    raw = np.asarray(imgs[tag][pos]).astype(np.float32)
    img = (np.clip(raw, 0.0, DEPTH_CLIP_MM) / DEPTH_CLIP_MM)[..., None]
    mask = np.asarray(masks[tag][pos])

    fig, axes = plt.subplots(1, len(conds), figsize=(2.0 * len(conds), 2.5))
    for ax, cond in zip(np.atleast_1d(axes), conds):
        try:
            v = apply_mask_condition(img, mask, cond, fill=1.0)
        except Exception as exc:                       # bg_plate needs the plates
            ax.text(0.5, 0.5, f"{cond}\n(n/a)", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            print(f"    {cond}: skipped ({type(exc).__name__})")
            continue
        ax.imshow(v[..., 0], cmap="magma", vmin=0, vmax=1)
        ax.set_title(cond, fontsize=9)
        ax.set_axis_off()
    fig.suptitle(f"depth, subject {man['subject'][row]}, frame row {row} "
                 f"— rendered by the trainer's own apply_mask_condition", fontsize=9)
    save(fig, out, "fig6_conditions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="stats_final.json")
    ap.add_argument("--range", dest="rangef", default=None, help="range_profile.json")
    ap.add_argument("--prep", default=None, help="prep dir; enables Figure 6")
    ap.add_argument("--subject", default=None, help="Figure 6: pick this subject")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.stats) as fh:
        stats = json.load(fh)
    rp = None
    if args.rangef and os.path.exists(args.rangef):
        with open(args.rangef) as fh:
            rp = json.load(fh)
    print(f"\n{len(stats.get('cells', []))} cells, {len(stats.get('paired', []))} "
          f"paired records -> {args.out}\n")

    fig_collapse(stats, args.out)
    fig_ladder(stats, args.out)
    fig_mechanism(stats, args.out)
    fig_bits(stats, args.out)
    fig_range(rp, args.out)
    if args.prep:
        fig_conditions(args.prep, args.out, subject=args.subject)
    else:
        print("  (fig6 skipped: pass --prep to render the condition panel)")
    print("\nAll figures are read-only over stats_final.json / range_profile.json.")
    print("Nothing is recomputed here, so a number in a figure and the same number")
    print("in the tables cannot disagree.")


if __name__ == "__main__":
    main()
