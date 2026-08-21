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
# hiride_train.py calls apply_mask_condition(img, m, condition, 0.0, ...)
TRAINER_FILL = 0.0


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
def fig_collapse(stats, out, ceiling=95.0):
    """Figure 1 -- the RGB advantage collapses at the cross-session rung.

    This is the paper's central claim in one panel. Each point is one
    (arch, condition) cell's mean rgb-minus-depth difference, paired by seed on
    identical test frames. The spread within a rung is across model variants,
    which is the point: the collapse is not one lucky configuration.
    """
    per = {}
    for r in stats["paired"]:
        key = (r["policy"], r.get("arch", "?"), r["condition"])
        per.setdefault(key, {"d": [], "rgb": [], "depth": []})
        per[key]["d"].append(r["diff"] * 100)
        per[key]["rgb"].append(r["rgb"] * 100)
        per[key]["depth"].append(r["depth"] * 100)
    pts = {}
    for (policy, arch, cond), v in per.items():
        # A cell where BOTH modalities sit against the ceiling carries no
        # information about which modality is better -- bg_plate at R0 is
        # 99.97 % vs 99.91 %, a 0.06 pp difference that would otherwise pull the
        # R0 mean toward zero with the same weight as a 61 pp one and make the
        # collapse look like it starts earlier than it does.
        sat = min(np.mean(v["rgb"]), np.mean(v["depth"])) >= ceiling
        pts.setdefault(policy, []).append((np.mean(v["d"]), f"{arch}/{cond}", sat))

    have = [p for p in RUNGS if pts.get(p)]
    if not have:
        print("  (fig1 skipped: no paired records)")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    rng = np.random.default_rng(0)
    n_sat = 0
    for i, policy in enumerate(have):
        rows = pts[policy]
        for marker_sat, face, edge in ((False, "#444444", "none"), (True, "none", "#999999")):
            vals = [v for v, _, sat in rows if sat == marker_sat]
            if not vals:
                continue
            n_sat += len(vals) if marker_sat else 0
            ax.scatter(np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals)), vals,
                       s=36, alpha=0.8, facecolor=face, edgecolor=edge,
                       linewidth=1.1, zorder=3)
        live = [v for v, _, sat in rows if not sat]
        if live:
            ax.hlines(np.mean(live), i - 0.3, i + 0.3, color="#000000", lw=2.4, zorder=4)
            ax.annotate(f"{np.mean(live):+.1f}", (i, np.mean(live)),
                        textcoords="offset points", xytext=(26, -4), fontsize=9.5,
                        fontweight="bold")
    ax.axhline(0, color="#888888", lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([RUNG_LABEL[p] for p in have])
    ax.set_ylabel("RGB accuracy − depth accuracy (pp)")
    ax.set_title("The RGB advantage is protocol-dependent, not modality-intrinsic",
                 fontsize=11)
    cap = ("each point = one (architecture, condition) cell, paired by seed on identical "
           "test frames; bar = mean")
    if n_sat:
        cap += (f"\nopen circles ({n_sat}) = both modalities above {ceiling:.0f} % "
                f"(saturated; excluded from the mean)")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=8, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out, "fig1_collapse")


def fig_ladder(stats, out, condition="scale_removed", arch="alexnet",
               overlay_arch="alexnet/stripe/aug8/tf10"):
    """Figure 2 -- accuracy down the ladder, both modalities, with CIs.

    NO architecture substitution. The first version of this figure fell back to
    a different architecture at rungs where the requested one had not been run,
    so R0 and R3 were the gap-head baseline while R1 and R4 were the best
    recipe -- under a title naming only the latter. That inflates the R1-to-R3
    drop by the head effect (about 13 pp at R1) and is exactly the kind of
    silent substitution the rest of this campaign has been spent removing. A
    rung the architecture was not run at is left empty and reported.

    The reference line matters as much as the curves: the honest floor is the
    majority-class rate, not 1/K, and a cell whose CI lower bound sits under it
    is not an identification result whatever its mean says.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    drawn = False
    for use_arch, style, alpha, tag in ((arch, "o-", 0.16, arch),
                                        (overlay_arch, "s--", 0.0, overlay_arch)):
        if not use_arch:
            continue
        for mod, colour in (("depth", DEPTH_C), ("rgb", RGB_C)):
            xs, ys, los, his, missing = [], [], [], [], []
            for i, policy in enumerate(RUNGS):
                c = cells_by(stats, policy=policy, modality=mod, arch=use_arch,
                             condition=condition, permuted=False)
                if not c:
                    missing.append(policy.split("_")[0])
                    continue
                c = max(c, key=lambda r: r["n_seeds"])
                xs.append(i); ys.append(c["frame_acc_mean"] * 100)
                los.append(c["subj_ci_lo_mean"] * 100); his.append(c["subj_ci_hi_mean"] * 100)
            if not xs:
                continue
            drawn = True
            filled = "none" if style.startswith("s") else colour
            marker, dash = style[0], style[1:]
            # Only connect rungs that are ADJACENT. The overlay arm exists at R1
            # and R4 but not R3, and a single line between them runs through
            # R3's x-position at a value nothing measured -- a drawn claim about
            # the rung the arm was never trained on. Gaps break the line; the
            # markers still show where the arm does exist.
            runs, cur = [], [0]
            for k in range(1, len(xs)):
                if xs[k] == xs[k - 1] + 1:
                    cur.append(k)
                else:
                    runs.append(cur); cur = [k]
            runs.append(cur)
            ax.plot(xs, ys, marker, color=colour, ms=6, markerfacecolor=filled,
                    ls="none", label=f"{mod}  {tag}", zorder=3)
            for run in runs:
                if len(run) > 1:
                    ax.plot([xs[k] for k in run], [ys[k] for k in run], ls=dash,
                            color=colour, lw=2, zorder=3)
            if alpha:
                for run in runs:
                    ax.fill_between([xs[k] for k in run], [los[k] for k in run],
                                    [his[k] for k in run], color=colour,
                                    alpha=alpha, zorder=2)
            if missing:
                print(f"    fig2: {use_arch} {mod} not run at {', '.join(missing)} "
                      f"-- left empty, NOT substituted")
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
    ax.set_title(f"{condition}  —  band = 95 % subject-cluster bootstrap on the solid arm",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
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
            # Match on the COMPOSED arch string, not a hand-listed set of
            # fields. The hand-listed version checked head, augment, frames and
            # encoding but forgot test_fuse, so alexnet/tf10 scale_removed
            # (48.20 % at R1) was drawn on top of the plain gap cell (55.40 %)
            # as a second point at 16 bits. arch_key already composes every
            # training-side axis, so equality against it cannot miss one.
            if (c["policy"] != policy or c["modality"] != "depth"
                    or c["permuted"] or c["arch"] != "alexnet"
                    or c.get("base_condition") != "scale_removed"
                    or c.get("frames", 1) != 1 or c.get("encoding", "raw") != "raw"
                    or c.get("eligibility", "cues") != "cues"):
                continue
            pts.append((c.get("bits", 16), c["frame_acc_mean"] * 100,
                        c["subj_ci_lo_mean"] * 100, c["subj_ci_hi_mean"] * 100,
                        c["n_seeds"]))
        if not pts:
            continue
        # Two cells landing on the same bit value means the filter above is not
        # unique -- the first version drew a vertical jump at 16 bits from two
        # stacked points and gave no clue which cells they were.
        by_bits = {}
        for row in pts:
            by_bits.setdefault(row[0], []).append(row)
        for bits, rows in sorted(by_bits.items()):
            if len(rows) > 1:
                names = [f"{c['arch']}/{c['condition']} n={c['n_seeds']}"
                         for c in stats["cells"]
                         if c["policy"] == policy and c["modality"] == "depth"
                         and c.get("bits", 16) == bits and c["arch"] == "alexnet"
                         and c.get("base_condition") == "scale_removed"
                         and c.get("frames", 1) == 1
                         and c.get("encoding", "raw") == "raw"
                         and c.get("eligibility", "cues") == "cues"
                         and not c["permuted"]]
                print(f"    fig4: {policy} has {len(rows)} cells at {bits} bits "
                      f"-- {names}; plotting the one with most seeds")
        pts = [max(rows, key=lambda r: r[4]) for _, rows in sorted(by_bits.items())]
        drawn = True
        b, y, lo, hi, _ = map(np.array, zip(*pts))
        ax.plot(b, y, marker + "-", color=colour, label=policy.split("_")[0], lw=2)
        ax.fill_between(b, lo, hi, color=colour, alpha=0.15)
    if not drawn:
        print("  (fig4 skipped: no bit-depth cells)")
        plt.close(fig)
        return
    ax.set_xticks([1, 2, 3, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "3", "4", "8", "16"])
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
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    # Every depth arm was drawn in the same blue, so five curves were
    # indistinguishable. Vary marker and dash instead of hue, keeping the
    # depth/RGB colour split that carries the actual contrast.
    styles = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":"), ("v", (0, (3, 1, 1, 1)))]
    di = 0
    for label, accs in sorted(rp["probe_acc"].items()):
        if "normals" in label or " f" in label:
            continue                      # keep the panel to the reported arms
        is_rgb = label.startswith("rgb")
        colour = RGB_C if is_rgb else DEPTH_C
        marker, dash = styles[di % len(styles)]
        di += 0 if is_rgb else 1
        ys = [None if a is None else a * 100 for a in accs]
        xs = [c for c, y in zip(centres, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, marker=marker, ls=dash, color=colour, alpha=0.9, lw=1.8,
                ms=5, label=label)
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
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5,
              loc="center left", bbox_to_anchor=(1.10, 0.5))
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
            # fill MUST match the trainer's call site (hiride_train.py passes
            # 0.0). With fill=1.0 the sil_scaled panel renders completely blank
            # -- silhouette interior and background both land on 1.0 -- and
            # every other panel gets an inverted background. Calling the real
            # function with the wrong argument is no better than reimplementing
            # it.
            v = apply_mask_condition(img, mask, cond, TRAINER_FILL)
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


def fig_operating_point(paths, out):
    """Figure 7 -- accuracy against OBSERVATION BUDGET, gated and ungated.

    This is the figure for the campaign's headline correction. Every other
    number in the paper is per frame, which silently assumes a system must
    answer from a single observation; a deployment watching someone walk has
    hundreds. Plotting accuracy against frames-per-decision shows where the
    curve FLATTENS, which is the honest operating point, and the decision count
    printed against each x tick shows where it stops being measurable.

    Two files overlay: with and without the full-body validity gate, so the two
    independent gains -- rejecting frames the sensor could not measure, and
    integrating over time -- are visible separately and in combination.
    """
    import json as _json
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ARMS = (("cnn", "#7f7f7f"), ("metric", DEPTH_C), ("geo", "#9467bd"))
    drawn, ticks, labels = False, [], []
    for pi, path in enumerate(paths):
        try:
            blob = _json.load(open(path))
        except (OSError, ValueError):
            print(f"  (fig7: cannot read {path})")
            continue
        meta = blob.get("_meta", {})
        gated = meta.get("full_body")
        conds = [k for k in blob if not k.startswith("_")]
        if not conds:
            continue
        cond = conds[0]
        rec = blob[cond]
        W = [w for w in rec["windows"]]
        xs = [len(rec["n_decisions"]) if w == 0 else w for w in W]
        # whole-tracklet (w=0) is placed just past the largest real window
        xmax = max(w for w in W if w > 0)
        xs = [xmax * 2 if w == 0 else w for w in W]
        for arm, colour in ARMS:
            ys = [100 * rec["acc"][arm][str(w)] for w in W]
            drawn = True
            ax.plot(xs, ys, marker="o" if pi == 0 else "s",
                    ls="-" if pi == 0 else "--", color=colour, lw=1.9, ms=5,
                    markerfacecolor=colour if pi == 0 else "none",
                    label=f"{arm}  {'gated' if gated else 'all frames'}")
        if pi == 0:
            ticks = list(xs)
            labels = [("all" if w == 0 else str(w))
                      + f"\nn={int(rec['n_decisions'][str(w)])}" for w in W]
    if not drawn:
        print("  (fig7 skipped: no sequence json)")
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    # AFTER set_xscale, which installs a log locator and would otherwise
    # discard these. Explicit ticks at the windows actually measured: an axis
    # labelled 2^0..2^8 cannot tell a reader which point is the 25-frame
    # operating point the paper quotes. The decision count belongs on the axis
    # too -- the curve rises toward its noisiest end, so the eye lands on the
    # whole-tracklet column exactly where there are 28 decisions, not 2,933.
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("frames per decision, and the number of decisions behind it",
                  labelpad=2)
    ax.set_ylabel("frame / window accuracy (%)")
    ax.set_title("Identity accumulates with observation, once invalid frames are refused",
                 fontsize=10.5)
    ax.axhline(6.34, color="#cc3311", lw=1, ls=":",
               label="majority-class rate (gated)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.04, "solid = frames with a complete body only (the sensor's own user "
                         "map); dashed = every frame\n\"all\" = one decision per subject, "
                         "28 of them -- read its interval, not its point estimate",
             ha="center", fontsize=8, color="#555555")
    save(fig, out, "fig7_operating_point")


def fig_cohort(path, out):
    """Figure 8 -- the operating point against enrolled cohort size.

    Absolute accuracy must fall as the gallery grows, so the raw curve alone
    reads as degradation. The chance line underneath is what makes it readable:
    the MARGIN over chance grows while accuracy falls, so the system becomes
    less accurate and more informative at the same time.

    The band is the min-max range across cohort draws, not a confidence
    interval. At small K a particular set of people can be easy or hard, and
    that variation is a property of the deployment question rather than noise
    to be averaged away -- a reader enrolling four residents wants to know the
    range they might land in.
    """
    import json as _json
    try:
        blob = _json.load(open(path))
    except (OSError, ValueError):
        print(f"  (fig8: cannot read {path})")
        return
    meta = blob.get("_meta", {})
    Ks = sorted(int(k) for k in blob if not k.startswith("_"))
    if not Ks:
        print("  (fig8 skipped: no cohorts in cohort.json)")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for arm, colour, lab in (("cnn_w", "#7f7f7f", "CNN"),
                             ("met_w", DEPTH_C, "metric features"),
                             ("geo_w", "#9467bd", "fusion")):
        ys = [blob[str(k)]["mean"].get(arm, float("nan")) for k in Ks]
        ax.plot(Ks, ys, "o-", color=colour, lw=2, ms=5, label=lab)
        lo = [100 * min(blob[str(k)]["draws"].get(arm, [float("nan")])) for k in Ks]
        hi = [100 * max(blob[str(k)]["draws"].get(arm, [float("nan")])) for k in Ks]
        ax.fill_between(Ks, lo, hi, color=colour, alpha=0.13, lw=0)
    ax.plot(Ks, [100.0 / k for k in Ks], ":", color="#cc3311", lw=1.6,
            label="chance (1/K)")
    for k in Ks:
        g = blob[str(k)]["mean"].get("geo_w")
        if g:
            ax.annotate(f"{g * k / 100:.0f}x", (k, g), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7.5, color="#9467bd")
    ax.set_xscale("log")
    ax.set_xticks(Ks); ax.set_xticklabels([str(k) for k in Ks])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("enrolled cohort size (subjects)")
    ax.set_ylabel(f"accuracy at {meta.get('window', 25)} frames/decision (%)")
    ax.set_title("Accuracy falls with cohort size; the margin over chance grows",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "band = min-max across cohort draws, not a confidence interval; "
                         "x-labels give the fusion's margin over chance",
             ha="center", fontsize=8, color="#555555")
    save(fig, out, "fig8_cohort")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="stats_final.json")
    ap.add_argument("--range", dest="rangef", default=None, help="range_profile.json")
    ap.add_argument("--prep", default=None, help="prep dir; enables Figure 6")
    ap.add_argument("--subject", default=None, help="Figure 6: pick this subject")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default=None,
                    help="cohort.json from hiride_cohort.py. Enables figure 8.")
    ap.add_argument("--sequence", action="append", default=None,
                    help="sequence*.json from hiride_sequence.py; repeatable, "
                         "gated first. Enables figure 7.")
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
    if args.cohort:
        fig_cohort(args.cohort, args.out)
    if args.sequence:
        fig_operating_point(args.sequence, args.out)
    else:
        print("  (fig7 skipped: pass --sequence to plot the operating point)")

    print("\nAll figures are read-only over stats_final.json / range_profile.json.")
    print("Nothing is recomputed here, so a number in a figure and the same number")
    print("in the tables cannot disagree.")


if __name__ == "__main__":
    main()
