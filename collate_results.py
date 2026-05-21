#!/usr/bin/env python3
"""Collate results_*.json files from multi-seed sweeps into a summary table.

Usage:
    python collate_results.py path/to/results_*.json
    python collate_results.py $SCRATCH/biwi-results/*/results_*.json
"""
import json
import sys
import glob
import statistics as stats
from collections import defaultdict


def main(patterns):
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(set(paths))
    if not paths:
        print("No results files matched.")
        return 1

    # Group runs by config tag (everything except the seed) so we collate across seeds.
    groups = defaultdict(list)
    n_skipped = 0
    for p in paths:
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  SKIP (unreadable/partial JSON): {p}  [{type(e).__name__}: {e}]")
            n_skipped += 1
            continue
        if not isinstance(data, dict):
            print(f"  SKIP (not a dict): {p}")
            n_skipped += 1
            continue
        cfg = data.get("config", {})
        key = (
            cfg.get("protocol"),
            cfg.get("test_subdir"),
            cfg.get("img_size"),
            cfg.get("width"),
            cfg.get("epochs"),
            cfg.get("clip_range"),
            cfg.get("hardneg_freq"),
            cfg.get("augment"),
            cfg.get("training_mode", "pair"),
            cfg.get("pk_p"),
            cfg.get("pk_k"),
            cfg.get("triplet_margin"),
            cfg.get("emb_dim"),
            cfg.get("probe_window"),
            cfg.get("use_bnneck"),
            cfg.get("ce_weight"),
            cfg.get("backbone", "smallresnet"),
            cfg.get("pretrained", True),
            cfg.get("anthro_weight", 0.0),
            bool(cfg.get("canonicalize_pose", False)),
            bool(cfg.get("anthro_3d", False)),
            bool(cfg.get("canonicalize_pose_3d", False)),
            cfg.get("clip_range", 600.0),
            cfg.get("temporal_stack_frames", 1),
            cfg.get("temporal_stack_stride", 5),
        )
        groups[key].append(data)

    def fmt(name, vals):
        if not vals:
            return f"{name:>8} n=0"
        if len(vals) == 1:
            return f"{name:>8} {vals[0]:.2f}      "
        return f"{name:>8} {stats.mean(vals):.2f} ± {stats.stdev(vals):.2f}"

    def ratio(name, depth_vals, rgb_vals):
        """Depth/RGB ratio — closer to 1.0 means more comparable. >1 means depth wins."""
        if not depth_vals or not rgb_vals:
            return f"{name:>8} n/a"
        d_mean = stats.mean(depth_vals)
        r_mean = stats.mean(rgb_vals)
        if r_mean <= 0:
            return f"{name:>8} n/a (RGB=0)"
        return f"{name:>8} {d_mean / r_mean:.3f}  (D/R ratio)"

    if n_skipped:
        print(f"\n  (skipped {n_skipped} unreadable file(s) — typically partial JSON from crashed jobs)")
    for key, runs in groups.items():
        (proto, tsub, sz, w, ep, clip, hn, aug, tmode, pkp, pkk, tmargin, emb, pw,
         bnneck, cew, backbone, pretrained, anthro_w, canon_pose, anthro_3d, canon_3d, clip_range_3d,
         tstack_frames, tstack_stride) = key
        seeds = [r.get("config", {}).get("seed") for r in runs]
        print()
        print("=" * 90)
        if tmode == "triplet":
            mode_info = f"mode={tmode}  P={pkp}  K={pkk}  m={tmargin}  emb={emb}"
            if bnneck is not None:
                mode_info += f"  bnneck={bnneck}  ce_w={cew}"
        else:
            mode_info = f"mode={tmode}  hardneg={hn}"
        seq_info = f"  probe_window={pw}" if pw else ""
        backbone_info = f"  backbone={backbone}" if backbone and backbone != "smallresnet" else ""
        if backbone and backbone != "smallresnet" and not pretrained:
            backbone_info += "  (scratch)"
        if anthro_w and float(anthro_w) > 0:
            backbone_info += f"  anthro_w={anthro_w}"
        if canon_pose:
            backbone_info += f"  canon_pose=True"
        if anthro_3d:
            backbone_info += f"  anthro_3d=True"
        if canon_3d:
            backbone_info += f"  canon_3d=True"
        if clip_range_3d and float(clip_range_3d) != 600.0:
            backbone_info += f"  clip_range={clip_range_3d}"
        if tstack_frames and int(tstack_frames) > 1:
            backbone_info += f"  tstack={tstack_frames}x{tstack_stride}"
        print(f"protocol={proto}  test={tsub}  size={sz}  w={w}  ep={ep}  clip={clip}  aug={aug}  {mode_info}{seq_info}{backbone_info}")
        print(f"seeds: {sorted(seeds)}  (n={len(runs)})")
        print("-" * 90)
        print("\n[within-val monitoring metric]")
        print(f"{'modality':<10} {'R1':>16} {'R5':>16} {'mAP':>16} {'SEP':>16} {'best_ep':>10}")
        for mod in ("depth", "rgb"):
            r1s, r5s, mAPs, seps, epochs = [], [], [], [], []
            for r in runs:
                m = r.get(mod) or {}
                if "best_rank1" in m: r1s.append(m["best_rank1"])
                if "best_rank5" in m: r5s.append(m["best_rank5"])
                if "best_mAP" in m: mAPs.append(m["best_mAP"])
                if "best_separation" in m: seps.append(m["best_separation"])
                if "best_epoch" in m: epochs.append(m["best_epoch"])
            print(f"{mod.upper():<10} {fmt('R1', r1s):>16} {fmt('R5', r5s):>16} "
                  f"{fmt('mAP', mAPs):>16} {fmt('SEP', seps):>16} {fmt('ep', epochs):>10}")

        # Helper to print one metric block (per-frame OR prototype) for given key prefix.
        def _print_block(title, key_prefix, has_sep=False):
            metric_suffixes = ["rank1", "rank5", "mAP"] + (["separation"] if has_sep else [])
            metric_short = ["R1", "R5", "mAP"] + (["SEP"] if has_sep else [])
            print(f"\n{title}")
            header_fields = " ".join(f"{ms:>16}" for ms in metric_short)
            print(f"{'modality':<10} {header_fields}")
            depth_vals = {ms: [] for ms in metric_short}
            rgb_vals = {ms: [] for ms in metric_short}
            any_data = False
            for mod in ("depth", "rgb"):
                row_vals = {ms: [] for ms in metric_short}
                for r in runs:
                    m = r.get(mod) or {}
                    for suf, short in zip(metric_suffixes, metric_short):
                        full_key = f"{key_prefix}_{suf}"
                        if full_key in m: row_vals[short].append(m[full_key])
                if row_vals[metric_short[0]]: any_data = True
                target = depth_vals if mod == "depth" else rgb_vals
                for ms in metric_short:
                    target[ms] = row_vals[ms]
                row_str = " ".join(f"{fmt(ms, row_vals[ms]):>16}" for ms in metric_short)
                print(f"{mod.upper():<10} {row_str}")
            if any_data:
                ratio_str = " ".join(
                    f"{ratio(ms, depth_vals[ms], rgb_vals[ms]):>16}" for ms in metric_short
                )
                print(f"{'D/RGB':<10} {ratio_str}")
            else:
                print("  (no metrics found — older runs from before this eval was added)")

        _print_block(
            "[within-test FRAME-MATCH — Still↔Walking per-frame nearest]",
            "wt",
        )
        _print_block(
            "[within-test PROTOTYPE — each (1) probe frame → nearest of 28 subject means]",
            "wt_proto",
        )
        _print_block(
            "[within-test SEQ-PROBE — N-frame video probe → nearest of 28 subject means (RECOMMENDED PAPER NUMBER)]",
            "wt_seq",
        )
        # K-reciprocal re-ranking variants (Zhong 2017). Each is a refinement of the
        # corresponding non-reranked block above.
        _print_block(
            "[within-test FRAME-MATCH + k-reciprocal RERANK]",
            "wt_rerank",
        )
        _print_block(
            "[within-test PROTOTYPE + k-reciprocal RERANK]",
            "wt_rerank_proto",
        )
        _print_block(
            "[within-test SEQ-PROBE + k-reciprocal RERANK (RECOMMENDED PAPER NUMBER)]",
            "wt_rerank_seq",
        )

        # ── OPEN-SET CROSS-CLOTHING (gallery = outfit A, probe = outfit B, unseen IDs) ──
        # The decisive comparison: does depth's clothing-invariant body-shape cue
        # match or beat RGB when clothing differs between gallery and probe?
        _print_block(
            "[X-CLOTH FRAME-MATCH — outfit A gallery ↔ outfit B probe, unseen IDs]",
            "xc",
        )
        _print_block(
            "[X-CLOTH PROTOTYPE — probe → nearest of 28 outfit-A subject means]",
            "xc_proto",
        )
        _print_block(
            "[X-CLOTH SEQ-PROBE — N-frame video probe → outfit-A subject means]",
            "xc_seq",
        )
        _print_block(
            "[X-CLOTH SEQ-PROBE + k-reciprocal RERANK (PAPER CROSS-CLOTHING NUMBER)]",
            "xc_rerank_seq",
        )
        _print_block(
            "[X-CLOTH PROTOTYPE + k-reciprocal RERANK]",
            "xc_rerank_proto",
        )

        _print_block(
            "[cross-clothing FRAME-MATCH — train↔test per-frame nearest, hardest]",
            "gp",
            has_sep=True,
        )
        _print_block(
            "[cross-clothing PROTOTYPE — each probe → nearest of 50 mean embeddings]",
            "gp_proto",
        )
        _print_block(
            "[cross-clothing PROTOTYPE + k-reciprocal RERANK]",
            "gp_rerank_proto",
        )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or ["results_*.json"]
    sys.exit(main(args))
