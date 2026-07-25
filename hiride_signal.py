"""Why is the depth-RGB margin what it is? A CPU diagnostic, no training.

    python hiride_signal.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

The measured position is that RGB beats depth by ~30 pp at R1 and ~45 pp at R3,
and that both collapse at R4. Theory says the depth margin should not be that
wide: a body's shape is a real, stable biometric. So before spending GPU on
"best-effort" waves, this asks what the two modalities actually LOOK LIKE to a
network, and where depth's signal is going.

THE CHAIN UNDER TEST
  L1  RGB identity signal = clothing colour/texture: dense, high contrast.
  L2  depth identity signal = body shape: small depth differences over the body
      (measured person depth extent p99-p1, median 419 mm) plus the outline.
  L3  our global normalisation maps 0-6000 mm onto the input range, so the body
      occupies ~7 % of it -- the shape arrives as a near-flat plate.
  L4  => the interior shape is PRESENT but nearly unreadable at that contrast.
  L5  20-30 % of depth pixels are invalid (stored 0), which after normalisation
      collides with "nearest surface" -- structured noise RGB does not have.
  L6  cross-session, RGB's signal CHANGES (clothes) while depth's SHOULD be
      invariant; both collapse anyway, so something else dominates.
  L7  what dominates is framing nuisance; `scale_removed` fixes position, size
      and standing distance but NOT the range compression.
  L8  => the outline (binary, contrast-free) beats the interior, which is what
      the runs show: sil_scaled 14.59 > scale_removed 12.45 at R4.
  L9  PREDICTION: body-relative range normalisation makes the interior usable,
      so scale_removed at a tight slab should overtake sil_scaled.

WHAT IS MEASURED, per representation and per rung
  1. CONTRAST -- std and spread of person pixels in the network's input units,
     and the number of distinguishable 8-bit levels the body spans. This is L3
     stated as a number, and it is directly comparable between modalities.
  2. SEPARABILITY ACROSS THE GAP -- same-subject distance over
     different-subject distance, with BOTH pair types crossing train->test so a
     common-mode session shift cancels. ratio ~= 1 means identity is swamped;
     < 1 means it separates. Reported with d-prime, since equal ratios can hide
     unequal spreads. (This is NOT an impossibility test: section 8.6's raw-space
     numbers give ratio 1.00 at R4 while the CNN still reaches 14.59 % there.)
  3. LINEAR DISCRIMINABILITY -- PCA fitted on train, then nearest-class-mean on
     test, plus the Fisher ratio (between-class scatter / within-class scatter).
     This is how much identity is LINEARLY present in the input, i.e. an
     information floor that owes nothing to architecture or optimisation.

Reading the outcome:
  * if the tight-slab depth probe beats the global-slab probe at R4, L9 holds
    and the GPU wave is worth running;
  * if the probe is flat across slabs, the interior carries nothing at this
    resolution and the outline is the whole depth signal -- which is a finding,
    and it saves the GPU time;
  * if `te_centred` is far above `whitened`, the cross-session failure is a
    COMMON-MODE OFFSET, not lost information -- and the fix is a normalisation,
    not a bigger network;
  * trust `whitened` for a null: the un-whitened probe is blind to low-variance
    directions, which is precisely what the compressed depth interior would be.

CPU only, ~10-20 min:
    sbatch --account=def-czarnuch_cpu --time=1:00:00 --mem=48000M --cpus-per-task=4 \
      -J hiride-signal -o logs/hiride-signal_%j.out --wrap 'cd ~/toolbox && \
      source ~/venvs/venv311/bin/activate && python hiride_signal.py \
      --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results'
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, make_split, block_train_counts, eligible_mask
from hiride_train import DEPTH_CLIP_MM, open_shards, load_split_arrays

# (label, modality, condition, slab_mm, frames, encoding)
# The three questions this set answers, beyond the earlier contrast sweep:
#   INTERIOR vs OUTLINE  interior_only (eroded mask, depth inside) against
#       sil_scaled (outline, no interior) and scale_removed (both). Every
#       previous condition confounded the two, so "the interior adds ~1 pp" was
#       a MARGINAL result given the outline, not evidence the interior is empty.
#   NOISE FLOOR          --frames N median fusion. Quantisation is ~25 mm at the
#       person's 2955 mm median and measured sigma is 40-70 mm, against 20-80 mm
#       of between-person curvature, so sqrt(N) denoising is the predicted lever.
#   DERIVATIVES          surface normals, which are scale-free (immune to the
#       contrast problem) but noise-amplifying -- so they are tried both raw and
#       on top of fusion, which is the order the physics demands.
REPS = [
    ("depth scale_rm @6000",   "depth", "scale_removed", DEPTH_CLIP_MM, 1,  "raw"),
    ("depth sil_scaled",       "depth", "sil_scaled",    DEPTH_CLIP_MM, 1,  "raw"),
    ("depth interior_only",    "depth", "interior_only", DEPTH_CLIP_MM, 1,  "raw"),
    ("depth interior @600",    "depth", "interior_only", 600.0,         1,  "raw"),
    ("depth scale_rm f5",      "depth", "scale_removed", DEPTH_CLIP_MM, 5,  "raw"),
    ("depth scale_rm f10",     "depth", "scale_removed", DEPTH_CLIP_MM, 10, "raw"),
    ("depth scale_rm f25",     "depth", "scale_removed", DEPTH_CLIP_MM, 25, "raw"),
    ("depth interior f10",     "depth", "interior_only", DEPTH_CLIP_MM, 10, "raw"),
    ("depth sil_scaled f10",   "depth", "sil_scaled",    DEPTH_CLIP_MM, 10, "raw"),
    ("depth normals",          "depth", "scale_removed", DEPTH_CLIP_MM, 1,  "normals"),
    ("depth normals f10",      "depth", "scale_removed", DEPTH_CLIP_MM, 10, "normals"),
    ("depth normals int f10",  "depth", "interior_only", DEPTH_CLIP_MM, 10, "normals"),
    ("rgb scale_rm",           "rgb",   "scale_removed", DEPTH_CLIP_MM, 1,  "raw"),
    ("rgb scale_rm f10",       "rgb",   "scale_removed", DEPTH_CLIP_MM, 10, "raw"),
]


def pool(x, k=4):
    """Mean-pool HxW by k, then flatten. Keeps the probe cheap and denoises."""
    n, h, w, c = x.shape
    return x.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4)).reshape(n, -1)


def balanced(idx, subj, per_class, rng):
    if per_class <= 0:                 # 0 = use every frame (no balancing)
        return np.sort(np.asarray(idx))
    out = []
    for s in np.unique(subj[idx]):
        pool_s = idx[subj[idx] == s]
        take = min(per_class, len(pool_s))
        out.append(rng.choice(pool_s, size=take, replace=False))
    return np.sort(np.concatenate(out))


def contrast(x, is_depth):
    """Spread of the PERSON's values in the network's own input units [-1, 1].

    load_split_arrays fills background with 0.0 and then maps [0,1] -> [-1,1],
    so background is exactly -1 in BOTH modalities and "is this a person pixel"
    is the same test for both: any channel above the array minimum. (An earlier
    version tested abs(x).sum(-1) > 1e-6 for rgb, which is 3.0 for a -1,-1,-1
    background pixel and therefore selected the whole frame.)

    `clipped` is the fraction of person pixels sitting on a rail. It matters for
    the slab sweep: the person's depth extent (p99-p1) has a median of 419 mm,
    so a 300-400 mm window necessarily pushes the extremities against the ends.
    Some of that is harmless (outstretched limbs far from the torso median are
    mostly noise) but if `clipped` is large the sweep is measuring truncation,
    not contrast, and the comparison across slabs stops being clean.
    """
    lo = float(x.min())
    m = x[..., 0] > lo + 1e-6 if is_depth else (x > lo + 1e-6).any(-1)
    v = x[m]
    if v.size == 0:
        return dict(std=float("nan"), p1_p99=float("nan"), levels8=float("nan"),
                    clipped=float("nan"), person_frac=0.0)
    p1, p99 = np.percentile(v, [1, 99])
    hi_rail = float((v >= 1.0 - 1e-5).mean())
    lo_rail = float((v <= lo + 2e-3 + 1e-5).mean())     # scale_remove floors at 1e-3 pre-scaling
    return dict(std=float(v.std()), p1_p99=float(p99 - p1),
                levels8=float((p99 - p1) / 2.0 * 255),   # of 255 8-bit steps
                clipped=hi_rail + lo_rail, person_frac=float(m.mean()))


def transfer_ratio(Xtr, ytr, Xte, yte, rng, n_pairs=3000):
    """Separability across the split gap: BOTH pair types cross train->test.

    An earlier version took the numerator across the gap (same subject, train ->
    test) but the denominator inside train (different subjects, train -> train).
    That is not a separability measure: any common-mode shift -- the camera
    moving between BIWI's sessions, which the census puts at +620 mm of
    background depth for all 28 shared subjects -- inflates the numerator alone,
    so the ratio reports the domain shift rather than whether identity is
    recoverable. HIRIDE_HANDOFF section 8.6 already shows how badly that misleads:
    same-subject Training->Walking is 1313.6 mm against different-subject
    within-Training 213.5 mm (ratio 6.2, which the old rule called
    "impossible"), while the MATCHED denominator, different-subject
    Training->Walking, is 1319.5 mm -- ratio 1.00. And depth sil_scaled still
    reaches 14.59 % at R4 with a CI clearing every trivial baseline. A rule that
    calls the paper's central result impossible is a broken rule.

    So: numerator = same subject across the gap, denominator = DIFFERENT
    subjects across the same gap. Both carry the shift, so it cancels.
      ratio  < 1  identity separates people more than frames of one person vary
      ratio ~= 1  identity is swamped -- the two distributions coincide
    Also returns d-prime, because two representations can share a mean ratio and
    differ in achievable accuracy if the distance distributions have different
    spreads; and the within-train denominator, kept only as a reference column.
    """
    common = np.intersect1d(np.unique(ytr), np.unique(yte))
    tr_by = {c: np.where(ytr == c)[0] for c in common}
    te_by = {c: np.where(yte == c)[0] for c in common}
    same, diff_x, diff_w = [], [], []
    for _ in range(n_pairs):
        c = rng.choice(common)
        same.append(np.linalg.norm(Xtr[rng.choice(tr_by[c])] - Xte[rng.choice(te_by[c])]))
        c1, c2 = rng.choice(common, 2, replace=False)
        diff_x.append(np.linalg.norm(Xtr[rng.choice(tr_by[c1])] - Xte[rng.choice(te_by[c2])]))
        diff_w.append(np.linalg.norm(Xtr[rng.choice(tr_by[c1])] - Xtr[rng.choice(tr_by[c2])]))
    s, dx, dw = np.array(same), np.array(diff_x), np.array(diff_w)
    sd = np.sqrt((s.var() + dx.var()) / 2.0)
    return dict(d_same=float(s.mean()), d_diff=float(dx.mean()),
                d_diff_within=float(dw.mean()),
                ratio=float(s.mean() / dx.mean()) if dx.mean() else float("nan"),
                dprime=float((dx.mean() - s.mean()) / sd) if sd else float("nan"))


def _ncm(Ztr, ytr, Zte, yte):
    cls = np.unique(ytr)
    M = np.stack([Ztr[ytr == c].mean(0) for c in cls])
    pred = cls[np.argmin(((Zte[:, None, :] - M[None]) ** 2).sum(-1), axis=1)]
    return (float((pred == yte).mean()),
            float(np.mean([(pred[yte == c] == c).mean() for c in np.unique(yte)])),
            pred)


def probe(Xtr, ytr, Xte, yte, n_comp=96):
    """How much identity is LINEARLY present, reported three ways.

    `raw`       PCA on train, un-whitened nearest-class-mean. Simple, but it
                measures distance in units of the leading variance directions,
                so it is BLIND to a signal that is real but low-variance --
                which is exactly what L9 predicts for the compressed depth
                interior. A flat `raw` result therefore cannot distinguish "no
                signal" from "probe cannot see it", so it must not be the only
                readout.
    `whitened`  the same, after dividing each component by its singular value.
                This is a Mahalanobis / LDA-style probe: low-variance directions
                get equal weight, so a genuine but faint interior signal shows
                up. This is the number to trust for a NULL.
    `te_centred` whitened, but the test set is centred on its OWN mean instead of
                the train mean. That removes any COMMON-MODE domain shift (the
                camera moved between BIWI sessions) while leaving identity
                structure intact. If this is much higher than `whitened`, the
                cross-session failure is a fixable offset rather than lost
                information -- a directly actionable distinction.
    """
    mu = Xtr.mean(0, keepdims=True)
    A = Xtr - mu
    n, d = A.shape
    if d > 4 * n:
        # GRAM-SIDE PCA. np.linalg.svd on an (n x d) matrix with d >> n is what
        # made the first --pool 1 attempt look hung: at pool 1 the feature
        # dimension is 65,536 for depth and 196,608 for rgb. Eigendecomposing the
        # n x n Gram matrix recovers the same subspace far more cheaply.
        G = A @ A.T
        lam, Q = np.linalg.eigh(G)
        order = np.argsort(lam)[::-1]
        k = int(min(n_comp, int((lam[order] > 1e-8).sum())))
        lam, Q = lam[order][:k], Q[:, order][:, :k]
        sv = np.sqrt(np.maximum(lam, 1e-16))
        W = (A.T @ Q) / sv                       # d x k, orthonormal columns
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        k = int(min(n_comp, Vt.shape[0]))
        W, sv = Vt[:k].T, S[:k]
    sv = np.where(sv > 1e-8, sv, 1e-8)
    Ztr, Zte = A @ W, (Xte - mu) @ W
    out = {}
    out["raw"], out["raw_subj"], _ = _ncm(Ztr, ytr, Zte, yte)
    out["whitened"], out["whitened_subj"], pred_w = _ncm(Ztr / sv, ytr, Zte / sv, yte)
    out["pred_whitened"] = pred_w
    Zte_c = ((Xte - Xte.mean(0, keepdims=True)) @ W) / sv
    Ztr_c = Ztr / sv
    out["te_centred"], out["te_centred_subj"], _ = _ncm(Ztr_c, ytr, Zte_c, yte)
    cls = np.unique(ytr)
    Zw = Ztr / sv
    gm = Zw.mean(0)
    sb = sum(len(Zw[ytr == c]) * ((Zw[ytr == c].mean(0) - gm) ** 2).sum() for c in cls)
    sw = sum(((Zw[ytr == c] - Zw[ytr == c].mean(0)) ** 2).sum() for c in cls)
    out["fisher_train"] = float(sb / sw) if sw else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--policies", default="R1_block,R4_cross_session")
    ap.add_argument("--per-class", type=int, default=90,
                    help="frames sampled per subject per split (keeps classes balanced)")
    ap.add_argument("--pool", type=int, default=4, help="spatial mean-pool factor")
    ap.add_argument("--range-bins", default="0,2000,2500,3000,3500,9999",
                    help="person-median-depth bin edges in mm. Quantisation grows as "
                         "z^2, from ~4.5 mm at 1.25 m to ~40.7 mm at 3.75 m, so if the "
                         "interior is sensor-limited its discriminability must RISE as "
                         "people get closer. BIWI spans 1240-3885 mm, so the prediction "
                         "is testable inside one dataset.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    keep = eligible_mask(z["cues"], [str(f) for f in z["feats"]])
    report = {}

    for policy in args.policies.split(","):
        kw = {}
        if policy.startswith("R1"):
            kw = dict(guard=150, match_ntrain=block_train_counts(man, guard=150, seed=0, keep=keep))
        tr, va, te = make_split(man, policy, seed=0, keep=keep, **kw)
        subj = man["subject"]
        tr = balanced(tr, subj, args.per_class, rng)
        te = balanced(te, subj, args.per_class, rng)
        classes = sorted(set(subj[tr].tolist()))
        cmap = {c: i for i, c in enumerate(classes)}
        ytr = np.array([cmap[s] for s in subj[tr]])
        yte = np.array([cmap.get(s, -1) for s in subj[te]])
        ok = yte >= 0
        te, yte = te[ok], yte[ok]
        print(f"\n{'=' * 108}\n{policy}: {len(tr)} train / {len(te)} test frames, "
              f"{len(classes)} subjects  (chance {100 / len(classes):.2f} %)\n{'=' * 108}")
        hdr = (f"{'representation':<22s}{'std':>7s}{'8bit':>6s}{'clip%':>6s}"
               f"{'ratio':>7s}{'dprime':>8s}{'NCM raw':>9s}{'whiten':>8s}{'te-ctr':>8s}"
               f"{'wh-subj':>8s}{'Fisher':>8s}")
        print(hdr); print("-" * len(hdr))

        p_med = z["cues"][:, [str(f) for f in z["feats"]].index("p_med")]
        edges = [float(x) for x in args.range_bins.split(",")]
        range_rows = []
        for label, modality, cond, slab, nfr, enc in REPS:
            shards = open_shards(args.prep, modality)
            Xtr = load_split_arrays(shards, tr, modality, cond, 16, None, man, slab,
                                    nfr, enc)
            Xte = load_split_arrays(shards, te, modality, cond, 16, None, man, slab,
                                    nfr, enc)
            c = contrast(Xtr, modality == "depth")
            Ftr, Fte = pool(Xtr, args.pool), pool(Xte, args.pool)
            del Xtr, Xte
            t = transfer_ratio(Ftr, ytr, Fte, yte, rng)
            pr = probe(Ftr, ytr, Fte, yte)
            del Ftr, Fte
            print(f"{label:<22s}{c['std']:>7.4f}{c['levels8']:>6.1f}{100 * c['clipped']:>5.1f}%"
                  f"{t['ratio']:>7.3f}{t['dprime']:>8.3f}{100 * pr['raw']:>8.2f}%"
                  f"{100 * pr['whitened']:>7.2f}%{100 * pr['te_centred']:>7.2f}%"
                  f"{100 * pr['whitened_subj']:>7.2f}%{pr['fisher_train']:>8.2f}")
            # per-range-bin accuracy of the whitened probe, on the SAME predictions
            pw = pr.pop("pred_whitened")
            zt = p_med[te]
            per_bin = {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (zt >= lo) & (zt < hi)
                if sel.sum() >= 30:
                    per_bin[f"{int(lo)}-{int(hi)}"] = dict(
                        n=int(sel.sum()), acc=float((pw[sel] == yte[sel]).mean()))
            range_rows.append((label, per_bin))
            report[f"{policy}|{label}"] = dict(
                policy=policy, representation=label, modality=modality, condition=cond,
                slab_mm=slab, frames=nfr, encoding=enc, **c, **t, **pr,
                range_bins=per_bin,
                n_train=int(len(tr)), n_test=int(len(te)), n_classes=len(classes))

        # ---- range dependence: the sensor-floor prediction -----------------
        keys = sorted({k for _, pb in range_rows for k in pb},
                      key=lambda s: int(s.split("-")[0]))
        if keys:
            print(f"\nWhitened-probe accuracy by PERSON DEPTH (quantisation grows as z^2):")
            hdr2 = f"{'representation':<22s}" + "".join(f"{k + ' mm':>16s}" for k in keys)
            print(hdr2); print("-" * len(hdr2))
            for label, pb in range_rows:
                row = f"{label:<22s}"
                for k in keys:
                    row += (f"{100 * pb[k]['acc']:9.2f}% n{pb[k]['n']:<5d}"
                            if k in pb else f"{'--':>16s}")
                print(row)

    print("\nHOW TO READ THIS")
    print("  person std / p1-p99 / 8-bit levels -- how much of the network's input range the")
    print("    BODY actually uses. Compare depth against rgb on the same row set: this is the")
    print("    contrast handicap (L3) as a number.")
    print("  ratio / dprime -- both pair types cross train->test, so the session shift cancels.")
    print("    ratio ~= 1 means identity is swamped by within-person variation; falling ratio")
    print("    (rising dprime) across slabs means range normalisation is buying separability.")
    print("  NCM whiten -- identity linearly present, no network. Trust THIS for a null: the")
    print("    raw column is blind to low-variance directions, i.e. to exactly the compressed")
    print("    interior L9 is about. te-ctr removes any common-mode session offset: if it is")
    print("    much higher than whiten, the failure is a fixable offset, not missing information.")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "signal_diagnostic.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'signal_diagnostic.json')}")


if __name__ == "__main__":
    main()
