"""RGB-derived person masks: what an RGB-ONLY system could actually obtain.

    python hiride_rgbseg.py --prep $SCRATCH/hiride2/prep

WHY. Every masked number in the mechanism suite cuts with the shipped userMap,
which is DEPTH-derived (the OpenNI/NITE user tracker runs on the depth stream).
So masked-depth rows describe a deployable depth-only system, but masked-RGB
rows describe an RGB-D system -- the oracle step advantages RGB, not depth
(HIRIDE_HANDOFF 13.15). This script converts that caveat into a measurement:
it segments the person from the RGB shards alone, with an off-the-shelf
pretrained segmenter, and writes `<tag>_maskrgb.npy` shards that
`hiride_train.py --mask-source rgbseg` (wave 20) cuts with instead.

WHAT IT RUNS. torchvision DeepLabV3-ResNet50 (COCO-with-VOC-labels weights),
person class, on each 256x256 RGB shard frame; optionally keeps only the
largest connected person component (default on -- a deployed system would).
This is a REPRESENTATIVE deployable segmenter, not the best possible one; the
paper's sentence is "with a standard RGB person segmenter of measured quality
X", and the quality is measured here against the userMap on the same frames.

ENVIRONMENT. Deliberately NOT venv311: installing torch beside TF 2.15 risks
exactly the dependency clobbering that 13.5 documents. Run inside a separate
venv (the submit script creates ~/venvs/venv-torch with
`pip install --no-index torch torchvision scipy`) with the weights pre-cached
on a login node:
    TORCH_HOME=$HOME/.torch python -c "from torchvision.models.segmentation \
        import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights; \
        deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)"

OUTPUT (all additive; nothing existing is touched):
    <tag>_maskrgb.npy   (N,S,S) uint8, aligned POSITIONALLY with <tag>_index.npz
    rgbseg_meta.json    per-sequence quality vs the userMap: IoU quartiles,
                        empty-mask rate, model + settings. Read it before
                        interpreting any wave-20 number.
"""
import os
import json
import argparse

import numpy as np

SEQ_TAGS = ("training", "testing_still", "testing_walking")
PERSON_CLASS = 15                    # VOC label set used by torchvision's weights


def largest_component(mask):
    """Largest 4-connected component of a boolean mask (scipy)."""
    from scipy import ndimage
    lb, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lb, range(1, n + 1))
    return lb == (int(np.argmax(sizes)) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="posterior threshold on the person class. 0.5 = argmax-"
                         "equivalent for a binary read; fixed, not tuned.")
    ap.add_argument("--no-lcc", action="store_true",
                    help="keep every person-labelled pixel instead of the "
                         "largest connected component")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torchvision.models.segmentation import (deeplabv3_resnet50,
                                                 DeepLabV3_ResNet50_Weights)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(dev).eval()
    print(f"[model] deeplabv3_resnet50 ({weights}) on {dev}")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    meta = dict(model="deeplabv3_resnet50", weights=str(weights),
                person_class=PERSON_CLASS, threshold=args.threshold,
                lcc=not args.no_lcc, sequences={})
    for tag in SEQ_TAGS:
        ipath = os.path.join(args.prep, f"{tag}_index.npz")
        rpath = os.path.join(args.prep, f"{tag}_rgb.npy")
        if not (os.path.exists(ipath) and os.path.exists(rpath)):
            continue
        rgb = np.load(rpath, mmap_mode="r")
        n = rgb.shape[0]
        out = np.zeros(rgb.shape[:3], dtype=np.uint8)
        with torch.no_grad():
            for lo in range(0, n, args.batch):
                xb = np.asarray(rgb[lo:lo + args.batch]).astype(np.float32) / 255.0
                t = torch.from_numpy(xb).permute(0, 3, 1, 2).to(dev)
                logits = model((t - mean) / std)["out"]
                prob = F.softmax(logits, dim=1)[:, PERSON_CLASS]
                mb = (prob >= args.threshold).cpu().numpy()
                for j in range(mb.shape[0]):
                    m = mb[j]
                    if m.any() and not args.no_lcc:
                        m = largest_component(m)
                    out[lo + j] = m.astype(np.uint8)
                if (lo // args.batch) % 50 == 0:
                    print(f"[{tag}] {lo}/{n}")
        np.save(os.path.join(args.prep, f"{tag}_maskrgb.npy"), out)

        # Quality against the userMap, where one exists and is non-empty. This
        # number goes in the paper next to every wave-20 row: the rgb-only
        # system's accuracy is conditional on segmentation of THIS quality.
        upath = os.path.join(args.prep, f"{tag}_mask.npy")
        stats = dict(n_frames=int(n),
                     empty_rate=float((out.reshape(n, -1).max(1) == 0).mean()))
        if os.path.exists(upath):
            um = np.load(upath, mmap_mode="r")
            ious = []
            for i in range(n):
                u = np.asarray(um[i]) > 0
                if not u.any():
                    continue
                r = out[i] > 0
                inter = float((u & r).sum())
                union = float((u | r).sum())
                ious.append(inter / union if union else 0.0)
            if ious:
                q = np.percentile(ious, [25, 50, 75])
                stats.update(iou_q25=float(q[0]), iou_median=float(q[1]),
                             iou_q75=float(q[2]), n_compared=len(ious))
                print(f"[{tag}] IoU vs userMap: median {q[1]:.3f} "
                      f"[{q[0]:.3f}, {q[2]:.3f}] on {len(ious)} frames; "
                      f"empty rgbseg on {100 * stats['empty_rate']:.1f} %")
        meta["sequences"][tag] = stats

    with open(os.path.join(args.prep, "rgbseg_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[done] shards + {os.path.join(args.prep, 'rgbseg_meta.json')}")


if __name__ == "__main__":
    main()
