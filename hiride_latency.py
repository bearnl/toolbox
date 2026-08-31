"""The efficiency claim, measured -- a negative result to replace a false one.

    python hiride_latency.py --out $SCRATCH/hiride2/results

The 2023 paper claimed depth input "significantly reduces the model size" and
"allows for faster processing". The audit (HIRIDE_HANDOFF 2.8) showed both are
arithmetically false: 1-vs-3 input channels changes 23,232 of 58,524,466
AlexNet parameters (0.0397 %), the Flatten->Dense(4096) head held ~65 % of the
model, and no latency benchmark existed anywhere. The journal rewrite drops
those claims; this script replaces them with a measurement so the retraction is
constructive: exact parameter counts and measured throughput for every
architecture the paper reports, 1 channel vs 3.

Numbers reported per (arch, head, channels):
    params            exact trainable + non-trainable count
    infer_ms_b1       median single-image forward latency (batch 1)
    infer_fps_b32     forward throughput at batch 32
    train_step_ms     median fit step time at batch 32 (same optimiser/loss as
                      the real trainer)
All on one 256x256 input, mixed_float16, the training configuration -- numbers
are for THIS hardware (stated in the output) and support relative statements
only. Runs in ~5 min on the 1g.10gb MIG slice.
"""
import os
import json
import time
import argparse

import numpy as np


def measure(model, ch, batch=32, steps=30):
    import tensorflow as tf
    x1 = np.random.rand(1, 256, 256, ch).astype(np.float32) * 2 - 1
    xb = np.random.rand(batch, 256, 256, ch).astype(np.float32) * 2 - 1
    yb = np.random.randint(0, 28, size=batch)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy")
    # warm-up compiles the graphs; never time the first calls
    model.predict(x1, verbose=0); model.predict(xb, verbose=0)
    model.train_on_batch(xb, yb)
    t1 = []
    for _ in range(steps):
        t0 = time.perf_counter(); model.predict(x1, verbose=0)
        t1.append(time.perf_counter() - t0)
    tb = []
    for _ in range(steps):
        t0 = time.perf_counter(); model.predict(xb, verbose=0)
        tb.append(time.perf_counter() - t0)
    tt = []
    for _ in range(steps):
        t0 = time.perf_counter(); model.train_on_batch(xb, yb)
        tt.append(time.perf_counter() - t0)
    return dict(params=int(model.count_params()),
                infer_ms_b1=float(np.median(t1) * 1e3),
                infer_fps_b32=float(batch / np.median(tb)),
                train_step_ms=float(np.median(tt) * 1e3))


def alexnet_2023_head(input_shape, n_classes):
    """The 2023 architecture's Flatten->Dense(4096)x2 head, for the parameter
    accounting only: it shows where the model's size actually lived."""
    import tensorflow as tf
    from hiride_train import build_alexnet
    L = tf.keras.layers
    base = build_alexnet(input_shape, n_classes, "gap")
    feat = base.layers[-4].output          # last conv block, before the head
    x = L.Flatten()(feat)
    x = L.Dense(4096, activation="relu")(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(4096, activation="relu")(x)
    out = L.Dense(n_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(base.input, out, name="alexnet_2023head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    import tensorflow as tf
    from hiride_train import build_alexnet, build_convnext
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    hw = tf.config.experimental.get_device_details(gpus[0]).get(
        "device_name", "unknown GPU") if gpus else "CPU"
    print(f"[hw] {hw}  TF {tf.__version__}")

    rows = {}
    cases = [("alexnet/gap", lambda ch: build_alexnet((256, 256, ch), 28, "gap")),
             ("alexnet/stripe", lambda ch: build_alexnet((256, 256, ch), 28, "stripe")),
             ("alexnet/2023head", lambda ch: alexnet_2023_head((256, 256, ch), 28)),
             ("convnext_tiny", lambda ch: build_convnext((256, 256, ch), 28,
                                                         pretrained=False))]
    for name, builder in cases:
        for ch in (1, 3):
            key = f"{name}@{ch}ch"
            tf.keras.backend.clear_session()
            m = builder(ch)
            rows[key] = measure(m, ch, steps=args.steps)
            r = rows[key]
            print(f"{key:<24s} params {r['params']:>12,d}  "
                  f"b1 {r['infer_ms_b1']:6.2f} ms  b32 {r['infer_fps_b32']:8.1f} fps  "
                  f"train {r['train_step_ms']:6.2f} ms/step")
    for name, _ in cases:
        a, b = rows.get(f"{name}@1ch"), rows.get(f"{name}@3ch")
        if a and b:
            dp = 100.0 * (b["params"] - a["params"]) / b["params"]
            df = 100.0 * (a["infer_fps_b32"] / b["infer_fps_b32"] - 1.0)
            print(f"  {name:<22s} 1ch saves {dp:5.2f} % of parameters; "
                  f"throughput {df:+5.1f} %")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "latency.json")
        json.dump(dict(hardware=hw, rows=rows), open(path, "w"), indent=1)
        print(f"[written] {path}")
    print("\nREAD: these support RELATIVE sentences on this hardware only. The honest")
    print("claim is that the channel count is irrelevant to model size and speed --")
    print("the 2023 head held ~65 % of the parameters, and removing IT is what changed")
    print("the model, not the input format.")


if __name__ == "__main__":
    main()
