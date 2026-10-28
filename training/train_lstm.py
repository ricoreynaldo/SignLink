"""Latih model urutan landmark (Conv1D + BiGRU) untuk 32 kata WL-BISINDO.

    .venv/Scripts/python.exe training/train_lstm.py --cv   # uji leave-one-signer-out (akurasi jujur)
    .venv/Scripts/python.exe training/train_lstm.py        # latih model final dari semua signer

Model final: models/signlink_wl.keras + models/signlink_wl_labels.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import FEATURES, HAND_POINTS, LABELS, POSE_POINTS, SEQ_LEN, normalize, resample  # noqa: E402

SRC = Path("data_landmarks")
NAME = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)")

# Pasangan titik pose kiri/kanan MediaPipe, dipakai untuk augmentasi cermin (orang kidal).
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def load():
    seqs, labels, signers = [], [], []
    for f in sorted(SRC.glob("*.npy")):
        m = NAME.search(f.stem)
        if not m:
            continue
        seqs.append(normalize(np.load(f)))
        signers.append(int(m.group(1)))
        labels.append(int(m.group(2)))
    return seqs, np.array(labels), np.array(signers)


def mirror(seq):
    p = seq.reshape(len(seq), -1, 3).copy()
    p[:, :, 0] *= -1
    for a, b in POSE_SWAP:
        p[:, [a, b]] = p[:, [b, a]]
    left = slice(POSE_POINTS, POSE_POINTS + HAND_POINTS)
    right = slice(POSE_POINTS + HAND_POINTS, POSE_POINTS + 2 * HAND_POINTS)
    p[:, left], p[:, right] = p[:, right].copy(), p[:, left].copy()
    return p.reshape(len(seq), -1)


def augment(seq, rng):
    """Variasi acak satu urutan (sudah dinormalisasi): potong waktu, cermin, skala, putar, geser."""
    n = len(seq)
    keep = rng.uniform(0.8, 1.0)
    start = rng.integers(0, max(1, int(n * (1 - keep))) + 1)
    seq = seq[start:start + max(8, int(n * keep))]
    if rng.random() < 0.3:
        seq = mirror(seq)
    p = seq.reshape(len(seq), -1, 3)
    present = np.any(p != 0, axis=2, keepdims=True)
    a = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    p = p.copy()
    p[:, :, :2] = p[:, :, :2] @ rot.T
    p *= rng.uniform(0.9, 1.1)
    p[:, :, :2] += rng.normal(0, 0.05, 2)
    p += rng.normal(0, 0.01, p.shape)
    p = np.where(present, p, 0)
    return resample(p.reshape(len(p), -1).astype(np.float32))


def batches(seqs, y, batch, rng):
    while True:
        idx = rng.permutation(len(seqs))
        for i in range(0, len(idx) - batch + 1, batch):
            b = idx[i:i + batch]
            yield np.stack([augment(seqs[j], rng) for j in b]), y[b]


def build_model():
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input((SEQ_LEN, FEATURES))
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.GRU(96, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.GRU(64))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(len(LABELS), activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit(seqs, y, epochs, batch=32, seed=0, val=None):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model()
    steps = max(1, len(seqs) // batch)
    sched = keras.callbacks.LearningRateScheduler(lambda e: 1e-3 * 0.5 * (1 + np.cos(np.pi * e / epochs)))
    model.fit(batches(seqs, y, batch, rng), steps_per_epoch=steps, epochs=epochs, validation_data=val, callbacks=[sched], verbose=2)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true", help="leave-one-signer-out, tidak menyimpan model")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    seqs, y, signers = load()
    print(f"{len(seqs)} urutan, {len(set(signers))} signer, {len(set(y))} kata")

    if args.cv:
        results = {}
        for s in sorted(set(signers)):
            train = np.where(signers != s)[0]
            test = np.where(signers == s)[0]
            X_test = np.stack([resample(seqs[i]) for i in test])
            print(f"\n=== Uji pada signer{s} ({len(test)} video), latih dari signer lain ({len(train)}) ===")
            model = fit([seqs[i] for i in train], y[train], args.epochs)
            pred = model.predict(X_test, verbose=0)
            top1 = float(np.mean(pred.argmax(1) == y[test]))
            top3 = float(np.mean([y[test][k] in np.argsort(p)[-3:] for k, p in enumerate(pred)]))
            results[s] = (top1, top3)
            print(f"signer{s}: top-1 {top1:.1%}  top-3 {top3:.1%}")
            wrong = {}
            for t, p in zip(y[test], pred.argmax(1)):
                if t != p:
                    wrong[(LABELS[t], LABELS[p])] = wrong.get((LABELS[t], LABELS[p]), 0) + 1
            for (t, p), n in sorted(wrong.items(), key=lambda kv: -kv[1])[:5]:
                print(f"   {t} -> dikira {p}: {n}x")
        print("\nRingkasan leave-one-signer-out:")
        for s, (t1, t3) in results.items():
            print(f"  signer{s}: top-1 {t1:.1%}  top-3 {t3:.1%}")
        print(f"  rata-rata: top-1 {np.mean([r[0] for r in results.values()]):.1%}")
        return

    model = fit(seqs, y, args.epochs)
    Path("models").mkdir(exist_ok=True)
    model.save("models/signlink_wl.keras")
    with open("models/signlink_wl_labels.json", "w", encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False)
    print("Model tersimpan: models/signlink_wl.keras")


if __name__ == "__main__":
    main()
