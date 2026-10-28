"""Ekstrak landmark MediaPipe dari semua video WL-BISINDO.

new_dataset/signer{s}_label{l}_sample{n}.mp4  ->  data_landmarks/signer{s}_label{l}_sample{n}.npy
Isi .npy: array (T, 201) koordinat mentah per frame. Bisa dihentikan dan dilanjutkan:
video yang sudah punya .npy dilewati.

Jalankan dari folder SignLink:
    .venv/Scripts/python.exe training/extract_landmarks.py [--workers 6]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import frame_features, make_holistic  # noqa: E402

SRC = Path("new_dataset")
DST = Path("data_landmarks")
MAX_WIDTH = 640  # video 1080p diperkecil dulu; MediaPipe tetap memproses di resolusi rendah

_holistic = None


def work(path):
    import cv2

    global _holistic
    if _holistic is None:
        _holistic = make_holistic()
    out = DST / (path.stem + ".npy")
    cap = cv2.VideoCapture(str(path))
    rows = []
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, round(h * MAX_WIDTH / w)))
        feat = frame_features(_holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if feat is not None:
            rows.append(feat)
    cap.release()
    if len(rows) < 8:
        return path.name, total, len(rows), False
    np.save(out, np.stack(rows))
    return path.name, total, len(rows), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    DST.mkdir(exist_ok=True)
    todo = [p for p in sorted(SRC.glob("*.mp4")) if not (DST / (p.stem + ".npy")).exists()]
    print(f"{len(todo)} video perlu diproses ({args.workers} worker)", flush=True)
    failed = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (name, total, kept, ok) in enumerate(pool.imap_unordered(work, todo), 1):
            if not ok:
                failed.append(name)
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"{i}/{len(todo)}  ~{(len(todo) - i) / rate / 60:.1f} menit lagi", flush=True)
    if failed:
        print(f"{len(failed)} video gagal (badan tidak terdeteksi):", *failed, sep="\n  ")
    print("Selesai.")


if __name__ == "__main__":
    main()
