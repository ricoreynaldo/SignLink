"""Bagian bersama untuk ekstraksi, training, dan webcam.

Satu frame -> satu vektor 201 angka:
  pose tubuh atas (25 titik x 3) + tangan kiri (21 x 3) + tangan kanan (21 x 3).
Tangan yang tidak terlihat diisi nol.
"""
import numpy as np

# Urutan label WL-BISINDO (github.com/AceKinnn/WL-BISINDO), label ke-i = LABELS[i].
LABELS = [
    "Air", "Belajar", "Cari", "Hari", "Ingat", "Lagi", "Maaf", "Makan",
    "Motor", "Saya", "Terima kasih", "Tuli", "Apa", "Siapa", "Kapan", "Di mana",
    "Mengapa", "Bagaimana", "Merah", "Kuning", "Hijau", "Hitam", "Dengar", "Berangkat",
    "Datang", "Teman", "Keluarga", "Rumah", "Pagi", "Siang", "Sore", "Malam",
]

POSE_POINTS = 25  # titik pose 0-24: wajah, bahu, siku, pergelangan, jari, pinggul
HAND_POINTS = 21
FEATURES = (POSE_POINTS + 2 * HAND_POINTS) * 3
SEQ_LEN = 32  # panjang urutan setelah di-resample
# Pergelangan lebih rendah dari ini (dalam lebar bahu, di bawah tengah bahu) = tangan istirahat
# di samping pinggul. Kamera yang framing-nya lebar ikut menangkapnya, yang sempit tidak;
# tanpa aturan ini model bingung pada orang/kamera yang belum pernah dilihat.
REST_Y = 2.0


def make_holistic(static=False):
    import mediapipe as mp

    return mp.solutions.holistic.Holistic(
        static_image_mode=static,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def frame_features(result):
    """Hasil MediaPipe Holistic -> vektor (FEATURES,), atau None kalau badan tidak terdeteksi."""
    if result.pose_landmarks is None:
        return None

    def pts(lms, n):
        if lms is None:
            return np.zeros((n, 3), np.float32)
        return np.array([[p.x, p.y, p.z] for p in lms.landmark[:n]], np.float32)

    return np.concatenate([
        pts(result.pose_landmarks, POSE_POINTS),
        pts(result.left_hand_landmarks, HAND_POINTS),
        pts(result.right_hand_landmarks, HAND_POINTS),
    ]).ravel()


def normalize(seq):
    """(T, FEATURES) koordinat gambar -> relatif ke tengah bahu, diskalakan lebar bahu.

    Dengan begitu posisi orang di frame, jarak ke kamera, dan resolusi video tidak berpengaruh.
    """
    seq = seq.reshape(len(seq), -1, 3).copy()
    present = np.any(seq != 0, axis=2, keepdims=True)  # titik tangan yang kosong tetap nol
    l_sh, r_sh = seq[:, 11], seq[:, 12]
    center = (l_sh + r_sh) / 2
    width = np.linalg.norm((l_sh - r_sh)[:, :2], axis=1)
    scale = np.median(width[width > 1e-6]) if np.any(width > 1e-6) else 1.0
    seq = (seq - center[:, None]) / scale
    seq = np.where(present, seq, 0)
    for start in (POSE_POINTS, POSE_POINTS + HAND_POINTS):
        hand = seq[:, start:start + HAND_POINTS]
        hand[hand[:, 0, 1] > REST_Y] = 0  # titik 0 tangan = pergelangan
    return seq.reshape(len(seq), -1).astype(np.float32)


def active_hand(raw_frame):
    """True kalau ada tangan terangkat (bukan istirahat di samping pinggul) di satu frame mentah."""
    return bool(np.any(normalize(raw_frame[None])[0, POSE_POINTS * 3:] != 0))


def resample(seq, n=SEQ_LEN):
    """Ubah panjang urutan jadi n frame dengan interpolasi linear di sumbu waktu."""
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, n)
    return np.stack([np.interp(t_new, t_old, seq[:, j]) for j in range(seq.shape[1])], axis=1).astype(np.float32)


def prepare(raw):
    """Urutan mentah dari video/webcam -> input model (SEQ_LEN, FEATURES)."""
    return resample(normalize(raw))
