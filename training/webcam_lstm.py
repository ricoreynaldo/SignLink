"""Terjemah isyarat WL-BISINDO dari webcam secara real-time.

Angkat tangan, peragakan satu kata, lalu turunkan tangan. Begitu tangan turun, urutan
gerakan tadi diprediksi dan kata yang yakin masuk ke kalimat di panel kanan.

Tombol (atau klik di panel):
  1-3 pilih/ganti tebakan · H daftar kata · P video panduan · S salin kalimat
  Backspace hapus kata · C hapus kalimat · Z batalkan hapus · Q / Esc keluar

    .venv/Scripts/python.exe training/webcam_lstm.py [--camera 0] [--video file.mp4]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import HAND_POINTS, POSE_POINTS, active_hand, frame_features, make_holistic, prepare  # noqa: E402

PRE_ROLL = 6  # frame sebelum tangan terangkat ikut dimasukkan, seperti awal video dataset
END_AFTER = 8  # tangan hilang selama sekian frame = isyarat selesai
MIN_FRAMES = 12
MIN_CONF = 0.5
MAX_WORDS = 12
LONG_SIGN = 5.0  # detik; lebih lama dari ini kemungkinan lupa menurunkan tangan
DETECT_WIDTH = 640  # frame diperkecil sebelum MediaPipe, sama seperti saat ekstraksi dataset
GUIDE = os.path.join(os.path.dirname(__file__), "..", "panduan", "index.html")

# ---------- Tampilan ----------
# Palet dan gaya mengikuti portfolio mymine: monokrom, latar putih, kartu gelap, font Satoshi,
# label kapital berjarak lebar, pill bergaris tipis, dan sorotan blok hitam (.hl).
CAM_H = 600  # tinggi awal jendela
MIN_H = 560  # panel butuh setinggi ini; jendela yang lebih pendek digambar segini lalu diperkecil OpenCV
PANEL_W = 380
PAD = 28

INK = (25, 25, 25)  # --ink #191919
BG = (255, 255, 255)  # --bg
MUTED = (110, 110, 110)  # --muted #6e6e6e
SOFT = (244, 244, 242)  # --soft #f4f4f2
LINE = (231, 231, 229)  # --line #e7e7e5
CARD = (42, 45, 49)  # --card #2a2d31
CARD_2 = (52, 56, 61)  # --card-2 #34383d
WHITE = (255, 255, 255)
ON_CARD = (165, 168, 172)  # teks redup di kartu gelap
DIM_ON_CARD = (130, 133, 138)
REC = (225, 72, 60)  # satu-satunya warna: titik rekam

FONT_DIRS = [
    os.path.join(os.path.dirname(__file__), "fonts"),  # taruh Satoshi-*.otf/ttf di sini (fontshare.com)
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
]


def font(names, size):
    for name in names:
        for d in FONT_DIRS:
            path = os.path.join(d, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def satoshi(weight, size):
    """Satoshi (font mymine) kalau ada di training/fonts, selain itu Segoe UI dengan ketebalan setara."""
    fallback = {"Regular": "segoeui.ttf", "Medium": "seguisb.ttf", "Bold": "segoeuib.ttf", "Black": "seguibl.ttf"}[weight]
    return font([f"Satoshi-{weight}.otf", f"Satoshi-{weight}.ttf", fallback, "segoeuib.ttf"], size)


F_TITLE = satoshi("Black", 30)
F_SMALL = satoshi("Regular", 13)
F_LABEL = satoshi("Bold", 11)
F_BODY = satoshi("Medium", 16)
F_WORD = satoshi("Black", 46)
F_SENT = satoshi("Bold", 22)
F_PILL = satoshi("Medium", 13)
F_KEY = satoshi("Bold", 11)


@dataclass
class State:
    status: str = "idle"  # loading | idle | recording | finishing | no_body
    rec_secs: float = 0.0  # lama rekaman yang sedang berjalan
    word: str = ""  # hasil terakhir
    confident: bool = False
    top3: list = field(default_factory=list)  # [(kata, peluang), ...]
    choice: int = -1  # tebakan yang sedang dipakai di kalimat (indeks top3), -1 = belum ada
    added: bool = False  # kata terakhir kalimat berasal dari hasil terakhir (bisa diganti dengan 1-3)
    sentence: list = field(default_factory=list)
    undo: list = None  # kalimat sebelum dihapus dengan C
    shown_at: float = 0.0  # waktu hasil terakhir muncul, untuk efek sorot
    toast: str = ""
    toast_at: float = 0.0
    show_help: bool = False
    hover: object = None  # aksi di bawah kursor, untuk efek hover
    fps: float = 0.0  # tampilan
    det_fps: float = 0.0  # frame yang diproses MediaPipe per detik

    def notify(self, text):
        self.toast, self.toast_at = text, time.time()


# ---------- Aksi (keyboard dan klik memakai jalur yang sama) ----------
def pick(s, i):
    """Masukkan tebakan ke-i ke kalimat; kalau hasil ini sudah masuk, ganti kata terakhirnya."""
    if i >= len(s.top3):
        return
    word = s.top3[i][0]
    if s.added and s.sentence:
        if s.sentence[-1] != word:
            s.notify(f"Diganti: {word}")
        s.sentence[-1] = word
    else:
        s.sentence = (s.sentence + [word])[-MAX_WORDS:]
        s.notify(f"Ditambahkan: {word}")
    s.choice, s.added = i, True


def act(s, action):
    if action is None:
        return
    if isinstance(action, tuple) and action[0] == "pick":
        pick(s, action[1])
    elif action == "backspace" and s.sentence:
        s.sentence.pop()
        if s.added:
            s.added, s.choice = False, -1
    elif action == "clear" and s.sentence:
        s.undo, s.sentence = s.sentence, []
        s.added, s.choice = False, -1
        s.notify("Kalimat dihapus · Z untuk batalkan")
    elif action == "undo" and s.undo:
        s.sentence, s.undo = s.undo, None
        s.notify("Kalimat dikembalikan")
    elif action == "copy":
        if not s.sentence:
            s.notify("Kalimat masih kosong")
        elif os.name == "nt":
            # Semua kata di daftar berupa ASCII, jadi aman dikirim apa adanya ke clip.exe.
            subprocess.run(["clip"], input=" ".join(s.sentence).encode("ascii", "replace"), check=False)
            s.notify("Kalimat disalin")
        else:
            s.notify("Salin hanya tersedia di Windows")
    elif action == "help":
        s.show_help = not s.show_help
    elif action == "guide":
        if os.path.exists(GUIDE) and hasattr(os, "startfile"):
            os.startfile(os.path.abspath(GUIDE))
            s.notify("Membuka video panduan di browser")
        else:
            s.notify("Panduan belum dibuat: jalankan buat_panduan.py")


KEYS = {ord("1"): ("pick", 0), ord("2"): ("pick", 1), ord("3"): ("pick", 2), 8: "backspace", ord("c"): "clear",
        ord("z"): "undo", ord("s"): "copy", ord("h"): "help", ord("p"): "guide"}


# ---------- Panel ----------
def tracked(draw, xy, text, fnt, fill, track=0.0):
    """Teks dengan jarak huruf (letter-spacing) dalam px; PIL tidak punya opsi ini. Mengembalikan lebarnya."""
    if not track:
        draw.text(xy, text, font=fnt, fill=fill)
        return draw.textlength(text, font=fnt)
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + track
    return x - xy[0] - track


def tracked_len(draw, text, fnt, track=0.0):
    return draw.textlength(text, font=fnt) + track * max(len(text) - 1, 0)


def label(draw, xy, text, fill=MUTED):
    """Gaya .label mymine: huruf kapital kecil, jarak huruf 0.14em."""
    tracked(draw, xy, text.upper(), F_LABEL, fill, 11 * 0.14)


def wrap(draw, words, fnt, width, track=0.0):
    lines, line = [], []
    for w in words:
        trial = line + [w]
        if tracked_len(draw, " ".join(trial), fnt, track) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = [w]
    return lines + ([line] if line else [])


def pill(draw, x, y, text, kind):
    """Status: 'outline' (siap), 'solid' (merekam), 'soft' (menunggu / posisi)."""
    w = draw.textlength(text, font=F_PILL) + 38
    box = (x, y, x + w, y + 30)
    if kind == "solid":
        draw.rounded_rectangle(box, radius=15, fill=INK)
        dot, color = REC, WHITE
    elif kind == "soft":
        draw.rounded_rectangle(box, radius=15, fill=SOFT)
        dot, color = MUTED, INK
    else:
        draw.rounded_rectangle(box, radius=15, fill=BG, outline=LINE, width=1)
        dot, color = INK, INK
    draw.ellipse((x + 14, y + 11, x + 22, y + 19), fill=dot)
    draw.text((x + 28, y + 7), text, font=F_PILL, fill=color)


def button(draw, hits, x, y, text, action, hover):
    """Tombol pill kecil bergaris (gaya tag mymine); hover = latar --soft. Mengembalikan kiri tombol."""
    w = draw.textlength(text, font=F_PILL) + 24
    box = (x - w, y, x, y + 26)
    draw.rounded_rectangle(box, radius=13, fill=SOFT if hover == action else BG, outline=INK if hover == action else LINE)
    draw.text((x - w + 12, y + 5), text, font=F_PILL, fill=INK)
    hits.append((box, action))
    return x - w


def keycap(draw, x, y, key, text):
    kw = max(22, draw.textlength(key, font=F_KEY) + 14)
    draw.rounded_rectangle((x, y, x + kw, y + 22), radius=6, fill=BG, outline=LINE, width=1)
    draw.text((x + (kw - draw.textlength(key, font=F_KEY)) / 2, y + 4), key, font=F_KEY, fill=INK)
    draw.text((x + kw + 7, y + 3), text, font=F_SMALL, fill=MUTED)
    return x + kw + 7 + draw.textlength(text, font=F_SMALL) + 14


def status_text(s):
    """Pill status + satu kalimat petunjuk langkah berikutnya."""
    if s.status == "loading":
        return ("Menyiapkan deteksi…", "soft"), "Sebentar, kamera sudah jalan"
    if s.status == "no_body":
        return ("Bahu belum terlihat", "soft"), "Mundur sedikit sampai bahu masuk kamera"
    if s.status == "recording":
        hint = "Turunkan tangan untuk menyelesaikan" if s.rec_secs > LONG_SIGN else "Turunkan tangan bila gerakan selesai"
        return (f"Merekam  ·  {s.rec_secs:.1f} dtk", "solid"), hint
    if s.status == "finishing":
        return ("Memproses…", "soft"), "Tahan tangan di bawah sebentar"
    if s.word and not s.confident and not s.added:
        return ("Siap", "outline"), "Kurang yakin: pilih 1–3, atau ulangi isyarat"
    if s.word:
        return ("Siap", "outline"), "Salah tebak? Pilih 1–3 untuk mengganti"
    return ("Siap", "outline"), "Angkat tangan untuk mulai"


def draw_panel(draw, x0, h, s):
    """Gambar panel; kembalikan daftar area klik [(kotak, aksi)] relatif ke panel."""
    hits = []
    x, w = x0 + PAD, PANEL_W - 2 * PAD
    right = x0 + PANEL_W - PAD
    draw.rectangle((x0, 0, x0 + PANEL_W, h), fill=BG)
    draw.line((x0, 0, x0, h), fill=LINE, width=1)

    tracked(draw, (x, 22), "SignLink", F_TITLE, INK, -30 * 0.035)
    draw.text((x, 60), "Penerjemah BISINDO · 32 kata", font=F_SMALL, fill=MUTED)
    if s.fps:
        for i, t in enumerate((f"{s.fps:.0f} fps", f"deteksi {s.det_fps:.0f}" if s.det_fps else "")):
            draw.text((right - draw.textlength(t, font=F_SMALL), 28 + i * 17), t, font=F_SMALL, fill=MUTED)

    (text, kind), hint = status_text(s)
    pill(draw, x, 92, text, kind)
    draw.text((x, 130), hint, font=F_SMALL, fill=INK if s.status == "recording" and s.rec_secs > LONG_SIGN else MUTED)

    # Kartu hasil: kartu gelap seperti kartu Works di mymine.
    y = 158
    cx0, cx1 = x0 + PAD - 4, x0 + PANEL_W - PAD + 4
    draw.rounded_rectangle((cx0, y, cx1, y + 240), radius=22, fill=CARD)
    ix, iw = cx0 + 20, cx1 - cx0 - 40
    label(draw, (ix, y + 18), "Hasil terakhir", ON_CARD)
    if s.word:
        shown = s.top3[s.choice][0] if s.added and 0 <= s.choice < len(s.top3) else s.word
        track = -46 * 0.035
        word = shown
        while tracked_len(draw, word, F_WORD, track) > iw - 16 and len(word) > 3:
            word = word[:-2] + "…"
        ww = tracked_len(draw, word, F_WORD, track)
        fresh = time.time() - s.shown_at < 1.2
        if s.added and fresh:
            # Sorotan blok (.hl), dibalik untuk kartu gelap: blok putih, teks ink.
            draw.rectangle((ix - 6, y + 46, ix + ww + 6, y + 96), fill=WHITE)
            tracked(draw, (ix, y + 36), word, F_WORD, INK, track)
        else:
            tracked(draw, (ix, y + 36), word, F_WORD, WHITE if s.added else ON_CARD, track)
        by = y + 106
        if not s.confident and not s.added:
            note = "Kurang yakin"
            nw = draw.textlength(note, font=F_SMALL) + 22
            draw.rounded_rectangle((ix + iw - nw, y + 14, ix + iw, y + 36), radius=11, outline=(110, 113, 118), width=1)
            draw.text((ix + iw - nw + 11, y + 17), note, font=F_SMALL, fill=(210, 212, 215))
        for i, (name, p) in enumerate(s.top3):
            row = (ix - 10, by - 6, ix + iw + 10, by + 34)
            chosen = s.added and i == s.choice
            if s.hover == ("pick", i):
                draw.rounded_rectangle(row, radius=10, fill=CARD_2)
            hits.append((row, ("pick", i)))
            fg = WHITE if chosen or (not s.added and i == 0) else (150, 153, 158)
            # Nomor = tombol pintas 1-3.
            draw.rounded_rectangle((ix, by + 1, ix + 18, by + 19), radius=5, outline=WHITE if chosen else DIM_ON_CARD,
                                   fill=WHITE if chosen else None)
            n = str(i + 1)
            draw.text((ix + 9 - draw.textlength(n, font=F_KEY) / 2, by + 3), n, font=F_KEY, fill=INK if chosen else ON_CARD)
            draw.text((ix + 28, by), name, font=F_BODY, fill=fg)
            pct = f"{p:.0%}"
            draw.text((ix + iw - draw.textlength(pct, font=F_BODY), by), pct, font=F_BODY, fill=fg)
            bar_y = by + 25
            draw.rounded_rectangle((ix + 28, bar_y, ix + iw, bar_y + 4), radius=2, fill=CARD_2)
            bw = max(4, int((iw - 28) * p))
            draw.rounded_rectangle((ix + 28, bar_y, ix + 28 + bw, bar_y + 4), radius=2, fill=fg if fg == WHITE else (110, 113, 118))
            by += 42
    else:
        draw.text((ix, y + 42), "Belum ada isyarat", font=F_BODY, fill=ON_CARD)
        hint = ["Pastikan bahu terlihat", "Angkat tangan, peragakan kata", "Turunkan tangan"]
        for i, t in enumerate(hint):
            ty = y + 88 + i * 44
            draw.text((ix, ty + 1), f"0{i + 1}", font=F_PILL, fill=DIM_ON_CARD)
            draw.text((ix + 32, ty - 1), t, font=F_BODY, fill=WHITE)
            if i < len(hint) - 1:
                draw.line((ix, ty + 30, ix + iw, ty + 30), fill=CARD_2, width=1)

    # Kalimat: kata terakhir disorot blok hitam seperti .hl di mymine; tombol salin/hapus di kanan.
    y = 418
    label(draw, (x, y + 7), "Kalimat")
    bx = button(draw, hits, right, y, "Hapus", "clear", s.hover)
    button(draw, hits, bx - 6, y, "Salin", "copy", s.hover)
    sy = y + 36
    if s.sentence:
        track = -22 * 0.02
        space = draw.textlength(" ", font=F_SENT)
        max_lines = max(1, (h - 80 - sy) // 34)  # berhenti sebelum baris keycap
        lines = wrap(draw, s.sentence, F_SENT, w - 12, track)[-max_lines:]
        for li, line in enumerate(lines):
            lx, ly = x, sy + li * 34
            for wi, word in enumerate(line):
                ww = tracked_len(draw, word, F_SENT, track)
                last = li == len(lines) - 1 and wi == len(line) - 1
                if last:
                    lx += 6 if wi else 4  # beri ruang supaya blok tidak menempel kata sebelumnya
                    draw.rectangle((lx - 4, ly + 5, lx + ww + 4, ly + 32), fill=INK)
                tracked(draw, (lx, ly), word, F_SENT, WHITE if last else INK, track)
                lx += ww + space
    else:
        draw.text((x, sy + 2), "Kata yang yakin akan muncul di sini", font=F_BODY, fill=MUTED)

    # Petunjuk tombol: H dan Q juga bisa diklik.
    ky = h - 72
    kx = keycap(draw, x, ky, "1–3", "pilih tebakan")
    hx = kx
    kx = keycap(draw, kx, ky, "H", "daftar kata")
    hits.append(((hx - 4, ky - 4, kx - 8, ky + 26), "help"))
    ky = h - 42
    kx = keycap(draw, x, ky, "S", "salin")
    kx = keycap(draw, kx, ky, "Backspace", "hapus kata")
    keycap(draw, kx, ky, "Q", "keluar")
    return hits


# ---------- Kamera + overlay ----------
def draw_skeleton(frame, feat):
    """Gambar bahu, lengan, dan tangan tipis-tipis dari vektor fitur mentah (sebelum dicerminkan)."""
    if feat is None:
        return
    h, w = frame.shape[:2]
    overlay = frame.copy()
    pts = feat.reshape(-1, 3)

    def px(block):
        return [(int(x * w), int(y * h)) for x, y, _ in block]

    p = px(pts[:POSE_POINTS])
    for a, b in [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
        cv2.line(overlay, p[a], p[b], WHITE, 2, cv2.LINE_AA)
    for start in (POSE_POINTS, POSE_POINTS + HAND_POINTS):
        hand = pts[start:start + HAND_POINTS]
        if not hand.any():
            continue
        p = px(hand)
        for a, b in HAND_EDGES:
            cv2.line(overlay, p[a], p[b], WHITE, 2, cv2.LINE_AA)
        for q in p:
            cv2.circle(overlay, q, 4, INK[::-1], -1, cv2.LINE_AA)
            cv2.circle(overlay, q, 4, WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# Posisi panel di jendela terakhir, untuk menerjemahkan klik mouse ke area panel.
_layout = {"area_w": 0, "scale": 1.0, "hits": []}


def default_size(frame):
    h, w = frame.shape[:2]
    return round(w * CAM_H / h) + PANEL_W, CAM_H


def render(frame, s, size=None, labels=None):
    """Frame kamera (BGR, sudah dicerminkan) + State -> gambar jendela (BGR) berukuran size (lebar, tinggi).

    Panel kanan tetap PANEL_W; kamera mengisi sisanya dengan rasio aslinya (tidak ditarik),
    sisi yang kosong diberi latar --soft.
    """
    win_w, win_h = size or default_size(frame)
    if win_h < MIN_H:
        # Terlalu pendek untuk panel: gambar lebih besar lalu perkecil merata (rasio tetap).
        big = render(frame, s, (round(win_w * MIN_H / win_h), MIN_H), labels)
        _layout["scale"] = MIN_H / win_h
        return cv2.resize(big, (win_w, win_h), interpolation=cv2.INTER_AREA)
    _layout["scale"] = 1.0
    area_w = max(win_w - PANEL_W, 160)
    h, w = frame.shape[:2]
    k = min(area_w / w, win_h / h)
    cam_w, cam_h = max(1, round(w * k)), max(1, round(h * k))
    x0, y0 = (area_w - cam_w) // 2, (win_h - cam_h) // 2
    cam = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    # Area kamera dirakit dengan numpy/OpenCV; hanya panel (sempit) yang digambar PIL, supaya
    # jendela besar/maksimal tetap ringan.
    canvas = np.empty((win_h, area_w + PANEL_W, 3), np.uint8)
    canvas[:] = SOFT[::-1]
    canvas[y0:y0 + cam_h, x0:x0 + cam_w] = cam
    x1, y1 = x0 + cam_w, y0 + cam_h

    if s.status in ("recording", "finishing"):
        cv2.rectangle(canvas, (x0 + 3, y0 + 3), (x1 - 4, y1 - 4), INK[::-1], 6)
        on = s.status == "finishing" or int(time.time() * 2) % 2
        blit(canvas, badge("rec_on" if on else "rec_off"), x0 + 18, y0 + 18)
    bottom = y1 - 70
    if s.status == "no_body":
        patch = badge("no_body")
        blit(canvas, patch, (x0 + x1 - patch[0].shape[1]) // 2, bottom)
        bottom -= 52
    if s.toast and time.time() - s.toast_at < 2.0:
        patch = toast_patch(s.toast)
        blit(canvas, patch, (x0 + x1 - patch[0].shape[1]) // 2, bottom)
    if s.show_help and labels:
        draw_help(canvas, x0, y0, x1, y1, labels)

    canvas[:, area_w:] = panel_image(s, win_h)
    _layout["area_w"] = area_w
    return canvas


_panel = {"key": None, "img": None, "hits": [], "fps": (0.0, 0.0), "fps_at": 0.0}


def panel_image(s, h):
    """Panel hanya digambar ulang kalau isinya berubah; angka fps diperbarui tiap 0,5 detik."""
    now = time.time()
    if now - _panel["fps_at"] >= 0.5:
        _panel["fps"], _panel["fps_at"] = (s.fps, s.det_fps), now
    fresh = now - s.shown_at < 1.2
    key = (h, s.status, round(s.rec_secs, 1), s.word, s.confident, tuple(s.top3), s.choice, s.added,
           tuple(s.sentence), fresh, s.hover, round(_panel["fps"][0]), round(_panel["fps"][1]))
    if key != _panel["key"]:
        shown = State(**{**s.__dict__, "fps": _panel["fps"][0], "det_fps": _panel["fps"][1]})
        img = Image.new("RGB", (PANEL_W, h), BG)
        _panel["hits"] = draw_panel(ImageDraw.Draw(img), 0, h, shown)
        _panel["key"], _panel["img"] = key, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    _layout["hits"] = _panel["hits"]
    return _panel["img"]


def hit_test(x, y):
    """Koordinat mouse di jendela -> aksi panel di bawahnya, atau None."""
    k = _layout["scale"]
    px, py = x * k - _layout["area_w"], y * k
    for (a, b, c, d), action in _layout["hits"]:
        if a <= px <= c and b <= py <= d:
            return action
    return None


_badges = {}


def _patch(img):
    a = np.asarray(img)
    return cv2.cvtColor(a[:, :, :3], cv2.COLOR_RGB2BGR), a[:, :, 3:] / 255.0


def badge(kind):
    """Label kecil di atas video, digambar sekali lalu disimpan: (BGR, alpha)."""
    if kind not in _badges:
        if kind == "no_body":
            msg = "Mundur sedikit sampai bahu terlihat"
            w = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(msg, font=F_BODY)) + 40
            img = Image.new("RGBA", (w, 42), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, w - 1, 41), radius=21, fill=WHITE + (255,))
            d.text((20, 10), msg, font=F_BODY, fill=INK)
        else:
            img = Image.new("RGBA", (136, 34), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((0, 0, 135, 33), radius=17, fill=INK + (255,))
            d.ellipse((15, 12, 25, 22), fill=(REC if kind == "rec_on" else (90, 90, 90)) + (255,))
            tracked(d, (36, 9), "MEREKAM", F_LABEL, WHITE, 11 * 0.14)
        _badges[kind] = _patch(img)
    return _badges[kind]


def toast_patch(text):
    """Notifikasi singkat: pill hitam di bawah video."""
    key = ("toast", text)
    if key not in _badges:
        w = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=F_BODY)) + 40
        img = Image.new("RGBA", (w, 42), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, 41), radius=21, fill=INK + (255,))
        d.text((20, 10), text, font=F_BODY, fill=WHITE)
        _badges[key] = _patch(img)
    return _badges[key]


def draw_help(canvas, x0, y0, x1, y1, labels):
    """Daftar 32 kata di atas video (H untuk membuka/menutup)."""
    key = ("help", x1 - x0, y1 - y0)
    if key not in _badges:
        w, h = x1 - x0, y1 - y0
        img = Image.new("RGBA", (w, h), (255, 255, 255, 235))
        d = ImageDraw.Draw(img)
        mx = max(24, w // 18)
        tracked(d, (mx, 26), "Daftar kata", F_TITLE, INK, -30 * 0.035)
        d.text((mx, 66), "32 kata yang dikenali.  H tutup  ·  P buka video panduan", font=F_SMALL, fill=MUTED)
        cols = 4 if w >= 640 else 3
        rows = -(-len(labels) // cols)
        cw = (w - 2 * mx) / cols
        rh = min(34, (h - 110 - 20) / rows)
        for i, name in enumerate(labels):
            c, r = divmod(i, rows)
            cx, cy = mx + c * cw, 104 + r * rh
            d.text((cx, cy + 2), f"{i + 1:02d}", font=F_PILL, fill=MUTED)
            d.text((cx + 30, cy), name, font=F_BODY, fill=INK)
            d.line((cx, cy + rh - 6, cx + cw - 16, cy + rh - 6), fill=LINE)
        _badges[key] = _patch(img)
    blit(canvas, _badges[key], x0, y0)


def blit(canvas, patch, x, y):
    bgr, alpha = patch
    h, w = bgr.shape[:2]
    x, y = max(0, int(x)), max(0, int(y))
    region = canvas[y:y + h, x:x + w]
    h, w = region.shape[:2]
    region[:] = (bgr[:h, :w] * alpha[:h, :w] + region * (1 - alpha[:h, :w])).astype(np.uint8)


def window_size(win):
    """Ukuran area gambar jendela saat ini, atau None kalau belum bisa dibaca."""
    try:
        _, _, w, h = cv2.getWindowImageRect(win)
    except cv2.error:
        return None
    return (w, h) if w > 0 and h > 0 else None


# ---------- Deteksi (proses terpisah) ----------
def detector(inq, outq):
    """Jalan di proses sendiri: MediaPipe menahan GIL, jadi thread biasa membuat tampilan tersendat."""
    if os.name == "nt":
        # Prioritas di bawah normal: kalau CPU penuh, tampilan didahulukan supaya video tetap lancar.
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    holistic = make_holistic()
    outq.put(("ready", None))
    while True:
        item = inq.get()
        if item is None:
            break
        idx, rgb = item
        outq.put((idx, frame_features(holistic.process(rgb))))


class Recognizer:
    """Potong aliran fitur jadi satu isyarat per angkat-turun tangan, lalu prediksi."""

    def __init__(self, model, labels, s):
        self.model, self.labels, self.s = model, labels, s
        self.history = deque(maxlen=PRE_ROLL)
        self.segment = []
        self.missing = 0
        self.started = 0.0

    def feed(self, feat):
        s = self.s
        if feat is None:
            s.status = "no_body"
            return
        if active_hand(feat):
            if not self.segment:
                self.segment = list(self.history)
                self.started = time.time()
            self.segment.append(feat)
            self.missing = 0
        elif self.segment:
            self.segment.append(feat)
            self.missing += 1
            if self.missing >= END_AFTER:
                seq = np.stack(self.segment[:-END_AFTER + 2])
                if len(seq) >= MIN_FRAMES:
                    self.predict(seq)
                else:
                    s.notify("Gerakan terlalu singkat, ulangi lebih lengkap")
                self.segment = []
                self.missing = 0
        self.history.append(feat)
        if not self.segment:
            s.status, s.rec_secs = "idle", 0.0
        else:
            s.status = "finishing" if self.missing else "recording"
            s.rec_secs = time.time() - self.started

    def predict(self, seq):
        s, labels = self.s, self.labels
        prob = self.model(prepare(seq)[None], training=False).numpy()[0]
        top = np.argsort(prob)[::-1][:3]
        s.word = labels[top[0]]
        s.confident = bool(prob[top[0]] >= MIN_CONF)
        s.top3 = [(labels[i], float(prob[i])) for i in top]
        s.shown_at = time.time()
        s.added, s.choice = False, -1
        if s.confident:
            s.sentence = (s.sentence + [s.word])[-MAX_WORDS:]
            s.added, s.choice = True, 0
        print(", ".join(f"{labels[i]} {prob[i]:.0%}" for i in top), flush=True)


def try_camera(index, backend):
    """Buka kamera dan pastikan benar-benar memberi gambar (bukan hitam). None kalau gagal."""
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    # MJPG 720p 30 fps: banyak webcam Windows tanpa ini cuma memberi 5-15 fps.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(10):  # beberapa webcam butuh beberapa frame sebelum gambarnya muncul
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 2:
            return cap
    cap.release()
    return None


def open_camera(index=None):
    """Kamera pilihan (--camera), atau yang pertama memberi gambar dari indeks 0-4.

    Di Windows, Media Foundation (dipakai juga oleh aplikasi Camera) dicoba dulu: sebagian webcam,
    misalnya V380 FHD Camera, hanya memberi frame hitam lewat DirectShow. Kamera virtual seperti
    OBS hanya terdaftar di DirectShow, jadi baru terpilih kalau tidak ada webcam lain yang jalan.
    Frame hitam dianggap gagal dan kamera berikutnya dicoba.
    """
    indexes = [index] if index is not None else range(5)
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW] if os.name == "nt" else [cv2.CAP_ANY]
    for backend in backends:
        for i in indexes:
            cap = try_camera(i, backend)
            if cap is not None:
                print(f"Memakai kamera {i}", flush=True)
                return cap
    return None


# ---------- Loop utama ----------
def main():
    import multiprocessing as mproc
    import queue

    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="nomor kamera; default: otomatis, kamera pertama yang ada gambarnya")
    ap.add_argument("--video", help="pakai file video, bukan webcam (untuk uji)")
    ap.add_argument("--workers", type=int, default=2, help="jumlah proses MediaPipe (lebih banyak = deteksi lebih rapat)")
    args = ap.parse_args()

    ctx = mproc.get_context("spawn")
    inq, outq = ctx.Queue(maxsize=args.workers), ctx.Queue()
    workers = [ctx.Process(target=detector, args=(inq, outq), daemon=True) for _ in range(args.workers)]
    for p in workers:
        p.start()  # dimulai duluan: memuat MediaPipe berjalan bersamaan dengan memuat TensorFlow

    from tensorflow import keras

    model = keras.models.load_model("models/signlink_wl.keras")
    with open("models/signlink_wl_labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    model(np.zeros((1, *model.input_shape[1:]), np.float32), training=False)  # pemanasan

    cap = cv2.VideoCapture(args.video) if args.video else open_camera(args.camera)
    if cap is None:
        print("Tidak ada kamera yang memberi gambar. Tutup aplikasi lain yang memakai kamera "
              "(aplikasi Camera, Zoom, Meet, OBS), atau pilih manual dengan --camera 1.")
        for p in workers:
            p.terminate()
        return
    video_dt = 1 / (cap.get(cv2.CAP_PROP_FPS) or 30) if args.video else 0
    win = "SignLink"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    s = State(status="loading")
    rec = Recognizer(model, labels, s)
    clicks = []

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            s.hover = hit_test(mx, my)
        elif event == cv2.EVENT_LBUTTONDOWN:
            clicks.append(hit_test(mx, my))

    cv2.setMouseCallback(win, on_mouse)
    first = True
    ready = 0
    next_id = expect = 0
    pending = {}
    last_feat = None
    last = time.time()
    det_times = deque(maxlen=30)

    try:
        while True:
            t_frame = time.time()
            ok, frame = cap.read()
            if not ok:
                if not args.video:
                    print("Kamera tidak bisa dibaca.")
                break

            # Kirim frame ke MediaPipe kalau ada proses yang senggang; kalau semua sibuk, frame ini
            # cuma ditampilkan. Fitur diambil dari frame asli (tidak dicerminkan), seperti dataset.
            if ready:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (DETECT_WIDTH, round(h * DETECT_WIDTH / w))) if w > DETECT_WIDTH else frame
                try:
                    inq.put_nowait((next_id, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                    next_id += 1
                except queue.Full:
                    pass

            # Ambil hasil yang sudah jadi, proses berurutan sesuai nomor frame.
            while True:
                try:
                    idx, feat = outq.get_nowait()
                except queue.Empty:
                    break
                if idx == "ready":
                    ready += 1
                    if s.status == "loading":
                        s.status = "idle"
                else:
                    pending[idx] = feat
            while expect in pending:
                last_feat = pending.pop(expect)
                expect += 1
                rec.feed(last_feat)
                det_times.append(time.time())
            if len(det_times) > 1:
                s.det_fps = (len(det_times) - 1) / max(det_times[-1] - det_times[0], 1e-3)

            now = time.time()
            s.fps = 0.9 * s.fps + 0.1 / max(now - last, 1e-3) if s.fps else 1 / max(now - last, 1e-3)
            last = now

            draw_skeleton(frame, last_feat)
            if first:
                cv2.resizeWindow(win, *default_size(frame))
                first = False
            # Digambar persis seukuran jendela, jadi OpenCV tidak perlu menarik gambarnya.
            cv2.imshow(win, render(cv2.flip(frame, 1), s, window_size(win), labels))
            wait = max(1, int((video_dt - (time.time() - t_frame)) * 1000)) if video_dt else 1
            key = cv2.waitKey(wait) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            act(s, KEYS.get(key))
            while clicks:
                act(s, clicks.pop(0))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in workers:
            try:
                inq.put(None, timeout=1)
            except queue.Full:
                pass
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        if args.video:
            print(f"tampilan {s.fps:.0f} fps, deteksi {s.det_fps:.0f} fps")


if __name__ == "__main__":
    main()
