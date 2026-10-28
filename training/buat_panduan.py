"""Buat halaman panduan gerakan: contoh video tiap kata dari dataset WL-BISINDO.

Video dataset (HEVC 1080p) diubah jadi MP4 H.264 kecil supaya bisa diputar di browser apa pun.
Hasil: panduan/index.html + panduan/video/*.mp4. Butuh ffmpeg di PATH.

    .venv/Scripts/python.exe training/buat_panduan.py
"""
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import LABELS  # noqa: E402

SRC = Path("new_dataset")
OUT = Path("panduan")
# Dua peraga berbeda per kata (keduanya punya ke-32 kata), supaya terlihat variasinya.
EXAMPLES = [(1, 1), (3, 1)]  # (signer, sample)


def convert(src, dst):
    if dst.exists():
        return
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-an",
         "-vf", "crop=ih:ih,scale=360:360",  # potong persegi di tengah: peraga lebih besar "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        check=True,
    )


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg tidak ditemukan di PATH.")
    (OUT / "video").mkdir(parents=True, exist_ok=True)
    words = []
    for label, word in enumerate(LABELS):
        clips = []
        for signer, sample in EXAMPLES:
            name = f"signer{signer}_label{label}_sample{sample}"
            convert(SRC / f"{name}.mp4", OUT / "video" / f"{name}.mp4")
            clips.append(f"video/{name}.mp4")
        words.append({"no": label + 1, "word": word, "clips": clips})
        print(f"{label + 1:2}/{len(LABELS)} {word}", flush=True)

    page = TEMPLATE.replace("__DATA__", json.dumps(words, ensure_ascii=False)).replace("__COUNT__", html.escape(str(len(words))))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Selesai: {OUT / 'index.html'}")


TEMPLATE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Panduan Isyarat SignLink</title>
<style>
  :root { --bg:#111318; --card:#1b1e26; --line:#2c303c; --text:#eceef2; --muted:#8b91a0; --accent:#50c88c; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 "Segoe UI", system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:12px 20px; align-items:center;
    padding:18px 28px; background:rgba(17,19,24,.92); backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
  h1 { margin:0; font-size:22px; }
  header p { margin:0; color:var(--muted); font-size:13px; flex-basis:100%; }
  input[type=search] { flex:1; min-width:200px; max-width:320px; padding:9px 14px; border-radius:10px;
    border:1px solid var(--line); background:var(--card); color:var(--text); font:inherit; }
  label { display:flex; gap:8px; align-items:center; color:var(--muted); font-size:14px; cursor:pointer; }
  main { display:grid; grid-template-columns:repeat(auto-fill, minmax(340px, 1fr)); gap:18px; padding:24px 28px 40px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:16px; overflow:hidden; }
  .card h2 { display:flex; align-items:baseline; gap:10px; margin:0; padding:12px 16px; font-size:19px; }
  .card h2 span { color:var(--muted); font-size:13px; font-weight:400; }
  .clips { display:grid; grid-template-columns:1fr 1fr; gap:2px; background:#000; }
  video { width:100%; aspect-ratio:1/1; object-fit:cover; display:block; background:#000; }
  body.mirror video { transform:scaleX(-1); }
  .empty { grid-column:1/-1; color:var(--muted); text-align:center; padding:40px; }
  @media (max-width:420px) { header, main { padding-left:16px; padding-right:16px; } main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>Panduan Isyarat</h1>
  <input type="search" id="q" placeholder="Cari kata…" autocomplete="off">
  <label><input type="checkbox" id="mirror" checked> Mode cermin</label>
  <p>__COUNT__ kata WL-BISINDO, dua peraga per kata. Mode cermin membalik video seperti cermin, jadi tiru saja gerakannya dengan tangan yang sama di layar.
     Video: dataset WL-BISINDO (CC BY-NC 4.0).</p>
</header>
<main id="grid"></main>
<script>
const words = __DATA__;
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const mirror = document.getElementById('mirror');

// Video hanya diputar saat terlihat di layar, supaya 64 video tidak jalan bersamaan.
const io = new IntersectionObserver((entries) => entries.forEach((e) => (e.isIntersecting ? e.target.play().catch(() => {}) : e.target.pause())));

for (const w of words) {
  const card = document.createElement('article');
  card.className = 'card';
  card.dataset.word = w.word.toLowerCase();
  const h = document.createElement('h2');
  h.textContent = w.word;
  const no = document.createElement('span');
  no.textContent = '#' + w.no;
  h.append(no);
  const clips = document.createElement('div');
  clips.className = 'clips';
  for (const src of w.clips) {
    const v = document.createElement('video');
    Object.assign(v, { src, muted: true, loop: true, playsInline: true, preload: 'metadata' });
    clips.append(v);
    io.observe(v);
  }
  card.append(h, clips);
  grid.append(card);
}

const empty = document.createElement('p');
empty.className = 'empty';
empty.textContent = 'Kata tidak ada di daftar.';
q.addEventListener('input', () => {
  const s = q.value.trim().toLowerCase();
  let shown = 0;
  for (const c of grid.querySelectorAll('.card')) {
    const hit = c.dataset.word.includes(s);
    c.hidden = !hit;
    shown += hit;
  }
  if (!shown) grid.append(empty); else empty.remove();
});
const setMirror = () => document.body.classList.toggle('mirror', mirror.checked);
mirror.addEventListener('change', setMirror);
setMirror();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
