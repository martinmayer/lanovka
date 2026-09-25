"""Složení finálního videa: záběry z Google Earth Studia + dokreslená lanovka + popisky.

Vstup: vyrenderované projekty z GES rozbalené do
  ges/01_dron/    (footage/*.jpeg + <projekt>.json se 3D tracking daty)
  ges/02_kabina/
Výstup: out/lanovka_ges.mp4

Kamera se bere z tracking JSON (poloha + rotace + FOV každého snímku), takže dokreslená
lanovka sedí přesně na to, co GES vyrenderovalo. Konvence os, výškový posun (MSL vs. elipsoid)
a odklon UTM sítě od zeměpisného severu se kalibrují automaticky proti out/kamera_*.json.

Použití: python3 composite.py [--parts 01_dron 02_kabina] [--from N --to M] [--png složka]
         python3 composite.py --selftest   (ověří výpočet kamery na syntetických datech)
"""
import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import geo
import render as R

GES = Path(os.environ.get("GES_DIR", geo.ROOT / "ges"))
OUT = geo.ROOT / "out"
FPS = R.FPS


def ecef2enu(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])


def enu2grid(lat, lon):
    """Rotace vektorů z ENU do lokální UTM sítě (odklon poledníku)."""
    e0, n0 = geo.to_utm(lon, lat)
    e1, n1 = geo.to_utm(lon, lat + 1e-4)
    gam = math.atan2(e1 - e0, n1 - n0)  # směr zeměpisného severu v síti
    c, s = math.cos(gam), math.sin(gam)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def find_tracking(folder):
    js = [p for p in Path(folder).rglob("*.json") if not p.name.startswith("transforms")]
    if len(js) != 1:
        raise SystemExit(f"{folder}: čekám právě jeden tracking JSON, nalezeno {len(js)}")
    return js[0]


def find_footage(folder):
    imgs = [p for p in Path(folder).rglob("*") if p.suffix.lower() in (".jpeg", ".jpg", ".png")]
    by_idx = {}
    for p in imgs:
        m = re.search(r"(\d+)$", p.stem)
        if m:
            by_idx[int(m.group(1))] = p
    return by_idx


def camera_axes(fr):
    """Osy kamery (ve 3 variantách znamének) v ENU a poloha z jednoho cameraFrame."""
    c = fr["coordinate"]
    lat, lon, alt = c["latitude"], c["longitude"], c["altitude"]
    rc = Rotation.from_euler("XYZ", [fr["rotation"]["x"], fr["rotation"]["y"], fr["rotation"]["z"]],
                             degrees=True).as_matrix()
    m = enu2grid(lat, lon) @ ecef2enu(lat, lon) @ rc
    return lat, lon, alt, m


class Calib:
    """Zjistí, která osa rotační matice je „dopředu“ a „nahoru“, a výškový posun."""

    def __init__(self, frames, intended):
        best = None
        n = min(len(frames), len(intended))
        sample = range(0, n, max(1, n // 60))
        for fwd_axis in range(3):
            for fs in (1, -1):
                for up_axis in range(3):
                    if up_axis == fwd_axis:
                        continue
                    for us in (1, -1):
                        score = 0.0
                        for i in sample:
                            _, _, _, m = camera_axes(frames[i])
                            want = intended_dir(intended[i])
                            f = fs * m[:, fwd_axis]
                            u = us * m[:, up_axis]
                            score += f @ want + 0.3 * u[2]
                        if best is None or score > best[0]:
                            best = (score, fwd_axis, fs, up_axis, us)
        _, self.fa, self.fs, self.ua, self.us = best
        dz = [frames[i]["coordinate"]["altitude"] - intended[i]["alt"] for i in sample]
        self.dz = float(np.median(dz))
        if abs(self.dz) < 2.0:
            self.dz = 0.0
        ang = [math.degrees(math.acos(np.clip(self.forward(frames[i]) @ intended_dir(intended[i]), -1, 1)))
               for i in sample]
        print(f"kalibrace: dopředu = {'+' if self.fs > 0 else '-'}osa{self.fa}, nahoru = "
              f"{'+' if self.us > 0 else '-'}osa{self.ua}, výškový posun {self.dz:+.1f} m, "
              f"odchylka směru od plánu: medián {np.median(ang):.2f}°, max {np.max(ang):.2f}°")

    def forward(self, fr):
        return self.fs * camera_axes(fr)[3][:, self.fa]

    def view(self, fr):
        lat, lon, alt, m = camera_axes(fr)
        e, n = geo.to_utm(lon, lat)
        eye = np.array([e - R.E0, n - R.N0, alt - self.dz])
        f = self.fs * m[:, self.fa]
        u = self.us * m[:, self.ua]
        return eye, R.look_at(eye, eye + f, up=u)


def intended_dir(c):
    e, n = geo.to_utm(c["lon"], c["lat"])
    te, tn = geo.to_utm(c["target_lon"], c["target_lat"])
    d = np.array([te - e, tn - n, c["target_alt"] - c["alt"]])
    return d / np.linalg.norm(d)


def run(parts, f0, f1, out, png, w=None, h=None):
    cams, tracks, foot = [], [], []
    for part in parts:
        intended = json.loads((OUT / f"kamera_{part}.json").read_text())
        tr = json.loads(find_tracking(GES / part).read_text())
        frames = tr["cameraFrames"]
        if len(frames) != len(intended):
            print(f"POZOR {part}: GES má {len(frames)} snímků, plán {len(intended)}")
        cal = Calib(frames, intended)
        ft = find_footage(GES / part)
        offset = 0 if 0 in ft else 1  # GES někdy čísluje od 1
        for i, fr in enumerate(frames[:len(intended)]):
            cams.append(dict(intended[i], part="dron" if part == "01_dron" else "kabina"))
            tracks.append((cal, fr))
            foot.append(ft.get(i + offset))
        w = w or tr.get("width", 1920)
        h = h or tr.get("height", 1080)
    n_drone = sum(1 for c in cams if c["part"] == "dron")
    rend = R.Renderer(w, h, 35.0)
    hud = R.Hud(w, h)
    fleet = R.Fleet()
    f1 = len(cams) if f1 is None else min(f1, len(cams))
    pipe = None
    if png:
        os.makedirs(png, exist_ok=True)
    else:
        pipe = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                 "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    t0 = time.time()
    for j, i in enumerate(range(f0, f1)):
        c = cams[i]
        cal, fr = tracks[i]
        eye, view = cal.view(fr)
        t_ride = (i - n_drone) / FPS
        cab = fleet.poses(t_ride, skip_own=c["part"] == "kabina")
        ov = rend.frame(eye, None, cab, view=view, fov=fr.get("fovVertical", 35.0), overlay=True)
        bg = Image.open(foot[i]).convert("RGB").resize((w, h), Image.LANCZOS) if foot[i] else \
            Image.new("RGB", (w, h), (90, 90, 90))
        img = Image.alpha_composite(bg.convert("RGBA"), ov).convert("RGB")
        ce, cn = geo.to_utm(c["lon"], c["lat"])
        agl = eye[2] - float(geo.DMR.sample(eye[0] + R.E0, eye[1] + R.N0))
        img = hud.draw(img, i / FPS, (ce, cn), c["s"], c["part"], agl)
        if pipe:
            pipe.stdin.write(img.tobytes())
        else:
            img.save(os.path.join(png, f"c{i:05d}.jpg"), quality=92)
        if j % 50 == 0:
            print(f"snímek {i} ({j + 1}/{f1 - f0}), {(time.time() - t0) / (j + 1):.2f} s/snímek", flush=True)
    if pipe:
        pipe.stdin.close()
        pipe.wait()
    print("hotovo", png or out)


def selftest():
    """Syntetický tracking JSON z plánované kamery (konvence OpenCV v ECEF) + náhled jako „záběry“."""
    part = "01_dron"
    intended = json.loads((OUT / f"kamera_{part}.json").read_text())
    frames = []
    for c in intended:
        d = intended_dir(c)
        e, n = geo.to_utm(c["lon"], c["lat"])
        G = enu2grid(c["lat"], c["lon"])
        f = G.T @ d                                     # síť → ENU
        r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
        dn = np.cross(f, r)
        m_enu = np.stack([r, dn, f], 1)                 # OpenCV: x vpravo, y dolů, z dopředu
        m_ecef = ecef2enu(c["lat"], c["lon"]).T @ m_enu
        rx, ry, rz = Rotation.from_matrix(m_ecef).as_euler("XYZ", degrees=True)
        frames.append(dict(coordinate=dict(latitude=c["lat"], longitude=c["lon"], altitude=c["alt"] + 44.6),
                           rotation=dict(x=rx, y=ry, z=rz), fovVertical=42.0))
    cal = Calib(frames, intended)
    for i in (0, 700, 1300):
        eye, view = cal.view(frames[i])
        ref = R.look_at(np.array([*(np.array(geo.to_utm(intended[i]["lon"], intended[i]["lat"])) -
                                    [R.E0, R.N0]), intended[i]["alt"]]),
                        np.array([*(np.array(geo.to_utm(intended[i]["target_lon"], intended[i]["target_lat"])) -
                                    [R.E0, R.N0]), intended[i]["target_alt"]]))
        print(f"snímek {i}: max rozdíl matice pohledu {np.abs(view - ref).max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["01_dron", "02_kabina"])
    ap.add_argument("--from", dest="f0", type=int, default=0)
    ap.add_argument("--to", dest="f1", type=int, default=None)
    ap.add_argument("--out", default=str(OUT / "lanovka_ges.mp4"))
    ap.add_argument("--png", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.parts, a.f0, a.f1, a.out, a.png)
