"""Geometrie lanové dráhy Pisárky – Kampus: terén, stanice, podpěry, lano.

Souřadnice jsou PŘIBLIŽNÁ rekonstrukce z textových popisů (DPMB, ÚS Akademická–Kamenice)
a polohy staveb v OSM, ne projektová data. Výšky jsou nadmořské (Bpv ≈ MSL) z ČÚZK DMR 5G / DMP 1G.
"""
import json
import math
from pathlib import Path

import numpy as np
import tifffile
from pyproj import Transformer
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).parent
DATA = ROOT / "data"

_to_utm = Transformer.from_crs(4326, 32633, always_xy=True)
_to_wgs = Transformer.from_crs(32633, 4326, always_xy=True)


def to_utm(lon, lat):
    return _to_utm.transform(lon, lat)


def to_wgs(e, n):
    return _to_wgs.transform(e, n)


class Raster:
    def __init__(self, name):
        self.a = tifffile.imread(DATA / f"{name}.tif").astype(np.float64)
        ext = json.loads((DATA / f"{name}.json").read_text())["extent"]
        self.x0, self.y1 = ext["xmin"], ext["ymax"]
        h, w = self.a.shape
        self.dx = (ext["xmax"] - ext["xmin"]) / w
        self.dy = (ext["ymax"] - ext["ymin"]) / h

    def sample(self, e, n):
        e, n = np.asarray(e, float), np.asarray(n, float)
        col = (np.atleast_1d(e) - self.x0) / self.dx - 0.5
        row = (self.y1 - np.atleast_1d(n)) / self.dy - 0.5
        v = map_coordinates(self.a, [row.ravel(), col.ravel()], order=1, mode="nearest")
        return v.reshape(e.shape)


DMR = Raster("dmr5g")  # terén
DMP = Raster("dmp1g")  # povrch vč. budov a stromů

# (název, lon, lat, výška lana nad terénem ve stanici [m])
STATIONS = [
    ("Pisárky – Lipová", 16.574932, 49.191548, 10.0),
    ("Riviéra", 16.572532, 49.188048, 12.0),
    ("Kampus", 16.5672, 49.1804, 14.0),
    ("Kampus – terminál", 16.5671, 49.1768, 12.0),
]

# Traťové podpěry (3 dle návrhu A Plus): (staničení [m], výška [m]).
# Polohy jsou odhad: P1 portálová nad silnicí u přechodu před Riviérou, nohy v gabionech postavených 2024, P2 na horní hraně zalesněného svahu nad Svratkou
# (musí přenést lano přes les na planině bez průseku), P3 podél Netroufalek u křižovatky s Kamenicí.
PYLONS = [
    (368.0, 18.0),
    (1000.0, 46.0),
    (1560.0, 18.0),
]

SAG = 0.012  # průvěs lana uprostřed pole jako podíl délky pole
TRACK_HALF = 2.7  # polovina rozchodu větví lana [m]; nahoru jede kabina po pravé větvi


def _station_utm():
    return np.array([to_utm(lon, lat) for _, lon, lat, _ in STATIONS])


def supports():
    """Seznam opěrných bodů lana: dict(kind, name, s, e, n, ground, rope)."""
    st = _station_utm()
    seg_len = np.linalg.norm(np.diff(st, axis=0), axis=1)
    seg_start = np.concatenate([[0.0], np.cumsum(seg_len)])
    pts = []
    for i, (name, _, _, h) in enumerate(STATIONS):
        e, n = st[i]
        g = float(DMR.sample(e, n))
        pts.append(dict(kind="station", name=name, s=seg_start[i], e=e, n=n, ground=g, rope=g + h))
    for k, (s, h) in enumerate(PYLONS):
        e, n, _ = horizontal(s)
        e, n = float(e), float(n)
        g = float(DMR.sample(e, n))
        pts.append(dict(kind="pylon", name=f"P{k + 1}", s=s, e=e, n=n, ground=g, rope=g + h, height=h))
    return sorted(pts, key=lambda p: p["s"])


def horizontal(s):
    """Staničení s [m] → (E, N) v UTM a azimut osy [°]."""
    st = _station_utm()
    seg_len = np.linalg.norm(np.diff(st, axis=0), axis=1)
    seg_start = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.clip(np.asarray(s, float), 0, seg_start[-1])
    i = np.clip(np.searchsorted(seg_start, s, side="right") - 1, 0, len(seg_len) - 1)
    f = (s - seg_start[i]) / seg_len[i]
    p = st[i] + f[..., None] * (st[i + 1] - st[i])
    d = st[i + 1] - st[i]
    az = np.degrees(np.arctan2(d[..., 0], d[..., 1])) % 360
    return p[..., 0], p[..., 1], az


def track_xy(s, side):
    """Vodorovná poloha větve lana: side = +1 pravá (nahoru), -1 levá (dolů), 0 osa."""
    e, n, az = horizontal(s)
    r = np.radians(az + 90)
    return e + side * TRACK_HALF * np.sin(r), n + side * TRACK_HALF * np.cos(r)


def rope_height(s):
    """Nadmořská výška lana ve staničení s (parabolický průvěs mezi opěrami)."""
    sup = supports()
    ss = np.array([p["s"] for p in sup])
    zz = np.array([p["rope"] for p in sup])
    s = np.clip(np.asarray(s, float), ss[0], ss[-1])
    i = np.clip(np.searchsorted(ss, s, side="right") - 1, 0, len(ss) - 2)
    L = ss[i + 1] - ss[i]
    f = (s - ss[i]) / L
    chord = zz[i] + f * (zz[i + 1] - zz[i])
    # ve stanicích je lano vedeno vodorovně na kladkách, proto průvěs jen v polích
    return chord - 4 * SAG * L * f * (1 - f)


def total_length():
    return supports()[-1]["s"]


if __name__ == "__main__":
    sup = supports()
    for p in sup:
        print(f"{p['kind']:7s} {p['name']:18s} s={p['s']:7.1f} terén={p['ground']:6.1f} lano={p['rope']:6.1f}")
    L = total_length()
    s = np.arange(0, L, 2.0)
    e, n, _ = horizontal(s)
    z = rope_height(s)
    g, surf = DMR.sample(e, n), DMP.sample(e, n)
    clr = z - surf
    j = np.argmin(clr)
    grad = np.abs(np.diff(z) / np.diff(s))
    print(f"délka vodorovně {L:.0f} m, převýšení {sup[-1]['rope'] - sup[0]['rope']:.1f} m (terén {sup[-1]['ground'] - sup[0]['ground']:.1f} m)")
    print(f"min. odstup lana od povrchu (stromy/budovy) {clr[j]:.1f} m v s={s[j]:.0f}")
    print(f"max. sklon lana {100 * grad.max():.1f} %, max výška nad terénem {np.max(z - g):.1f} m")
