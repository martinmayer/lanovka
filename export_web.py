"""Export dat pro 3D prohlížeč (web/): terén jako PNG výškovka, ortofota, geometrie lanovky.

Výšky v PNG: v = round((h - H_BASE) * 100), R = v >> 8, G = v & 255 (přesnost 1 cm).
Lokální soustava: x = E - E0, y = N - N0 (UTM 33N), z = nadmořská výška.
"""
import json

import numpy as np
import tifffile
from PIL import Image

import geo

WEB = geo.ROOT / "web"
DATA = WEB / "data"
H_BASE = 150.0
E0, N0 = geo.to_utm(geo.STATIONS[0][1], geo.STATIONS[0][2])


def height_png(z, path):
    v = np.clip(np.round((z - H_BASE) * 100), 0, 65535).astype(np.uint32)
    rgb = np.stack([(v >> 8) & 255, v & 255, np.zeros_like(v)], -1).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path, optimize=True)


def extent(name):
    e = json.loads((geo.DATA / f"{name}.json").read_text())["extent"]
    return dict(x0=e["xmin"] - E0, x1=e["xmax"] - E0, y0=e["ymin"] - N0, y1=e["ymax"] - N0)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    # blízký terén: povrch DMP 1G (budovy, stromy), 2 m
    height_png(geo.DMP.a, DATA / "near_h.png")
    ortho = Image.open(geo.DATA / "ortofoto_hi.jpg")
    k = 4096 / max(ortho.size)
    ortho.resize((round(ortho.width * k), round(ortho.height * k)), Image.LANCZOS) \
        .save(DATA / "near_ortho.jpg", quality=84, optimize=True, progressive=True)
    # vzdálené okolí: DMR 5G, 20 m, o 1,5 m níž (blízký terén má přednost)
    far = tifffile.imread(geo.DATA / "dmr5g_far.tif").astype(np.float64)[::2, ::2] - 1.5
    height_png(far, DATA / "far_h.png")
    Image.open(geo.DATA / "ortofoto_far.jpg").resize((2048, 1909), Image.LANCZOS) \
        .save(DATA / "far_ortho.jpg", quality=82, optimize=True, progressive=True)

    L = geo.total_length()
    s = np.arange(0, L, 2.0)
    s = np.append(s, L)
    e, n, az = geo.horizontal(s)
    scene = dict(
        origin=dict(e=E0, n=N0, epsg=32633),
        hBase=H_BASE,
        near=dict(extent("dmp1g"), w=geo.DMP.a.shape[1], h=geo.DMP.a.shape[0]),
        far=dict(extent("dmr5g_far"), w=far.shape[1], h=far.shape[0]),
        trackHalf=geo.TRACK_HALF,
        length=L,
        axis=dict(
            s=np.round(s, 2).tolist(),
            x=np.round(e - E0, 2).tolist(),
            y=np.round(n - N0, 2).tolist(),
            az=np.round(az, 3).tolist(),
            rope=np.round(geo.rope_height(s), 2).tolist(),
            ground=np.round(geo.DMR.sample(e, n), 2).tolist(),
            surface=np.round(geo.DMP.sample(e, n), 2).tolist(),
        ),
        supports=[dict(kind=p["kind"], name=p["name"], s=round(p["s"], 2), x=round(p["e"] - E0, 2),
                       y=round(p["n"] - N0, 2), ground=round(p["ground"], 2), rope=round(p["rope"], 2),
                       az=round(float(geo.horizontal(min(max(p["s"], 0.5), L - 0.5))[2]), 3))
                  for p in geo.supports()],
    )
    (DATA / "scene.json").write_text(json.dumps(scene, separators=(",", ":")))
    for f in sorted(DATA.iterdir()):
        print(f"{f.name:16s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
