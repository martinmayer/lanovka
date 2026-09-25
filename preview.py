"""Kontrolní náhled: trasa a dráha kamery nad ortofotem ČÚZK + výškový profil."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import geo

OUT = geo.ROOT / "out"
ext = json.loads((geo.DATA / "dmr5g.json").read_text())["extent"]
img = Image.open(geo.DATA / "ortofoto.jpg")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(18, 11), gridspec_kw=dict(width_ratios=[1, 1.35]))
ax.imshow(img, extent=[ext["xmin"], ext["xmax"], ext["ymin"], ext["ymax"]])
L = geo.total_length()
s = np.arange(0, L, 2.0)
e, n, _ = geo.horizontal(s)
ax.plot(e, n, "-", color="#ff3030", lw=2.5, label="lano")
for name, col in [("01_dron", "#30c0ff"), ("02_kabina", "#ffe030")]:
    fr = json.loads((OUT / f"kamera_{name}.json").read_text())
    ce, cn = geo.to_wgs.__self__ if False else (None, None)
    ce, cn = geo.to_utm(np.array([f["lon"] for f in fr]), np.array([f["lat"] for f in fr]))
    ax.plot(ce, cn, "--", color=col, lw=1.4, label=f"kamera {name}")
    te, tn = geo.to_utm(np.array([f["target_lon"] for f in fr]), np.array([f["target_lat"] for f in fr]))
    for i in range(0, len(fr), 90):
        ax.plot([ce[i], te[i]], [cn[i], tn[i]], "-", color=col, lw=0.5, alpha=0.6)
for p in geo.supports():
    ax.plot(p["e"], p["n"], "s" if p["kind"] == "station" else "^", ms=9,
            color="white" if p["kind"] == "station" else "orange", mec="k")
    ax.annotate(p["name"], (p["e"], p["n"]), xytext=(8, 4), textcoords="offset points",
                color="white", fontsize=10, weight="bold")
ax.set_xlim(ext["xmin"] + 250, ext["xmax"] - 150)
ax.set_ylim(ext["ymin"] + 150, ext["ymax"] - 250)
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Trasa (přibližná) nad ortofotem ČÚZK")
ax.set_xticks([]); ax.set_yticks([])

s = np.arange(0, L, 2.0)
e, n, _ = geo.horizontal(s)
bx.fill_between(s, 180, geo.DMP.sample(e, n), color="#6a9f4a", alpha=0.6, label="povrch (stromy, budovy) DMP 1G")
bx.fill_between(s, 180, geo.DMR.sample(e, n), color="#8a6d4a", label="terén DMR 5G")
bx.plot(s, geo.rope_height(s), "k-", lw=1.5, label="dopravní lano")
bx.plot(s, geo.rope_height(s) - 4.2, "--", color="#d0a000", lw=1, label="oči v kabině")
for p in geo.supports():
    bx.plot([p["s"], p["s"]], [p["ground"], p["rope"]], "-", color="#555", lw=4 if p["kind"] == "pylon" else 8)
    bx.annotate(p["name"], (p["s"], p["rope"] + 3), ha="center", fontsize=9)
bx.set_ylim(195, 330)
bx.set_xlabel("staničení [m]"); bx.set_ylabel("m n. m.")
bx.legend(loc="upper left", fontsize=9)
bx.set_title("Podélný profil")
bx.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "nahled_trasy.png", dpi=90)
print(OUT / "nahled_trasy.png")
