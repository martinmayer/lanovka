"""Export dat pro stylizovanou 3D scénu (web/data/): třídy povrchu, budovy, stromy.

- surface.png: třída povrchu na mřížce blízkého terénu (2 m), 8bit index (viz CLASSES)
- near_dmr_h.png: holý terén DMR 5G (budovy a stromy jsou samostatné objekty)
- buildings.json: obrysy z OSM, výška z OSM nebo z rozdílu DMP − DMR uvnitř obrysu
- trees.json: stromy tam, kde povrch DMP převyšuje terén o víc než 3,5 m mimo budovy
- far_col.png: rozmazané barvy okolí z ortofota (jen tón vzdálené krajiny)
"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import binary_dilation, maximum_filter

import geo
from export_web import DATA, E0, N0, height_png

CLASSES = ["zástavba", "tráva", "les", "zahrady a pole", "voda", "silnice", "cesta", "koleje",
           "zpevněné plochy", "staveniště", "hřiště"]
C = {n: i for i, n in enumerate(CLASSES)}

RAST = geo.DMP  # stejná mřížka jako DMP / DMR
H, W = RAST.a.shape


def px(lon, lat):
    e, n = geo.to_utm(np.asarray(lon), np.asarray(lat))
    return (e - RAST.x0) / RAST.dx, (RAST.y1 - n) / RAST.dy


def rings(el):
    """Vnější prstence prvku (way nebo multipolygon) jako seznamy (lon, lat)."""
    if el["type"] == "way":
        g = el.get("geometry") or []
        return [[(p["lon"], p["lat"]) for p in g]] if len(g) >= 3 else []
    segs = [[(p["lon"], p["lat"]) for p in m.get("geometry") or []]
            for m in el.get("members", []) if m.get("role", "outer") in ("outer", "") and m.get("geometry")]
    out = []
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, s in enumerate(segs):
                if s[0] == ring[-1]:
                    ring += s[1:]
                elif s[-1] == ring[-1]:
                    ring += s[::-1][1:]
                elif s[-1] == ring[0]:
                    ring = s + ring[1:]
                elif s[0] == ring[0]:
                    ring = s[::-1] + ring[1:]
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if len(ring) >= 4:
            out.append(ring)
    return out


def area_class(t):
    lu, nat, lei = t.get("landuse"), t.get("natural"), t.get("leisure")
    if nat in ("water",) or lu in ("reservoir", "basin") or lei in ("swimming_pool",):
        return "voda"
    if nat in ("wood", "scrub") or lu in ("forest",):
        return "les"
    if lu in ("grass", "meadow", "recreation_ground", "village_green", "cemetery", "flowerbed") or \
            nat in ("grassland", "heath") or lei in ("park", "garden", "playground"):
        return "tráva"
    if lu in ("farmland", "allotments", "orchard", "vineyard", "plant_nursery"):
        return "zahrady a pole"
    if lu in ("construction", "brownfield", "greenfield", "landfill") or nat == "sand":
        return "staveniště"
    if lei in ("pitch", "track", "sports_centre"):
        return "hřiště"
    if t.get("amenity") == "parking" or lu in ("industrial", "railway", "garages", "commercial", "retail"):
        return "zpevněné plochy"
    if lei == "water_park":
        return "tráva"
    return None


ROAD_W = {"motorway": 22, "trunk": 16, "primary": 12, "secondary": 10, "tertiary": 8, "unclassified": 6,
          "residential": 6, "living_street": 5, "service": 4, "pedestrian": 4, "track": 3,
          "motorway_link": 8, "trunk_link": 8, "primary_link": 7, "secondary_link": 7, "tertiary_link": 6}
PATH = {"footway", "path", "cycleway", "steps", "bridleway"}


def main():
    osm = json.loads((geo.DATA / "osm_near.json").read_text())["elements"]
    img = Image.new("L", (W, H), C["zástavba"])
    d = ImageDraw.Draw(img)
    # pořadí: velké plochy, pak menší (zjednodušeně podle plochy), voda a komunikace navrch
    areas = []
    for el in osm:
        t = el.get("tags", {})
        if "building" in t or "highway" in t:
            continue
        cls = area_class(t)
        if not cls:
            continue
        for r in rings(el):
            x, y = px([p[0] for p in r], [p[1] for p in r])
            a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append((a, cls, list(zip(x, y))))
    for a, cls, pts in sorted(areas, key=lambda q: -q[0]):
        if cls != "voda":
            d.polygon(pts, fill=C[cls])
    for el in osm:
        t = el.get("tags", {})
        if t.get("waterway") in ("river", "canal", "stream") and el["type"] == "way":
            x, y = px([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
            d.line(list(zip(x, y)), fill=C["voda"], width=int((22 if t["waterway"] == "river" else 4) / RAST.dx))
    for a, cls, pts in areas:
        if cls == "voda":
            d.polygon(pts, fill=C["voda"])
    for el in osm:
        t = el.get("tags", {})
        hw, rw = t.get("highway"), t.get("railway")
        if el["type"] != "way" or not (hw or rw) or t.get("tunnel") == "yes" or t.get("area") == "yes":
            continue
        x, y = px([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
        pts = list(zip(x, y))
        if rw in ("tram", "rail"):
            d.line(pts, fill=C["koleje"], width=max(1, round(3 / RAST.dx)))
        elif hw in ROAD_W:
            d.line(pts, fill=C["silnice"], width=max(1, round(ROAD_W[hw] / RAST.dx)), joint="curve")
        elif hw in PATH:
            d.line(pts, fill=C["cesta"], width=1)
    # nezmapované plochy: zeleň podle ortofota, les tam, kde jsou i koruny stromů (DMP − DMR)
    from scipy.ndimage import median_filter
    rgb = np.asarray(Image.open(geo.DATA / "ortofoto.jpg").convert("RGB").resize((W, H)), float)
    exg = 2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]
    green = median_filter(exg > 18, size=5)
    tall = median_filter((geo.DMP.a - geo.DMR.a) > 3.0, size=5)
    surf = np.array(img)
    free = surf == C["zástavba"]
    surf[free & green] = C["tráva"]
    bm = Image.new("L", (W, H), 0)
    for el in osm:
        if "building" in el.get("tags", {}):
            for r in rings(el):
                x, y = px([q[0] for q in r], [q[1] for q in r])
                ImageDraw.Draw(bm).polygon(list(zip(x, y)), fill=1)
    notbld = ~binary_dilation(np.array(bm).astype(bool), iterations=3)
    surf[free & tall & notbld] = C["les"]
    img = Image.fromarray(surf)
    img.save(DATA / "surface.png", optimize=True)

    # holý terén
    height_png(geo.DMR.a, DATA / "near_dmr_h.png")

    # budovy
    ndsm = geo.DMP.a - geo.DMR.a
    bmask = Image.new("L", (W, H), 0)
    bd = ImageDraw.Draw(bmask)
    blds = []
    for el in osm:
        t = el.get("tags", {})
        if "building" not in t or t["building"] in ("roof", "no", "construction") or t.get("location") == "underground":
            continue
        for r in rings(el):
            lon, lat = np.array([p[0] for p in r]), np.array([p[1] for p in r])
            x, y = px(lon, lat)
            x0, y0 = max(int(x.min()), 0), max(int(y.min()), 0)
            x1, y1 = min(int(x.max()) + 2, W), min(int(y.max()) + 2, H)
            if x1 <= x0 or y1 <= y0:
                continue
            m = Image.new("L", (x1 - x0, y1 - y0), 0)
            ImageDraw.Draw(m).polygon(list(zip(x - x0, y - y0)), fill=1)
            bd.polygon(list(zip(x, y)), fill=1)
            sl = (slice(y0, y1), slice(x0, x1))
            mk = np.array(m).astype(bool)
            if mk.sum() < 2:
                continue
            ground = geo.DMR.a[sl][mk]
            if "height" in t:
                try:
                    h = float(t["height"].split()[0].replace(",", "."))
                except ValueError:
                    h = None
            elif "building:levels" in t:
                try:
                    h = 3.2 * float(t["building:levels"]) + 1.0
                except ValueError:
                    h = None
            else:
                h = None
            if h is None:
                h = float(np.percentile(ndsm[sl][mk], 75))
            h = float(np.clip(h, 3.0, 80.0))
            e, n = geo.to_utm(lon, lat)
            ring = np.round(np.stack([e - E0, n - N0], 1), 1)
            if np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            blds.append(dict(r=ring.ravel().tolist(), b=round(float(ground.min()) - 0.5, 1), h=round(h, 1),
                             k=t["building"]))
    (DATA / "buildings.json").write_text(json.dumps(blds, separators=(",", ":"), ensure_ascii=False))

    # stromy: vrcholy korun v nDSM mimo budovy a silnice
    surf = np.array(img)
    block = binary_dilation(np.array(bmask).astype(bool), iterations=2) | (surf == C["silnice"]) | (surf == C["voda"])
    canopy = np.where(block, 0, ndsm)
    rng = np.random.default_rng(7)
    step = 3.2  # px ≈ 6,5 m
    gy, gx = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing="ij")
    rr = np.clip(np.round(gy + rng.uniform(-1.4, 1.4, gy.shape)), 0, H - 1).astype(int).ravel()
    cc = np.clip(np.round(gx + rng.uniform(-1.4, 1.4, gx.shape)), 0, W - 1).astype(int).ravel()
    ok = canopy[rr, cc] > 3.5
    rr, cc = rr[ok], cc[ok]
    e = RAST.x0 + (cc + 0.5) * RAST.dx
    n = RAST.y1 - (rr + 0.5) * RAST.dy
    trees = np.stack([np.round(e - E0, 1), np.round(n - N0, 1), np.round(geo.DMR.a[rr, cc], 1),
                      np.round(np.minimum(canopy[rr, cc], 35), 1)], 1)
    (DATA / "trees.json").write_text(json.dumps(trees.ravel().tolist(), separators=(",", ":")))

    far = Image.open(geo.DATA / "ortofoto_far.jpg").convert("RGB").resize((330, 308), Image.LANCZOS)
    far.filter(ImageFilter.GaussianBlur(2)).save(DATA / "far_col.png", optimize=True)

    scene = json.loads((DATA / "scene.json").read_text())
    scene["classes"] = CLASSES
    (DATA / "scene.json").write_text(json.dumps(scene, separators=(",", ":"), ensure_ascii=False))
    print(f"budov {len(blds)}, stromů {len(trees)}")
    for f in ["surface.png", "near_dmr_h.png", "buildings.json", "trees.json", "far_col.png"]:
        print(f"{f:16s} {(DATA / f).stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
