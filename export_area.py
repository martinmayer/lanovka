"""Export rozšířeného okolí (~5,5 × 5 km) pro low-poly 3D scénu (web/data/).

- area_h.png: holý terén DMR 5G po 4 m
- surface.png: třída povrchu po 2 m (OSM + zeleň z ortofota + koruny z DMP)
- buildings.json: obrysy z OSM, výška (OSM / DMP − DMR), tvar střechy, typ pro barvu
- trees.json: stromy z korun (DMP − DMR), hustší u trasy, řidší dál
- trams.json: trasy tramvajových linek z OSM (relace route=tram), body po 4 m se zastávkami
"""
import json
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, median_filter

import geo
from export_style import CLASSES, C, ROAD_W, PATH, area_class, rings
from export_web import DATA, E0, N0, height_png

DMR = geo.Raster("dmr5g_area")
DMP = geo.Raster("dmp1g_area")
H, W = DMR.a.shape
OSM = geo.DATA

ARENA = dict(name="T ARENA", h=29.5, eave=23.0)  # 151 × 108 m, výška 29,5 m (arenabrno.cz)


def px(lon, lat):
    e, n = geo.to_utm(np.asarray(lon), np.asarray(lat))
    return (e - DMR.x0) / DMR.dx, (DMR.y1 - n) / DMR.dy


def load(name):
    return json.loads((OSM / name).read_text())["elements"]


def surface(land, roads, blds):
    img = Image.new("L", (W, H), C["zástavba"])
    d = ImageDraw.Draw(img)
    areas = []
    for el in land:
        t = el.get("tags", {})
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
    for el in land:
        t = el.get("tags", {})
        if t.get("waterway") in ("river", "canal", "stream") and el["type"] == "way":
            x, y = px([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
            d.line(list(zip(x, y)), fill=C["voda"], width=int((22 if t["waterway"] == "river" else 4) / DMR.dx))
    for a, cls, pts in areas:
        if cls == "voda":
            d.polygon(pts, fill=C["voda"])
    surf = np.array(img)
    # nezmapované plochy: zeleň z ortofota, les tam, kde jsou koruny
    rgb = np.asarray(Image.open(OSM / "ortofoto_area.jpg").convert("RGB").resize((W, H)), float)
    green = median_filter((2 * rgb[..., 1] - rgb[..., 0] - rgb[..., 2]) > 18, size=5)
    tall = median_filter((DMP.a - DMR.a) > 3.0, size=5)
    bm = Image.new("L", (W, H), 0)
    for el in blds:
        for r in rings(el):
            x, y = px([q[0] for q in r], [q[1] for q in r])
            ImageDraw.Draw(bm).polygon(list(zip(x, y)), fill=1)
    notbld = ~binary_dilation(np.array(bm).astype(bool), iterations=3)
    free = surf == C["zástavba"]
    surf[free & green] = C["tráva"]
    surf[free & tall & notbld] = C["les"]
    img = Image.fromarray(surf)
    d = ImageDraw.Draw(img)
    for el in roads:
        t = el.get("tags", {})
        hw, rw = t.get("highway"), t.get("railway")
        if el["type"] != "way" or t.get("tunnel") == "yes" or t.get("area") == "yes":
            continue
        x, y = px([p["lon"] for p in el["geometry"]], [p["lat"] for p in el["geometry"]])
        pts = list(zip(x, y))
        if rw in ("tram", "rail"):
            d.line(pts, fill=C["koleje"], width=max(1, round(3 / DMR.dx)))
        elif hw in ROAD_W:
            d.line(pts, fill=C["silnice"], width=max(1, round(ROAD_W[hw] / DMR.dx)), joint="curve")
        elif hw in PATH:
            d.line(pts, fill=C["cesta"], width=1)
    surf = np.array(img)
    # okolí T Areny už není staveniště, ale zpevněné náměstí
    ae, an = geo.to_utm(16.5722, 49.1902)
    yy, xx = np.mgrid[0:H, 0:W]
    near_arena = ((DMR.x0 + (xx + 0.5) * DMR.dx - ae) ** 2 + (DMR.y1 - (yy + 0.5) * DMR.dy - an) ** 2) < 260 ** 2
    surf[near_arena & (surf == C["staveniště"])] = C["zpevněné plochy"]
    Image.fromarray(surf).save(DATA / "surface.png", optimize=True)
    return surf, np.array(bm).astype(bool)


KIND = {  # typ budovy → skupina barev
    "house": 0, "detached": 0, "semidetached_house": 0, "terrace": 0, "bungalow": 0, "cabin": 0, "farm": 0,
    "residential": 1, "apartments": 1, "dormitory": 1, "hotel": 1,
    "industrial": 2, "warehouse": 2, "service": 2, "manufacture": 2, "hangar": 2,
    "garage": 4, "garages": 4, "shed": 4, "carport": 4,
}
ROOF = {"flat": "f", "gabled": "g", "hipped": "h", "half-hipped": "h", "pyramidal": "p", "dome": "d",
        "skillion": "f", "round": "d", "onion": "d"}


def num(v):
    try:
        return float(str(v).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return None


def buildings(blds):
    ndsm = DMP.a - DMR.a
    out = []
    for el in blds:
        t = el.get("tags", {})
        b = t.get("building")
        if b in ("roof", "no", "ruins") or t.get("location") == "underground":
            continue
        is_arena = t.get("name") == ARENA["name"]
        if b == "construction" and not (is_arena or t.get("name") or t.get("construction") in ("house", "building")):
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
            mk = np.array(m).astype(bool)
            if mk.sum() < 2:
                continue
            sl = (slice(y0, y1), slice(x0, x1))
            ground = DMR.a[sl][mk]
            h = num(t.get("height"))
            if h is None and num(t.get("building:levels")) is not None:
                h = 3.0 * num(t["building:levels"]) + 1.0 + 2.5 * (num(t.get("roof:levels")) or 0)
            dsm_h = float(np.percentile(ndsm[sl][mk], 75))
            if b == "construction":
                pass  # nová stavba: povrchový model ji ještě nezná
            elif h is None:
                h = dsm_h
            elif num(t.get("height")) is None:
                h = max(h, dsm_h)  # počet podlaží × 3 m podhodnocuje vysoká patra (kampus, školy, haly)
            roof = ROOF.get(t.get("roof:shape"), "")
            kind = KIND.get(b, 3)
            if is_arena:
                h, roof, kind = ARENA["h"], "a", 5
            h = float(np.clip(h or 3.0, 2.5, 90.0))
            e, n = geo.to_utm(lon, lat)
            ring = np.round(np.stack([e - E0, n - N0], 1), 1)
            if np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            if not roof:  # odhad: malé domy sedlová, jinak plochá
                roof = "g" if (kind == 0 or (kind in (1, 3) and h < 12 and mk.sum() * DMR.dx * DMR.dy < 260)) else "f"
            rec = dict(r=ring.ravel().tolist(), b=round(float(ground.min()) - 0.6, 1), h=round(h, 1), f=roof, k=kind)
            if is_arena:  # výška 29,5 m platí od náměstí (nejvyšší terén v půdorysu), ne od dna svahu k řece
                rec["g"] = round(float(ground.max()), 1)
            out.append(rec)
    return out


def trees(surf, bmask, lite_core):
    ndsm = DMP.a - DMR.a
    block = binary_dilation(bmask, iterations=2) | (surf == C["silnice"]) | (surf == C["voda"]) | (surf == C["koleje"])
    canopy = np.where(block, 0, ndsm)
    rng = np.random.default_rng(7)
    pts = []
    for step, keep in ((3.2, lite_core), (5.5, ~lite_core)):
        gy, gx = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step), indexing="ij")
        rr = np.clip(np.round(gy + rng.uniform(-1.4, 1.4, gy.shape)), 0, H - 1).astype(int).ravel()
        cc = np.clip(np.round(gx + rng.uniform(-1.4, 1.4, gx.shape)), 0, W - 1).astype(int).ravel()
        ok = (canopy[rr, cc] > 3.5) & keep[rr, cc]
        rr, cc = rr[ok], cc[ok]
        e = DMR.x0 + (cc + 0.5) * DMR.dx
        n = DMR.y1 - (rr + 0.5) * DMR.dy
        big = 1.0 if step < 4 else 1.35  # dál od trasy řidší, ale větší koruny
        pts.append(np.stack([np.round(e - E0, 1), np.round(n - N0, 1), np.round(DMR.a[rr, cc], 1),
                             np.round(np.minimum(canopy[rr, cc], 35) * big, 1)], 1))
    return np.vstack(pts)


def stitch(ways):
    """Seřazené členy relace → souvislá polyline (lon, lat)."""
    line = []
    for g in ways:
        if not line:
            line = g[:]
            continue
        if line[-1] == g[0]:
            line += g[1:]
        elif line[-1] == g[-1]:
            line += g[::-1][1:]
        elif line[0] == g[-1] and len(line) == len(ways[0]):
            line = g + line[1:]
        elif line[0] == g[0] and len(line) == len(ways[0]):
            line = g[::-1] + line[1:]
        else:
            d0 = math.dist(line[-1], g[0]); d1 = math.dist(line[-1], g[-1])
            line += g if d0 <= d1 else g[::-1]
    return line


def trams():
    rels = load("osm_tram.json")
    ext = dict(x0=DMR.x0 - E0, x1=DMR.x0 + W * DMR.dx - E0, y0=DMR.y1 - H * DMR.dy - N0, y1=DMR.y1 - N0)
    out = []
    for r in rels:
        ways = [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in r["members"]
                if m["type"] == "way" and m.get("role", "") in ("", "forward", "backward") and m.get("geometry")]
        stops = [(m["lon"], m["lat"]) for m in r["members"] if m["type"] == "node" and m.get("role", "").startswith("stop")
                 and "lon" in m]
        if not ways:
            continue
        line = stitch(ways)
        e, n = geo.to_utm(np.array([p[0] for p in line]), np.array([p[1] for p in line]))
        x, y = e - E0, n - N0
        seg = np.hypot(np.diff(x), np.diff(y))
        s = np.concatenate([[0], np.cumsum(seg)])
        ss = np.arange(0, s[-1], 4.0)
        xs, ys = np.interp(ss, s, x), np.interp(ss, s, y)
        inside = (xs > ext["x0"] + 20) & (xs < ext["x1"] - 20) & (ys > ext["y0"] + 20) & (ys < ext["y1"] - 20)
        if inside.sum() < 50:
            continue
        # nejdelší souvislý úsek uvnitř území
        idx = np.flatnonzero(inside)
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        xs, ys = xs[run], ys[run]
        zs = DMR.sample(xs + E0, ys + N0)
        se, sn = (geo.to_utm(np.array([p[0] for p in stops]), np.array([p[1] for p in stops])) if stops else ([], []))
        stop_s = []
        for a, b in zip(np.asarray(se) - E0, np.asarray(sn) - N0):
            dd = np.hypot(xs - a, ys - b)
            j = int(np.argmin(dd))
            if dd[j] < 25:
                stop_s.append(j * 4.0)
        out.append(dict(ref=r["tags"].get("ref"), name=r["tags"].get("name"),
                        p=np.round(np.stack([xs, ys, zs], 1), 1).ravel().tolist(), stops=sorted(set(stop_s))))
    return out


def main():
    blds = load("osm_a_bld.json")
    land = load("osm_a_land.json")
    roads = load("osm_a_roads.json")
    surf, bmask = surface(land, roads, blds)
    height_png(DMR.a[::2, ::2], DATA / "area_h.png")
    B = buildings(blds)
    (DATA / "buildings.json").write_text(json.dumps(B, separators=(",", ":")))
    # jádro = do 900 m od osy lanovky
    L = geo.total_length()
    s = np.arange(0, L, 20.0)
    ax, ay, _ = geo.horizontal(s)
    yy, xx = np.mgrid[0:H, 0:W]
    ex, ny = DMR.x0 + (xx + 0.5) * DMR.dx, DMR.y1 - (yy + 0.5) * DMR.dy
    dmin = np.full((H, W), np.inf)
    for a, b in zip(ax, ay):
        dmin = np.minimum(dmin, (ex - a) ** 2 + (ny - b) ** 2)
    core = dmin < 900 ** 2
    T = trees(surf, bmask, core)
    (DATA / "trees.json").write_text(json.dumps(T.ravel().tolist(), separators=(",", ":")))
    TR = trams()
    (DATA / "trams.json").write_text(json.dumps(TR, separators=(",", ":"), ensure_ascii=False))

    scene = json.loads((DATA / "scene.json").read_text())
    ext = json.loads((OSM / "dmr5g_area.json").read_text())["extent"]
    hh, ww = DMR.a[::2, ::2].shape
    scene["area"] = dict(x0=ext["xmin"] - E0, x1=ext["xmin"] - E0 + ww * 2 * DMR.dx, y1=ext["ymax"] - N0,
                         y0=ext["ymax"] - N0 - hh * 2 * DMR.dy, w=ww, h=hh)
    scene["surf"] = dict(w=W, h=H)
    scene["classes"] = CLASSES
    (DATA / "scene.json").write_text(json.dumps(scene, separators=(",", ":"), ensure_ascii=False))
    print(f"budov {len(B)}, stromů {len(T)}, tramvajových tras {len(TR)}: "
          + ", ".join(f"{t['ref']}({len(t['p']) // 3 * 4 / 1000:.1f} km, {len(t['stops'])} zast.)" for t in TR))
    for f in ["surface.png", "area_h.png", "buildings.json", "trees.json", "trams.json"]:
        print(f"{f:16s} {(DATA / f).stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
