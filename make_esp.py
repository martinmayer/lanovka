"""Generuje projekty Google Earth Studio (.esp) pro video jízdy lanovkou Pisárky – Kampus.

Výstup (složka out/):
  01_dron.esp        úvodní přelet dronem od kampusu nad trasou k nástupní stanici Pisárky – Lipová
  02_kabina.esp      zrychlená jízda z kabiny (4×), průjezd mezistanicemi, závěrečné vystoupání
  lanovka_trasa.kml  lano, stanice a podpěry pro kontrolu v GES (Overlays → KML)
  kamera_*.json      zamýšlená dráha kamery po snímcích (pro skládání a kontrolu)

Kamera se řídí polohou + cílem pohledu (cameraTargetEffect), nikoli úhly.
Formát .esp podle https://github.com/Anonmous765/city_annotation (ověřeno proti projektům z GES):
hodnoty normalizované do [0, 1], výška absolutní nad mořem.
"""
import json
import math
from pathlib import Path

import numpy as np

import geo

OUT = geo.ROOT / "out"
FPS = 30
WIDTH, HEIGHT = 1920, 1080
KEY_EVERY = 3  # klíčový snímek každé 3 snímky (10 Hz)

SPEEDUP = 4.0
V_LINE = 5.0 * SPEEDUP      # rychlost na trati ve videu [m/s]
V_STATION = 6.0             # průjezd stanicí ve videu [m/s]
STATION_ZONE = 12.0         # polovina délky nástupiště [m]
RAMP = 45.0                 # délka rozjezdu / brzdění [m]
EYE_BELOW_ROPE = 4.0        # oči cestujícího pod dopravním lanem [m]
LOOK_AHEAD = 140.0          # cíl pohledu před kabinou [m]
LOOK_DOWN = 10.0            # cíl pohledu je o tolik níž než oči [m]

ALT_MIN, ALT_MAX = -500, 65_117_481


def lon_rel(lon):
    return (lon + 180.0) / 360.0


def lat_rel(lat):
    return (lat + 90.0) / 180.0


def alt_rel(alt):
    return (alt - ALT_MIN) / (ALT_MAX - ALT_MIN)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- jízda kabiny

def cabin_speed(s):
    st = np.array([p["s"] for p in geo.supports() if p["kind"] == "station"])
    d = np.min(np.abs(np.asarray(s)[..., None] - st), axis=-1)
    return V_STATION + (V_LINE - V_STATION) * smoothstep((d - STATION_ZONE) / RAMP)


def cabin_s_of_t():
    """Staničení kabiny pro každý snímek jízdy (od stanice Lipová do terminálu)."""
    L = geo.total_length()
    s = np.linspace(0, L, 20001)
    v = cabin_speed(s)
    t = np.concatenate([[0], np.cumsum(np.diff(s) * 2 / (v[1:] + v[:-1]))])
    n = int(math.ceil(t[-1] * FPS)) + 1
    return np.interp(np.arange(n) / FPS, t, s)


def eye_alt(s):
    return geo.rope_height(s) - EYE_BELOW_ROPE


def smooth_along(s, fn, window=40.0):
    """Průměr fn přes okno staničení: vyhladí zlom osy v mezistanici Kampus."""
    offs = np.linspace(-window / 2, window / 2, 9)
    return np.mean([fn(s + o) for o in offs], axis=0)


def cabin_frames():
    s = cabin_s_of_t()
    L = geo.total_length()
    cam_e = smooth_along(s, lambda x: geo.track_xy(x, +1)[0], 24)
    cam_n = smooth_along(s, lambda x: geo.track_xy(x, +1)[1], 24)
    cam_z = eye_alt(s)
    # cíl pohledu: bod na trase před kabinou; u konce se prodlouží za terminál ve směru osy
    sa = s + LOOK_AHEAD
    tgt_e, tgt_n, az = geo.horizontal(np.minimum(sa, L))
    over = np.maximum(sa - L, 0)
    tgt_e = smooth_along(sa, lambda x: geo.horizontal(np.minimum(x, L))[0], 80) + over * np.sin(np.radians(az))
    tgt_n = smooth_along(sa, lambda x: geo.horizontal(np.minimum(x, L))[1], 80) + over * np.cos(np.radians(az))
    tgt_z = eye_alt(np.minimum(sa, L)) - LOOK_DOWN
    frames = dict(e=cam_e, n=cam_n, z=cam_z, te=tgt_e, tn=tgt_n, tz=tgt_z, s=s)

    # výdrž ve stanici a závěrečné vystoupání nad terminálem
    hold = int(1.5 * FPS)
    rise = int(8.0 * FPS)
    last = {k: v[-1] for k, v in frames.items()}
    _, _, az_end = geo.horizontal(L)
    back = np.array([-np.sin(np.radians(az_end)), -np.cos(np.radians(az_end))])
    u = smoothstep(np.arange(1, rise + 1) / rise)
    term = [p for p in geo.supports() if p["kind"] == "station"][-1]
    ext = dict(
        e=np.concatenate([np.full(hold, last["e"]), last["e"] + back[0] * 160 * u]),
        n=np.concatenate([np.full(hold, last["n"]), last["n"] + back[1] * 160 * u]),
        z=np.concatenate([np.full(hold, last["z"]), last["z"] + 140 * u]),
        te=np.concatenate([np.full(hold, last["te"]), last["te"] + (term["e"] - last["te"]) * u]),
        tn=np.concatenate([np.full(hold, last["tn"]), last["tn"] + (term["n"] - last["tn"]) * u]),
        tz=np.concatenate([np.full(hold, last["tz"]), last["tz"] + (term["ground"] - last["tz"]) * u]),
        s=np.full(hold + rise, L),
    )
    return {k: np.concatenate([frames[k], ext[k]]) for k in frames}


# ---------------------------------------------------------------- přelet dronem

def route_point(s, side=0.0, up=0.0, ground=False):
    """Bod u trasy: staničení s, boční odsazení (+ = vpravo ve směru jízdy), výška nad lanem/terénem."""
    L = geo.total_length()
    e, n, az = geo.horizontal(s)
    if s < 0 or s > L:  # prodloužení osy za koncové stanice
        d = s if s < 0 else s - L
        e, n = e + d * np.sin(np.radians(az)), n + d * np.cos(np.radians(az))
    r = np.radians(az + 90)
    e, n = float(e + side * np.sin(r)), float(n + side * np.cos(r))
    base = float(geo.DMR.sample(e, n)) if ground else float(geo.rope_height(s))
    return np.array([e, n, base + up])


def catmull_rom(times, pts, t):
    """Centripetální-ish Catmull-Rom (uniformní) přes klíčové body v čase."""
    times = np.asarray(times, float)
    P = np.asarray(pts, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for tt in t:
        i = int(np.clip(np.searchsorted(times, tt, side="right") - 1, 0, len(times) - 2))
        u = (tt - times[i]) / (times[i + 1] - times[i])
        u = smoothstep(u) * 0.35 + u * 0.65  # jemné zpomalení kolem klíčů
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    return np.array(out)


def drone_frames(cabin_start):
    L = geo.total_length()
    lip = [p for p in geo.supports() if p["kind"] == "station"][0]
    # vpravo ve směru jízdy (azimut ~204°) = severozápad; dron letí po levé (jihovýchodní) straně
    keys = [  # (čas [s], kamera, cíl pohledu)
        (0.0, route_point(L + 380, side=-180, up=330, ground=True), route_point(900, ground=True)),
        (9.0, route_point(L - 60, side=-230, up=250, ground=True), route_point(760, ground=True)),
        (20.0, route_point(1000, side=-240, up=190, ground=True), route_point(430, ground=True)),
        (30.0, route_point(430, side=-200, up=120, ground=True), route_point(80, ground=True)),
        (37.0, route_point(-70, side=-130, up=45, ground=True), route_point(0, up=-6)),
        (43.0, route_point(-90, side=0, up=6), route_point(40, up=-6)),
        (48.0, np.array([cabin_start["e"], cabin_start["n"], cabin_start["z"]]),
         np.array([cabin_start["te"], cabin_start["tn"], cabin_start["tz"]])),
    ]
    T = keys[-1][0]
    t = np.arange(int(T * FPS)) / FPS
    cam = catmull_rom([k[0] for k in keys], [k[1] for k in keys], t)
    tgt = catmull_rom([k[0] for k in keys], [k[2] for k in keys], t)
    return dict(e=cam[:, 0], n=cam[:, 1], z=cam[:, 2], te=tgt[:, 0], tn=tgt[:, 1], tz=tgt[:, 2],
                s=np.full(len(t), -1.0))


# ---------------------------------------------------------------- zápis .esp

def keyframes(values, n_frames, rel):
    idx = list(range(0, n_frames, KEY_EVERY))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    return [{"time": i / (n_frames - 1), "value": rel(float(values[i]))} for i in idx]


def build_esp(name, fr):
    n = len(fr["e"])
    lon, lat = geo.to_wgs(fr["e"], fr["n"])
    tlon, tlat = geo.to_wgs(fr["te"], fr["tn"])
    alt = fr["z"]
    tz = fr["tz"]

    def attr(typ, kfs, extra=None):
        v = {"relative": kfs[0]["value"]}
        if extra:
            v.update(extra)
        return {"type": typ, "value": v, "keyframes": kfs, "inTimeline": True}

    altx = {"maxValueRange": ALT_MAX, "minValueRange": ALT_MIN, "logarithmic": False}
    return {
        "modelVersion": 18,
        "settings": {"name": name, "frameRate": FPS, "dimensions": {"width": WIDTH, "height": HEIGHT},
                     "duration": n - 1, "timeFormat": "frames"},
        "scenes": [{
            "animationModel": {"roving": False, "logarithmic": False, "groupedPosition": True},
            "duration": n - 1,
            "attributes": [
                {"type": "cameraGroup", "inTimeline": True, "attributes": [
                    {"type": "cameraPositionGroup", "inTimeline": True, "attributes": [
                        {"type": "position", "inTimeline": True, "attributes": [
                            attr("longitude", keyframes(lon, n, lon_rel)),
                            attr("latitude", keyframes(lat, n, lat_rel)),
                            attr("altitude", keyframes(alt, n, alt_rel), altx),
                        ]}]},
                    {"type": "cameraTargetEffect", "inTimeline": True, "attributes": [
                        {"type": "enabled", "value": {"relative": 1}, "inTimeline": True},
                        {"type": "poi", "inTimeline": True, "attributes": [
                            attr("longitudePOI", keyframes(tlon, n, lon_rel)),
                            attr("latitudePOI", keyframes(tlat, n, lat_rel)),
                            attr("altitudePOI", keyframes(tz, n, alt_rel), altx),
                        ]},
                        {"type": "influence", "value": {"relative": 1}, "inTimeline": True},
                    ]},
                    {"type": "cameraRotationGroup", "inTimeline": True, "attributes": [
                        {"type": "rotationX", "value": {}, "inTimeline": True},
                        {"type": "rotationY", "value": {}, "inTimeline": True},
                        {"type": "rotationZ", "value": {}},
                    ]},
                    {"type": "cameraLensGroup", "attributes": [
                        {"type": "fov", "value": {}}, {"type": "exposure", "value": {}},
                        {"type": "aperture", "value": {}}, {"type": "minFocusLength", "value": {}},
                    ]},
                ]},
                {"type": "environmentGroup", "attributes": [
                    {"type": "sunGroup", "attributes": [{"type": "sunVisibility", "value": {}},
                                                        {"type": "worldTime", "value": {}}]},
                    {"type": "cloudGroup", "attributes": [{"type": "cloudVisibility", "value": {}},
                                                          {"type": "cloudopacity", "value": {}},
                                                          {"type": "cloudheight", "value": {}},
                                                          {"type": "clouddate", "value": {}}]},
                    {"type": "starsPlanetsGroup", "attributes": [{"type": "starsEnabled", "value": {}}]},
                    {"type": "seawaterGroup", "attributes": [{"type": "seawater", "value": {}},
                                                             {"type": "influence", "value": {"relative": 1}}]},
                    {"type": "buildingsEnabled", "value": {}},
                ]},
            ],
            "cameraExport": {"logarithmic": False, "modelVersion": 2},
        }],
        "playbackManager": {"range": {"start": 0, "end": n - 1}},
    }


def frames_json(fr):
    lon, lat = geo.to_wgs(fr["e"], fr["n"])
    tlon, tlat = geo.to_wgs(fr["te"], fr["tn"])
    return [dict(frame=i, lon=round(float(lon[i]), 7), lat=round(float(lat[i]), 7), alt=round(float(fr["z"][i]), 2),
                 target_lon=round(float(tlon[i]), 7), target_lat=round(float(tlat[i]), 7),
                 target_alt=round(float(fr["tz"][i]), 2), s=round(float(fr["s"][i]), 2))
            for i in range(len(lon))]


def write_kml(path):
    L = geo.total_length()
    s = np.arange(0, L + 1, 5.0)
    e, n, _ = geo.horizontal(s)
    lon, lat = geo.to_wgs(e, n)
    z = geo.rope_height(s)
    line = " ".join(f"{a:.7f},{b:.7f},{c:.1f}" for a, b, c in zip(lon, lat, z))
    marks = []
    for p in geo.supports():
        plon, plat = geo.to_wgs(p["e"], p["n"])
        marks.append(f"""  <Placemark><name>{p['name']}</name><styleUrl>#{p['kind']}</styleUrl>
    <Point><altitudeMode>absolute</altitudeMode><coordinates>{plon:.7f},{plat:.7f},{p['rope']:.1f}</coordinates></Point></Placemark>""")
        marks.append(f"""  <Placemark><name>{p['name']} (sloup)</name><styleUrl>#mast</styleUrl>
    <LineString><altitudeMode>absolute</altitudeMode><coordinates>{plon:.7f},{plat:.7f},{p['ground']:.1f} {plon:.7f},{plat:.7f},{p['rope']:.1f}</coordinates></LineString></Placemark>""")
    path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
  <name>Lanová dráha Pisárky – Kampus (přibližná trasa)</name>
  <Style id="rope"><LineStyle><color>ff202020</color><width>3</width></LineStyle></Style>
  <Style id="mast"><LineStyle><color>ffb0b0b0</color><width>5</width></LineStyle></Style>
  <Style id="station"><IconStyle><color>ff0060ff</color></IconStyle></Style>
  <Style id="pylon"><IconStyle><color>ffa0a0a0</color></IconStyle></Style>
  <Placemark><name>Dopravní lano</name><styleUrl>#rope</styleUrl>
    <LineString><altitudeMode>absolute</altitudeMode><coordinates>{line}</coordinates></LineString></Placemark>
{chr(10).join(marks)}
</Document></kml>
""", encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    cab = cabin_frames()
    start = {k: v[0] for k, v in cab.items()}
    dr = drone_frames(start)
    for name, fr in [("01_dron", dr), ("02_kabina", cab)]:
        (OUT / f"{name}.esp").write_text(json.dumps(build_esp(name, fr), separators=(",", ":")))
        (OUT / f"kamera_{name}.json").write_text(json.dumps(frames_json(fr)))
        print(f"{name}: {len(fr['e'])} snímků = {len(fr['e']) / FPS:.1f} s")
    write_kml(OUT / "lanovka_trasa.kml")


if __name__ == "__main__":
    main()
