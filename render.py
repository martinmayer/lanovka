"""Náhledové video (previz) jízdy lanovkou Pisárky – Kampus z otevřených dat.

Terén: ČÚZK DMP 1G (povrch vč. budov a stromů) potažený ortofotem, okolí DMR 5G.
Lanovka: dvě větve lana, 3 podpěry, 4 stanice, kabiny po 72 m (2 000 os/h), zrychleno 4×.
Kamera: out/kamera_01_dron.json + out/kamera_02_kabina.json (stejná dráha jako v .esp projektech).

Použití: python3 render.py [--w 1280 --h 720] [--from 0 --to N] [--out out/nahled.mp4]
"""
import argparse
import json
import math
import os
import subprocess

os.environ.setdefault("GLCONTEXT_DEVICE_INDEX", "1")  # EGL zařízení 1 = llvmpipe (bez GPU)

import moderngl
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import geo
import make_esp as esp

OUT = geo.ROOT / "out"
FPS = esp.FPS
SKY_TOP = np.array([0.42, 0.62, 0.86])
SKY_HORIZON = np.array([0.80, 0.86, 0.92])
SUN = np.array([-0.45, -0.35, 0.82])
SUN = SUN / np.linalg.norm(SUN)
CABIN_SPACING = 72.0
ROPE_V = esp.V_LINE
TURN_TIME = 3.0  # objetí úvraťového kola ve videu [s]

# počátek lokální soustavy (m): nástupní stanice Lipová
E0, N0 = geo.to_utm(geo.STATIONS[0][1], geo.STATIONS[0][2])


# ---------------------------------------------------------------- matice

def perspective(fovy_deg, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], dtype="f8")


def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


# ---------------------------------------------------------------- geometrie

def grid_mesh(z, x0, y1, dx, dy, step=1):
    z = z[::step, ::step]
    h, w = z.shape
    xs = x0 + (np.arange(w) * step + 0.5) * dx - E0
    ys = y1 - (np.arange(h) * step + 0.5) * dy - N0
    X, Y = np.meshgrid(xs, ys)
    U, V = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    verts = np.stack([X, Y, z, U, V], -1).reshape(-1, 5).astype("f4")
    i = np.arange(h * w).reshape(h, w)
    a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
    idx = np.stack([a, c, b, b, c, d], -1).reshape(-1).astype("i4")
    return verts, idx


class Mesh:
    """Trojúhelníky s normálou a barvou: (x, y, z, nx, ny, nz, r, g, b)."""

    def __init__(self):
        self.parts = []

    def tri(self, p, color):
        p = np.asarray(p, float).reshape(-1, 3, 3)
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        n = np.repeat(n, 3, axis=0)
        c = np.broadcast_to(np.asarray(color, float), (len(n), 3))
        self.parts.append(np.hstack([p.reshape(-1, 3), n, c]))

    def box(self, center, size, yaw_deg, color, pitch_deg=0.0):
        """Kvádr; yaw = azimut delší osy (od severu po směru hod. ručiček)."""
        hx, hy, hz = np.asarray(size) / 2
        c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = rot(yaw_deg, pitch_deg)
        v = c @ R.T + center
        faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        tris = []
        for a, b, cc, d in faces:
            tris += [v[a], v[b], v[cc], v[a], v[cc], v[d]]
        self.tri(tris, color)

    def cylinder(self, p0, p1, r0, r1, color, seg=10):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0
        ax /= np.linalg.norm(ax)
        t = np.cross(ax, [1, 0, 0] if abs(ax[0]) < 0.9 else [0, 1, 0])
        t /= np.linalg.norm(t)
        b = np.cross(ax, t)
        tris = []
        for k in range(seg):
            a0, a1 = 2 * math.pi * k / seg, 2 * math.pi * (k + 1) / seg
            d0, d1 = math.cos(a0) * t + math.sin(a0) * b, math.cos(a1) * t + math.sin(a1) * b
            q = [p0 + r0 * d0, p0 + r0 * d1, p1 + r1 * d1, p1 + r1 * d0]
            tris += [q[0], q[2], q[1], q[0], q[3], q[2]]
            tris += [p1, p1 + r1 * d0, p1 + r1 * d1]
        self.tri(tris, color)

    def array(self):
        return np.vstack(self.parts).astype("f4") if self.parts else np.zeros((0, 9), "f4")


def rot(yaw_deg, pitch_deg=0.0):
    """Místní osy: x = napříč, y = podél azimutu, z = nahoru."""
    y = math.radians(yaw_deg)
    fwd = np.array([math.sin(y), math.cos(y), 0.0])
    right = np.array([math.cos(y), -math.sin(y), 0.0])
    p = math.radians(pitch_deg)
    fwd2 = fwd * math.cos(p) + np.array([0, 0, 1.0]) * math.sin(p)
    up = np.cross(right, fwd2)
    return np.stack([right, fwd2, up], 1)


def local(e, n):
    return np.asarray(e) - E0, np.asarray(n) - N0


def static_scene():
    m = Mesh()
    sup = geo.supports()
    for p in sup:
        x, y = local(p["e"], p["n"])
        _, _, az = geo.horizontal(p["s"] + (0.5 if p["s"] < geo.total_length() else -0.5))
        az = float(az)
        if p["kind"] == "pylon":
            top = p["rope"] - 1.2
            m.cylinder([x, y, p["ground"] - 1], [x, y, top], 0.9, 0.55, (0.78, 0.79, 0.80), 12)
            m.box([x, y, top + 0.3], [2 * geo.TRACK_HALF + 2.6, 1.0, 0.8], az, (0.70, 0.71, 0.72))
            for sd in (-1, 1):
                ex, ey = geo.track_xy(p["s"], sd)
                ex, ey = local(ex, ey)
                m.box([float(ex), float(ey), p["rope"] - 0.35], [0.5, 7.0, 0.5], az, (0.35, 0.36, 0.38))
            m.box([x, y, p["ground"] + 0.4], [3.2, 3.2, 1.6], az, (0.62, 0.60, 0.55))
        else:
            station_model(m, p, x, y, az)
    return m.array()


def station_model(m, p, x, y, az):
    L_, W_ = 40.0, 16.0
    floor = p["rope"] - 5.4
    roof = p["rope"] + 2.8
    R = rot(az)
    def at(dx, dy, z):
        return np.array([x, y, 0]) + R @ np.array([dx, dy, 0]) + np.array([0, 0, z])
    m.box(at(0, 0, floor - 0.3), [W_, L_, 0.6], az, (0.55, 0.55, 0.56))            # nástupiště
    m.box(at(0, 0, roof + 0.4), [W_ + 2, L_ + 4, 0.8], az, (0.93, 0.93, 0.94))      # střecha
    m.box(at(0, 0, roof + 1.4), [W_ - 3, L_ - 6, 1.4], az, (0.80, 0.16, 0.16))      # červená nástavba (pohon)
    for dx in (-W_ / 2 + 0.5, W_ / 2 - 0.5):
        for dy in (-L_ / 2 + 1, 0, L_ / 2 - 1):
            c = at(dx, dy, 0)
            m.cylinder([c[0], c[1], p["ground"] - 1], [c[0], c[1], roof], 0.35, 0.35, (0.65, 0.66, 0.68), 8)
    # prosklené boční stěny (tmavý pás) a zábradlí
    for dx in (-W_ / 2, W_ / 2):
        m.box(at(dx, 0, floor + 1.1), [0.15, L_, 1.0], az, (0.40, 0.46, 0.52))
    if p["name"] in (geo.STATIONS[0][0], geo.STATIONS[-1][0]):
        end = -1 if p["name"] == geo.STATIONS[0][0] else 1
        c = at(0, end * (L_ / 2 - 3), p["rope"])
        m.cylinder([c[0], c[1], p["rope"] - 0.25], [c[0], c[1], p["rope"] + 0.25],
                   geo.TRACK_HALF + 0.2, geo.TRACK_HALF + 0.2, (0.25, 0.25, 0.27), 24)


# ---------------------------------------------------------------- kabiny

class Fleet:
    def __init__(self):
        cab = esp.cabin_s_of_t()
        self.t_up = (len(cab) - 1) / FPS
        self.s_tab = cab
        self.period = 2 * self.t_up + 2 * TURN_TIME
        self.dt = CABIN_SPACING / ROPE_V
        self.n = int(round(self.period / self.dt))
        self.dt = self.period / self.n

    def s_at(self, t):
        return np.interp(t * FPS, np.arange(len(self.s_tab)), self.s_tab)

    def poses(self, t_ride, skip_own):
        """Poloha úchytu na laně (x, y, z) a azimut kabiny pro všechny kabiny."""
        L = geo.total_length()
        out = []
        for k in range(self.n):
            if skip_own and k == 0:
                continue
            ph = (t_ride - k * self.dt) % self.period
            if ph < self.t_up:
                s, side, heading = self.s_at(ph), 1, 0
            elif ph < self.t_up + TURN_TIME:
                out.append(self._turn(L, (ph - self.t_up) / TURN_TIME, end=True))
                continue
            elif ph < 2 * self.t_up + TURN_TIME:
                s, side, heading = self.s_at(2 * self.t_up + TURN_TIME - ph), -1, 180
            else:
                out.append(self._turn(0.0, (ph - 2 * self.t_up - TURN_TIME) / TURN_TIME, end=False))
                continue
            e, n = geo.track_xy(s, side)
            _, _, az = geo.horizontal(s)
            x, y = local(e, n)
            out.append((float(x), float(y), float(geo.rope_height(s)), float(az) + heading))
        return out

    def _turn(self, s_end, u, end):
        _, _, az = geo.horizontal(s_end)
        az = float(az)
        e, n, _ = geo.horizontal(s_end)
        # úvraťové kolo leží 17 m za středem stanice
        d = 17.0 if end else -17.0
        ce = e + d * math.sin(math.radians(az))
        cn = n + d * math.cos(math.radians(az))
        # pravá větev → objetí kola → levá větev (na konci trasy), opačně na začátku
        a0 = az + 90 if end else az - 90
        ang = a0 - 180 * u if end else a0 - 180 * u
        x, y = local(ce + geo.TRACK_HALF * math.sin(math.radians(ang)),
                     cn + geo.TRACK_HALF * math.cos(math.radians(ang)))
        z = float(geo.rope_height(s_end))
        return (float(x), float(y), z, ang - 90)


def cabin_mesh(poses):
    m = Mesh()
    for x, y, z, az in poses:
        base = z - 5.2
        m.box([x, y, base + 0.45], [1.9, 2.0, 0.9], az, (0.78, 0.10, 0.12))
        m.box([x, y, base + 1.42], [1.86, 1.96, 1.04], az, (0.16, 0.22, 0.30))
        m.box([x, y, base + 2.06], [1.7, 1.8, 0.26], az, (0.95, 0.95, 0.95))
        m.box([x, y, base + 3.6], [0.12, 0.12, 2.9], az, (0.30, 0.30, 0.32))
        m.box([x, y, z - 0.35], [0.35, 0.9, 0.35], az, (0.25, 0.25, 0.27))
    return m.array()


# ---------------------------------------------------------------- lano

def rope_polylines():
    L = geo.total_length()
    s = np.arange(0, L + 0.1, 2.0)
    lines = []
    for side in (-1, 1):
        e, n = geo.track_xy(s, side)
        x, y = local(e, n)
        lines.append(np.stack([x, y, geo.rope_height(s)], -1))
    return lines


def ribbons(lines, eye, px_angle, radius=0.026, min_px=1.1, color=(0.08, 0.08, 0.09)):
    tris = []
    for P in lines:
        T = np.gradient(P, axis=0)
        V = P - eye
        dist = np.linalg.norm(V, axis=1)
        side = np.cross(T, V)
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
        w = np.maximum(radius, 0.5 * min_px * px_angle * dist)[:, None]
        a, b = P - side * w, P + side * w
        q = np.stack([a[:-1], b[:-1], b[1:], a[:-1], b[1:], a[1:]], 1).reshape(-1, 3)
        tris.append(q)
    q = np.vstack(tris)
    n = np.tile([0, 0, 1.0], (len(q), 1))
    c = np.tile(color, (len(q), 1))
    return np.hstack([q, n, c]).astype("f4")


# ---------------------------------------------------------------- shadery

TERRAIN_VS = """#version 330
uniform mat4 mvp; uniform vec3 eye;
in vec3 in_pos; in vec2 in_uv;
out vec2 uv; out float dist;
void main(){ gl_Position = mvp*vec4(in_pos,1.0); uv=in_uv; dist=length(in_pos-eye); }"""
TERRAIN_FS = """#version 330
uniform sampler2D tex; uniform vec3 haze; uniform float fogk;
in vec2 uv; in float dist; out vec4 f;
void main(){ vec3 c = texture(tex, uv).rgb; c = pow(c, vec3(0.95))*1.04;
  float h = 1.0-exp(-dist*fogk); f = vec4(mix(c, haze, h*0.85),1.0); }"""
MODEL_VS = """#version 330
uniform mat4 mvp; uniform vec3 eye;
in vec3 in_pos; in vec3 in_n; in vec3 in_c;
out vec3 n; out vec3 c; out float dist;
void main(){ gl_Position = mvp*vec4(in_pos,1.0); n=in_n; c=in_c; dist=length(in_pos-eye); }"""
MODEL_FS = """#version 330
uniform vec3 sun; uniform vec3 haze; uniform float fogk;
in vec3 n; in vec3 c; in float dist; out vec4 f;
void main(){ float l = 0.45 + 0.6*abs(dot(normalize(n), sun));
  float h = 1.0-exp(-dist*fogk); f = vec4(mix(c*l, haze, h*0.85),1.0); }"""
SKY_VS = """#version 330
in vec2 p; out vec2 q; void main(){ gl_Position=vec4(p,0.9999,1.0); q=p; }"""
SKY_FS = """#version 330
uniform mat4 inv; uniform vec3 top; uniform vec3 hor; in vec2 q; out vec4 f;
void main(){ vec4 w = inv*vec4(q,1.0,1.0); vec3 d = normalize(w.xyz/w.w);
  float t = clamp(d.z*3.0,0.0,1.0); f = vec4(mix(hor, top, sqrt(t)),1.0); }"""


class Renderer:
    def __init__(self, w, h, fov):
        self.w, self.h, self.fov = w, h, fov
        self.ctx = moderngl.create_standalone_context(backend="egl")
        ctx = self.ctx
        self.fbo = ctx.framebuffer(ctx.renderbuffer((w, h), components=4, samples=4),
                                   ctx.depth_renderbuffer((w, h), samples=4))
        self.res = ctx.framebuffer(ctx.renderbuffer((w, h), components=4))
        self.tprog = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.mprog = ctx.program(vertex_shader=MODEL_VS, fragment_shader=MODEL_FS)
        self.sprog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.sky = ctx.vertex_array(self.sprog, [(ctx.buffer(np.array(
            [-1, -1, 1, -1, -1, 1, 1, 1], "f4")), "2f", "p")], mode=moderngl.TRIANGLE_STRIP)
        self.terrains = []
        far = json.loads((geo.DATA / "dmr5g_far.json").read_text())["extent"]
        zf = tifffile.imread(geo.DATA / "dmr5g_far.tif").astype("f4") - 1.5
        self._add_terrain(zf, far, "ortofoto_far.jpg", step=1)
        near = json.loads((geo.DATA / "dmp1g.json").read_text())["extent"]
        zn = geo.DMP.a.astype("f4")
        self._add_terrain(zn, near, "ortofoto_hi.jpg", step=1)
        st = static_scene()
        self.static = ctx.vertex_array(self.mprog, [(ctx.buffer(st), "3f 3f 3f", "in_pos", "in_n", "in_c")])
        self.dyn_buf = ctx.buffer(reserve=16 * 1024 * 1024)
        self.dyn = ctx.vertex_array(self.mprog, [(self.dyn_buf, "3f 3f 3f", "in_pos", "in_n", "in_c")])
        self.ropes = rope_polylines()

    def _add_terrain(self, z, ext, img, step):
        h, w = z.shape
        dx, dy = (ext["xmax"] - ext["xmin"]) / w, (ext["ymax"] - ext["ymin"]) / h
        verts, idx = grid_mesh(z, ext["xmin"], ext["ymax"], dx, dy, step)
        im = Image.open(geo.DATA / img).convert("RGB")
        tex = self.ctx.texture(im.size, 3, im.tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.anisotropy = 8.0
        vao = self.ctx.vertex_array(self.tprog, [(self.ctx.buffer(verts), "3f 2f", "in_pos", "in_uv")],
                                    index_buffer=self.ctx.buffer(idx))
        self.terrains.append((vao, tex))

    def frame(self, eye, target, cabins, view=None, fov=None, overlay=False):
        """Vykreslí snímek. overlay=True: terén jen zakrývá (bez barvy), výstup RGBA jen s lanovkou."""
        ctx = self.ctx
        proj = perspective(fov or self.fov, self.w / self.h, 0.3, 30000)
        if view is None:
            view = look_at(eye, target)
        mvp = (proj @ view).T.astype("f4").tobytes()
        self.fbo.use()
        if overlay:
            ctx.clear(0.0, 0.0, 0.0, 0.0)
        else:
            ctx.clear(*SKY_HORIZON, 1.0)
            ctx.disable(moderngl.DEPTH_TEST)
            inv = np.linalg.inv(proj @ np.block([[view[:3, :3], np.zeros((3, 1))], [0, 0, 0, 1]]))
            self.sprog["inv"].write(inv.T.astype("f4").tobytes())
            self.sprog["top"].value = tuple(SKY_TOP)
            self.sprog["hor"].value = tuple(SKY_HORIZON)
            self.sky.render()
        ctx.enable(moderngl.DEPTH_TEST)
        for prog in (self.tprog, self.mprog):
            prog["mvp"].write(mvp)
            prog["eye"].value = tuple(eye)
            prog["haze"].value = tuple(SKY_HORIZON)
            prog["fogk"].value = 1 / 9000
        self.mprog["sun"].value = tuple(SUN)
        if overlay:
            # zákryt terénem: jen hloubka, mírně odsunutá, aby lanovka „vyhrávala“ u nepřesností modelů
            self.fbo.color_mask = (False, False, False, False)
            ctx.polygon_offset = (2.0, 40.0)
        for vao, tex in self.terrains:
            tex.use(0)
            vao.render()
        if overlay:
            self.fbo.color_mask = (True, True, True, True)
            ctx.polygon_offset = (0.0, 0.0)
        self.static.render()
        px_angle = 2 * math.tan(math.radians(self.fov) / 2) / self.h
        dyn = np.vstack([ribbons(self.ropes, eye, px_angle), cabin_mesh(cabins)])
        hangers = [np.array([[x, y, z], [x, y, z - 3.0]]) for x, y, z, _ in cabins]
        if hangers:
            dyn = np.vstack([dyn, ribbons(hangers, eye, px_angle, radius=0.05, color=(0.3, 0.3, 0.32))])
        self.dyn_buf.orphan(dyn.nbytes)
        self.dyn_buf.write(dyn.tobytes())
        self.dyn.render(vertices=len(dyn))
        ctx.copy_framebuffer(self.res, self.fbo)
        if overlay:
            img = Image.frombytes("RGBA", (self.w, self.h), self.res.read(components=4))
        else:
            img = Image.frombytes("RGB", (self.w, self.h), self.res.read(components=3))
        return img.transpose(Image.FLIP_TOP_BOTTOM)


# ---------------------------------------------------------------- popisky

def font(size, bold=False):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


class Hud:
    def __init__(self, w, h):
        self.w, self.h = w, h
        k = h / 720
        self.k = k
        self.f_title, self.f_sub = font(int(40 * k), True), font(int(20 * k))
        self.f_st, self.f_small = font(int(30 * k), True), font(int(14 * k))
        self.f_info = font(int(17 * k), True)
        ext = json.loads((geo.DATA / "dmr5g.json").read_text())["extent"]
        self.ext = ext
        L = geo.total_length()
        s = np.arange(0, L, 4.0)
        e, n, _ = geo.horizontal(s)
        xs, ys = self._px(e, n)
        box = (int(xs.min() - 90), int(ys.min() - 60), int(xs.max() + 90), int(ys.max() + 60))
        self.box = box
        ortho = Image.open(geo.DATA / "ortofoto.jpg").convert("RGB").crop(box)
        mh = int(250 * k)
        self.ms = mh / ortho.height
        mw = int(ortho.width * self.ms)
        ortho = ortho.resize((mw, mh), Image.LANCZOS)
        ortho = Image.blend(ortho, Image.new("RGB", ortho.size, (20, 24, 30)), 0.35)
        d = ImageDraw.Draw(ortho)
        pts = [((x - box[0]) * self.ms, (y - box[1]) * self.ms) for x, y in zip(xs, ys)]
        d.line(pts, fill=(255, 70, 60), width=max(2, int(3 * k)))
        for p in geo.supports():
            x, y = self._px(p["e"], p["n"])
            x, y = (x - box[0]) * self.ms, (y - box[1]) * self.ms
            r = 4 * k if p["kind"] == "station" else 2.5 * k
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255) if p["kind"] == "station" else (255, 180, 60))
        self.minimap = ortho

    def _px(self, e, n):
        e, n = np.asarray(e), np.asarray(n)
        return ((e - self.ext["xmin"]) / geo.DMR.dx, (self.ext["ymax"] - n) / geo.DMR.dy)

    def draw(self, img, t, eye_en, s_cabin, part, alt_agl):
        k = self.k
        base = img.convert("RGBA")
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # minimapa
        mm = self.minimap.copy()
        md = ImageDraw.Draw(mm)
        x, y = self._px(*eye_en)
        x, y = (x - self.box[0]) * self.ms, (y - self.box[1]) * self.ms
        r = 6 * k
        md.ellipse([x - r, y - r, x + r, y + r], fill=(255, 225, 40), outline=(0, 0, 0), width=2)
        mx, my = int(self.w - mm.width - 24 * k), int(24 * k)
        d.rectangle([mx - 3, my - 3, mx + mm.width + 2, my + mm.height + 2], fill=(255, 255, 255, 160))
        ov.paste(mm, (mx, my))
        # titulky
        if part == "dron":
            a = fade(t, 1.0, 9.5, 1.2)
            if a > 0:
                shadow_text(d, (int(40 * k), int(self.h - 150 * k)), "Lanová dráha Pisárky – Kampus", self.f_title, a)
                shadow_text(d, (int(42 * k), int(self.h - 100 * k)),
                            "záměr DPMB · 1,7 km · 4 stanice · převýšení 70 m · kabiny pro 8 osob",
                            self.f_sub, a)
        else:
            for p in geo.supports():
                if p["kind"] != "station":
                    continue
                dist = abs(s_cabin - p["s"])
                a = float(np.clip((90 - dist) / 40, 0, 1))
                if a > 0:
                    shadow_text(d, (int(40 * k), int(40 * k)), p["name"], self.f_st, a)
                    shadow_text(d, (int(42 * k), int(80 * k)), "stanice" if p["name"] in (
                        geo.STATIONS[0][0], geo.STATIONS[-1][0]) else "mezistanice", self.f_sub, a)
            info = f"jízda z kabiny · zrychleno {esp.SPEEDUP:.0f}× · výška nad terénem {alt_agl:3.0f} m"
            shadow_text(d, (int(40 * k), int(self.h - 70 * k)), info, self.f_info, 1.0)
        shadow_text(d, (int(40 * k), int(self.h - 36 * k)),
                    "Vizualizace nerealizovaného záměru; poloha stanic a podpěr přibližná. "
                    "Data: ČÚZK (ortofoto, DMR 5G, DMP 1G), OpenStreetMap", self.f_small, 0.9)
        return Image.alpha_composite(base, ov).convert("RGB")


def fade(t, t0, t1, ramp):
    return float(np.clip(min((t - t0) / ramp, (t1 - t) / ramp), 0, 1))


def shadow_text(d, xy, text, f, a):
    x, y = xy
    d.text((x + 2, y + 2), text, font=f, fill=(0, 0, 0, int(170 * a)))
    d.text((x, y), text, font=f, fill=(255, 255, 255, int(255 * a)))


# ---------------------------------------------------------------- hlavní smyčka

def load_camera():
    frames = []
    for part in ("01_dron", "02_kabina"):
        fr = json.loads((OUT / f"kamera_{part}.json").read_text())
        for f in fr:
            f["part"] = "dron" if part == "01_dron" else "kabina"
        frames += fr
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fov", type=float, default=42.0)
    ap.add_argument("--from", dest="f0", type=int, default=0)
    ap.add_argument("--to", dest="f1", type=int, default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", default=str(OUT / "nahled.mp4"))
    ap.add_argument("--png", default=None, help="uloží jednotlivé snímky místo videa (složka)")
    a = ap.parse_args()

    cams = load_camera()
    n_drone = sum(1 for c in cams if c["part"] == "dron")
    rend = Renderer(a.w, a.h, a.fov)
    hud = Hud(a.w, a.h)
    fleet = Fleet()
    f1 = a.f1 if a.f1 is not None else len(cams)
    idx = range(a.f0, f1, a.step)
    if a.png:
        os.makedirs(a.png, exist_ok=True)
        pipe = None
    else:
        pipe = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-s", f"{a.w}x{a.h}", "-r", str(FPS // a.step), "-i", "-",
                                 "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                                 a.out], stdin=subprocess.PIPE)
    import time
    t_start = time.time()
    for j, i in enumerate(idx):
        c = cams[i]
        ce, cn = geo.to_utm(c["lon"], c["lat"])
        te, tn = geo.to_utm(c["target_lon"], c["target_lat"])
        eye = np.array([ce - E0, cn - N0, c["alt"]])
        tgt = np.array([te - E0, tn - N0, c["target_alt"]])
        t_ride = (i - n_drone) / FPS
        cab = fleet.poses(t_ride, skip_own=c["part"] == "kabina")
        img = rend.frame(eye, tgt, cab)
        agl = c["alt"] - float(geo.DMR.sample(ce, cn))
        img = hud.draw(img, i / FPS, (ce, cn), c["s"], c["part"], agl)
        if pipe:
            pipe.stdin.write(img.tobytes())
        else:
            img.save(os.path.join(a.png, f"f{i:05d}.jpg"), quality=90)
        if j % 50 == 0:
            el = time.time() - t_start
            print(f"snímek {i} ({j + 1}/{len(idx)}), {el / (j + 1):.2f} s/snímek", flush=True)
    if pipe:
        pipe.stdin.close()
        pipe.wait()
    print("hotovo", a.png or a.out)


if __name__ == "__main__":
    main()
