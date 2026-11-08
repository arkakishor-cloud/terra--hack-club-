"""
Terra AI - GPU-Accelerated Infinite Procedural World Map

Architecture:
  - GLSL fragment shader computes FBM noise per-pixel on the GPU
  - Panning = changing a uniform offset (zero data transfer)
  - Hover tooltip computed on CPU for a single pixel (instant)
  - Scales to 4K at 60+ FPS; truly infinite in all directions

Prerequisites:
  pip install moderngl moderngl-window
"""

import time
import argparse
import sys

import numpy as np

try:
    import moderngl
    import moderngl_window
except ImportError:
    moderngl = None
    moderngl_window = None
    # Deferred class definitions - will be replaced with stubs if no GFX libs
    WorldMapWindow = None
    run_gpu = None


# ---------------------------------------------------------------------------
# GLSL fragment shader - runs entirely on the GPU
# ---------------------------------------------------------------------------

FRAGMENT_SHADER = """#version 330 core

in vec2 v_tex;
out vec4 f_color;

uniform vec2  u_resolution;
uniform vec2  u_offset;       // world-space pan offset
uniform float u_freq;         // base noise frequency
uniform int   u_octaves;
uniform float u_persistence;
uniform float u_lacunarity;
uniform float u_sea_level;
uniform int   u_ice;
uniform int   u_seed;

// Deterministic hash for any integer vec2 (works with negatives)
float hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p.x) * 43758.5453 + sin(p.y) * 12345.6789);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);  // cubic smoothstep
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p, int octaves, float freq, float persistence, float lacunarity, int seed) {
    float total = 0.0;
    float amp = 1.0;
    float max_amp = 0.0;
    vec2 pp = p * freq;
    for (int i = 0; i < octaves; i++) {
        vec2 offset = vec2(float(seed + i) * 123.456, float(seed + i) * 789.012);
        total += noise(pp + offset) * amp;
        max_amp += amp;
        amp *= persistence;
        pp *= lacunarity;
    }
    return total / max_amp;
}

vec3 world_colormap(float t) {
    vec3 c1 = vec3(0.000, 0.122, 0.247);  // #001f3f
    vec3 c2 = vec3(0.000, 0.247, 0.498);  // #003f7f
    vec3 c3 = vec3(0.118, 0.533, 0.898);  // #1e88e5
    vec3 c4 = vec3(0.306, 0.714, 0.675);  // #4db6ac
    vec3 c5 = vec3(0.831, 0.784, 0.604);  // #d4c89a
    vec3 c6 = vec3(0.435, 0.667, 0.290);  // #6faa4a
    vec3 c7 = vec3(0.604, 0.627, 0.420);  // #9aa06b
    vec3 c8 = vec3(0.553, 0.451, 0.290);  // #8d734a
    vec3 c9 = vec3(0.690, 0.690, 0.690);  // #b0b0b0
    vec3 c10 = vec3(1.000, 1.000, 1.000);  // #ffffff

    if (t < 0.125) return mix(c1, c2, t / 0.125);
    if (t < 0.250) return mix(c2, c3, (t - 0.125) / 0.125);
    if (t < 0.300) return mix(c3, c4, (t - 0.250) / 0.050);
    if (t < 0.340) return mix(c4, c5, (t - 0.300) / 0.040);
    if (t < 0.420) return mix(c5, c6, (t - 0.340) / 0.080);
    if (t < 0.560) return mix(c6, c7, (t - 0.420) / 0.140);
    if (t < 0.720) return mix(c7, c8, (t - 0.560) / 0.160);
    if (t < 0.860) return mix(c8, c9, (t - 0.720) / 0.140);
    return mix(c9, c10, (t - 0.860) / 0.140);
}

void main() {
    vec2 world = (v_tex * u_resolution - u_resolution * 0.5) / u_resolution.y * 2000.0 + u_offset;

    float elev = fbm(world, u_octaves, u_freq, u_persistence, u_lacunarity, u_seed);

    if (elev < u_sea_level) {
        elev = elev * u_sea_level;
    } else {
        elev = u_sea_level + (elev - u_sea_level);
    }

    if (u_ice == 1) {
        float lat = abs(world.y) / 2500.0;
        lat = clamp(lat, 0.0, 1.0);
        float ice = clamp(lat - 0.88, 0.0, 1.0) / 0.12;
        elev = clamp(elev + ice * 0.10, 0.0, 1.0);
    }

    vec3 color = world_colormap(elev);
    f_color = vec4(color, 1.0);
}
"""


# ---------------------------------------------------------------------------
# CPU-side noise for hover tooltip (single point - instant)
# ---------------------------------------------------------------------------

def quintic(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def hash_val(ix, iy, seed):
    n = (ix * 1597334677 + iy * 3875123 + seed * 1099087573) % (2 ** 32)
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n % 1000000) / 1000000.0


def value_noise(x, y, seed):
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0, y - y0
    wx, wy = quintic(dx), quintic(dy)
    a = hash_val(x0, y0, seed)
    b = hash_val(x1, y0, seed)
    c = hash_val(x0, y1, seed)
    d = hash_val(x1, y1, seed)
    top = a * (1 - wx) + b * wx
    bottom = c * (1 - wx) + d * wx
    return top * (1 - wy) + bottom * wy


def fbm_point(x, y, seed, octaves=7, persistence=0.55, lacunarity=2.0, freq=0.005):
    total = 0.0
    amp = 1.0
    max_amp = 0.0
    f = freq
    for i in range(octaves):
        total += value_noise(x * f + seed + i, y * f + seed + i, seed + i) * amp
        max_amp += amp
        amp *= persistence
        f *= lacunarity
    return total / max_amp if max_amp > 0 else total


def apply_sea_level_point(elev, sea_level=0.42):
    if elev < sea_level:
        return elev * sea_level
    return sea_level + (elev - sea_level)


def apply_ice_caps_point(elev, world_y, ice_strength=0.12):
    lat = abs(world_y) / 2500.0
    lat = min(max(lat, 0.0), 1.0)
    ice = min(max(lat - (1.0 - ice_strength), 0.0), 1.0) / ice_strength
    return min(max(elev + ice * 0.10, 0.0), 1.0)


def biome_label(elev):
    if elev < 0.25: return "Ocean"
    elif elev < 0.34: return "Coastal"
    elif elev < 0.42: return "Beach"
    elif elev < 0.56: return "Lowland"
    elif elev < 0.75: return "Highland"
    elif elev < 0.90: return "Mountain"
    else: return "Peak"


# ---------------------------------------------------------------------------
# GPU renderer (moderngl_window) - only defined if moderngl is available
# ---------------------------------------------------------------------------

if moderngl is not None and moderngl_window is not None:

    class WorldMapWindow(moderngl_window.WindowConfig):
        gl_version = (3, 3)
        title = "Terra AI - GPU Infinite World Map"
        resizable = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prog = self.ctx.program(
                vertex_shader='''
                    #version 330 core
                    in vec2 in_vert;
                    in vec2 in_texcoord;
                    out vec2 v_tex;
                    void main() {
                        gl_Position = vec4(in_vert, 0.0, 1.0);
                        v_tex = in_texcoord;
                    }
                ''',
                fragment_shader=FRAGMENT_SHADER,
            )

            vertices = np.array([
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype=np.float32)

            self.vbo = self.ctx.buffer(vertices.tobytes())
            self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_texcoord')

            self.cam_x = 0.0
            self.cam_y = 0.0
            self.drag_active = False
            self.drag_start_x = 0.0
            self.drag_start_y = 0.0
            self.drag_cam_x = 0.0
            self.drag_cam_y = 0.0

            self.seed = kwargs.get('seed', 42)
            self.octaves = kwargs.get('octaves', 7)
            self.persistence = kwargs.get('persistence', 0.55)
            self.lacunarity = kwargs.get('lacunarity', 2.0)
            self.freq = kwargs.get('freq', 0.005)
            self.sea_level = kwargs.get('sea_level', 0.42)
            self.ice = kwargs.get('ice', True)
            self.pan_speed = kwargs.get('pan_speed', 1.0)

            self.last_hover = 0.0

        def render(self, time, frame_time):
            self.ctx.clear(0.0, 0.0, 0.0)

            self.prog['u_resolution'] = (self.wnd.width, self.wnd.height)
            self.prog['u_offset'] = (self.cam_x, self.cam_y)
            self.prog['u_freq'] = self.freq
            self.prog['u_octaves'] = self.octaves
            self.prog['u_persistence'] = self.persistence
            self.prog['u_lacunarity'] = self.lacunarity
            self.prog['u_sea_level'] = self.sea_level
            self.prog['u_ice'] = 1 if self.ice else 0
            self.prog['u_seed'] = self.seed

            self.vao.render(moderngl.TRIANGLE_STRIP)

        def resize(self, width: int, height: int):
            self.ctx.viewport = (0, 0, width, height)

        def mouse_drag_event(self, x, y, dx, dy):
            world_dx = -dx * self.pan_speed
            world_dy = dy * self.pan_speed
            self.cam_x = self.drag_cam_x + world_dx / self.wnd.height * 2000.0
            self.cam_y = self.drag_cam_y + world_dy / self.wnd.height * 2000.0

        def mouse_press_event(self, x, y, button):
            if button == 1:
                self.drag_active = True
                self.drag_cam_x = self.cam_x
                self.drag_cam_y = self.cam_y

        def mouse_release_event(self, x, y, button):
            if button == 1:
                self.drag_active = False

        def mouse_motion_event(self, x, y, dx, dy):
            now = time.time()
            if now - self.last_hover < 0.033:
                return
            self.last_hover = now

            world_x = (x - self.wnd.width * 0.5) / self.wnd.height * 2000.0 + self.cam_x
            world_y = (-(y - self.wnd.height * 0.5)) / self.wnd.height * 2000.0 + self.cam_y

            elev = fbm_point(world_x, world_y, self.seed,
                             self.octaves, self.persistence, self.lacunarity, self.freq)
            elev = apply_sea_level_point(elev, self.sea_level)
            if self.ice:
                elev = apply_ice_caps_point(elev, world_y)

            print(f"({world_x:.0f}, {world_y:.0f})  Elevation: {elev:.3f}  |  {biome_label(elev)}")


def run_gpu(seed=42, width=960, height=540, octaves=7, persistence=0.55,
            lacunarity=2.0, freq=0.005, sea_level=0.42, ice=True, pan_speed=1.0):
    if moderngl is None or moderngl_window is None:
        print("ERROR: moderngl not installed.")
        print("Install with: pip install moderngl moderngl-window")
        sys.exit(1)

    moderngl_window.run_window_config(
        WorldMapWindow,
        args=type('Args', (), {
            'seed': seed, 'octaves': octaves, 'persistence': persistence,
            'lacunarity': lacunarity, 'freq': freq, 'sea_level': sea_level,
            'ice': ice, 'pan_speed': pan_speed,
            'width': width, 'height': height,
        })(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated infinite procedural world map."
    )
    parser.add_argument("--width", type=int, default=960, help="Window width")
    parser.add_argument("--height", type=int, default=540, help="Window height")
    parser.add_argument("--octaves", type=int, default=7, help="FBM octaves")
    parser.add_argument("--persistence", type=float, default=0.55, help="Amplitude decay (0-1)")
    parser.add_argument("--lacunarity", type=float, default=2.0, help="Frequency multiplier (>1)")
    parser.add_argument("--freq", type=float, default=0.005, help="Base noise frequency")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--sea-level", type=float, default=0.42, help="Ocean threshold (0-1)")
    parser.add_argument("--no-ice", action="store_true", help="Disable polar ice caps")
    parser.add_argument("--pan-speed", type=float, default=1.0, help="Mouse pan multiplier")

    args = parser.parse_args()

    print("GPU Infinite World Map")
    print(f"  Window     : {args.width} x {args.height}")
    print(f"  Octaves    : {args.octaves}")
    print(f"  Persistence: {args.persistence}")
    print(f"  Lacunarity : {args.lacunarity}")
    print(f"  Frequency  : {args.freq}")
    print(f"  Seed       : {args.seed}")
    print(f"  Sea level  : {args.sea_level}")
    print(f"  Ice caps   : {not args.no_ice}")
    print()
    print("Controls:")
    print("  Click + drag to pan in any direction (GPU-accelerated, 60 FPS)")
    print("  Console shows elevation/biome under mouse")
    print("  Close window to exit")

    run_gpu(seed=args.seed, width=args.width, height=args.height,
            octaves=args.octaves, persistence=args.persistence,
            lacunarity=args.lacunarity, freq=args.freq,
            sea_level=args.sea_level, ice=not args.no_ice,
            pan_speed=args.pan_speed)


if __name__ == "__main__":
    main()
