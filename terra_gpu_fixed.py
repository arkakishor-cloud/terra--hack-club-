"""
Terra AI — GPU-accelerated procedural terrain viewer

The terrain is evaluated per-pixel in a GLSL fragment shader.
Panning changes a uniform offset, so the terrain is not regenerated on the CPU.

Install:
    python -m pip install numpy moderngl moderngl-window

Run:
    python terra_gpu.py
    python terra_gpu.py --width 1600 --height 900 --octaves 8 --seed 42

Controls:
    Left mouse drag : pan
    Mouse move      : print elevation / biome under cursor
    Esc             : close
"""

import argparse
import math
import sys
import time

import numpy as np

try:
    import moderngl
    import moderngl_window as mglw
except ImportError as exc:
    moderngl = None
    mglw = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# GPU shader
# ---------------------------------------------------------------------------

FRAGMENT_SHADER = """
#version 330 core

in vec2 v_tex;
out vec4 f_color;

uniform vec2  u_resolution;
uniform vec2  u_offset;
uniform float u_freq;
uniform int   u_octaves;
uniform float u_persistence;
uniform float u_lacunarity;
uniform float u_sea_level;
uniform int   u_ice;
uniform float u_ice_strength;
uniform int   u_seed;

// Keep this hash consistent with hash_val() on the CPU.
float hash_grid(vec2 p) {
    float n = dot(p, vec2(127.1, 311.7))
            + float(u_seed) * 74.7;
    return fract(sin(n) * 43758.5453123);
}

float value_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 f = fract(p);

    // Cubic interpolation.
    f = f * f * (3.0 - 2.0 * f);

    float a = hash_grid(cell);
    float b = hash_grid(cell + vec2(1.0, 0.0));
    float c = hash_grid(cell + vec2(0.0, 1.0));
    float d = hash_grid(cell + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float total = 0.0;
    float amplitude = 1.0;
    float amplitude_sum = 0.0;
    float frequency = u_freq;

    for (int i = 0; i < u_octaves; ++i) {
        total += value_noise(p * frequency) * amplitude;
        amplitude_sum += amplitude;
        amplitude *= u_persistence;
        frequency *= u_lacunarity;
    }

    return total / max(amplitude_sum, 0.000001);
}

vec3 terrain_color(float h) {
    // Elevation palette: deep ocean -> shallow ocean -> beach -> land -> peaks.
    const vec3 deep_ocean = vec3(0.000, 0.055, 0.180);
    const vec3 ocean      = vec3(0.000, 0.247, 0.498);
    const vec3 shallow    = vec3(0.118, 0.533, 0.898);
    const vec3 coast      = vec3(0.306, 0.714, 0.675);
    const vec3 beach      = vec3(0.831, 0.784, 0.604);
    const vec3 lowland    = vec3(0.435, 0.667, 0.290);
    const vec3 highland   = vec3(0.604, 0.627, 0.420);
    const vec3 mountain   = vec3(0.553, 0.451, 0.290);
    const vec3 rock       = vec3(0.690, 0.690, 0.690);
    const vec3 snow       = vec3(1.000, 1.000, 1.000);

    if (h < 0.12) return mix(deep_ocean, ocean, h / 0.12);
    if (h < 0.25) return mix(ocean, shallow, (h - 0.12) / 0.13);
    if (h < 0.30) return mix(shallow, coast, (h - 0.25) / 0.05);
    if (h < 0.34) return mix(coast, beach, (h - 0.30) / 0.04);
    if (h < 0.48) return mix(beach, lowland, (h - 0.34) / 0.14);
    if (h < 0.64) return mix(lowland, highland, (h - 0.48) / 0.16);
    if (h < 0.78) return mix(highland, mountain, (h - 0.64) / 0.14);
    if (h < 0.91) return mix(mountain, rock, (h - 0.78) / 0.13);
    return mix(rock, snow, (h - 0.91) / 0.09);
}

void main() {
    // Keep the same world-coordinate mapping in the CPU hover calculation.
    vec2 screen = (v_tex * u_resolution - 0.5 * u_resolution);
    vec2 world = screen / u_resolution.y * 2000.0 + u_offset;

    float elevation = fbm(world);

    // Compress underwater elevations into the ocean range.
    if (elevation < u_sea_level) {
        elevation = (elevation / max(u_sea_level, 0.0001)) * 0.30;
    } else {
        elevation = 0.30 + (elevation - u_sea_level)
                         / max(1.0 - u_sea_level, 0.0001) * 0.70;
    }

    // Optional stylized polar ice. This is a visual latitude effect, not a globe.
    if (u_ice == 1) {
        float latitude = clamp(abs(world.y) / 2500.0, 0.0, 1.0);
        float ice_amount = smoothstep(1.0 - u_ice_strength, 1.0, latitude);
        elevation = clamp(mix(elevation, 1.0, ice_amount * 0.85), 0.0, 1.0);
    }

    f_color = vec4(terrain_color(clamp(elevation, 0.0, 1.0)), 1.0);
}
"""


# ---------------------------------------------------------------------------
# CPU implementation for the hover readout.
# It intentionally mirrors the shader's hash, interpolation and FBM.
# ---------------------------------------------------------------------------

def hash_val(ix: int, iy: int, seed: int) -> float:
    n = ix * 127.1 + iy * 311.7 + seed * 74.7
    value = math.sin(n) * 43758.5453123
    return value - math.floor(value)


def value_noise(x: float, y: float, seed: int) -> float:
    x0, y0 = math.floor(x), math.floor(y)
    fx, fy = x - x0, y - y0

    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    a = hash_val(x0,     y0,     seed)
    b = hash_val(x0 + 1, y0,     seed)
    c = hash_val(x0,     y0 + 1, seed)
    d = hash_val(x0 + 1, y0 + 1, seed)

    top = a + (b - a) * fx
    bottom = c + (d - c) * fx
    return top + (bottom - top) * fy


def fbm_point(x, y, seed, octaves, persistence, lacunarity, freq):
    total = 0.0
    amplitude = 1.0
    amplitude_sum = 0.0
    frequency = freq

    for _ in range(octaves):
        total += value_noise(x * frequency, y * frequency, seed) * amplitude
        amplitude_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / amplitude_sum if amplitude_sum else 0.0


def shape_elevation(elevation, sea_level):
    if elevation < sea_level:
        return (elevation / max(sea_level, 1e-6)) * 0.30
    return 0.30 + ((elevation - sea_level) /
                   max(1.0 - sea_level, 1e-6)) * 0.70


def biome_label(elevation):
    if elevation < 0.12:
        return "Deep ocean"
    if elevation < 0.25:
        return "Ocean"
    if elevation < 0.30:
        return "Shallow water"
    if elevation < 0.34:
        return "Coast / beach"
    if elevation < 0.48:
        return "Lowland"
    if elevation < 0.64:
        return "Highland"
    if elevation < 0.78:
        return "Mountain"
    if elevation < 0.91:
        return "Rock / alpine"
    return "Peak / snow"


# ---------------------------------------------------------------------------
# Window / renderer
# ---------------------------------------------------------------------------

if mglw is not None:

    class WorldMapWindow(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "Terra AI — GPU Procedural Terrain"
        window_size = (1280, 720)
        resizable = True
        aspect_ratio = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            # Settings are assigned to the class in run_gpu().
            self.seed = int(type(self).terrain_seed)
            self.octaves = max(1, min(12, int(type(self).terrain_octaves)))
            self.persistence = float(type(self).terrain_persistence)
            self.lacunarity = float(type(self).terrain_lacunarity)
            self.freq = float(type(self).terrain_freq)
            self.sea_level = float(type(self).terrain_sea_level)
            self.ice = bool(type(self).terrain_ice)
            self.ice_strength = float(type(self).terrain_ice_strength)

            self.cam_x = 0.0
            self.cam_y = 0.0
            self.pan_speed = float(type(self).terrain_pan_speed)
            self.last_hover_time = 0.0

            vertex_shader = """
                #version 330 core
                in vec2 in_vert;
                in vec2 in_texcoord;
                out vec2 v_tex;

                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_tex = in_texcoord;
                }
            """

            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=FRAGMENT_SHADER,
            )

            vertices = np.array([
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                -1.0,  1.0, 0.0, 1.0,
                 1.0,  1.0, 1.0, 1.0,
            ], dtype="f4")

            self.vbo = self.ctx.buffer(vertices.tobytes())
            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, "2f 2f", "in_vert", "in_texcoord")],
            )

            info = self.ctx.info
            print("\nOpenGL renderer:", info.get("GL_RENDERER", "Unknown"))
            print("OpenGL version :", info.get("GL_VERSION", "Unknown"))
            print("Terrain shader is evaluated by the OpenGL GPU renderer.\n")

        def render(self, current_time, frame_time):
            self.ctx.clear(0.02, 0.02, 0.025, 1.0)

            self.program["u_resolution"].value = (
                float(self.wnd.buffer_size[0]),
                float(self.wnd.buffer_size[1]),
            )
            self.program["u_offset"].value = (self.cam_x, self.cam_y)
            self.program["u_freq"].value = self.freq
            self.program["u_octaves"].value = self.octaves
            self.program["u_persistence"].value = self.persistence
            self.program["u_lacunarity"].value = self.lacunarity
            self.program["u_sea_level"].value = self.sea_level
            self.program["u_ice"].value = 1 if self.ice else 0
            self.program["u_ice_strength"].value = self.ice_strength
            self.program["u_seed"].value = self.seed

            self.vao.render(mode=moderngl.TRIANGLE_STRIP)

        def resize(self, width, height):
            self.ctx.viewport = (0, 0, width, height)

        def mouse_drag_event(self, x, y, dx, dy):
            # dx/dy are per-event deltas, so accumulate them.
            height = max(1, self.wnd.height)
            self.cam_x -= dx * self.pan_speed / height * 2000.0
            self.cam_y += dy * self.pan_speed / height * 2000.0

        def mouse_position_event(self, x, y, dx, dy):
            now = time.monotonic()
            if now - self.last_hover_time < 0.08:
                return
            self.last_hover_time = now

            height = max(1, self.wnd.height)
            world_x = ((x - self.wnd.width * 0.5) / height) * 2000.0 + self.cam_x
            world_y = ((self.wnd.height * 0.5 - y) / height) * 2000.0 + self.cam_y

            raw = fbm_point(
                world_x, world_y, self.seed, self.octaves,
                self.persistence, self.lacunarity, self.freq,
            )
            elevation = shape_elevation(raw, self.sea_level)

            if self.ice:
                latitude = min(abs(world_y) / 2500.0, 1.0)
                edge = 1.0 - self.ice_strength
                ice_amount = 0.0 if latitude <= edge else (
                    (latitude - edge) / max(self.ice_strength, 1e-6)
                )
                elevation = min(1.0, elevation + ice_amount * 0.85 *
                                (1.0 - elevation))

            print(
                f"\rWorld ({world_x:8.1f}, {world_y:8.1f}) | "
                f"Elevation {elevation:.3f} | {biome_label(elevation):14s}",
                end="",
                flush=True,
            )


        # ModernGL-Window 3.x callback names. The aliases preserve 2.x support.
        on_render = render
        on_resize = resize
        on_mouse_drag_event = mouse_drag_event
        on_mouse_position_event = mouse_position_event


def run_gpu(args):
    if moderngl is None or mglw is None:
        print("Missing graphics dependencies.")
        print("Install them with:")
        print("  python -m pip install numpy moderngl moderngl-window")
        if _IMPORT_ERROR:
            print("\nImport error:", _IMPORT_ERROR)
        raise SystemExit(1)

    # run_window_config(args=...) expects CLI strings, not argparse.Namespace.
    # Store our own settings on the config class, then let moderngl-window
    # parse an empty argument list for its built-in window options.
    WorldMapWindow.window_size = (args.width, args.height)
    WorldMapWindow.terrain_seed = args.seed
    WorldMapWindow.terrain_octaves = args.octaves
    WorldMapWindow.terrain_persistence = args.persistence
    WorldMapWindow.terrain_lacunarity = args.lacunarity
    WorldMapWindow.terrain_freq = args.freq
    WorldMapWindow.terrain_sea_level = args.sea_level
    WorldMapWindow.terrain_ice = args.ice
    WorldMapWindow.terrain_ice_strength = 0.12
    WorldMapWindow.terrain_pan_speed = args.pan_speed

    mglw.run_window_config(WorldMapWindow, args=[])


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated procedural terrain viewer."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--octaves", type=int, default=7)
    parser.add_argument("--persistence", type=float, default=0.55)
    parser.add_argument("--lacunarity", type=float, default=2.0)
    parser.add_argument("--freq", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sea-level", type=float, default=0.42)
    parser.add_argument("--ice", action="store_true",
                        help="Enable stylized polar ice caps")
    parser.add_argument("--pan-speed", type=float, default=1.0)

    args = parser.parse_args()

    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if not 0.0 < args.persistence <= 1.0:
        parser.error("persistence must be in the range (0, 1]")
    if args.lacunarity <= 1.0:
        parser.error("lacunarity must be greater than 1")
    if args.freq <= 0.0:
        parser.error("frequency must be positive")
    if not 0.0 < args.sea_level < 1.0:
        parser.error("sea-level must be between 0 and 1")
    if args.octaves < 1 or args.octaves > 12:
        parser.error("octaves must be between 1 and 12")

    print("Terra AI — GPU Procedural Terrain")
    print(f"Window: {args.width} x {args.height}")
    print(f"Octaves: {args.octaves} | Persistence: {args.persistence}")
    print(f"Lacunarity: {args.lacunarity} | Frequency: {args.freq}")
    print(f"Seed: {args.seed} | Sea level: {args.sea_level}")
    print(f"Ice caps: {args.ice}")
    print("\nDrag with the left mouse button to explore. Move the mouse")
    print("to print the elevation and biome under the cursor.\n")

    run_gpu(args)


if __name__ == "__main__":
    main()
