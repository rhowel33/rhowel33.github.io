// Particle + sink rendering. Two pipelines share this module:
//   point_vs / point_fs — stars, tinted by nearest-sink gravitational redshift
//   sink_vs  / sink_fs  — sinks, drawn as fiery accretion-disk sprites
//
// Per-particle data lives in the same storage buffer the compute kernels
// write. The star draw issues instances 0..nStars-1; the sink draw uses
// firstInstance=nStars so instance_index directly indexes the sink slot.

const MAX_SINKS: u32 = 5u;

struct Uniforms {
  viewProj:     mat4x4<f32>,
  pointNdc:     vec2<f32>,
  sinkPointNdc: vec2<f32>,
  perGalaxy:    u32,                    // stars per galaxy (last slot absorbs remainder)
  galaxyCount:  u32,                    // 1..5
  nSinks:       u32,
  _pad:         u32,
  sinks:        array<vec4<f32>, 5>,    // xyz = position, w = Schwarzschild radius
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;

fn cornerAt(vi: u32) -> vec2<f32> {
  var cs = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
  );
  return cs[vi];
}

// ─── Stars ──────────────────────────────────────────────────────────────

struct StarVOut {
  @builtin(position) pos:   vec4<f32>,
  @location(0)       uv:    vec2<f32>,
  @location(1)       color: vec3<f32>,
  @location(2)       alpha: f32,
};

@vertex
fn point_vs(@builtin(vertex_index) vi: u32,
            @builtin(instance_index) ii: u32) -> StarVOut {
  var out: StarVOut;
  let p = positions[ii];

  // Dead particles (mass = 0 after accretion) → degenerate off-screen.
  if (p.w <= 0.0) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.color = vec3<f32>(0.0);
    out.alpha = 0.0;
    return out;
  }

  let clip = u.viewProj * vec4<f32>(p.xyz, 1.0);
  if (clip.w <= 0.0) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.color = vec3<f32>(0.0);
    out.alpha = 0.0;
    return out;
  }

  // N galaxies: pick colour from a fixed palette by which slice the instance
  // falls into. Palette alternates spiral / lenticular tones.
  var palette = array<vec3<f32>, 5>(
    vec3<f32>(0.62, 0.78, 1.00),   // spiral: cool blue-white
    vec3<f32>(1.00, 0.85, 0.55),   // lenticular: amber
    vec3<f32>(0.70, 1.00, 0.70),   // mint
    vec3<f32>(1.00, 0.60, 0.85),   // rose
    vec3<f32>(0.85, 0.70, 1.00),   // violet
  );
  var slot: u32 = 0u;
  if (u.perGalaxy > 0u) { slot = ii / u.perGalaxy; }
  if (slot >= u.galaxyCount) { slot = u.galaxyCount - 1u; }
  let base = palette[slot];

  // Gravitational redshift (worldline approx) — z = 1 − √(1 − rs/r), using
  // whichever sink's potential is deepest.
  var z_red: f32 = 0.0;
  for (var s: u32 = 0u; s < u.nSinks; s = s + 1u) {
    let sink = u.sinks[s];
    let r  = distance(p.xyz, sink.xyz);
    let rs = sink.w;
    var z: f32 = 1.0;
    if (r > rs * 1.01) {
      z = 1.0 - sqrt(max(0.0, 1.0 - rs / r));
    }
    z_red = max(z_red, z);
  }

  let red_tint = vec3<f32>(0.85, 0.18, 0.06);
  let tinted   = mix(base, red_tint, z_red);
  let color    = tinted * (1.0 - z_red * 0.92);

  let corner = cornerAt(vi);
  let offset = corner * u.pointNdc * 0.5 * clip.w;
  out.pos    = vec4<f32>(clip.xy + offset, clip.z, clip.w);
  out.uv     = corner;
  out.color  = color;
  out.alpha  = 1.0;
  return out;
}

@fragment
fn point_fs(in: StarVOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let a = exp(-r2 * 5.5) * in.alpha;
  return vec4<f32>(in.color * a, a);
}

// ─── Sinks ──────────────────────────────────────────────────────────────

struct SinkVOut {
  @builtin(position) pos:       vec4<f32>,
  @location(0)       uv:        vec2<f32>,
  @location(1)       intensity: f32,
};

@vertex
fn sink_vs(@builtin(vertex_index) vi: u32,
           @builtin(instance_index) ii: u32) -> SinkVOut {
  var out: SinkVOut;
  let p = positions[ii];   // caller sets firstInstance = nStars, so ii is the sink slot

  if (p.w <= 0.0) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.intensity = 0.0;
    return out;
  }

  let clip = u.viewProj * vec4<f32>(p.xyz, 1.0);
  if (clip.w <= 0.0) {
    out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.intensity = 0.0;
    return out;
  }

  let corner = cornerAt(vi);
  let offset = corner * u.sinkPointNdc * 0.5 * clip.w;
  out.pos    = vec4<f32>(clip.xy + offset, clip.z, clip.w);
  out.uv     = corner;
  out.intensity = clamp(p.w * 40.0, 0.5, 4.0);
  return out;
}

@fragment
fn sink_fs(in: SinkVOut) -> @location(0) vec4<f32> {
  let uv_len = length(in.uv);
  if (uv_len > 1.0) { discard; }
  // Reproject uv into Metal's [0, 0.5] point-coord range so the profile
  // constants below match shaders/render.metal literally.
  let r = uv_len * 0.5;

  let ring    = exp(-pow((r - 0.30) * 9.0, 2.0));
  let halo    = exp(-pow(r * 3.0, 2.0)) * 0.35;
  let horizon = smoothstep(0.18, 0.00, r);

  let hot   = vec3<f32>(2.4, 1.3, 0.35);
  let warm  = vec3<f32>(1.8, 0.7, 0.15);
  let glow  = mix(warm, hot, ring) * (ring + halo);
  let color = glow * (1.0 - horizon * 0.95);

  let alpha = ring + halo;
  return vec4<f32>(color * in.intensity, alpha);
}
