import type { SinkInit } from './simulation.ts';

export interface HaloSpec {
  cx: number; cy: number; cz: number;
  v0: number; rc: number;
}

export type GalaxyType = 'spiral' | 'lenticular';

export interface GalaxyIC {
  pos:   Float32Array;   // 3 × nStars
  vel:   Float32Array;   // 3 × nStars
  mass:  Float32Array;   // nStars
  sinks: SinkInit[];
  halos: HaloSpec[];
}

interface Vec3 { x: number; y: number; z: number }

interface GalaxyParams {
  n:                  number;
  M_total:            number;
  central_point_mass: number;
  bulge_mass_frac:    number;
  bulge_a:            number;
  disk_Rd:            number;
  disk_hz:            number;
  n_arms:             number;  // 0 → lenticular
  arm_strength:       number;
  arm_pitch_rad:      number;
  sigma_R_frac:       number;
  sigma_z_frac:       number;
  R_out_factor:       number;
  prograde:           boolean;
  tilt_x_rad:         number;
  center:             Vec3;
  bulk_velocity:      Vec3;
  halo_v0:            number;  // 0 disables
  halo_rc:            number;
  seed:               number;
}

const G = 1.0;
const TWO_PI = 2 * Math.PI;

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard Box-Muller (one sample, discard the pair).
function normal(u: () => number): number {
  const u1 = Math.max(u(), 1e-12);
  const u2 = u();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(TWO_PI * u2);
}

// gamma(shape=2, scale=θ) as the sum of two exponentials with mean θ.
function gamma2(u: () => number, scale: number): number {
  const u1 = Math.max(u(), 1e-12);
  const u2 = Math.max(u(), 1e-12);
  return -scale * (Math.log(u1) + Math.log(u2));
}

// Spherical enclosed mass: bulge (Plummer) + exponential disk + central
// point mass. Mirrors enclosedMass() in initial_conditions.cpp.
function enclosedMass(g: GalaxyParams, R: number): number {
  const Mb = g.M_total * g.bulge_mass_frac;
  const Md = g.M_total * (1 - g.bulge_mass_frac);
  const a  = g.bulge_a;
  const M_b_enc = Mb * (R * R * R) / Math.pow(R * R + a * a, 1.5);
  const x       = R / Math.max(g.disk_Rd, 1e-6);
  const M_d_enc = Md * (1 - (1 + x) * Math.exp(-x));
  return M_b_enc + M_d_enc + g.central_point_mass;
}

function circularSpeed(g: GalaxyParams, R: number): number {
  const R_floor = 0.02 * g.disk_Rd;
  const Re = Math.max(R, R_floor);
  let vc2 = G * enclosedMass(g, Re) / Re;
  if (g.halo_v0 > 0) {
    const R2  = Re * Re;
    const rc2 = g.halo_rc * g.halo_rc;
    vc2 += (g.halo_v0 * g.halo_v0) * R2 / (R2 + rc2);
  }
  return Math.sqrt(Math.max(vc2, 0));
}

function samplePlummerBulge(
  u: () => number,
  M: number, a: number, n: number,
  outPos: Float32Array, outVel: Float32Array, base: number,
): void {
  for (let i = 0; i < n; i++) {
    const uu = Math.min(u(), 0.999999);
    const r  = a / Math.sqrt(Math.pow(uu, -2 / 3) - 1);

    const ct = 1 - 2 * u();
    const st = Math.sqrt(Math.max(0, 1 - ct * ct));
    const ph = TWO_PI * u();
    const px = r * st * Math.cos(ph);
    const py = r * st * Math.sin(ph);
    const pz = r * ct;

    // q ∝ q²(1−q²)^(7/2), rejection-sampled (Aarseth–Hénon–Wielen).
    let q = 0;
    for (;;) {
      const x1 = u();
      const x2 = u() * 0.1;
      const om = Math.max(0, 1 - x1 * x1);
      if (x2 <= x1 * x1 * Math.pow(om, 3.5)) { q = x1; break; }
    }
    const v_esc = Math.sqrt(2 * G * M / Math.sqrt(r * r + a * a));
    const speed = q * v_esc;

    const vct = 1 - 2 * u();
    const vst = Math.sqrt(Math.max(0, 1 - vct * vct));
    const vph = TWO_PI * u();
    const vx  = speed * vst * Math.cos(vph);
    const vy  = speed * vst * Math.sin(vph);
    const vz  = speed * vct;

    const w = base + i;
    outPos[3 * w + 0] = px; outPos[3 * w + 1] = py; outPos[3 * w + 2] = pz;
    outVel[3 * w + 0] = vx; outVel[3 * w + 1] = vy; outVel[3 * w + 2] = vz;
  }
}

function sampleDisk(
  g: GalaxyParams,
  u: () => number,
  outPos: Float32Array, outVel: Float32Array,
  base: number, n: number,
): void {
  const R_max         = g.R_out_factor * g.disk_Rd;
  const pitch         = Math.max(g.arm_pitch_rad, 1e-3);
  const inv_tan_pitch = 1 / Math.tan(pitch);
  const R_ref         = g.disk_Rd;
  const A             = g.n_arms > 0 ? Math.min(0.95, Math.max(0, g.arm_strength)) : 0;

  for (let i = 0; i < n; i++) {
    let R = 0;
    do { R = gamma2(u, g.disk_Rd); } while (R > R_max);

    let phi = 0;
    if (g.n_arms > 0) {
      for (;;) {
        phi = TWO_PI * u();
        const psi  = Math.log(Math.max(R / R_ref, 1e-6)) * inv_tan_pitch;
        const dens = 1 + A * Math.cos(g.n_arms * (phi - psi));
        if (u() * (1 + A) <= dens) break;
      }
    } else {
      phi = TWO_PI * u();
    }

    let uz = 2 * u() - 1;
    uz = Math.max(-0.9999, Math.min(0.9999, uz));
    const z = 0.5 * g.disk_hz * Math.log((1 + uz) / (1 - uz));

    const cp = Math.cos(phi);
    const sp = Math.sin(phi);
    const x  = R * cp;
    const y  = R * sp;

    const vc   = circularSpeed(g, R);
    const sigR = g.sigma_R_frac * vc;
    const sigT = g.sigma_R_frac * vc * 0.707;
    const sigZ = g.sigma_z_frac * vc;

    // Asymmetric-drift correction (see comment in initial_conditions.cpp).
    let v_rot = g.prograde ? +vc : -vc;
    if (vc > 1e-6) {
      const ad = 0.5 * sigR * sigR / vc;
      v_rot -= g.prograde ? ad : -ad;
    }

    const vR = sigR * normal(u);
    const vT = v_rot + sigT * normal(u);
    const vz = sigZ * normal(u);

    const vx = vR * cp - vT * sp;
    const vy = vR * sp + vT * cp;

    const w = base + i;
    outPos[3 * w + 0] = x;  outPos[3 * w + 1] = y;  outPos[3 * w + 2] = z;
    outVel[3 * w + 0] = vx; outVel[3 * w + 1] = vy; outVel[3 * w + 2] = vz;
  }
}

function sampleGalaxyInto(
  g:       GalaxyParams,
  outPos:  Float32Array,
  outVel:  Float32Array,
  outMass: Float32Array,
  offset:  number,
): void {
  const u = mulberry32(g.seed);

  let n_bulge = Math.round(g.n * g.bulge_mass_frac);
  if (n_bulge > g.n) n_bulge = g.n;
  const n_disk  = g.n - n_bulge;
  const m_per   = g.M_total / g.n;
  const M_bulge = m_per * n_bulge;

  // Sample into a galaxy-local frame first, tilt + translate afterward.
  const tmpPos = new Float32Array(3 * g.n);
  const tmpVel = new Float32Array(3 * g.n);
  samplePlummerBulge(u, M_bulge, g.bulge_a, n_bulge, tmpPos, tmpVel, 0);
  sampleDisk(g, u, tmpPos, tmpVel, n_bulge, n_disk);

  const ct = Math.cos(g.tilt_x_rad);
  const st = Math.sin(g.tilt_x_rad);
  for (let i = 0; i < g.n; i++) {
    const px = tmpPos[3 * i + 0], py = tmpPos[3 * i + 1], pz = tmpPos[3 * i + 2];
    const vx = tmpVel[3 * i + 0], vy = tmpVel[3 * i + 1], vz = tmpVel[3 * i + 2];
    const prx = px,               pry = py * ct - pz * st, prz = py * st + pz * ct;
    const vrx = vx,               vry = vy * ct - vz * st, vrz = vy * st + vz * ct;
    const w = offset + i;
    outPos[3 * w + 0] = prx + g.center.x;
    outPos[3 * w + 1] = pry + g.center.y;
    outPos[3 * w + 2] = prz + g.center.z;
    outVel[3 * w + 0] = vrx + g.bulk_velocity.x;
    outVel[3 * w + 1] = vry + g.bulk_velocity.y;
    outVel[3 * w + 2] = vrz + g.bulk_velocity.z;
    outMass[w] = m_per;
  }
}

function spiralPreset(
  n: number, M_stars: number, M_sink: number,
  halo_v0: number, halo_rc: number, seed: number,
): GalaxyParams {
  return {
    n, M_total: M_stars, central_point_mass: M_sink,
    bulge_mass_frac: 0.15, bulge_a: 0.15,
    disk_Rd: 0.40,          disk_hz: 0.03,
    n_arms: 2, arm_strength: 0.55, arm_pitch_rad: 0.35,
    sigma_R_frac: 0.10, sigma_z_frac: 0.05,
    R_out_factor: 5.0,
    prograde: true, tilt_x_rad: 0.15,
    center: { x: 0, y: 0, z: 0 },
    bulk_velocity: { x: 0, y: 0, z: 0 },
    halo_v0, halo_rc, seed,
  };
}

function lenticularPreset(
  n: number, M_stars: number, M_sink: number,
  halo_v0: number, halo_rc: number, seed: number,
): GalaxyParams {
  return {
    n, M_total: M_stars, central_point_mass: M_sink,
    bulge_mass_frac: 0.35, bulge_a: 0.20,
    disk_Rd: 0.35,          disk_hz: 0.06,
    n_arms: 0, arm_strength: 0.0, arm_pitch_rad: 0.35,
    sigma_R_frac: 0.25, sigma_z_frac: 0.10,
    R_out_factor: 4.5,
    prograde: false, tilt_x_rad: 0.9,
    center: { x: 0, y: 0, z: 0 },
    bulk_velocity: { x: 0, y: 0, z: 0 },
    halo_v0, halo_rc, seed,
  };
}

export interface CollidingGalaxiesOpts {
  nStars:          number;
  separation:      number;
  approachSpeed:   number;
  impactParameter: number;
  sinkMassFrac:    number;
  haloV0:          number;
  haloRc:          number;
  seed:            number;
}

export function generateCollidingGalaxies(o: CollidingGalaxiesOpts): GalaxyIC {
  const n_a = Math.floor(o.nStars / 2);
  const n_b = o.nStars - n_a;

  const half_s = 0.5 * o.separation;
  const half_b = 0.5 * o.impactParameter;
  const half_v = 0.5 * o.approachSpeed;
  const M_galaxy = 1.0;
  const M_sink   = Math.max(0, o.sinkMassFrac) * M_galaxy;
  const M_stars  = M_galaxy - M_sink;

  const spiral = spiralPreset(n_a, M_stars, M_sink, o.haloV0, o.haloRc, o.seed);
  spiral.center        = { x: -half_s, y: -half_b, z: 0 };
  spiral.bulk_velocity = { x: +half_v, y:  0,      z: 0 };

  // 32-bit analog of the C++ u64 golden-ratio xor; we only need a distinct
  // stream from the spiral, not bit-for-bit-equivalent output.
  const seedB = (o.seed ^ 0x9E3779B9) >>> 0;
  const lent = lenticularPreset(n_b, M_stars, M_sink, o.haloV0, o.haloRc, seedB);
  lent.center        = { x: +half_s, y: +half_b, z: 0 };
  lent.bulk_velocity = { x: -half_v, y:  0,      z: 0 };

  const pos  = new Float32Array(3 * o.nStars);
  const vel  = new Float32Array(3 * o.nStars);
  const mass = new Float32Array(o.nStars);
  sampleGalaxyInto(spiral, pos, vel, mass, 0);
  sampleGalaxyInto(lent,   pos, vel, mass, n_a);

  const sinks: SinkInit[] = [];
  if (M_sink > 0) {
    sinks.push({
      x: spiral.center.x, y: spiral.center.y, z: spiral.center.z,
      vx: spiral.bulk_velocity.x, vy: spiral.bulk_velocity.y, vz: spiral.bulk_velocity.z,
      mass: M_sink,
    });
    sinks.push({
      x: lent.center.x, y: lent.center.y, z: lent.center.z,
      vx: lent.bulk_velocity.x, vy: lent.bulk_velocity.y, vz: lent.bulk_velocity.z,
      mass: M_sink,
    });
  }

  const halos: HaloSpec[] = [];
  if (o.haloV0 > 0) {
    halos.push({ cx: spiral.center.x, cy: spiral.center.y, cz: spiral.center.z, v0: o.haloV0, rc: o.haloRc });
    halos.push({ cx: lent.center.x,   cy: lent.center.y,   cz: lent.center.z,   v0: o.haloV0, rc: o.haloRc });
  }

  return { pos, vel, mass, sinks, halos };
}

export type ScenarioMode = 'collision' | 'grazing' | 'orbit';

export interface MultiGalaxyOpts {
  nStars:          number;
  nGalaxies:       number;      // 1..5
  mode:            ScenarioMode;
  separation:      number;      // radius of the ring galaxies sit on (diameter for N=2)
  approachSpeed:   number;      // inward speed for collision/grazing
  impactParameter: number;      // tangential miss distance for grazing
  sinkMassFrac:    number;
  haloV0:          number;
  haloRc:          number;
  seed:            number;
}

// Circular-orbit speed for N equal-mass points arranged in a regular N-gon of
// radius R around their common barycenter, each body with mass m.
//   v² = G·m / (4R) · Σ_{k=1..N−1} 1 / sin(π k / N)
// Derived from radial-force balance; for N=2 reduces to v² = Gm/(4R).
function ringOrbitSpeed(N: number, m: number, R: number, G: number): number {
  if (N < 2 || R <= 0) return 0;
  let sum = 0;
  for (let k = 1; k < N; k++) sum += 1 / Math.sin((Math.PI * k) / N);
  return Math.sqrt((G * m) / (4 * R) * sum);
}

export function generateMultiGalaxyScenario(o: MultiGalaxyOpts): GalaxyIC {
  const N = Math.max(1, Math.min(5, Math.floor(o.nGalaxies)));
  const M_galaxy = 1.0;
  const M_sink   = Math.max(0, o.sinkMassFrac) * M_galaxy;
  const M_stars  = M_galaxy - M_sink;

  // Split stars as evenly as possible; last galaxy takes the remainder.
  const base = Math.floor(o.nStars / N);
  const counts: number[] = [];
  let assigned = 0;
  for (let i = 0; i < N; i++) {
    const n = i === N - 1 ? o.nStars - assigned : base;
    counts.push(n);
    assigned += n;
  }

  const pos  = new Float32Array(3 * o.nStars);
  const vel  = new Float32Array(3 * o.nStars);
  const mass = new Float32Array(o.nStars);
  const sinks: SinkInit[] = [];
  const halos: HaloSpec[] = [];

  // N=1 is a single galaxy at rest; ignore mode.
  if (N === 1) {
    const g = spiralPreset(counts[0], M_stars, M_sink, o.haloV0, o.haloRc, o.seed);
    g.center        = { x: 0, y: 0, z: 0 };
    g.bulk_velocity = { x: 0, y: 0, z: 0 };
    g.tilt_x_rad    = 0;
    sampleGalaxyInto(g, pos, vel, mass, 0);
    if (M_sink > 0) sinks.push({ x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: M_sink });
    if (o.haloV0 > 0) halos.push({ cx: 0, cy: 0, cz: 0, v0: o.haloV0, rc: o.haloRc });
    return { pos, vel, mass, sinks, halos };
  }

  const R = Math.max(0.5 * o.separation, 1e-3);
  const vOrbit = ringOrbitSpeed(N, M_galaxy, R, G);
  // Angle from the inward (radial-to-center) direction at which velocity is
  // aimed so bodies pass the origin at distance ≈ impactParameter.
  const grazeAng = Math.asin(Math.max(-1, Math.min(1, o.impactParameter / (2 * R))));

  let offset = 0;
  for (let i = 0; i < N; i++) {
    const theta = (2 * Math.PI * i) / N;
    const ct = Math.cos(theta), st = Math.sin(theta);
    const cx = R * ct, cy = R * st, cz = 0;

    // Radial outward unit vector at this position; inward is its negation.
    const rx = ct, ry = st;
    // Tangential unit (counter-clockwise around +z).
    const tx = -st, ty = ct;

    let vx = 0, vy = 0, vz = 0;
    if (o.mode === 'collision') {
      vx = -o.approachSpeed * rx;
      vy = -o.approachSpeed * ry;
    } else if (o.mode === 'grazing') {
      // Inward velocity rotated by ±grazeAng; alternate sign between galaxies
      // so N=2 reproduces the classic "pass each other" configuration.
      const s = i % 2 === 0 ? +1 : -1;
      const cosA = Math.cos(grazeAng), sinA = s * Math.sin(grazeAng);
      vx = o.approachSpeed * (-rx * cosA + tx * sinA);
      vy = o.approachSpeed * (-ry * cosA + ty * sinA);
    } else {
      vx = vOrbit * tx;
      vy = vOrbit * ty;
    }

    const seedI = (o.seed ^ Math.imul(i + 1, 0x9E3779B9)) >>> 0;
    const nI = counts[i];
    const isSpiral = i % 2 === 0;
    const g = isSpiral
      ? spiralPreset(nI, M_stars, M_sink, o.haloV0, o.haloRc, seedI)
      : lenticularPreset(nI, M_stars, M_sink, o.haloV0, o.haloRc, seedI);
    g.center        = { x: cx, y: cy, z: cz };
    g.bulk_velocity = { x: vx, y: vy, z: vz };
    // Small per-galaxy tilt so they don't all sit exactly coplanar.
    g.tilt_x_rad    = isSpiral ? 0.15 + 0.1 * i : 0.9 - 0.1 * i;
    sampleGalaxyInto(g, pos, vel, mass, offset);
    offset += nI;

    if (M_sink > 0) sinks.push({ x: cx, y: cy, z: cz, vx, vy, vz, mass: M_sink });
    if (o.haloV0 > 0) halos.push({ cx, cy, cz, v0: o.haloV0, rc: o.haloRc });
  }

  return { pos, vel, mass, sinks, halos };
}

export interface SingleGalaxyOpts {
  type:         GalaxyType;
  nStars:       number;
  sinkMassFrac: number;
  haloV0:       number;
  haloRc:       number;
  seed:         number;
}

export function generateSingleGalaxy(o: SingleGalaxyOpts): GalaxyIC {
  const M_galaxy = 1.0;
  const M_sink   = Math.max(0, o.sinkMassFrac) * M_galaxy;
  const M_stars  = M_galaxy - M_sink;

  const g = o.type === 'spiral'
    ? spiralPreset(o.nStars, M_stars, M_sink, o.haloV0, o.haloRc, o.seed)
    : lenticularPreset(o.nStars, M_stars, M_sink, o.haloV0, o.haloRc, o.seed);
  g.center        = { x: 0, y: 0, z: 0 };
  g.bulk_velocity = { x: 0, y: 0, z: 0 };
  g.tilt_x_rad    = 0;

  const pos  = new Float32Array(3 * o.nStars);
  const vel  = new Float32Array(3 * o.nStars);
  const mass = new Float32Array(o.nStars);
  sampleGalaxyInto(g, pos, vel, mass, 0);

  const sinks: SinkInit[] = [];
  if (M_sink > 0) {
    sinks.push({ x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: M_sink });
  }
  const halos: HaloSpec[] = [];
  if (o.haloV0 > 0) {
    halos.push({ cx: 0, cy: 0, cz: 0, v0: o.haloV0, rc: o.haloRc });
  }
  return { pos, vel, mass, sinks, halos };
}
