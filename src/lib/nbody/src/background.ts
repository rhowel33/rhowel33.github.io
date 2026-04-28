// Embeddable n-body background. Wraps the engine (Simulation + Renderer)
// in a self-contained controller suitable for dropping behind website
// content: no DOM controls, slow auto-orbit camera, graceful no-WebGPU bail,
// pause when the tab is hidden, ResizeObserver for responsive resize.
//
// Usage:
//   const ctrl = await startNBodyBackground({ canvas });
//   if (!ctrl) {  /* WebGPU not available — fall back to static bg */  }
//   // later: ctrl.destroy()

import { Simulation, type SinkInit } from './simulation.ts';
import { createRenderer, type Renderer } from './renderer.ts';
import { generateSingleGalaxy } from './initial_conditions.ts';

/**
 * Live, externally-mutable camera state. Caller can pass an instance in via
 * BackgroundOptions.state and mutate it from anywhere — sliders, animation
 * frames, whatever. The engine reads the live fields each frame, so updates
 * take effect on the next render. Auto-rotate is just `azimuth +=
 * autoRotateSpeed * dt` inside the engine loop, so setting `autoRotateSpeed`
 * to 0 freezes the auto-orbit while leaving manual control alive.
 */
export interface BackgroundState {
  azimuth:         number;
  elevation:       number;
  distance:        number;
  target:          [number, number, number];
  autoRotateSpeed: number;
}

export interface BackgroundOptions {
  canvas:           HTMLCanvasElement;
  /** Live camera state shared with caller. If omitted, internal defaults. */
  state?:           BackgroundState;
  /** Particle count. ~12k is the sweet spot for a battery-friendly background. */
  nBodies?:         number;
  /** Auto-orbit speed in radians/second. */
  autoRotateSpeed?: number;
  /** Camera radius from origin. */
  cameraDistance?:  number;
  /** Camera elevation in radians (0 = equatorial, π/2 = pole). */
  cameraElevation?: number;
  /** Sim steps per rendered frame. 1 keeps things calm. */
  stepsPerFrame?:   number;
  /** RNG seed for reproducible initial conditions. */
  seed?:            number;
  /** Galaxy type. 'spiral' has visible arms; 'lenticular' is smoother. */
  galaxyType?:      'spiral' | 'lenticular';
  /** When set, dispatch tree-based force solver. Off by default; cheap N²
   *  is faster at the small body counts we use as a background. */
  useBarnesHut?:    boolean;
}

export interface BackgroundController {
  destroy(): void;
  pause(): void;
  resume(): void;
  /** Live camera state. Mutate fields directly; changes take effect next frame. */
  state: BackgroundState;
  /** Toggle click-to-place user body. Off cancels any in-progress drag and
   *  clears the existing user body so it doesn't keep perturbing the sim
   *  while the visitor is on a non-interactive route. */
  setInteractive(active: boolean): void;
}

const DEFAULTS = {
  nBodies:         12000,
  autoRotateSpeed: 0.06,
  cameraDistance:  3.8,
  cameraElevation: 0.55,   // ≈ 31° tilt
  stepsPerFrame:   1,
  seed:            42,
  galaxyType:      'spiral' as const,
  useBarnesHut:    false,
};

export async function startNBodyBackground(
  opts: BackgroundOptions,
): Promise<BackgroundController | null> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  if (!adapter) return null;

  let device: GPUDevice;
  try { device = await adapter.requestDevice(); } catch { return null; }

  const ctx = opts.canvas.getContext('webgpu');
  if (!ctx) { device.destroy(); return null; }
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'premultiplied' });

  const cfg = { ...DEFAULTS, ...opts };

  // ---- Initial conditions (single galaxy at origin) -----------------------
  const ic = generateSingleGalaxy({
    type:           cfg.galaxyType,
    nStars:         cfg.nBodies,
    sinkMassFrac:   0.02,
    haloV0:         1.0,
    haloRc:         0.3,
    seed:           cfg.seed,
  });

  // Reserve one extra sink slot for the user-placed body. Parked far away with
  // mass=0 until the user clicks; activated via sim.setUserBody().
  const PARKED_USER_BODY: SinkInit = {
    x: 1e6, y: 1e6, z: 1e6, vx: 0, vy: 0, vz: 0, mass: 0,
  };
  const allSinks: SinkInit[] = [...ic.sinks, PARKED_USER_BODY];

  // 2× central sink mass (or a sane default if the IC has no sinks at all).
  const centralSinkMass = ic.sinks.length > 0 ? ic.sinks[0].mass : 0.02;
  const userBodyMass    = 2 * centralSinkMass;

  // ---- Simulation ---------------------------------------------------------
  const sim = new Simulation(
    device,
    ic.pos.length / 3,
    allSinks.length,
    {
      dt:            0.01,
      softening:     0.05,
      G:             1.0,
      c:             1.0,
      pnAmp:         1.0,
      theta:         0.7,
      useBarnesHut:  cfg.useBarnesHut,
      captureRadius: 0.03,
      halos:         ic.halos,
    },
  );
  sim.setState(ic.pos, ic.vel, ic.mass, allSinks);

  // ---- Renderer -----------------------------------------------------------
  const renderer: Renderer = createRenderer(
    device, format, sim.posBuf, sim.nStars, sim.nSinks,
  );

  // ---- Camera state (live, externally mutable) ---------------------------
  // If the caller provided one, share it; otherwise create a private object
  // initialized from defaults.
  const state: BackgroundState = opts.state ?? {
    azimuth:         0,
    elevation:       cfg.cameraElevation,
    distance:        cfg.cameraDistance,
    target:          [0, 0, 0],
    autoRotateSpeed: cfg.autoRotateSpeed,
  };

  const canvas = opts.canvas;
  const eye: [number, number, number] = [0, 0, 0];
  const up:  [number, number, number] = [0, 1, 0];

  const dprFor = (): number => Math.min(window.devicePixelRatio || 1, 2);
  const resize = (): void => {
    const dpr = dprFor();
    const w = Math.max(1, Math.floor(canvas.clientWidth  * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w)  canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;
  };
  resize();

  const ro = new ResizeObserver(() => resize());
  ro.observe(canvas);

  // ---- Middle-click pan ---------------------------------------------------
  // Drag-the-world: dragging the mouse to the right slides the universe
  // right, which means our camera target slides LEFT in world coords.
  let panning = false;
  let lastPx = 0, lastPy = 0;

  // ---- Left-click: place user body ---------------------------------------
  // Click anywhere on the canvas to drop a heavy body at the cursor's world
  // position; release with cursor moving to throw it. Only one user body
  // exists at a time — clicking again replaces it. Mass = 2× central sink.
  const FOVY = Math.PI / 3;
  const VEL_WINDOW_MS = 80;  // sample window for release-velocity smoothing

  // Cast a ray from the camera through the cursor pixel and intersect it
  // with the galaxy disk plane (z=0). Falls back to the screen-aligned
  // plane through the camera target when the ray is near-parallel to the
  // disk (e.g. extreme edge-on views).
  const cursorToWorld = (clientX: number, clientY: number):
    [number, number, number] => {
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    const nx =  ((clientX - rect.left) / w) * 2 - 1;
    const ny = -(((clientY - rect.top)  / h) * 2 - 1);

    const cosE = Math.cos(state.elevation);
    const sinE = Math.sin(state.elevation);
    const sinA = Math.sin(state.azimuth);
    const cosA = Math.cos(state.azimuth);

    const eyeX = state.target[0] + state.distance * cosE * sinA;
    const eyeY = state.target[1] + state.distance * sinE;
    const eyeZ = state.target[2] + state.distance * cosE * cosA;

    // Forward = (target − eye) / |.|
    const fx = -cosE * sinA, fy = -sinE, fz = -cosE * cosA;
    // right = forward × up, up = (0,1,0)
    let sx = fy * 0 - fz * 1;
    let sy = fz * 0 - fx * 0;
    let sz = fx * 1 - fy * 0;
    const sl = Math.hypot(sx, sy, sz) || 1;
    sx /= sl; sy /= sl; sz /= sl;
    // trueUp = right × forward
    const ux = sy * fz - sz * fy;
    const uy = sz * fx - sx * fz;
    const uz = sx * fy - sy * fx;

    const halfH = Math.tan(FOVY / 2);
    const halfW = halfH * (w / h);

    let dx = fx + nx * halfW * sx + ny * halfH * ux;
    let dy = fy + nx * halfW * sy + ny * halfH * uy;
    let dz = fz + nx * halfW * sz + ny * halfH * uz;
    const dl = Math.hypot(dx, dy, dz) || 1;
    dx /= dl; dy /= dl; dz /= dl;

    if (Math.abs(dz) > 1e-3) {
      const t = -eyeZ / dz;
      if (t > 0.01 && t < 200) {
        return [eyeX + t * dx, eyeY + t * dy, 0];
      }
    }
    const t = state.distance;
    return [eyeX + t * dx, eyeY + t * dy, eyeZ + t * dz];
  };

  let placing = false;
  let interactive = false;
  let prevAutoRotate = 0;
  type Sample = { t: number; x: number; y: number; z: number };
  let placeHistory: Sample[] = [];

  const trimHistory = (now: number): void => {
    const cutoff = now - VEL_WINDOW_MS;
    while (placeHistory.length > 1 && placeHistory[0].t < cutoff) {
      placeHistory.shift();
    }
  };

  const onPointerDown = (e: PointerEvent): void => {
    if (e.button === 1) {
      e.preventDefault();
      panning = true;
      lastPx = e.clientX;
      lastPy = e.clientY;
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button === 0) {
      if (!interactive) return;
      // Don't hijack clicks on links / buttons sitting above the canvas.
      const target = e.target;
      if (target instanceof Element && target !== canvas) return;
      e.preventDefault();
      placing = true;
      // Freeze auto-rotate so cursor world-positions sampled across the drag
      // share a consistent camera frame; resumes on release.
      prevAutoRotate = state.autoRotateSpeed;
      state.autoRotateSpeed = 0;
      const [wx, wy, wz] = cursorToWorld(e.clientX, e.clientY);
      placeHistory = [{ t: performance.now(), x: wx, y: wy, z: wz }];
      canvas.setPointerCapture(e.pointerId);
    }
  };
  const onPointerMove = (e: PointerEvent): void => {
    if (panning) {
      const dx = e.clientX - lastPx;
      const dy = e.clientY - lastPy;
      lastPx = e.clientX;
      lastPy = e.clientY;
      // Camera basis: right = (cosA, 0, -sinA);  up = (-sinA·sinE, cosE, -cosA·sinE)
      const cosE = Math.cos(state.elevation);
      const sinE = Math.sin(state.elevation);
      const sinA = Math.sin(state.azimuth);
      const cosA = Math.cos(state.azimuth);
      const rX =  cosA, rY = 0,    rZ = -sinA;
      const uX = -sinA * sinE, uY = cosE, uZ = -cosA * sinE;
      // Pan scale: 1 px ≈ distance / canvas-height world units, so panning
      // feels speed-correct at any zoom.
      const scale = state.distance / Math.max(1, canvas.clientHeight);
      state.target[0] += (-dx * rX + dy * uX) * scale;
      state.target[1] += (-dx * rY + dy * uY) * scale;
      state.target[2] += (-dx * rZ + dy * uZ) * scale;
    }
    if (placing) {
      const now = performance.now();
      const [wx, wy, wz] = cursorToWorld(e.clientX, e.clientY);
      placeHistory.push({ t: now, x: wx, y: wy, z: wz });
      trimHistory(now);
    }
  };
  const onPointerUp = (e: PointerEvent): void => {
    if (e.button === 1 && panning) {
      panning = false;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
      return;
    }
    if (e.button === 0 && placing) {
      placing = false;
      const now = performance.now();
      const [wx, wy, wz] = cursorToWorld(e.clientX, e.clientY);
      placeHistory.push({ t: now, x: wx, y: wy, z: wz });
      trimHistory(now);

      // Velocity = (latest − oldest in trimmed window) / Δt. Falls back to
      // zero for a pure click (no movement).
      let vx = 0, vy = 0, vz = 0;
      if (placeHistory.length >= 2) {
        const first = placeHistory[0];
        const last  = placeHistory[placeHistory.length - 1];
        const dt = (last.t - first.t) / 1000;
        if (dt > 0.005) {
          vx = (last.x - first.x) / dt;
          vy = (last.y - first.y) / dt;
          vz = (last.z - first.z) / dt;
        }
      }

      // Slot 0 is the central anchor; slot 1 is the replaceable user body.
      // After a disruption event both are mass=0 — the next click reseeds
      // slot 0 (with the original central-sink mass) and only later clicks
      // start filling/replacing the user-body slot.
      const centralAlive = sim.getSinkMass(0) > 0;
      const slot = centralAlive ? 1 : 0;
      const mass = centralAlive ? userBodyMass : centralSinkMass;
      sim.setSink(slot, { x: wx, y: wy, z: wz, vx, vy, vz, mass });

      state.autoRotateSpeed = prevAutoRotate;
      placeHistory = [];
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    }
  };
  const onPointerCancel = (): void => {
    panning = false;
    if (placing) {
      placing = false;
      state.autoRotateSpeed = prevAutoRotate;
      placeHistory = [];
    }
  };
  // Suppress browser default for middle-button auxclick (Linux paste, etc.).
  const onAuxClick = (e: MouseEvent): void => { if (e.button === 1) e.preventDefault(); };

  canvas.addEventListener('pointerdown',   onPointerDown);
  canvas.addEventListener('pointermove',   onPointerMove);
  canvas.addEventListener('pointerup',     onPointerUp);
  canvas.addEventListener('pointercancel', onPointerCancel);
  canvas.addEventListener('auxclick',      onAuxClick);

  // ---- Animation loop -----------------------------------------------------
  let paused = false;
  let stepping = false;
  let destroyed = false;
  let rafId = 0;
  let lastTime = performance.now();

  const onVisibility = (): void => {
    if (document.hidden) paused = true;
    else                 paused = false;
  };
  document.addEventListener('visibilitychange', onVisibility);

  const stepAsync = async (): Promise<void> => {
    if (stepping || paused || destroyed) return;
    stepping = true;
    try { await sim.stepMany(cfg.stepsPerFrame); }
    finally { stepping = false; }
  };

  const tick = (): void => {
    if (destroyed) return;

    const now = performance.now();
    const dt  = Math.min(0.1, (now - lastTime) / 1000);
    lastTime  = now;
    if (!paused) state.azimuth += state.autoRotateSpeed * dt;

    void stepAsync();

    // Eye on an orbit around the live target.
    const r    = state.distance;
    const cosE = Math.cos(state.elevation);
    const sinE = Math.sin(state.elevation);
    eye[0] = state.target[0] + r * cosE * Math.sin(state.azimuth);
    eye[1] = state.target[1] + r * sinE;
    eye[2] = state.target[2] + r * cosE * Math.cos(state.azimuth);

    const aspect = canvas.width / Math.max(1, canvas.height);
    const view   = lookAt(eye, state.target, up);
    const proj   = perspective(Math.PI / 3, aspect, 0.02, 2000);
    const vp     = mul(proj, view);

    let textureView: GPUTextureView;
    try {
      textureView = ctx.getCurrentTexture().createView();
    } catch {
      rafId = requestAnimationFrame(tick);
      return;
    }

    renderer.render(textureView, {
      viewProj:    vp,
      widthPx:     canvas.width,
      heightPx:    canvas.height,
      perGalaxy:   sim.nStars,
      galaxyCount: 1,
      sinks:       sim.sinks,
      G:           1.0,
      c:           1.0,
    });

    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);

  return {
    destroy: () => {
      if (destroyed) return;
      destroyed = true;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      document.removeEventListener('visibilitychange', onVisibility);
      canvas.removeEventListener('pointerdown',   onPointerDown);
      canvas.removeEventListener('pointermove',   onPointerMove);
      canvas.removeEventListener('pointerup',     onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerCancel);
      canvas.removeEventListener('auxclick',      onAuxClick);
      sim.destroy();
      device.destroy();
    },
    pause:  () => { paused = true;  },
    resume: () => { paused = false; },
    state,
    setInteractive: (active: boolean) => {
      interactive = active;
      if (!active) {
        // Cancel any in-progress drag and clear the existing user body so the
        // sim isn't being perturbed while the visitor is on another page.
        if (placing) {
          placing = false;
          state.autoRotateSpeed = prevAutoRotate;
          placeHistory = [];
        }
        sim.setUserBody(null);
      }
    },
  };
}

// ---- Inlined matrix helpers (kept here so Svelte can import a single file) -

function lookAt(
  eye:    readonly [number, number, number],
  target: readonly [number, number, number],
  up:     readonly [number, number, number],
): Float32Array {
  const fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  const f0 = fx / fl, f1 = fy / fl, f2 = fz / fl;

  const s0 = f1 * up[2] - f2 * up[1];
  const s1 = f2 * up[0] - f0 * up[2];
  const s2 = f0 * up[1] - f1 * up[0];
  const sl = Math.hypot(s0, s1, s2) || 1;
  const r0 = s0 / sl, r1 = s1 / sl, r2 = s2 / sl;

  const u0 = r1 * f2 - r2 * f1;
  const u1 = r2 * f0 - r0 * f2;
  const u2 = r0 * f1 - r1 * f0;

  const m = new Float32Array(16);
  m[0]  = r0;   m[4]  = r1;   m[8]  = r2;   m[12] = -(r0 * eye[0] + r1 * eye[1] + r2 * eye[2]);
  m[1]  = u0;   m[5]  = u1;   m[9]  = u2;   m[13] = -(u0 * eye[0] + u1 * eye[1] + u2 * eye[2]);
  m[2]  = -f0;  m[6]  = -f1;  m[10] = -f2;  m[14] =  (f0 * eye[0] + f1 * eye[1] + f2 * eye[2]);
  m[15] = 1;
  return m;
}

function perspective(fovy: number, aspect: number, near: number, far: number): Float32Array {
  const f  = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const m  = new Float32Array(16);
  m[0]  = f / aspect;
  m[5]  = f;
  m[10] = far * nf;
  m[11] = -1;
  m[14] = far * near * nf;
  return m;
}

function mul(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + i] * b[j * 4 + k];
      out[j * 4 + i] = s;
    }
  }
  return out;
}
