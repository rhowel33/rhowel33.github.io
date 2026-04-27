<!--
  WebGPU n-body galaxy simulation as a fixed-position background, with a
  small camera-control panel pinned to the bottom-right corner.

  Sliders drive a $state object that the engine reads each frame. Touching
  any slider zeroes autoRotateSpeed (full manual control); the ↻ button
  re-enables the slow auto-orbit.

  Middle-click drag on empty page background pans the lookAt target.
  Content (header, links, body text) sits at higher z-index than the canvas
  so normal clicks pass through to those elements unaffected.

  Falls back to a one-shot static starfield (2D canvas) when WebGPU is
  unavailable, on tiny screens, or when the user prefers reduced motion —
  so every visitor sees stars instead of a flat background.
-->
<script>
  import { onMount, onDestroy } from 'svelte';

  // Live camera state — same object the engine reads each frame.
  let camera = $state({
    azimuth:         0,
    elevation:       0.55,
    distance:        3.8,
    target:          [0, 0, 0],
    autoRotateSpeed: 0.05,
  });

  let canvas  = $state(null);
  let cleanup = null;
  let active  = $state(false);
  let supported = $state(false);
  let resizeHandler = null;

  function shouldRun() {
    if (typeof window === 'undefined') return false;
    if (!('gpu' in navigator) || !navigator.gpu) return false;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return false;
    if (window.matchMedia?.('(max-width: 600px)')?.matches) return false;
    return true;
  }

  // Deterministic PRNG so the starfield is identical between page paints
  // (avoids a flash of differently-placed stars on hydration / resize).
  function mulberry32(seed) {
    return () => {
      seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function drawStarfield(c) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = window.innerWidth;
    const h = window.innerHeight;
    c.width  = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    c.style.width  = w + 'px';
    c.style.height = h + 'px';

    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    // Background: subtle radial nebula tint over the page's #0f0f13.
    const grad = ctx.createRadialGradient(w * 0.5, h * 0.55, 0, w * 0.5, h * 0.55, Math.max(w, h) * 0.7);
    grad.addColorStop(0, 'rgba(60, 30, 80, 0.18)');
    grad.addColorStop(0.5, 'rgba(20, 15, 40, 0.08)');
    grad.addColorStop(1, 'rgba(15, 15, 19, 0)');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    const rand = mulberry32(7);
    const density = 0.00035; // stars per square px
    const count = Math.max(180, Math.min(900, Math.floor(w * h * density)));
    const palette = ['#ffffff', '#ffe9c8', '#cfe0ff', '#ffd1d1', '#e8d8ff'];

    for (let i = 0; i < count; i++) {
      const x = rand() * w;
      const y = rand() * h;
      const r = Math.pow(rand(), 3.5) * 1.6 + 0.25;
      const a = 0.35 + rand() * 0.6;
      ctx.fillStyle = palette[Math.floor(rand() * palette.length)];
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      // Halo on the brightest ~5%
      if (r > 1.3) {
        ctx.globalAlpha = 0.12;
        ctx.beginPath();
        ctx.arc(x, y, r * 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  onMount(async () => {
    if (!canvas) return;

    if (shouldRun()) {
      supported = true;
      const { startNBodyBackground } = await import('./nbody/src/background.ts');
      cleanup = await startNBodyBackground({
        canvas,
        state:        camera,
        nBodies:      12000,
        stepsPerFrame: 1,
        galaxyType:   'spiral',
        seed:         7,
      });
      if (cleanup) active = true;
      return;
    }

    // Fallback: static starfield. One-shot draw; redraw on resize.
    drawStarfield(canvas);
    active = true;
    resizeHandler = () => drawStarfield(canvas);
    window.addEventListener('resize', resizeHandler);
  });

  onDestroy(() => {
    if (cleanup) cleanup.destroy();
    cleanup = null;
    if (resizeHandler) window.removeEventListener('resize', resizeHandler);
    resizeHandler = null;
    active = false;
  });

  // Touching any slider freezes auto-rotate so the camera doesn't slide
  // out from under the user. The ↻ button revives it at the original rate.
  function manualMode() {
    if (camera.autoRotateSpeed !== 0) camera.autoRotateSpeed = 0;
  }
  function resumeAuto() {
    camera.autoRotateSpeed = 0.05;
  }

  // Slider value <-> radians. Azimuth wraps; elevation clamped just under ±π/2.
  const TAU       = Math.PI * 2;
  const ELEV_LIM  = Math.PI / 2 - 0.05;
</script>

<canvas
  bind:this={canvas}
  class="nbody-bg"
  class:active
  aria-hidden="true"
></canvas>

{#if active && supported}
  <div class="nbody-controls" role="group" aria-label="Camera controls">
    <div class="row">
      <span class="lbl">XY</span>
      <input
        type="range"
        min="0" max={TAU} step="0.005"
        bind:value={camera.azimuth}
        oninput={manualMode}
        aria-label="Azimuth (rotate around vertical axis)"
      />
    </div>
    <div class="row">
      <span class="lbl">XZ</span>
      <input
        type="range"
        min={-ELEV_LIM} max={ELEV_LIM} step="0.005"
        bind:value={camera.elevation}
        oninput={manualMode}
        aria-label="Elevation (tilt up/down)"
      />
    </div>
    <div class="row">
      <span class="lbl">Z</span>
      <input
        type="range"
        min="1.5" max="25" step="0.05"
        bind:value={camera.distance}
        oninput={manualMode}
        aria-label="Zoom (distance from origin)"
      />
    </div>
    {#if camera.autoRotateSpeed === 0}
      <button class="resume" type="button" onclick={resumeAuto} aria-label="Resume auto-rotate">↻</button>
    {/if}
  </div>
{/if}

<style>
  .nbody-bg {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    /* pointer-events auto so middle-click pan reaches the canvas in
       margin/empty regions. Content (header, links, text) sits at higher
       z-index and continues to receive clicks normally. */
    pointer-events: auto;
    opacity: 0;
    transition: opacity 700ms ease;
    filter: brightness(0.8);
  }
  .nbody-bg.active {
    opacity: 0.9;
  }

  .nbody-controls {
    position: fixed;
    right: 14px;
    bottom: 14px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 10px 12px;
    background: rgba(15, 15, 19, 0.55);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 11px;
    color: #c8c8d4;
    user-select: none;
    pointer-events: auto;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .lbl {
    width: 18px;
    text-align: right;
    color: #9999b0;
    font-variant-numeric: tabular-nums;
  }
  input[type='range'] {
    width: 110px;
    accent-color: #d5000d;
    cursor: pointer;
  }
  .resume {
    align-self: flex-end;
    background: transparent;
    color: #d5000d;
    border: 1px solid rgba(213, 0, 13, 0.4);
    border-radius: 4px;
    padding: 1px 8px;
    cursor: pointer;
    font: inherit;
  }
  .resume:hover {
    background: rgba(213, 0, 13, 0.15);
  }
  @media (max-width: 600px) {
    .nbody-controls { display: none; }
  }
</style>
