<!--
  WebGPU n-body galaxy simulation as a fixed-position background, with a
  small camera-control panel pinned to the bottom-right corner.

  Sliders drive a $state object that the engine reads each frame. Touching
  any slider zeroes autoRotateSpeed (full manual control); the ↻ button
  re-enables the slow auto-orbit.

  Middle-click drag on empty page background pans the lookAt target.
  Content (header, links, body text) sits at higher z-index than the canvas
  so normal clicks pass through to those elements unaffected.

  Bails silently on browsers without WebGPU, on tiny screens, and on
  prefers-reduced-motion: in those cases the body's `#0f0f13` shows through.
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

  function shouldRun() {
    if (typeof window === 'undefined') return false;
    if (!('gpu' in navigator) || !navigator.gpu) return false;
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches) return false;
    if (window.matchMedia?.('(max-width: 600px)')?.matches) return false;
    return true;
  }

  onMount(async () => {
    if (!shouldRun() || !canvas) return;
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
  });

  onDestroy(() => {
    if (cleanup) cleanup.destroy();
    cleanup = null;
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

{#if active}
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
