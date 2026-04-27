<script>
  import { onMount, onDestroy } from 'svelte';

  // ── Constants ────────────────────────────────────────────────────────────
  const G = 1200;             // Gravitational constant (tuned to canvas scale)
  const EPSILON = 8;          // Softening parameter (px) — prevents singularities
  const DT = 0.35;            // Base time step per frame
  const TRAIL_LENGTH = 180;   // Trail history length
  const MAX_BODIES = 8;
  const AUTO_START_AT = 1;    // Auto-start simulation at this many bodies
  const ESCAPE_MARGIN = 120;  // px beyond canvas edge before a body is considered escaped

  // Body color palette: red accent + complementary colors
  const PALETTE = [
    '#d5000d',  // accent red (matches site theme)
    '#3d9ef5',  // blue
    '#f5a623',  // amber
    '#7ed321',  // green
    '#9b59b6',  // purple
    '#1abc9c',  // teal
    '#e67e22',  // orange
    '#e91e63',  // pink
  ];

  // Pre-parse palette colors to [r,g,b] for performance
  const PALETTE_RGB = PALETTE.map(hex => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ]);

  // ── Reactive State (Svelte 5 runes) ──────────────────────────────────────
  let bodies = $state([]);       // active (in-plane) bodies
  let escaped = $state([]);      // bodies that have left the canvas
  let isPlaying = $state(false);
  let speedMultiplier = $state(1);

  // ── Derived ──────────────────────────────────────────────────────────────
  let totalPlaced = $derived(bodies.length + escaped.length);

  let bodyCountText = $derived(
    bodies.length === 0 && escaped.length === 0
      ? 'No bodies placed'
      : bodies.length === 1
        ? `1 active body${escaped.length > 0 ? `, ${escaped.length} escaped` : ''}`
        : `${bodies.length} active${escaped.length > 0 ? `, ${escaped.length} escaped` : ''}`
  );

  let canSimulate = $derived(bodies.length >= AUTO_START_AT);

  // Slots available = MAX_BODIES minus active bodies (escaped slots can be reclaimed)
  let slotsAvailable = $derived(totalPlaced < MAX_BODIES);

  let instructionText = $derived(
    !slotsAvailable
      ? 'Maximum bodies reached — Reset or re-add escaped ones'
      : bodies.length === 0
        ? 'All bodies escaped — re-add them below or Reset'
        : isPlaying
          ? `Simulating — click canvas to add more bodies (${bodies.length} active)`
          : 'Press Play to start the simulation'
  );

  // ── DOM refs ─────────────────────────────────────────────────────────────
  let canvas = $state(null);
  let containerEl = $state(null);
  let ctx = null;
  let animationId = null;
  let resizeObserver = null;
  let dpr = 1;

  // ── Body Factory ─────────────────────────────────────────────────────────
  // Gives each new body an orbital velocity around the current center of mass.
  // This keeps bodies gravitationally bound and moving within the canvas plane.
  function createBody(x, y, index) {
    const mass = 80 + Math.random() * 40;

    let vx = 0;
    let vy = 0;

    if (bodies.length > 0) {
      // Compute center of mass of existing bodies
      let totalMass = 0, cx = 0, cy = 0;
      for (const b of bodies) {
        cx += b.x * b.mass;
        cy += b.y * b.mass;
        totalMass += b.mass;
      }
      cx /= totalMass;
      cy /= totalMass;

      // Vector from center of mass to new body
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy) || 1;

      // Circular orbital speed: v = sqrt(G * M_total / r), scaled down slightly
      // so bodies aren't perfectly circular (adds visual interest)
      const orbitalSpeed = Math.sqrt((G * totalMass) / r) * (0.055 + Math.random() * 0.02);

      // Perpendicular to the radius vector (tangential = orbital direction)
      // Pick the direction that gives a consistent rotation sense
      vx = (-dy / r) * orbitalSpeed;
      vy = ( dx / r) * orbitalSpeed;
    }
    // First body gets zero velocity — it acts as a gravitational anchor
    // until others are placed

    return {
      x, y,
      vx, vy,
      mass,
      colorIndex: index % PALETTE.length,
      trail: [],
      radius: 6,
    };
  }

  // ── Re-add an escaped body at the center of the active system ────────────
  function readdBody(escapedBody) {
    // Place it near the center of mass of active bodies (or canvas center)
    let cx, cy;
    if (bodies.length > 0) {
      let totalMass = 0;
      cx = 0; cy = 0;
      for (const b of bodies) {
        cx += b.x * b.mass;
        cy += b.y * b.mass;
        totalMass += b.mass;
      }
      cx /= totalMass;
      cy /= totalMass;
    } else {
      cx = (canvas.width  / dpr) / 2;
      cy = (canvas.height / dpr) / 2;
    }

    // Offset slightly from exact center so it doesn't overlap
    const offsetAngle = Math.random() * Math.PI * 2;
    const offsetR = 40 + Math.random() * 40;
    const x = cx + Math.cos(offsetAngle) * offsetR;
    const y = cy + Math.sin(offsetAngle) * offsetR;

    // Compute orbital velocity around current center of mass (same as createBody)
    let vx = 0, vy = 0;
    if (bodies.length > 0) {
      let totalMass = bodies.reduce((s, b) => s + b.mass, 0);
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy) || 1;
      const orbitalSpeed = Math.sqrt((G * totalMass) / r) * (0.055 + Math.random() * 0.02);
      vx = (-dy / r) * orbitalSpeed;
      vy = ( dx / r) * orbitalSpeed;
    }

    const newBody = {
      x, y, vx, vy,
      mass: escapedBody.mass,
      colorIndex: escapedBody.colorIndex,
      trail: [],
      radius: 6,
    };

    bodies = [...bodies, newBody];
    escaped = escaped.filter(e => e !== escapedBody);

    // Auto-start if we now have enough bodies
    if (bodies.length >= AUTO_START_AT && !isPlaying) {
      isPlaying = true;
    }
  }

  // ── Physics: compute acceleration on point (px,py) from a list of sources ─
  function computeAccel(px, py, sources) {
    let ax = 0, ay = 0;
    for (const s of sources) {
      const dx = s.x - px;
      const dy = s.y - py;
      const distSq = dx * dx + dy * dy + EPSILON * EPSILON;
      const dist = Math.sqrt(distSq);
      const mag = (G * s.mass) / distSq;
      ax += mag * (dx / dist);
      ay += mag * (dy / dist);
    }
    return [ax, ay];
  }

  // ── RK4 integrator for one body ──────────────────────────────────────────
  function rk4Step(body, others, dt) {
    const { x, y, vx, vy } = body;

    // k1
    const [a1x, a1y] = computeAccel(x, y, others);

    // k2
    const x2  = x  + 0.5 * dt * vx;
    const y2  = y  + 0.5 * dt * vy;
    const vx2 = vx + 0.5 * dt * a1x;
    const vy2 = vy + 0.5 * dt * a1y;
    const [a2x, a2y] = computeAccel(x2, y2, others);

    // k3
    const x3  = x  + 0.5 * dt * vx2;
    const y3  = y  + 0.5 * dt * vy2;
    const vx3 = vx + 0.5 * dt * a2x;
    const vy3 = vy + 0.5 * dt * a2y;
    const [a3x, a3y] = computeAccel(x3, y3, others);

    // k4
    const x4  = x  + dt * vx3;
    const y4  = y  + dt * vy3;
    const vx4 = vx + dt * a3x;
    const vy4 = vy + dt * a3y;
    const [a4x, a4y] = computeAccel(x4, y4, others);

    // Weighted combination
    let newVx = vx + (dt / 6) * (a1x + 2*a2x + 2*a3x + a4x);
    let newVy = vy + (dt / 6) * (a1y + 2*a2y + 2*a3y + a4y);

    return {
      x:  x  + (dt / 6) * (vx  + 2*vx2 + 2*vx3 + vx4),
      y:  y  + (dt / 6) * (vy  + 2*vy2 + 2*vy3 + vy4),
      vx: newVx,
      vy: newVy,
    };
  }

  // ── Simulation step ──────────────────────────────────────────────────────
  function stepSimulation() {
    const dt = DT * speedMultiplier;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    // Snapshot positions at frame-start for consistent force calculation
    const snapshot = bodies.map(b => ({ x: b.x, y: b.y, mass: b.mass }));

    const newlyEscaped = [];
    const stillActive = [];

    bodies.forEach((body, i) => {
      const others = snapshot.filter((_, j) => j !== i);
      const next = rk4Step(body, others, dt);

      // Append current position to trail
      const trail = [...body.trail, { x: body.x, y: body.y }];
      if (trail.length > TRAIL_LENGTH) trail.shift();

      const updated = { ...body, ...next, trail };

      // Check if the body has escaped beyond the canvas + margin
      const outOfBounds =
        updated.x < -ESCAPE_MARGIN ||
        updated.x > W + ESCAPE_MARGIN ||
        updated.y < -ESCAPE_MARGIN ||
        updated.y > H + ESCAPE_MARGIN;

      if (outOfBounds) {
        newlyEscaped.push({ mass: body.mass, colorIndex: body.colorIndex });
      } else {
        stillActive.push(updated);
      }
    });

    bodies = stillActive;
    if (newlyEscaped.length > 0) {
      escaped = [...escaped, ...newlyEscaped];
    }
  }

  // ── Rendering ────────────────────────────────────────────────────────────
  function draw() {
    if (!ctx || !canvas) return;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    // Clear
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f0f13';
    ctx.fillRect(0, 0, W, H);

    // Subtle grid
    ctx.strokeStyle = 'rgba(42, 42, 56, 0.5)';
    ctx.lineWidth = 0.5;
    const spacing = 40;
    for (let gx = 0; gx < W; gx += spacing) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }
    for (let gy = 0; gy < H; gy += spacing) {
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    }

    // Draw trails
    for (const body of bodies) {
      if (body.trail.length < 2) continue;
      const [r, g, b] = PALETTE_RGB[body.colorIndex];
      for (let i = 1; i < body.trail.length; i++) {
        const t = i / body.trail.length;
        ctx.beginPath();
        ctx.strokeStyle = `rgba(${r},${g},${b},${t * 0.65})`;
        ctx.lineWidth = t * 2.5;
        ctx.lineCap = 'round';
        ctx.moveTo(body.trail[i - 1].x, body.trail[i - 1].y);
        ctx.lineTo(body.trail[i].x, body.trail[i].y);
        ctx.stroke();
      }
    }

    // Draw bodies
    for (const body of bodies) {
      const { x, y, radius, colorIndex } = body;
      const [r, g, b] = PALETTE_RGB[colorIndex];

      // Outer glow
      const glow1 = ctx.createRadialGradient(x, y, 0, x, y, radius * 6);
      glow1.addColorStop(0, `rgba(${r},${g},${b},0.25)`);
      glow1.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(x, y, radius * 6, 0, Math.PI * 2);
      ctx.fillStyle = glow1;
      ctx.fill();

      // Inner glow
      const glow2 = ctx.createRadialGradient(x, y, 0, x, y, radius * 2.5);
      glow2.addColorStop(0, `rgba(${r},${g},${b},0.85)`);
      glow2.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.beginPath();
      ctx.arc(x, y, radius * 2.5, 0, Math.PI * 2);
      ctx.fillStyle = glow2;
      ctx.fill();

      // Solid core
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }

    // Placement hint: crosshair at mouse position — handled via CSS cursor
    // Ghost label when no bodies
    if (bodies.length === 0) {
      ctx.fillStyle = 'rgba(90, 90, 120, 0.5)';
      ctx.font = `600 14px Inter, system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText('Click to place bodies', W / 2, H / 2);
      ctx.font = `400 12px Inter, system-ui, sans-serif`;
      ctx.fillStyle = 'rgba(90, 90, 120, 0.35)';
      ctx.fillText('Newtonian gravity · RK4 integration', W / 2, H / 2 + 22);
      ctx.textAlign = 'left';
    }
  }

  // ── Animation loop ───────────────────────────────────────────────────────
  function loop() {
    if (isPlaying && bodies.length >= AUTO_START_AT) {
      stepSimulation();
    }
    draw();
    animationId = requestAnimationFrame(loop);
  }

  // ── Canvas sizing ────────────────────────────────────────────────────────
  function resizeCanvas() {
    if (!canvas || !containerEl) return;
    dpr = window.devicePixelRatio || 1;
    const W = containerEl.clientWidth;
    const H = Math.round(W * (10 / 16));
    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';
    ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    draw();
  }

  // ── Event handlers ───────────────────────────────────────────────────────
  function handleCanvasClick(e) {
    if (!slotsAvailable) return;
    const rect = canvas.getBoundingClientRect();
    // Scale from CSS pixels to canvas logical pixels (accounts for DPI)
    const x = (e.clientX - rect.left) * (canvas.width / dpr / rect.width);
    const y = (e.clientY - rect.top)  * (canvas.height / dpr / rect.height);
    // colorIndex starts at 2 since default bodies use 0 (red) and 1 (blue)
    const newBody = createBody(x, y, totalPlaced + 2);
    bodies = [...bodies, newBody];

    // Auto-start when threshold reached
    if (bodies.length >= AUTO_START_AT && !isPlaying) {
      isPlaying = true;
    }
  }

  function togglePlay() {
    if (!canSimulate) return;
    isPlaying = !isPlaying;
  }

  function reset() {
    escaped = [];
    spawnDefaultBodies();
  }

  // ── Default two-body orbit ────────────────────────────────────────────────
  // Spawns two equal-mass bodies in a perfect circular orbit around their
  // shared center of mass so the user immediately sees what the sim can do.
  function spawnDefaultBodies() {
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;
    const cx = W / 2;
    const cy = H / 2;

    const mass = 100;         // equal masses for a symmetric orbit
    const separation = 140;   // px between the two bodies (each is 70px from center)
    const r = separation / 2;

    // Exact circular orbital speed: v = sqrt(G * m / (4 * r))
    // For two equal masses, each orbits the barycenter at radius r with
    // v = sqrt(G * m / (4 * r))
    const v = Math.sqrt((G * mass) / (4 * r));

    // Body 0 — left, moving downward
    // Body 1 — right, moving upward
    bodies = [
      {
        x: cx - r, y: cy,
        vx: 0, vy: v,
        mass, colorIndex: 0,
        trail: [], radius: 6,
      },
      {
        x: cx + r, y: cy,
        vx: 0, vy: -v,
        mass, colorIndex: 1,
        trail: [], radius: 6,
      },
    ];

    isPlaying = true;
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────
  onMount(() => {
    resizeCanvas();
    spawnDefaultBodies();

    resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });
    resizeObserver.observe(containerEl);

    animationId = requestAnimationFrame(loop);
  });

  onDestroy(() => {
    if (animationId) cancelAnimationFrame(animationId);
    if (resizeObserver) resizeObserver.disconnect();
  });
</script>

<svelte:head>
  <title>N-body Simulation — Reagan Howell</title>
  <meta name="description" content="Interactive gravitational N-body simulation. Click to place bodies and watch Newtonian gravity unfold — built on research into three-body equilibrium configurations." />
</svelte:head>

<section class="page-header">
  <div class="page-eyebrow">Interactive</div>
  <h1>N-body Simulation</h1>
  <p class="page-subtitle">
    Click the canvas to place bodies. Gravity does the rest — powered by RK4 numerical integration,
    the same method used in my BYU research on three-body equilibrium configurations.
  </p>
</section>

<div class="sim-wrapper" bind:this={containerEl}>
  <canvas
    bind:this={canvas}
    class="sim-canvas"
    onclick={handleCanvasClick}
    style="cursor: {!slotsAvailable ? 'not-allowed' : 'crosshair'}"
  ></canvas>
</div>

<div class="controls">
  <div class="controls-left">
    <span class="body-count">{bodyCountText}</span>
    <span class="instruction">{instructionText}</span>
  </div>
  <div class="controls-right">
    <div class="speed-control">
      <label for="speed" class="speed-label">Speed</label>
      <input
        id="speed"
        type="range"
        min="0.5"
        max="3"
        step="0.5"
        bind:value={speedMultiplier}
        class="speed-slider"
        aria-label="Simulation speed"
      />
      <span class="speed-value">{speedMultiplier}×</span>
    </div>
    <button
      class="btn btn-primary"
      onclick={togglePlay}
      disabled={!canSimulate}
      aria-label={isPlaying ? 'Pause simulation' : 'Play simulation'}
    >
      {#if isPlaying}
        <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor" aria-hidden="true">
          <rect x="0" y="0" width="4" height="14" rx="1"/>
          <rect x="8" y="0" width="4" height="14" rx="1"/>
        </svg>
        Pause
      {:else}
        <svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor" aria-hidden="true">
          <path d="M1 0l11 7-11 7V0z"/>
        </svg>
        Play
      {/if}
    </button>
    <button class="btn btn-secondary" onclick={reset} aria-label="Reset simulation">
      Reset
    </button>
  </div>
</div>

{#if escaped.length > 0}
<div class="escaped-panel">
  <span class="escaped-label">
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path d="M6 1L11 6L6 11M1 6h10" stroke="#d5000d" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    Escaped bodies
  </span>
  <div class="escaped-list">
    {#each escaped as body, i (i)}
      <button
        class="escaped-btn"
        onclick={() => readdBody(body)}
        style="--body-color: {PALETTE[body.colorIndex]}"
        aria-label="Re-add escaped body"
        title="Re-add to simulation"
      >
        <span class="escaped-dot"></span>
        Re-add
      </button>
    {/each}
  </div>
</div>
{/if}

<section class="about-section">
  <h2 class="section-heading">About This Simulation</h2>
  <p>
    The <strong>N-body problem</strong> asks: given N masses interacting purely through Newtonian
    gravity, how does the system evolve over time? For N&nbsp;≥&nbsp;3, no general closed-form
    solution exists — the motion is chaotic and exquisitely sensitive to initial conditions.
  </p>
  <p>
    This simulation uses <strong>Runge-Kutta 4 (RK4)</strong> integration with a softening
    parameter ε&nbsp;=&nbsp;8&nbsp;px to prevent force singularities when bodies pass close
    together. The gravitational constant G is scaled to the canvas rather than SI units.
    Bodies are given a small random initial velocity so they don't immediately fall into each other.
  </p>

  <div class="tip-grid">
    <div class="tip">
      <span class="tip-icon">⬤</span>
      <span class="tip-text">1 body drifts freely; 2 bodies orbit each other; 3+ creates chaotic interactions</span>
    </div>
    <div class="tip">
      <span class="tip-icon">⬤</span>
      <span class="tip-text">The first body anchors the system; place it near the canvas center</span>
    </div>
    <div class="tip">
      <span class="tip-icon">⬤</span>
      <span class="tip-text">Adding bodies mid-simulation perturbs the system — watch for chaotic slingshots</span>
    </div>
    <div class="tip">
      <span class="tip-icon">⬤</span>
      <span class="tip-text">Slow speed to 0.5× to watch close approaches and exchanges in detail</span>
    </div>
  </div>
</section>

<style>
  /* ── Page header ─────────────────────────────────────────── */
  .page-header {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #2a2a38;
    margin-bottom: 2rem;
  }

  .page-eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #d5000d;
    margin-bottom: 0.4rem;
  }

  .page-header h1 {
    font-family: 'Chivo', sans-serif;
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 900;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
    line-height: 1.15;
  }

  .page-subtitle {
    color: #7070a0;
    font-size: 0.97rem;
    line-height: 1.65;
    margin: 0;
    max-width: 560px;
  }

  /* ── Canvas wrapper ──────────────────────────────────────── */
  /* Negative margins escape the layout's 1.5rem horizontal padding,
     making the canvas fill the full 740px content column. */
  .sim-wrapper {
    margin-left:  -1.5rem;
    margin-right: -1.5rem;
    background: #0f0f13;
    border-top:    1px solid #2a2a38;
    border-bottom: 1px solid #2a2a38;
    line-height: 0; /* removes inline-block gap under canvas */
  }

  .sim-canvas {
    display: block;
    width: 100%;
    /* height set dynamically via JS */
  }

  /* ── Controls bar ────────────────────────────────────────── */
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.1rem 0 1.25rem;
    border-bottom: 1px solid #2a2a38;
    margin-bottom: 2.5rem;
  }

  .controls-left {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    min-width: 0;
  }

  .body-count {
    font-family: 'Chivo', sans-serif;
    font-size: 0.92rem;
    font-weight: 700;
    color: #c5c5e8;
    white-space: nowrap;
  }

  .instruction {
    font-size: 0.78rem;
    color: #5a5a78;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .controls-right {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-shrink: 0;
    flex-wrap: wrap;
  }

  /* ── Speed control ───────────────────────────────────────── */
  .speed-control {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .speed-label {
    font-size: 0.72rem;
    color: #5a5a78;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    white-space: nowrap;
  }

  .speed-value {
    font-size: 0.72rem;
    color: #9999b0;
    width: 2.2ch;
    text-align: right;
  }

  .speed-slider {
    -webkit-appearance: none;
    appearance: none;
    width: 72px;
    height: 3px;
    border-radius: 2px;
    background: #2a2a38;
    outline: none;
    cursor: pointer;
  }

  .speed-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 13px;
    height: 13px;
    border-radius: 50%;
    background: #d5000d;
    cursor: pointer;
    transition: transform 0.1s ease;
  }

  .speed-slider::-webkit-slider-thumb:hover {
    transform: scale(1.25);
  }

  .speed-slider::-moz-range-thumb {
    width: 13px;
    height: 13px;
    border: none;
    border-radius: 50%;
    background: #d5000d;
    cursor: pointer;
  }

  /* ── Buttons ─────────────────────────────────────────────── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'Chivo', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    padding: 0.42rem 1rem;
    border-radius: 6px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: background 0.15s ease, color 0.15s ease,
                border-color 0.15s ease, opacity 0.15s ease;
    white-space: nowrap;
    line-height: 1;
  }

  .btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .btn-primary {
    background: #d5000d;
    color: #ffffff;
    border-color: #d5000d;
  }

  .btn-primary:hover:not(:disabled) {
    background: #ff1a1a;
    border-color: #ff1a1a;
  }

  .btn-secondary {
    background: transparent;
    color: #9999b0;
    border-color: #2a2a38;
  }

  .btn-secondary:hover:not(:disabled) {
    color: #e8e8f0;
    border-color: #5a5a78;
  }

  /* ── About section ───────────────────────────────────────── */
  .about-section {
    margin-bottom: 3rem;
  }

  .section-heading {
    font-family: 'Chivo', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #c5c5e8;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #2a2a38;
    letter-spacing: -0.01em;
  }

  .about-section p {
    color: #c0c0d8;
    font-size: 0.95rem;
    line-height: 1.75;
    margin-bottom: 1rem;
  }

  .about-section strong {
    color: #e8e8f0;
  }

  /* ── Tips grid ───────────────────────────────────────────── */
  .tip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1.25rem;
  }

  .tip {
    background: #16161e;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    padding: 0.65rem 0.85rem;
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    transition: border-color 0.2s ease;
  }

  .tip:hover {
    border-color: #d5000d;
  }

  .tip-icon {
    font-size: 0.5rem;
    color: #d5000d;
    margin-top: 0.35rem;
    flex-shrink: 0;
  }

  .tip-text {
    font-size: 0.8rem;
    color: #9090b8;
    line-height: 1.5;
  }

  /* ── Escaped bodies panel ────────────────────────────────── */
  .escaped-panel {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.6rem;
    padding: 0.75rem 1rem;
    background: #16161e;
    border: 1px solid #3a1515;
    border-radius: 8px;
    margin-bottom: 2rem;
  }

  .escaped-label {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a3030;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .escaped-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }

  .escaped-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: 'Chivo', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 0.28rem 0.7rem;
    border-radius: 5px;
    border: 1px solid color-mix(in srgb, var(--body-color) 40%, transparent);
    background: color-mix(in srgb, var(--body-color) 12%, transparent);
    color: var(--body-color);
    cursor: pointer;
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .escaped-btn:hover {
    background: color-mix(in srgb, var(--body-color) 22%, transparent);
    border-color: var(--body-color);
  }

  .escaped-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--body-color);
    box-shadow: 0 0 5px var(--body-color);
    flex-shrink: 0;
  }

  /* ── Responsive ──────────────────────────────────────────── */
  @media (max-width: 600px) {
    .controls {
      flex-direction: column;
      align-items: flex-start;
    }

    .controls-right {
      width: 100%;
      justify-content: space-between;
    }

    .tip-grid {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 420px) {
    .sim-wrapper {
      margin-left:  -1rem;
      margin-right: -1rem;
    }

    .speed-slider {
      width: 52px;
    }
  }
</style>
