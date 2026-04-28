<!--
  /galaxy — interactive stage for the WebGPU n-body simulation.

  The simulation is mounted in the layout (NBodyBackground.svelte) and runs
  on every page; the layout passes `interactive={isGalaxy}` so click-to-place
  is wired up only here. The page itself ships no body content so the empty
  <main> lets clicks fall through to the canvas behind. Left-click drops a
  heavy user body at the cursor with the release-time mouse velocity;
  middle-click drag pans the camera; sliders in the bottom-right corner
  control the camera orbit.
-->
<svelte:head>
  <title>Galaxy · Reagan Howell</title>
  <meta
    name="description"
    content="WebGPU n-body galaxy simulation: direct-summation gravity over ~12k stars in a Plummer-bulge + exponential-disk spiral with a static dark-matter halo. Click to drop a heavy body and perturb the disk."
  />
</svelte:head>
