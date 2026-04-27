<!--
  /galaxy — bare stage for the WebGPU n-body simulation.

  The simulation lives in the layout (NBodyBackground.svelte), so this page
  intentionally ships no body content; <main> just holds blank space so the
  galaxy fills the viewport behind the header and footer. Camera-control
  sliders remain accessible bottom-right since they're rendered outside
  the layout's .site-wrapper.
-->
<svelte:head>
  <title>Galaxy · Reagan Howell</title>
  <meta
    name="description"
    content="WebGPU n-body galaxy simulation: direct-summation gravity over ~12k stars in a Plummer-bulge + exponential-disk spiral with a static dark-matter halo."
  />
</svelte:head>
