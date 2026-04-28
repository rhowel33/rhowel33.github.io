<script>
  import { page } from '$app/stores';
  import NBodyBackground from '$lib/NBodyBackground.svelte';
  let { children } = $props();

  let isGalaxy = $derived($page.url.pathname.startsWith('/galaxy'));
</script>

<NBodyBackground interactive={isGalaxy} />

<div class="site-wrapper">
  <header class="site-header">
    <div class="header-inner">
      <a href="/" class="site-title">Reagan Howell</a>
      <nav class="header-nav">
        <a href="/projects" class:active={$page.url.pathname.startsWith('/projects')}>Projects</a>
        <a href="/galaxy" class:active={$page.url.pathname.startsWith('/galaxy')}>Galaxy</a>
        <a href="/resume/resume.pdf" target="_blank" rel="noopener noreferrer">Resume</a>
        <a href="https://github.com/rhowel33" target="_blank" rel="noopener noreferrer">GitHub</a>
      </nav>
    </div>
  </header>

  <main class="main-content">
    {@render children()}
  </main>

  <footer class="site-footer">
    <div class="footer-inner">
      <p class="footer-label">Personal Links</p>
      <nav class="footer-nav">
        <a href="/resume/resume.pdf" target="_blank" rel="noopener noreferrer">Resume</a>
        <a href="https://github.com/rhowel33" target="_blank" rel="noopener noreferrer">GitHub</a>
        <a href="https://letterboxd.com/ironisblack/" target="_blank" rel="noopener noreferrer">Letterboxd</a>
        <a href="https://shelflife.tokyodelights.com/user/cmmwbn5v1000047pam7j9kc9u" target="_blank" rel="noopener noreferrer">Shelf Life</a>
        <a href="https://www.chess.com/member/rhowel33" target="_blank" rel="noopener noreferrer">Chess.com</a>
      </nav>
      <p class="footer-credit">Built with SvelteKit · Hosted on GitHub Pages</p>
    </div>
  </footer>
</div>

<style>
  /* ── Global resets & base styles ── */
  :global(*) {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  :global(body) {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: #0f0f13;
    color: #e8e8f0;
    line-height: 1.7;
    min-height: 100vh;
  }

  :global(h1, h2, h3, h4) {
    font-family: 'Chivo', sans-serif;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }

  :global(a) {
    color: #d5000d;
    text-decoration: none;
    transition: color 0.2s ease;
  }

  :global(a:hover) {
    color: #ff1a1a;
  }

  :global(p) {
    margin-bottom: 1rem;
  }

  /* ── Layout shell ── */
  .site-wrapper {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  .main-content {
    flex: 1;
    width: 100%;
    max-width: 740px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
    /* Sit above the n-body canvas (z:0) so text and links on every page
       paint and receive clicks normally. The element itself is transparent
       to pointer events so empty padding lets clicks fall through to the
       canvas behind; direct children opt back in for normal interactivity. */
    position: relative;
    z-index: 2;
    pointer-events: none;
  }
  .main-content > * {
    pointer-events: auto;
  }
  /* The simulation canvas is position:fixed at z:0, so the otherwise-static
     footer would paint underneath it. Lifting it into a stacking context
     above the canvas keeps it visible and clickable on every page. */
  .site-footer {
    position: relative;
    z-index: 1;
  }

  /* ── Header ── */
  .site-header {
    background: #16161e;
    border-bottom: 1px solid #2a2a38;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-inner {
    max-width: 740px;
    margin: 0 auto;
    padding: 0 1.5rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .site-title {
    font-family: 'Chivo', sans-serif;
    font-weight: 900;
    font-size: 1.15rem;
    color: #ffffff;
    letter-spacing: -0.02em;
    text-decoration: none;
  }

  .site-title:hover {
    color: #d5000d;
  }

  .header-nav {
    display: flex;
    gap: 1.5rem;
    align-items: center;
  }

  .header-nav a {
    font-size: 0.9rem;
    font-weight: 500;
    color: #9999b0;
    padding: 0.25rem 0;
    border-bottom: 2px solid transparent;
    transition: color 0.2s ease, border-color 0.2s ease;
  }

  .header-nav a:hover,
  .header-nav a.active {
    color: #e8e8f0;
    border-bottom-color: #d5000d;
  }

  /* ── Footer ── */
  .site-footer {
    background: #16161e;
    border-top: 1px solid #2a2a38;
    padding: 2rem 1.5rem;
    margin-top: 3rem;
  }

  .footer-inner {
    max-width: 740px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  .footer-label {
    font-family: 'Chivo', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5a5a78;
    margin: 0;
  }

  .footer-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    justify-content: center;
  }

  .footer-nav a {
    font-size: 0.9rem;
    color: #9999b0;
    transition: color 0.2s ease;
  }

  .footer-nav a:hover {
    color: #d5000d;
  }

  .footer-credit {
    font-size: 0.75rem;
    color: #3d3d55;
    margin: 0;
  }

  /* ── Responsive ── */
  @media (max-width: 480px) {
    .header-nav {
      gap: 1rem;
    }

    .header-nav a {
      font-size: 0.82rem;
    }
  }
</style>
