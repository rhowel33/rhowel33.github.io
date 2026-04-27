<svelte:head>
  <title>Reagan Howell</title>
  <meta name="description" content="Applied Mathematics graduate, software and hardware enthusiast." />
</svelte:head>

<section class="hero">
  <div class="hero-eyebrow">Hello, I'm</div>
  <h1 class="hero-name">Reagan Howell</h1>
  <p class="hero-tagline">
    Applied Mathematics · Software &amp; Hardware · Server Tinkerer
  </p>
</section>

<section class="about-section">
  <h2 class="section-heading">About</h2>
  <p>
    I'm a graduate from Brigham Young University with a Bachelor of Science in Applied Mathematics,
    concentration in mechanical engineering. During my time at BYU I did research on the N-body
    problem — specifically studying equilibrium configurations and their stabilities, such as the
    figure-8 configuration. I also worked on a team investigating possible locations of earthquake
    epicenters based on historical records of earthquakes and tsunamis before the seismograph.
  </p>
  <p>
    My passion lies in both software and hardware, particularly projects centered around machine
    learning and computational applications that make a positive impact. I thrive on problem-solving
    and relish fixing anything that's broken — mechanical systems or code alike. I host my own server
    for media, backups, and intensive computing programs.
  </p>
</section>

<section class="personal-section">
  <h2 class="section-heading">Personal</h2>
  <p>
    Beyond technical work, I enjoy watching movies, reading, biking, playing chess, and cooking.
    <em>Ratatouille</em> — a blend of great animation, cooking, and storytelling — is one of my
    all-time favorites.
  </p>

  <div class="interest-pills">
    <span class="pill">Movies</span>
    <span class="pill">Reading</span>
    <span class="pill">Biking</span>
    <span class="pill">Chess</span>
    <span class="pill">Cooking</span>
    <span class="pill">Self-hosting</span>
  </div>
</section>

<style>
  /* ── Hero ── */
  .hero {
    padding: 3rem 0 2.5rem;
    border-bottom: 1px solid #2a2a38;
    margin-bottom: 2.5rem;
  }

  .hero-eyebrow {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #d5000d;
    margin-bottom: 0.5rem;
  }

  .hero-name {
    font-size: clamp(2.5rem, 6vw, 4rem);
    color: #ffffff;
    margin-bottom: 0.75rem;
  }

  .hero-tagline {
    font-size: 1rem;
    color: #7070a0;
    margin: 0;
  }

  /* ── Shared section styles ── */
  .section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #c5c5e8;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #2a2a38;
  }

  .about-section,
  .personal-section {
    margin-bottom: 3rem;
  }

  /* ── Interest pills ── */
  .interest-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
  }

  .pill {
    background: #1e1e2e;
    border: 1px solid #3a3a55;
    color: #9090b8;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
  }

  .pill:hover {
    background: #2a1515;
    color: #e8e8f0;
    border-color: #d5000d;
  }

</style>
