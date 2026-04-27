<svelte:head>
  <title>Projects — Reagan Howell</title>
  <meta name="description" content="Personal projects: Audiobookshelf home server setup and an Arduino chess clock build." />
</svelte:head>

<section class="page-header">
  <h1>Personal Projects</h1>
  <p class="page-subtitle">Hardware tinkering and self-hosted software.</p>
</section>

<div class="projects-list">

  <!-- Project 1: Audiobookshelf -->
  <article class="project-card">
    <div class="project-card-header">
      <div class="project-meta">
        <span class="project-tag">Self-Hosted</span>
        <span class="project-tag">Raspberry Pi</span>
        <span class="project-tag">Docker</span>
      </div>
      <h2 class="project-title">Audiobooks Home Server</h2>
      <p class="project-subtitle">A personal Audible — running on a repurposed desktop.</p>
    </div>

    <div class="project-image-wrap">
      <img src="/images/abs.png" alt="Audiobookshelf interface screenshot" loading="lazy" />
    </div>

    <div class="project-body">
      <p>
        Over the years my family has collected a large audiobook library. After we moved out,
        accessing books we already own became difficult. I found
        <a href="https://www.audiobookshelf.org/" target="_blank" rel="noopener noreferrer">Audiobookshelf</a>
        — an open-source, self-hosted audiobook server — a personal Audible, a <em>personaudible</em>,
        if you will.
      </p>
      <p>
        I started with a spare Raspberry Pi 4B running the server in a Docker container. It handled
        one or two simultaneous listeners fine, but the family's usage quickly exceeded the Pi's
        limits. I sourced a cheap old desktop, repurposed it as an Ubuntu server, and migrated the
        library there. Much better.
      </p>
    </div>
  </article>

  <!-- Project 2: Chess Clock -->
  <article class="project-card">
    <div class="project-card-header">
      <div class="project-meta">
        <span class="project-tag">Arduino</span>
        <span class="project-tag">Hardware</span>
        <span class="project-tag">C++</span>
      </div>
      <h2 class="project-title">Homemade Chess Clock</h2>
      <p class="project-subtitle">Why pay $60 when you have an Arduino and too much confidence?</p>
    </div>

    <div class="project-image-wrap chess-image">
      <img src="/images/Chess.png" alt="Hand-drawn chess set used at tournaments" loading="lazy" />
    </div>

    <div class="project-body">
      <p>
        Early in the Covid-19 pandemic I became enamored by chess. Like many people, I made a
        <a href="https://www.chess.com/member/rhowel33" target="_blank" rel="noopener noreferrer">chess.com</a>
        account and worked through puzzles and games. Puzzle solving was more compelling to me —
        given enough time, the right move exists. The thrill of winning (and the utter disappointment
        of losing) kept me hooked.
      </p>
      <p>
        I eventually got a US Chess Federation membership and started playing small tournaments.
        Most events ask players to bring a board and a clock to keep costs down. Clocks sell for
        around $60 — so naturally I thought: <em>how hard could it be to make my own?</em>
      </p>
      <p>
        I dug out my old Arduino Uno from an EE class and sourced a breadboard, buttons, an LCD
        screen from eBay, resistors, and what felt like a million jumper wires. Refreshing my C++
        muscle memory was straightforward enough. The real challenge was troubleshooting hardware
        from questionable sellers — poor solder joints, unreliable components, and messy code all
        compounded. I got a half-working clock before the LCD decided to only display its left half.
        The project is shelved until I can source a reliable replacement screen.
      </p>
    </div>

    <div class="project-status">
      <span class="status-badge status-shelved">Shelved — awaiting LCD replacement</span>
    </div>
  </article>

</div>

<style>
  /* ── Page header ── */
  .page-header {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #2a2a38;
    margin-bottom: 2.5rem;
  }

  .page-header h1 {
    font-size: clamp(2rem, 5vw, 3rem);
    color: #ffffff;
    margin-bottom: 0.5rem;
  }

  .page-subtitle {
    color: #7070a0;
    font-size: 1rem;
    margin: 0;
  }

  /* ── Projects list ── */
  .projects-list {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  /* ── Project card ── */
  .project-card {
    background: #16161e;
    border: 1px solid #2a2a38;
    border-radius: 14px;
    overflow: hidden;
    transition: border-color 0.25s ease, box-shadow 0.25s ease;
  }

  .project-card:hover {
    border-color: #d5000d;
    box-shadow: 0 0 0 1px rgba(213, 0, 13, 0.25), 0 8px 32px rgba(213, 0, 13, 0.06);
  }

  .project-card-header {
    padding: 1.75rem 1.75rem 0;
  }

  .project-meta {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
  }

  .project-tag {
    background: #1e1e2e;
    border: 1px solid #3a3a55;
    color: #d5000d;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
  }

  .project-title {
    font-size: 1.5rem;
    color: #e8e8f0;
    margin-bottom: 0.3rem;
  }

  .project-subtitle {
    color: #7070a0;
    font-size: 0.92rem;
    margin-bottom: 0;
  }

  /* ── Project image ── */
  .project-image-wrap {
    margin: 1.25rem 1.75rem;
    border-radius: 8px;
    overflow: hidden;
    background: #0f0f13;
    display: flex;
    align-items: center;
    justify-content: center;
    max-height: 280px;
  }

  .project-image-wrap img {
    max-width: 100%;
    max-height: 280px;
    object-fit: contain;
    display: block;
  }

  .chess-image {
    max-height: 220px;
  }

  .chess-image img {
    max-height: 220px;
  }

  /* ── Project body text ── */
  .project-body {
    padding: 0 1.75rem 1.5rem;
  }

  .project-body p {
    color: #c0c0d8;
    font-size: 0.95rem;
    line-height: 1.75;
  }

  .project-body p:last-child {
    margin-bottom: 0;
  }

  /* ── Status badge ── */
  .project-status {
    padding: 0 1.75rem 1.5rem;
  }

  .status-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    letter-spacing: 0.03em;
  }

  .status-shelved {
    background: #2a1010;
    border: 1px solid #5a1515;
    color: #d56060;
  }

  /* ── Responsive ── */
  @media (max-width: 520px) {
    .project-card-header,
    .project-body,
    .project-status {
      padding-left: 1.25rem;
      padding-right: 1.25rem;
    }

    .project-image-wrap {
      margin: 1rem 1.25rem;
    }
  }
</style>
