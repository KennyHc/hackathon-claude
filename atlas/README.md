# Dataset atlas

A one-page visual briefing on the hackathon dataset for the team (and the project video): the task, phases and drift,
how the files connect, where the score points are, exceptions, items per line, entities and formats.

- **Live:** https://kennyhc.github.io/hackathon-claude/atlas/ (GitHub Pages, `gh-pages` branch: a copy of `index.html`
  and `og.png`). Locally: open `atlas/index.html` in a browser (self-contained, ~110 KB; fonts from Google Fonts).
- **Glossary:** 73 finance, banking and scoring terms, defined at the end of the page; every occurrence in the text has a dotted underline with the definition on hover/tap.
- **Every number is computed from `data/`.** FINAL figures come from the FINAL input files (no answers); the results
  (96.71, per-case-type accuracy) come from the official leaderboard breakdown and `insights.md`.
  The "LIVE-like window" is TRAIN bank lines booked 2025-01-01 → 2025-06-30 (199,987 lines, 735,474 pairs).

## Rebuild

```bash
python3 atlas/compute.py   # needs data/ incl. data/cache/*.parquet; writes atlas/agg.json (~1 min)
python3 atlas/compute_final.py   # FINAL inputs only -> atlas/agg_final.json (~10 s)
python3 atlas/build.py     # injects both into atlas.src.html -> atlas/index.html (head above <!--/HEAD-->)
```

`agg.json` and `agg_final.json` are committed so `build.py` works without the data. `og.png` (the 1200×630 link
preview) is a screenshot of `og.html` at 1200×630.

**Publish:** copy `index.html` and `og.png` to `atlas/` on the `gh-pages` branch and push.

**Solution guide** (`guide/`): a companion page on how the pipeline works (matchers, classifier, ranker, FINAL day), live at
https://kennyhc.github.io/hackathon-claude/guide/. It is hand-written HTML with no build step: edit `guide/index.html`;
`guide/og.html` is the source of `guide/og.png`. Publish by copying both to `guide/` on the `gh-pages` branch.

## Takeaways

1. Remittance advices + collection batches: 4 % of lines, 38 % of (line, item) pairs (≈ 20 of 100 points).
2. Customer transfers (≈ 27 pts) and supplier payments (≈ 23 pts) are still the biggest block; `ar_ref_noise` alone ≈ 12.7.
3. Exception macro-F1 weighs every code equally: 171 `SHORT_BANK_CHARGES` rows = 15,793 `FX_DIFFERENCE` rows = 2 points each.

**Infographic** (`infographic/`): the one-page LinkedIn summary (1080×1350, shown at 2×), live at
https://kennyhc.github.io/hackathon-claude/infographic/. `index.html` shows `infographic-light.png` or `-dark.png` by
colour scheme; `source-*.html` are the HTML sources the PNGs were rendered from. Publish by copying `index.html` and both
PNGs to `infographic/` on the `gh-pages` branch.
