# Dataset atlas

A one-page visual briefing on the hackathon dataset for the team (and the project video): the task, phases and drift,
how the files connect, where the score points are, exceptions, items per line, entities and formats.

- **View:** open `atlas/index.html` in a browser (self-contained, ~75 KB; fonts from Google Fonts).
  Published copy: https://claude.ai/artifact/ACHuugFCU13rUYNaWD4m29 (private; ask Kenny to share).
- **Glossary:** 73 finance, banking and scoring terms, defined at the end of the page; every occurrence in the text has a dotted underline with the definition on hover/tap.
- **Every number is computed from `data/`.** Only FINAL's ~220k lines is the organisers' announced figure.
  The "LIVE-like window" is TRAIN bank lines booked 2025-01-01 → 2025-06-30 (199,987 lines, 735,474 pairs).

## Rebuild

```bash
python3 atlas/compute.py   # needs data/ incl. data/cache/*.parquet; writes atlas/agg.json (~1 min)
python3 atlas/build.py     # injects agg.json into atlas.src.html -> atlas/index.html
```

`agg.json` is committed so `build.py` works without the data.

## Takeaways

1. Remittance advices + collection batches: 4 % of lines, 38 % of (line, item) pairs (≈ 20 of 100 points).
2. Customer transfers (≈ 27 pts) and supplier payments (≈ 23 pts) are still the biggest block; `ar_ref_noise` alone ≈ 12.7.
3. Exception macro-F1 weighs every code equally: 171 `SHORT_BANK_CHARGES` rows = 15,793 `FX_DIFFERENCE` rows = 2 points each.
