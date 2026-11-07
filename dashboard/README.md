# Close the Books Mission Control (dashboard)

Live version (private until shared): https://claude.ai/artifact/3kyFbwq7iGgqbqfjwYS2c8

Source of the metrics dashboard: score tiles (latest LIVE upload, validation, holdout, organisers' benchmark), today's deadline
clock, score components by run, score progression, validation-vs-LIVE accuracy per segment, points lost per segment,
a sortable segment table, matcher precision and exception F1 (validation vs holdout).

- `index.html` is the page. It runs inside the claude.ai Dashboard artifact type, which provides `dash.data`,
  `dash.draw`, `dash.calc`, d3 and the theme tokens, so it does not open on its own in a browser.
- `datasets/*.json` are the datasets it reads (uploaded to the artifact as fixed files):
  `runs`, `progression`, `segments`, `matchers`, `exceptions`, `timeline`, `benchmarks`.
- `build_data.py` regenerates `runs`, `progression`, `segments`, `matchers` and `exceptions` from the pipeline outputs
  (`integration/out/{val,hold}`, the score logs) and the LIVE leaderboard per-segment results in
  `datasets/live_v1_segments.json`, `live_v2_segments.json`, `live_v3_segments.json` (LIVE v3 = latest, read as `accuracy_live`). Run it from its own folder after a new val/hold run.
