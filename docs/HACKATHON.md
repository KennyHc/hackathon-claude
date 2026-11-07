# Hackathon logistics (from https://hackathon.usetalky.com, read 2026-11-07)

Event: **Saturday 7 Nov 2026, 09:00–21:00, La Nave, Madrid**. All times are Madrid time.

## Today's deadlines

| Time | What | Who acts |
|---|---|---|
| 10:00 | Hackathon starts, LIVE phase opens | everyone |
| 18:00 | **FINAL data released** (220 k lines, 2026-01 → 2026-06) → download into `data/final/` | integration owner posts INFO on the board |
| 19:30 | **Submissions close** (the one FINAL submission must be in before this) | integration owner |
| 19:45 | **Project delivery**: video + repo + description (page `/proyecto`) | whole team |
| 20:30 | Final leaderboard reveal | |

Plan backwards: the full pipeline must run end to end on `live` **well before 18:00**, so FINAL is just "rerun on
`final` → validate → submit once" (it's ~220 k lines, so allow for runtime).

## Submitting (integration owner only)

- Limits are **per team**: **1 submission every 3 min per phase, 20 per day per phase**, file ≤ 60 MB and ≤ 3 M rows,
  `.csv` or `.csv.gz`. **Asking for an upload URL already uses up a slot**, even if you never upload.
  → Only the integration owner submits. Others post a NEED on the board with the file to try.
- LIVE: scored in minutes, your best score counts. FINAL: **one shot**: the first *valid* file counts, can't be
  redone, score hidden until 20:30. A file with format errors doesn't use up the FINAL shot.
  Teams without a FINAL submission are not ranked.
- Team token: in an env var only (`read -rs HK_TOKEN && export HK_TOKEN`). **Never in code, files, commits or board
  posts.** Lost or leaked → ask the organisers to reset it.
- API (`https://hackathon.usetalky.com/api`, header `Authorization: Bearer $HK_TOKEN`):
  1. `POST /submissions/upload-url` `{"phase":"LIVE"|"FINAL","filename":"…"}` → `sub_id` + presigned S3 `upload`
     (201)
  2. `POST` the file to `upload.url` with all `upload.fields` (form fields first, file last)
  3. `POST /submissions/{sub_id}/score` `{}`
  4. `GET /submissions/{sub_id}`: poll **no more than every 5 s** (the whole venue shares one IP) until `status` ∈
     `scored | received | invalid | error`
  The site's reference Python script for this is on the `/enviar` page; integration keeps our copy in
  `integration/submit.py`.

## Reference scores (from the rules page)

| Approach | Score |
|---|---|
| Starter as shipped | ~10–15 |
| Naive: exact amount + reference | ≈ 19 |
| Well-worked heuristic rules | ≈ 61 |
| Perfect (the golden) | 100 |

The result page also shows (unweighted) amount-weighted accuracy, per-segment accuracy (1:1, 1:N, FX, returns…) and
counts: integration posts these on the board after each LIVE submission so each piece knows where points are lost.

## Project delivery (19:45): the jury judges the approach too

1. **Video** ≤ 5 min, 720p: what is KEY in our solution, the approach, the results.
2. **Repo link**: if private, the jury must be able to see it (make it public at delivery or share access when asked).
   → Keep the repo presentable: each piece's README says what it does and how to run it.
3. **Description**: key ideas · stack and models used · what we'd do next · team members.

Only the organisers and the jury see the project. Fair play: don't share the token, don't attack the platform, one
person per team, don't try to obtain LIVE/FINAL golden. Any tool, language or model (including Claude) is allowed.
