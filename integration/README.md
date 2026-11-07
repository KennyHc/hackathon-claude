# integration (piece D)

**Mission:** own the end-to-end pipeline and the **one FINAL submission**.

1. `docs/CONTRACTS.md`: keep it current; you're the only one who edits it (others ask on the board).
2. `integration/run.py --phase <p>`: merge `matching/out` + `classification/out` into `integration/out/<p>/submission.csv.gz`
   (each bank line gets either item rows or one category row, never both; every line of the phase is present).
3. `integration/validate.py`: reproduce every rule of `docs/FORMAT.md` (a rejected FINAL file doesn't count).
4. `integration/score.py`: wrap `docs/starter/bankrec_score.py`; report score + per-segment accuracy on sample/train_lite/train,
   so the team knows where points are lost. Post scores on the board.
5. `integration/submit.py`: copy of the reference script on https://hackathon.usetalky.com/enviar (token from `HK_TOKEN`
   env var only; poll ≥ 5 s). You are the only one who submits: the quota is per team (docs/HACKATHON.md).
6. First thing: get `docs/starter/starter.py` output through validate → score → LIVE upload, so the pipeline works end to end
   before anything clever exists.
