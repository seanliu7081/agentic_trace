# Provenance

- `src/atrace/` was forked on 2026-11-22 from CP-5 (`/workspace/astra_harness_v5_cp5/src/cp5`), build `13686994640183a4`
  (hash rule of CP-5's run_episode.py: sha256 of src/cp5/*.py sorted + scripts/run_episode.py, first 16 hex).
- `scripts/run_episode.py`, `scripts/dev_episode.py`, `scripts/s1_stub.py` were forked from the same build.
- `.venv`, `third_party` and `artifacts` are symlinks into the CP-5 checkout (read-only use: Python 3.10 env with
  torch 2.10+cu128, MuJoCo 3.2.6, LIBERO assets, SAM 3.1 weights). Do not edit CP-5 from here.
- `src/traceplan/` is new: TRACE (TRACE_method.md v0.1) as changed by TRACE integration.md.
