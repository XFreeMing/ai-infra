# AI Infra research workspace

## Purpose

Help the owner read first-party AI infrastructure discussions and source code on a Mac,
form testable hypotheses, validate them on PGX GB10, and publish reproducible evidence.
Use Chinese for research notes and user-facing documentation, preserving English
technical terms and short English summaries when useful for learning.

## Repository boundaries

- `sources/` contains upstream repositories managed as Git submodules. Follow each
  upstream repository's instructions when working inside it. Keep study notes and
  experiment harnesses in this parent repository.
- `research/` stores sources, people, topics, and study records. `experiments/` stores
  versioned experiment definitions and scripts. `reports/` stores reviewed findings.
- `artifacts/`, `models/`, `.venv/`, and `.cache/` are machine-local and are not committed.
  Do not add model weights, credentials, large traces, or caches to ordinary Git.
- The project tooling uses Python 3.12+ and the standard library. Keep dependencies
  small and add them only when needed for an actual workflow.

## Research evidence

- Cite upstream URLs and exact commits, files, and symbols for source-based claims.
  Distinguish the version being read from newer documentation or discussions.
- Label original authors' claims, AI inferences, and observations from executed
  experiments separately. Never describe a planned or CPU-only check as a GB10 result.
- End each study record with what is understood, unresolved questions, and a concrete
  next reading or experiment step so another session can continue it.
- Track relevant PR authors and reviewers with evidence links; do not infer maintainer
  authority from activity alone.

## Experiments and two-machine work

- Tie each execution to an experiment ID, unique run ID, parent commit, required
  submodule commits, complete configuration, input/model versions, and actual environment.
- Run long experiments from their own clean checkout and isolated environment. Do not
  update code or dependencies used by a running experiment.
- Commit and make upstream modifications retrievable before relying on a submodule
  pointer on another machine. A parent commit does not include uncommitted child edits.
- Check actual framework import paths and build provenance against the intended source.
  Record this in experiment evidence; a pinned checkout alone does not prove what ran.
- Begin with a small smoke test, then set an explicit runtime budget and stopping rule.
  Serialize performance measurements on the GB10 unless concurrency is the subject.
- Preserve failures, raw measurements, quality checks, and limits. Record external
  artifact locations and checksums; public reports need accessible supporting evidence.
- Work on Mac research may continue while PGX executes an older fixed snapshot.

## Validation

- Run `uv run --offline python -m unittest discover -s tests -v` for tooling changes.
  A Python 3.12+ interpreter can also run the same unittest command without uv.
- Run `uv run --offline python main.py doctor` to inspect local readiness. It does not
  install frameworks or demonstrate GPU compatibility.
- The `smoke` experiment validates recording and execution only; it is not an inference
  benchmark. GPU experiments must report their own environment and correctness checks.
