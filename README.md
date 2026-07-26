# camtrap-longtail-dino
Evaluating design decisions in long-tailed camera-trap species classification.

## Experiments

See [`experiments/`](experiments/) for per-cycle write-ups and run instructions.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Linting

```bash
uv run ruff check --fix   # lint + auto-fix
uv run ruff format        # format
uv run pre-commit run --all-files  # run all hooks manually
```

Pre-commit hooks run automatically on `git commit` (ruff lint + format).

`experiments/cycle1/analysis.ipynb` and `bootstrap_methodology.ipynb` show the analysis
methodology (paths swapped to placeholders, outputs stripped) — metric definitions are
still being finalized, so results/figures aren't included yet. Re-run locally to reproduce.

## Dataset paths

Dataset roots (`base_dir` in `src/configs/dataset/*.yaml`, `ROOT`/`WILDS_ROOT`/`CACHE_FILE`
in a few scripts under `src/`) are placeholders — set them to wherever you keep
`iwildcam_224`, `serengeti_binary`, and the WILDS cache locally before running anything.
