# Development

PoseSearch v0.1.0 uses Python 3.12 as the primary CI baseline.

## Test baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
python -m compileall -q src tests
```

The initial local baseline before repository import passed 16 tests and the compile check.

## Inference setup

Inference dependencies and model weights are deliberately separate from the unit-test baseline:

```bash
pip install -e '.[inference]'
python scripts/resolve_models.py
posesearch models-verify
```

Do not commit downloaded model weights, local SQLite databases, media libraries, benchmark output, or private evaluation data.
