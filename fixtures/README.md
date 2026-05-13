# Fixtures

This directory holds regression-test artifacts and the tooling that captures /
re-runs them.

## Layout

```
fixtures/
├── runner/
│   ├── digest_lib.py       — digest schema + capture/compare utilities
│   ├── run_v2.py           — re-run a fixture against the v2 pipeline and
│   │                          compare to the committed digest
│   └── digests/
│       └── <fixture_id>/expected.json
└── configs/                — v2 YAML configs that reproduce each fixture
    └── <fixture_id>.yaml
```

`digests/<fixture_id>/expected.json` is the committed regression target for
each fixture: a JSON record of output shapes, dtypes, summary statistics, and
a small set of sampled pixel values. The digest schema is documented in
`runner/digest_lib.py`.

## Capturing a digest

Digests are captured once per fixture in a Python environment that matches the
HCIPy release the fixture was originally written against. The capture script
is intentionally not part of the polished package and is invoked manually
during development.

## Re-running a fixture

```bash
pytest tests/fixtures/         # all fixture tests
pytest tests/fixtures/test_<fixture_id>.py
```

Each fixture test loads `fixtures/configs/<fixture_id>.yaml`, runs it through
the v2 pipeline with a fixed RNG seed, and compares against
`fixtures/runner/digests/<fixture_id>/expected.json`.
