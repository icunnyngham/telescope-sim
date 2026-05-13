# Conda environments

Reproducible conda environment YAMLs for development and the regression suite.

| File | Purpose |
|---|---|
| `env-dev.yaml` | Current development environment |

Additional pinned environments may be added here as the regression suite grows
(e.g. for capturing reference outputs against specific HCIPy releases).

## Usage

```bash
conda env create -f envs/env-dev.yaml
conda activate telescope-sim-dev
pip install -e ".[dev,doc]"
pytest
```
