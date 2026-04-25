# GMT Graph Memory Transformer

Public reference repository for the base v7 Graph Memory Transformer (GMT).

This repository is being structured as a clean research implementation of the
base GMT v7 model. The first public target is a minimal, reproducible codebase
for training and evaluating the v7 architecture without historical variants,
paper build artifacts, or private experiment outputs.

## Planned Layout

```text
GMT-GraphMemoryTransformer/
├── configs/
│   └── gmt_v7_base.yaml
├── scripts/
│   └── train_gmt_v7.py
├── src/
│   └── gmt/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       └── train.py
└── tests/
    └── test_smoke.py
```

The implementation will be ported from the validated base v7 code path, keeping
the model behavior faithful while cleaning naming, configuration, data paths,
and public documentation.
