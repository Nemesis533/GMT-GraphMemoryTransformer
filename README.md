# GMT Graph Memory Transformer

Graph Memory Transformer (GMT) is a decoder-only language model architecture in
which the feed-forward transformation is replaced by a graph-structured memory
cell. Tokens are softly routed to source memory states, moved through a learned
memory graph, and transformed by the displacement between source and target
states.

This repository contains the base v7 GMT implementation layout. The code is
organized around a small package, explicit configuration, script entrypoints,
and portable tests.

## Repository Layout

```text
GMT-GraphMemoryTransformer/
├── LICENSE
├── README.md
├── configs/
│   └── gmt_v7_base.yaml
├── pyproject.toml
├── requirements.txt
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

The next implementation step is to fill this layout with the base v7 model,
training loop, configuration values, and portable validation tests.
