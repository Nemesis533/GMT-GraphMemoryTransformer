# GMT Graph Memory Transformer

Graph Memory Transformer (GMT) is a decoder-only language model architecture in
which the feed-forward transformation is replaced by a graph-structured memory
cell. Tokens are softly routed to source memory states, moved through a learned
memory graph, and transformed by the displacement between source and target
states.

This codebase contains the base v7 GMT implementation: causal self-attention,
one-hop graph-memory traversal, centroid write-back, memory-maintenance
utilities, the auxiliary losses used to stabilize routing, and a training
entrypoint for token-stream language modeling.

## Install

```bash
pip install -e .
```

## Training Data

The training entrypoint expects contiguous `uint16` token streams:

```text
data/prepared_owt/
├── train.bin
└── val.bin
```

Each sequence is read as `seq_len + 1` tokens and shifted into next-token
prediction pairs.

## Train

```bash
python scripts/train_gmt_v7.py --config configs/gmt_v7_base.yaml --resume
```

The default configuration uses the base v7 settings: 16 layers, hidden size
768, 12 attention heads, 128 memory slots per block, and a 128-dimensional
navigation space.

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
