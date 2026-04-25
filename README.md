# GMT Graph Memory Transformer

Graph Memory Transformer (GMT) v7 is a decoder-only language model in which the
usual feed-forward block is replaced by a graph-structured memory cell. After
causal self-attention, each token is softly assigned to a source memory state,
routed through a learned one-hop memory graph, and transformed by the gated
displacement between source and target memory states.

The implementation here follows the base v7 design: 16 decoder blocks, hidden
size 768, 12 attention heads, 128 memory slots per block, one-hop graph
traversal, centroid write-back, memory maintenance, and the auxiliary losses
used to keep routing and memory usage stable.

## Install

```bash
pip install -e .
```

For tests, install the optional test dependencies:

```bash
pip install -e .[test]
pytest
```

## Prepare Data

Training expects one contiguous `uint16` token stream for each split:

```text
data/prepared_owt/
├── train.bin
└── val.bin
```

Each file should contain token IDs written directly as unsigned 16-bit integers.
During loading, the stream is divided into windows of `seq_len + 1` tokens and
shifted into next-token prediction pairs. The default configuration uses
`seq_len: 1024` and `vocab_size: 50257`; if you use a different tokenizer,
adjust the configuration so the vocabulary size matches the token IDs.

## Train

The default configuration is in `configs/gmt_v7_base.yaml`:

```bash
python scripts/train_gmt_v7.py --config configs/gmt_v7_base.yaml
```

To resume from `latest.pt` or `best.pt` in the output directory:

```bash
python scripts/train_gmt_v7.py --config configs/gmt_v7_base.yaml --resume
```

Data and output directories can be overridden from the command line:

```bash
python scripts/train_gmt_v7.py \
  --data-dir data/prepared_owt \
  --output-dir runs/gmt_v7_base
```

## Evaluate

Validation runs during training and reports validation loss and perplexity. To
evaluate a saved checkpoint without running another training epoch, use
`--eval-only` with `--resume`:

```bash
python scripts/train_gmt_v7.py \
  --config configs/gmt_v7_base.yaml \
  --resume \
  --eval-only
```

The evaluation path uses `val.bin` from the configured data directory. It reads
the checkpoint from the configured output directory and reports validation loss
and perplexity without writing a new best checkpoint.

## Outputs

By default, training writes to `runs/gmt_v7_base/`:

```text
runs/gmt_v7_base/
├── best.pt
├── latest.pt
└── progress.csv
```

`latest.pt` is the most recent periodic checkpoint, `best.pt` is the checkpoint
with the lowest validation loss observed during training, and `progress.csv`
records validation metrics and memory diagnostics over time. Generated data,
checkpoints, logs, and run directories are ignored by git.

## Repository Layout

```text
GMT-GraphMemoryTransformer/
├── configs/
│   └── gmt_v7_base.yaml
├── scripts/
│   └── train_gmt_v7.py
├── src/
│   └── gmt/
│       ├── data.py
│       ├── model.py
│       └── train.py
└── tests/
    └── test_smoke.py
```
