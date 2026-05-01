#!/usr/bin/env python3
"""
eval_arc_easy.py  —  Unified ARC-Easy evaluator (baseline / v7)

Architecture is auto-detected from the checkpoint state dict.
For v7 checkpoints the model is loaded via src.gmt.model.GMTV7;
for baseline checkpoints a GPT-2-style model is defined inline.

Temperature: v7 models have a learnable routing temperature buffer
(model.temp).  These scripts set it to TEMP_MIN (0.1) for evaluation
to sharpen routing — matching the paper's protocol.

Usage:
------
    python eval_scripts/eval_arc_easy.py --model /path/to/checkpoint_dir
    python eval_scripts/eval_arc_easy.py --ckpt  /path/to/best.pt
    python eval_scripts/eval_arc_easy.py --model checkpoints/gmt_v7 --split validation --limit 200

Options:
  --model     Checkpoint folder; loads best.pt from it
  --ckpt      Explicit checkpoint path (overrides --model)
  --split     test / validation / train           (default: test)
  --limit     Evaluate only the first N examples
  --device    cuda / cpu                          (default: auto-detect)
  --output    Path to write JSON results (auto-generated if omitted)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── imports for v7 model ─────────────────────────────────────────────────────
try:
    from src.gmt.model import GMTV7, GMTV7Config, TEMP_MIN
except ImportError:
    # allow running from inside eval_scripts/
    sys.path.insert(0, str(_SCRIPT_DIR.parent))
    from src.gmt.model import GMTV7, GMTV7Config, TEMP_MIN

# ── checkpoint path resolution ──────────────────────────────────────────────

def resolve_checkpoint(model_arg: str) -> Path:
    p = Path(model_arg)
    candidates = [
        p / "best.pt",
        _SCRIPT_DIR / p / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    sys.exit(
        f"Cannot find best.pt for model '{model_arg}'.\nTried:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )

# ── baseline model (inline) ─────────────────────────────────────────────────

class _Attn(nn.Module):
    def __init__(self, hdim: int, n_heads: int):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hdim // n_heads
        self.qkv      = nn.Linear(hdim, 3 * hdim, bias=False)
        self.out_proj = nn.Linear(hdim, hdim, bias=False)

    def forward(self, x):
        B, T, H = x.shape
        q, k, v = self.qkv(x).split(H, dim=-1)
        def _r(t): return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(_r(q), _r(k), _r(v), is_causal=True)
        return self.out_proj(h.transpose(1, 2).contiguous().view(B, T, H))


class _MLP(nn.Module):
    def __init__(self, hdim: int, idim: int):
        super().__init__()
        self.fc1 = nn.Linear(hdim, idim, bias=False)
        self.fc2 = nn.Linear(idim, hdim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, hdim: int, idim: int, n_heads: int):
        super().__init__()
        self.ln1  = nn.LayerNorm(hdim)
        self.attn = _Attn(hdim, n_heads)
        self.ln2  = nn.LayerNorm(hdim)
        self.mlp  = _MLP(hdim, idim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _BaselineGPT(nn.Module):
    def __init__(self, vocab_size, hdim, n_layers, n_heads, idim, seq_len):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size, hdim)
        self.pos_embed  = nn.Embedding(seq_len, hdim)
        self.blocks     = nn.ModuleList(
            [_Block(hdim, idim, n_heads) for _ in range(n_layers)])
        self.norm_final = nn.LayerNorm(hdim)
        self.lm_head    = nn.Linear(hdim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm_final(h))

# ── model type detection ────────────────────────────────────────────────────

def detect_model_type(sd: dict) -> str:
    """Infer model family from checkpoint state-dict keys."""
    if any(k.startswith("blocks.") and ".cell." in k for k in sd):
        return "v7"
    return "baseline"

# ── model loader ────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    """Load any supported model from a checkpoint. Returns (model, model_type)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd   = ckpt["model"]

    mtype = detect_model_type(sd)
    step  = ckpt.get("step", "?")
    loss  = ckpt.get("loss", float("nan"))

    if mtype == "v7":
        model = GMTV7()
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        model.temp.fill_(TEMP_MIN)
        logger.info(f"Type       : v7  ({model.n_layers}L  hdim={model.hdim})")
    else:
        hdim     = sd["embed.weight"].shape[1]
        vocab    = sd["embed.weight"].shape[0]
        seq_len  = sd["pos_embed.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.startswith("blocks.")
                       and k.endswith(".ln1.weight"))
        idim     = sd["blocks.0.mlp.fc1.weight"].shape[0]
        n_heads  = 12

        model = _BaselineGPT(
            vocab_size=vocab, hdim=hdim, n_layers=n_layers,
            n_heads=n_heads, idim=idim, seq_len=seq_len,
        )
        remapped = {k.replace("ln_f.", "norm_final."): v for k, v in sd.items()}
        model.load_state_dict(remapped, strict=False)
        model.to(device).eval()
        logger.info(f"Type       : baseline  ({n_layers}L  hdim={hdim}  idim={idim})")

    logger.info(f"Checkpoint : {ckpt_path}")
    logger.info(f"Step       : {step}   val_loss={loss:.4f}")
    return model, mtype

# ── forward wrapper ─────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_logits(model, ids: torch.Tensor, mtype: str) -> torch.Tensor:
    if mtype == "baseline":
        return model(ids)
    else:
        logits, _ = model(ids, return_mem_loss=False)
        return logits

# ── batch scoring ───────────────────────────────────────────────────────────

def _batch_score(model, mtype: str, ctx_tokens: list, choices: list,
                 device: torch.device, max_len: int):
    seqs  = [(ctx_tokens + ch)[-max_len:] for ch in choices]
    max_t = max(len(s) for s in seqs)
    enc_  = tiktoken.get_encoding("gpt2")
    eot   = enc_.eot_token

    ids = torch.full((len(seqs), max_t), eot, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    logits = _forward_logits(model, ids, mtype)
    lp     = F.log_softmax(logits.float(), dim=-1)

    raw_scores, norm_scores = [], []
    for i, (seq, ch) in enumerate(zip(seqs, choices)):
        n_ctx  = len(seq) - len(ch)
        n_cont = len(ch)
        pred   = lp[i, n_ctx - 1 : n_ctx + n_cont - 1]
        tgts   = torch.tensor(seq[n_ctx:n_ctx + n_cont],
                               dtype=torch.long, device=device)
        ll = pred[torch.arange(n_cont, device=device), tgts].sum().item()
        raw_scores.append(ll)
        norm_scores.append(ll / max(n_cont, 1))
    return raw_scores, norm_scores

# ── ARC-Easy helpers ────────────────────────────────────────────────────────

def _format_question(question: str, choices: dict):
    ctx          = f"Question: {question}\nAnswer:"
    choice_texts = [f" {text}" for text in choices["text"]]
    return ctx, choice_texts


def _answer_index(answer_key: str, labels: list) -> int:
    if answer_key in labels:
        return labels.index(answer_key)
    return {"1": 0, "2": 1, "3": 2, "4": 3}.get(answer_key, 0)

# ── evaluation loop ─────────────────────────────────────────────────────────

def evaluate(model, mtype: str, enc, dataset, device: torch.device,
             max_len: int) -> dict:
    n_correct_raw = n_correct_norm = n_total = 0
    per_example = []

    for ex in tqdm(dataset, desc="ARC-Easy", unit="q"):
        correct_idx         = _answer_index(ex["answerKey"], ex["choices"]["label"])
        ctx, choice_texts   = _format_question(ex["question"], ex["choices"])
        ctx_toks            = enc.encode(ctx)
        choice_toks         = [enc.encode(c) for c in choice_texts]

        raw, norm = _batch_score(model, mtype, ctx_toks, choice_toks, device, max_len)

        pred_raw  = int(torch.tensor(raw).argmax().item())
        pred_norm = int(torch.tensor(norm).argmax().item())
        n_correct_raw  += int(pred_raw  == correct_idx)
        n_correct_norm += int(pred_norm == correct_idx)
        n_total        += 1

        per_example.append({
            "id":          ex.get("id", n_total),
            "correct_idx": correct_idx,
            "pred_raw":    pred_raw,
            "pred_norm":   pred_norm,
            "raw_scores":  raw,
            "norm_scores": norm,
        })

    return {
        "n_total":        n_total,
        "n_correct_raw":  n_correct_raw,
        "n_correct_norm": n_correct_norm,
        "acc_raw":        n_correct_raw  / n_total,
        "acc_norm":       n_correct_norm / n_total,
        "per_example":    per_example,
    }

# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified ARC-Easy evaluator (baseline / v7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model", type=str, default=None,
        help="Checkpoint folder; loads best.pt from it")
    group.add_argument(
        "--ckpt", type=str, default=None,
        help="Explicit checkpoint path (overrides --model)")
    parser.add_argument("--split",  type=str, default="test",
                        choices=["test", "validation", "train"])
    parser.add_argument("--limit",  type=int, default=None,
                        help="Evaluate only the first N examples")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON (auto-generated in results/ if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            sys.exit(f"Checkpoint not found: {ckpt_path}")
    elif args.model:
        ckpt_path = resolve_checkpoint(args.model)
    else:
        sys.exit("Provide --model <name> or --ckpt <path>")

    model_label = ckpt_path.parent.name
    model, mtype = load_model(ckpt_path, device)
    enc     = tiktoken.get_encoding("gpt2")
    max_len = model.pos_embed.weight.shape[0]

    logger.info(f"Loading ARC-Easy ({args.split} split) …")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    logger.info(f"Examples: {len(dataset)}")

    results = evaluate(model, mtype, enc, dataset, device, max_len)

    print("\n" + "=" * 60)
    print(f"  {model_label}  [{mtype}]  —  ARC-Easy results")
    print("=" * 60)
    print(f"  Split           : {args.split}")
    print(f"  Checkpoint      : {ckpt_path}")
    print(f"  Examples        : {results['n_total']}")
    print(f"  Accuracy (raw)  : {results['acc_raw']:.4f}"
          f"  ({results['n_correct_raw']} / {results['n_total']})")
    print(f"  Accuracy (norm) : {results['acc_norm']:.4f}"
          f"  ({results['n_correct_norm']} / {results['n_total']})")
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"arc_easy_{model_label}_{args.split}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "task":        "arc_easy",
        "model":       model_label,
        "model_type":  mtype,
        "split":       args.split,
        "checkpoint":  str(ckpt_path),
        "n_total":     results["n_total"],
        "acc_raw":     results["acc_raw"],
        "acc_norm":    results["acc_norm"],
        "per_example": results["per_example"],
    }
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
