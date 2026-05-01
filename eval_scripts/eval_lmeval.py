#!/usr/bin/env python3
"""
eval_lmeval.py  —  Unified lm-evaluation-harness wrapper (baseline / v7)

Architecture is auto-detected from the checkpoint state dict.
For v7 checkpoints the model is loaded via src.gmt.model.GMTV7;
for baseline checkpoints a GPT-2-style model is defined inline.

Temperature: v7 models have a learnable routing temperature buffer
(model.temp).  These scripts set it to TEMP_MIN (0.1) for evaluation
to sharpen routing — matching the paper's protocol.

Usage:
------
    python eval_scripts/eval_lmeval.py --model /path/to/checkpoint_dir --tasks hellaswag
    python eval_scripts/eval_lmeval.py --ckpt  /path/to/best.pt --tasks hellaswag,piqa
    python eval_scripts/eval_lmeval.py --model checkpoints/gmt_v7 --tasks arc_easy --limit 500

Options:
  --model         Checkpoint folder; loads best.pt from it
  --ckpt          Explicit checkpoint path (overrides --model)
  --tasks         Comma-separated lm_eval task names  (default: hellaswag)
  --num_fewshot   Few-shot examples                   (default: 0)
  --batch_size    Inference batch size                (default: 8)
  --device        cuda / cpu                          (default: auto-detect)
  --output_path   Directory to write JSON results (auto-generated if omitted)
  --limit         Max samples per task (int or 0–1 fraction)
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent

# ── lm_eval imports (graceful fail) ────
try:
    from lm_eval.api.model import LM
    from lm_eval import evaluator
    _LMEVAL_OK = True
except ImportError:
    _LMEVAL_OK = False
    LM = object

# ── imports for v7 model ─────────────────────────────────────────────────────
try:
    from src.gmt.model import GMTV7, GMTV7Config, TEMP_MIN
except ImportError:
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
    if any(k.startswith("blocks.") and ".cell." in k for k in sd):
        return "v7"
    return "baseline"

# ── model loader ────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
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

# ── lm_eval wrapper ─────────────────────────────────────────────────────────

class UnifiedLM(LM):
    """lm-evaluation-harness (≥0.4.x) wrapper for baseline and v7 models."""

    def __init__(self, model, mtype: str, device: torch.device,
                 batch_size: int = 8):
        super().__init__()
        self.model       = model
        self._mtype      = mtype
        self._device     = device
        self._batch_size = batch_size
        self.enc         = tiktoken.get_encoding("gpt2")
        self._max_len    = model.pos_embed.weight.shape[0]
        self._eot        = self.enc.eot_token
        self._vocab_size = model.embed.weight.shape[0]

    @property
    def eot_token_id(self) -> int:    return self._eot
    @property
    def max_length(self) -> int:      return self._max_len
    @property
    def max_gen_toks(self) -> int:    return 256
    @property
    def batch_size(self) -> int:      return self._batch_size
    @batch_size.setter
    def batch_size(self, v: int):     self._batch_size = v
    @property
    def device(self):                 return self._device

    def tok_encode(self, string: str) -> list:
        return self.enc.encode(string, allowed_special={"<|endoftext|>"})

    def tok_decode(self, tokens: list) -> str:
        return self.enc.decode(tokens)

    @torch.no_grad()
    def _run_batch(self, seqs: list) -> list:
        max_t = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), max_t), self._eot,
                         dtype=torch.long, device=self._device)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self._device)
        if self._mtype == "baseline":
            logits = self.model(ids)
        else:
            logits, _ = self.model(ids, return_mem_loss=False)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        return [log_probs[i, :len(s)] for i, s in enumerate(seqs)]

    def loglikelihood(self, requests) -> list:
        results: list = []
        pending: list = []

        def _flush(pending):
            seqs      = [ctx + cont for ctx, cont in pending]
            ctx_lens  = [len(ctx)  for ctx, cont in pending]
            cont_lens = [len(cont) for ctx, cont in pending]

            seqs_trunc = [s[-self._max_len:] for s in seqs]
            adj_ctx = [
                max(cl - (len(s) - len(st)), 0)
                for cl, s, st in zip(ctx_lens, seqs, seqs_trunc)
            ]

            lp_list = self._run_batch(seqs_trunc)
            out = []
            for lp, cs, cl, trunc in zip(lp_list, adj_ctx, cont_lens, seqs_trunc):
                if cs == 0 or cs + cl > lp.shape[0] + 1:
                    out.append((-float("inf"), False))
                    continue
                pred  = lp[cs - 1 : cs + cl - 1]
                tgts  = torch.tensor(trunc[cs : cs + cl],
                                     dtype=torch.long, device=self._device)
                ll        = pred[torch.arange(cl, device=self._device), tgts].sum().item()
                is_greedy = (pred.argmax(dim=-1) == tgts).all().item()
                out.append((ll, bool(is_greedy)))
            return out

        for req in requests:
            ctx_str, cont_str = req.args
            ctx  = self.tok_encode(ctx_str)
            cont = self.tok_encode(cont_str)
            if len(cont) == 0:
                results.append((-float("inf"), False))
                continue
            pending.append((ctx, cont))
            if len(pending) == self._batch_size:
                results.extend(_flush(pending))
                pending = []
        if pending:
            results.extend(_flush(pending))
        return results

    def loglikelihood_rolling(self, requests) -> list:
        results: list = []
        for req in requests:
            tokens = self.tok_encode(req.args[0])
            if len(tokens) <= 1:
                results.append(0.0)
                continue
            total_ll = 0.0
            start = 0
            while start < len(tokens) - 1:
                end   = min(start + self._max_len, len(tokens))
                chunk = tokens[start:end]
                ids   = torch.tensor([chunk], dtype=torch.long, device=self._device)
                if self._mtype == "baseline":
                    logits = self.model(ids)
                else:
                    logits, _ = self.model(ids, return_mem_loss=False)
                lp = F.log_softmax(logits[0].float(), dim=-1)
                score_from = 1 if start == 0 else 0
                for i in range(score_from, len(chunk) - 1):
                    total_ll += lp[i, chunk[i + 1]].item()
                start += self._max_len - 1
            results.append(total_ll)
        return results

    def generate_until(self, requests) -> list:
        results: list = []
        for req in requests:
            ctx_str, gen_kwargs = req.args
            max_new = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
            temp    = float(gen_kwargs.get("temperature", 1.0))
            until   = gen_kwargs.get("until", [])
            top_k   = int(gen_kwargs.get("top_k", 0))
            top_p   = float(gen_kwargs.get("top_p", 1.0))

            ctx_tokens = self.tok_encode(ctx_str)
            input_ids  = torch.tensor([ctx_tokens], dtype=torch.long,
                                      device=self._device)
            gen_tokens: list = []

            with torch.no_grad():
                for _ in range(max_new):
                    inp = input_ids[:, -self._max_len:]
                    if self._mtype == "baseline":
                        out = self.model(inp)
                    else:
                        out, _ = self.model(inp, return_mem_loss=False)
                    logits = out[0, -1, :].float()

                    if temp == 0.0:
                        next_tok = logits.argmax().unsqueeze(0)
                    else:
                        logits /= max(temp, 1e-8)
                        if top_k > 0:
                            kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][-1]
                            logits[logits < kth] = float("-inf")
                        if top_p < 1.0:
                            sorted_l, sorted_i = torch.sort(logits, descending=True)
                            cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                            remove = cum > top_p
                            remove[1:] = remove[:-1].clone()
                            remove[0]  = False
                            logits[sorted_i[remove]] = float("-inf")
                        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1)

                    tok_id = int(next_tok.item())
                    if tok_id == self._eot:
                        break
                    gen_tokens.append(tok_id)
                    input_ids = torch.cat([input_ids, next_tok.view(1, 1)], dim=1)

                    partial = self.enc.decode(gen_tokens)
                    hit = next((s for s in until if partial.endswith(s)), None)
                    if hit:
                        results.append(partial[: -len(hit)])
                        break
                else:
                    results.append(self.enc.decode(gen_tokens))
        return results

# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if not _LMEVAL_OK:
        sys.exit(
            "lm-evaluation-harness is not installed.\n"
            "Install with:  pip install lm-eval"
        )

    parser = argparse.ArgumentParser(
        description="Unified lm-evaluation-harness wrapper (baseline / v7)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model", type=str, default=None,
        help="Checkpoint folder; loads best.pt from it")
    group.add_argument(
        "--ckpt", type=str, default=None,
        help="Explicit checkpoint path (overrides --model)")
    parser.add_argument(
        "--tasks", type=str, default="hellaswag",
        help="Comma-separated lm_eval task names")
    parser.add_argument("--num_fewshot",  type=int,   default=0)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--device",       type=str,   default=None)
    parser.add_argument("--output_path",  type=str,   default=None,
                        help="Directory to write JSON (auto-generated in results/ if omitted)")
    parser.add_argument("--limit",        type=float, default=None,
                        help="Max samples per task (int count or 0–1 fraction)")
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
    lm = UnifiedLM(model=model, mtype=mtype, device=device,
                   batch_size=args.batch_size)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    logger.info(f"Tasks: {tasks}  |  few-shot={args.num_fewshot}"
                f"  |  limit={args.limit}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=str(device),
        limit=args.limit,
    )

    print("\n" + "=" * 66)
    print(f"  {model_label}  [{mtype}]  —  lm-evaluation-harness results")
    print("=" * 66)
    for task, metrics in results["results"].items():
        print(f"\n  {task}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric:<32s}  {value:.4f}")
            else:
                print(f"    {metric:<32s}  {value}")
    print("=" * 66)

    task_slug = args.tasks.replace(",", "_")[:60]
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_path:
        out_dir = Path(args.output_path)
    else:
        out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"lmeval_{model_label}_{task_slug}_{ts}.json"
    with out_file.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"Results saved → {out_file}")


if __name__ == "__main__":
    main()
