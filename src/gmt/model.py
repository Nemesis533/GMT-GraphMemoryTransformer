"""Base v7 Graph Memory Transformer model."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SEQ_LEN = 1024
HDIM = 768
N_HEADS = 12
N_LAYERS = 16
MEM_SLOTS = 128
NAV_DIM = 128

ATTN_DROPOUT = 0.1
EMBED_DROPOUT = 0.1

GRAV_EPS = 0.01
TEMP_MAX = 1.0
TEMP_MIN = 0.1

LAMBDA_TRACK = 1.0
ORTHO_BETA = 0.05
LAMBDA_CLUSTER = 0.3
LAMBDA_EDGE = 0.1
LAMBDA_CONTRAST = 0.5
EDGE_ENT_TARGET = 4.0

DEAD_THRESHOLD = 1e-3
MERGE_THRESH = 0.95
CENTROID_COOLDOWN = 100


def _compile_disabled(fn):
    compiler = getattr(torch, "compiler", None)
    disable = getattr(compiler, "disable", None)
    return disable(fn) if disable is not None else fn


@dataclass(frozen=True)
class GMTV7Config:
    """Configuration for the base v7 GMT architecture."""

    vocab_size: int = 50257
    seq_len: int = SEQ_LEN
    hidden_dim: int = HDIM
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    memory_slots: int = MEM_SLOTS
    nav_dim: int = NAV_DIM
    attention_dropout: float = ATTN_DROPOUT
    embedding_dropout: float = EMBED_DROPOUT
    grav_eps: float = GRAV_EPS
    temp_max: float = TEMP_MAX
    temp_min: float = TEMP_MIN
    edge_entropy_target: float = EDGE_ENT_TARGET
    dead_threshold: float = DEAD_THRESHOLD
    merge_threshold: float = MERGE_THRESH
    centroid_cooldown: int = CENTROID_COOLDOWN

    @property
    def cluster_target(self) -> float:
        return float(self.memory_slots // 4)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention wrapper used by GMT blocks."""

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_bias,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        is_causal=is_causal if attn_bias is None else False,
    )


class CausalSelfAttention(nn.Module):
    """Causal self-attention used before each GMT memory block."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = ATTN_DROPOUT):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_dim = x.shape
        q, k, v = self.qkv(x).split(hidden_dim, dim=-1)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape_heads(q), reshape_heads(k), reshape_heads(v)
        h = flash_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        h = h.transpose(1, 2).contiguous().view(batch, seq_len, hidden_dim)
        return self.out_proj(h)


class GraphMemoryCell(nn.Module):
    """Graph-structured memory cell used as the GMT feed-forward replacement."""

    def __init__(
        self,
        hidden_dim: int,
        n_slots: int,
        nav_dim: int,
        block_idx: int,
        grav_eps: float = GRAV_EPS,
        cluster_target: Optional[float] = None,
        edge_entropy_target: float = EDGE_ENT_TARGET,
        dead_threshold: float = DEAD_THRESHOLD,
        merge_threshold: float = MERGE_THRESH,
        centroid_cooldown: int = CENTROID_COOLDOWN,
    ):
        super().__init__()
        self.hdim = hidden_dim
        self.F = n_slots
        self.nav_dim = nav_dim
        self.block_idx = block_idx
        self.scale = math.sqrt(nav_dim)
        self.grav_eps = grav_eps
        self.cluster_target = (
            float(n_slots // 4) if cluster_target is None else float(cluster_target)
        )
        self.edge_entropy_target = edge_entropy_target
        self.dead_threshold = dead_threshold
        self.merge_threshold = merge_threshold
        self.centroid_cooldown = centroid_cooldown

        self.C = nn.Parameter(F.normalize(torch.randn(n_slots, hidden_dim), dim=-1))
        self.E = nn.Parameter(torch.randn(n_slots, n_slots) * 1.0)

        diag_mask = torch.zeros(n_slots, n_slots)
        diag_mask.fill_diagonal_(float("-inf"))
        self.register_buffer("diag_mask", diag_mask)

        self.Q_proj = nn.Linear(hidden_dim, nav_dim, bias=False)
        self.K_proj = nn.Linear(hidden_dim, nav_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(1.0))
        self.write_momentum = nn.Parameter(torch.tensor(4.6))
        self.norm_C = nn.LayerNorm(hidden_dim)
        self.norm_disp = nn.LayerNorm(hidden_dim)

        self.register_buffer("usage", torch.ones(n_slots) / n_slots)
        self.register_buffer("centroid_age", torch.zeros(n_slots))

    @_compile_disabled
    def _source_weights(
        self,
        h_n: torch.Tensor,
        C_n: torch.Tensor,
        temp: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        sim = h_n @ C_n.T
        dist = (1.0 - sim).clamp(min=self.grav_eps)
        return F.softmax((1.0 / dist) / temp, dim=-1)

    @_compile_disabled
    def _traverse(
        self,
        x: torch.Tensor,
        w_src: torch.Tensor,
        C_normed: torch.Tensor,
    ) -> torch.Tensor:
        E_soft = F.softmax(self.E + self.diag_mask, dim=-1)
        w_edge = w_src @ E_soft
        q = self.Q_proj(x)
        k = self.K_proj(C_normed)
        ctx_scores = (q @ k.T) / self.scale
        return F.softmax(w_edge + ctx_scores, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        temp: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        C_normed = self.norm_C(self.C)
        C_n = F.normalize(C_normed, dim=-1)
        h_n = F.normalize(x, dim=-1)

        w_src = self._source_weights(h_n, C_n, temp)
        w_tgt = self._traverse(x, w_src, C_normed)

        C_src = w_src @ C_normed
        C_tgt = w_tgt @ C_normed
        disp = self.norm_disp(C_tgt - C_src)

        return torch.sigmoid(self.gate) * disp, w_src, w_tgt, C_normed

    def tracking_loss(
        self,
        x: torch.Tensor,
        w_src: torch.Tensor,
        C_normed: torch.Tensor,
    ) -> torch.Tensor:
        m = torch.sigmoid(self.write_momentum)
        weighted = w_src.reshape(-1, self.F) @ C_normed
        x_flat = x.reshape(-1, self.hdim)
        return (1.0 - m) * F.mse_loss(x_flat.detach(), weighted)

    def orthogonality_loss(self) -> torch.Tensor:
        C_n = F.normalize(self.C, dim=-1)
        gram = C_n @ C_n.T
        mask = ~torch.eye(self.F, dtype=torch.bool, device=self.C.device)
        return gram[mask].pow(2).mean()

    def clustering_penalty(self, w_src: torch.Tensor) -> torch.Tensor:
        usage = w_src.reshape(-1, self.F).mean(0)
        usage = usage / (usage.sum() + 1e-8)
        entropy = -(usage * (usage + 1e-8).log()).sum()
        n_effective = entropy.exp()
        return (
            self.cluster_target / n_effective.clamp(min=1.0) - 1.0
        ).clamp(min=0.0)

    def edge_entropy_loss(self) -> torch.Tensor:
        E_soft = F.softmax(self.E + self.diag_mask, dim=-1)
        row_entropy = -(E_soft * (E_soft + 1e-8).log()).sum(dim=-1)
        return F.relu(self.edge_entropy_target - row_entropy).mean()

    def edge_contrastive_loss(self) -> torch.Tensor:
        E_soft = F.softmax(self.E + self.diag_mask, dim=-1)
        E_n = F.normalize(E_soft, dim=-1)
        row_sim = E_n @ E_n.T
        mask = ~torch.eye(self.F, dtype=torch.bool, device=self.C.device)
        return row_sim[mask].mean()

    @_compile_disabled
    @torch.no_grad()
    def write_back(self, h: torch.Tensor, w_src: torch.Tensor) -> None:
        m = torch.sigmoid(self.write_momentum).item()
        h_flat = h.reshape(-1, self.hdim).detach()
        w_flat = w_src.reshape(-1, self.F).detach()

        assigns = w_flat.argmax(dim=-1)
        one_hot = F.one_hot(assigns, num_classes=self.F).float()

        h_agg = one_hot.T @ h_flat
        counts = one_hot.sum(0).clamp(min=1e-8)
        h_mean = h_agg / counts.unsqueeze(-1)

        updated = m * self.C.data + (1.0 - m) * h_mean
        self.C.data = F.normalize(updated, dim=-1)
        self.centroid_age.add_(1.0)

    @torch.no_grad()
    def update_usage(self, w_src: torch.Tensor) -> None:
        batch_usage = w_src.reshape(-1, self.F).mean(0).detach()
        self.usage.mul_(0.99).add_(batch_usage * 0.01)

    @_compile_disabled
    @torch.no_grad()
    def merge_similar_centroids(self, x: torch.Tensor) -> int:
        C_n = F.normalize(self.C, dim=-1)
        gram = C_n @ C_n.T
        gram.fill_diagonal_(0.0)
        young = self.centroid_age < self.centroid_cooldown
        gram[young, :] = 0.0
        gram[:, young] = 0.0
        upper = torch.triu(gram, diagonal=1)
        pairs = (upper > self.merge_threshold).nonzero(as_tuple=False)
        if len(pairs) == 0:
            return 0

        x_flat = x.reshape(-1, self.hdim).detach()
        n_merged = 0
        for i, j in pairs.tolist():
            Ci = F.normalize(self.C[i : i + 1], dim=-1)
            Cj = F.normalize(self.C[j : j + 1], dim=-1)
            if (Ci @ Cj.T).item() < self.merge_threshold:
                continue
            reset_idx = j if self.usage[i] >= self.usage[j] else i
            src = torch.randint(0, x_flat.shape[0], (1,), device=x_flat.device).item()
            self.C.data[reset_idx] = F.normalize(
                x_flat[src : src + 1], dim=-1
            ).squeeze(0)
            self.usage[reset_idx] = 1.0 / self.F
            self.centroid_age[reset_idx] = 0.0
            n_merged += 1
        return n_merged

    @_compile_disabled
    @torch.no_grad()
    def reset_dead_centroids(self, x: torch.Tensor) -> int:
        dead = (self.usage < self.dead_threshold).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0

        x_flat = x.reshape(-1, self.hdim).detach()
        if len(dead) > self.F * 0.5:
            resets = F.normalize(
                torch.randn(len(dead), self.hdim, device=x_flat.device), dim=-1
            )
        else:
            idx = torch.randint(0, x_flat.shape[0], (len(dead),), device=x_flat.device)
            resets = F.normalize(x_flat[idx], dim=-1)

        self.C.data[dead] = resets
        self.usage[dead] = 1.0 / self.F
        self.centroid_age[dead] = 0.0
        return len(dead)

    @_compile_disabled
    @torch.no_grad()
    def diagnostics(self) -> Dict[str, float]:
        C_n = F.normalize(self.C, dim=-1)
        gram = (C_n @ C_n.T).clone()
        mask = ~torch.eye(self.F, dtype=torch.bool, device=self.C.device)
        cos_sim = gram[mask].mean().item()
        usage = self.usage / (self.usage.sum() + 1e-8)
        entropy = -(usage * (usage + 1e-8).log()).sum().item()
        n_effective = math.exp(min(entropy, 20))
        dead = (self.usage < self.dead_threshold).sum().item()
        nearest = gram.fill_diagonal_(float("-inf")).max(dim=-1).values
        coverage = (1 - nearest).mean().item()
        gate_val = torch.sigmoid(self.gate).item()
        momentum_val = torch.sigmoid(self.write_momentum).item()
        E_soft = F.softmax(self.E + self.diag_mask, dim=-1)
        edge_entropy = -(E_soft * (E_soft + 1e-8).log()).sum(dim=-1)
        edge_max = E_soft.max(dim=-1).values.mean().item()
        E_n = F.normalize(E_soft, dim=-1)
        e_row_sim = (E_n @ E_n.T)[mask].mean().item()
        return {
            "N_eff": n_effective,
            "dead": int(dead),
            "cos_sim": cos_sim,
            "coverage": coverage,
            "gate": gate_val,
            "momentum": momentum_val,
            "edge_ent": edge_entropy.mean().item(),
            "edge_max": edge_max,
            "e_row_sim": e_row_sim,
        }


class GMTBlock(nn.Module):
    """Transformer block with causal attention followed by graph memory."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_slots: int,
        nav_dim: int,
        block_idx: int,
        attention_dropout: float = ATTN_DROPOUT,
        grav_eps: float = GRAV_EPS,
        cluster_target: Optional[float] = None,
        edge_entropy_target: float = EDGE_ENT_TARGET,
        dead_threshold: float = DEAD_THRESHOLD,
        merge_threshold: float = MERGE_THRESH,
        centroid_cooldown: int = CENTROID_COOLDOWN,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, attention_dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.cell = GraphMemoryCell(
            hidden_dim,
            n_slots,
            nav_dim,
            block_idx,
            grav_eps=grav_eps,
            cluster_target=cluster_target,
            edge_entropy_target=edge_entropy_target,
            dead_threshold=dead_threshold,
            merge_threshold=merge_threshold,
            centroid_cooldown=centroid_cooldown,
        )

    def forward(
        self,
        x: torch.Tensor,
        temp: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x + self.attn(self.ln1(x))
        disp, w_src, w_tgt, C_normed = self.cell(self.ln2(h), temp)
        return h + disp, w_src, w_tgt, C_normed


class GMTV7(nn.Module):
    """Base v7 Graph Memory Transformer language model."""

    def __init__(self, config: Optional[GMTV7Config] = None, **overrides):
        super().__init__()
        if config is None:
            config = GMTV7Config(**overrides)
        elif overrides:
            config = replace(config, **overrides)

        self.config = config
        self.hdim = config.hidden_dim
        self.n_layers = config.n_layers

        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_dim)
        self.drop = nn.Dropout(config.embedding_dropout)

        self.blocks = nn.ModuleList(
            [
                GMTBlock(
                    config.hidden_dim,
                    config.n_heads,
                    config.memory_slots,
                    config.nav_dim,
                    block_idx=i,
                    attention_dropout=config.attention_dropout,
                    grav_eps=config.grav_eps,
                    cluster_target=config.cluster_target,
                    edge_entropy_target=config.edge_entropy_target,
                    dead_threshold=config.dead_threshold,
                    merge_threshold=config.merge_threshold,
                    centroid_cooldown=config.centroid_cooldown,
                )
                for i in range(config.n_layers)
            ]
        )

        self.norm_final = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.register_buffer("temp", torch.tensor(config.temp_max))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    @staticmethod
    def routing_temperature(
        step: int,
        total_steps: int,
        temp_max: float = TEMP_MAX,
        temp_min: float = TEMP_MIN,
    ) -> float:
        progress = min(step / max(total_steps, 1), 1.0)
        log_progress = math.log1p(progress * (math.e - 1))
        return temp_max * (temp_min / temp_max) ** log_progress

    def forward(
        self,
        x: torch.Tensor,
        return_mem_loss: bool = False,
        update_memory: bool = True,
    ):
        _, seq_len = x.shape
        if seq_len > self.config.seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds configured "
                f"seq_len={self.config.seq_len}"
            )

        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.drop(self.embed(x) + self.pos_embed(pos))

        all_w_src: List[torch.Tensor] = []
        all_h: List[torch.Tensor] = []
        all_C_normed: List[torch.Tensor] = []

        for block in self.blocks:
            h, w_src, w_tgt, C_normed = block(h, self.temp)
            if update_memory:
                block.cell.update_usage(w_src)
                block.cell.write_back(h, w_src)
            all_w_src.append(w_src)
            all_h.append(h)
            all_C_normed.append(C_normed)

        logits = self.lm_head(self.norm_final(h))

        if return_mem_loss:
            L_track = torch.stack(
                [
                    block.cell.tracking_loss(h_i, w_i, cn)
                    for block, h_i, w_i, cn in zip(
                        self.blocks, all_h, all_w_src, all_C_normed
                    )
                ]
            ).mean()
            L_ortho = torch.stack(
                [block.cell.orthogonality_loss() for block in self.blocks]
            ).mean()
            L_cluster = torch.stack(
                [
                    block.cell.clustering_penalty(w_i)
                    for block, w_i in zip(self.blocks, all_w_src)
                ]
            ).mean()
            L_edge = torch.stack(
                [block.cell.edge_entropy_loss() for block in self.blocks]
            ).mean()
            L_contrast = torch.stack(
                [block.cell.edge_contrastive_loss() for block in self.blocks]
            ).mean()
            return logits, L_track, L_ortho, L_cluster, L_edge, L_contrast, h

        return logits, h

    @_compile_disabled
    @torch.no_grad()
    def diagnostics(self) -> Dict[str, object]:
        per_block = [block.cell.diagnostics() for block in self.blocks]

        def mean(key: str) -> float:
            return sum(d[key] for d in per_block) / len(per_block)

        def mn(key: str) -> float:
            return min(d[key] for d in per_block)

        def mx(key: str) -> float:
            return max(d[key] for d in per_block)

        return {
            "N_eff_mean": mean("N_eff"),
            "N_eff_min": mn("N_eff"),
            "dead_total": sum(d["dead"] for d in per_block),
            "cos_sim_mean": mean("cos_sim"),
            "coverage_mean": mean("coverage"),
            "gate_mean": mean("gate"),
            "gate_min": mn("gate"),
            "gate_max": mx("gate"),
            "momentum_mean": mean("momentum"),
            "momentum_min": mn("momentum"),
            "momentum_max": mx("momentum"),
            "edge_ent_mean": mean("edge_ent"),
            "edge_max_mean": mean("edge_max"),
            "e_row_sim_mean": mean("e_row_sim"),
            "per_block": per_block,
        }

    def log_diagnostics(
        self,
        diag: Dict[str, object],
        track: float,
        ortho: float,
        cluster: float,
        edge_ent: float,
        contrast: float,
        temp: float,
    ) -> None:
        per_block = diag["per_block"]
        logger.info("GMT v7 diagnostics")
        logger.info(
            "N_eff: mean=%.1f min=%.1f target>=%.0f",
            diag["N_eff_mean"],
            diag["N_eff_min"],
            self.config.cluster_target,
        )
        logger.info(
            "Dead centroids: %s/%s",
            diag["dead_total"],
            self.n_layers * self.config.memory_slots,
        )
        logger.info("Cos sim: %.4f", diag["cos_sim_mean"])
        logger.info("Coverage: %.4f", diag["coverage_mean"])
        logger.info(
            "Gate: mean=%.4f [%.4f, %.4f]",
            diag["gate_mean"],
            diag["gate_min"],
            diag["gate_max"],
        )
        logger.info(
            "Momentum: mean=%.4f [%.4f, %.4f]",
            diag["momentum_mean"],
            diag["momentum_min"],
            diag["momentum_max"],
        )
        logger.info(
            "Edge entropy: %.3f target>=%.1f max_w=%.3f",
            diag["edge_ent_mean"],
            self.config.edge_entropy_target,
            diag["edge_max_mean"],
        )
        logger.info("E row sim: %.4f", diag["e_row_sim_mean"])
        logger.info("Temp: %.4f", temp)
        logger.info(
            "Per-block gates: %s",
            " ".join(f"B{i}={d['gate']:.3f}" for i, d in enumerate(per_block)),
        )
        logger.info(
            "Per-block momentum: %s",
            " ".join(
                f"B{i}={d['momentum']:.3f}" for i, d in enumerate(per_block)
            ),
        )
        logger.info(
            "Losses: track=%.5f ortho=%.5f cluster=%.5f edge=%.5f contrast=%.5f",
            track,
            ortho,
            cluster,
            edge_ent,
            contrast,
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
