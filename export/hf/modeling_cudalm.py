"""cudalm decoder for Hugging Face transformers (custom architecture, load with trust_remote_code=True).

Mirrors the cudalm forward (gpu_forward in cudalm.cu) in plain PyTorch:
  embedding -> [hashed n-gram injection before layer `ngram.layer`] -> L x (RMSNorm, GQA attention with RoPE
  (half-split pairs), optional sliding window, optional output gate 2*sigmoid(x Wg); RMSNorm, SwiGLU FFN or top-k MoE
  with a selection-only router bias and raw softmax gates) -> RMSNorm -> tied lm head.
fp32 is the reference path; it matches `cudalm logits` from the CPU build.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_cudalm import CudalmConfig


class CudalmRMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rope(x, pos, theta):                                  # x [B,h,n,hd], pos [B,n]; pairs (i, i+hd/2)
    half = x.shape[-1] // 2
    inv = theta ** (-2.0 * torch.arange(half, device=x.device, dtype=torch.float32) / x.shape[-1])
    ang = pos.float()[..., None] * inv
    c, s = ang.cos()[:, None], ang.sin()[:, None]
    a, b = x[..., :half].float(), x[..., half:].float()
    return torch.cat([a * c - b * s, b * c + a * s], -1).to(x.dtype)


class CudalmAttention(nn.Module):
    def __init__(self, cfg, l):
        super().__init__()
        D, self.H, self.KV, self.hd, self.l = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, l
        self.theta, self.vres = cfg.rope_theta, cfg.value_residual
        self.q_proj = nn.Linear(D, self.H * self.hd, bias=False)
        self.k_proj = nn.Linear(D, self.KV * self.hd, bias=False)
        self.v_proj = nn.Linear(D, self.KV * self.hd, bias=False)
        self.o_proj = nn.Linear(self.H * self.hd, D, bias=False)
        self.gate_proj = nn.Linear(D, self.H * self.hd, bias=False) if cfg.attn_gate else None

    def forward(self, x, pos, mask, cache, v0):
        B, n, _ = x.shape
        q = self.q_proj(x).view(B, n, self.H, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, n, self.KV, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, n, self.KV, self.hd).transpose(1, 2)
        if self.vres:                                     # v_l = (v_l + v_0) / 2 for l >= 1
            if self.l == 0:
                v0 = v
            else:
                v = (v + v0) * 0.5
        q, k = rope(q, pos, self.theta), rope(k, pos, self.theta)
        if cache is not None:
            k, v = cache.update(k, v, self.l)
        g = self.H // self.KV
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, n, self.H * self.hd)
        if self.gate_proj is not None:
            o = o * (2.0 * torch.sigmoid(self.gate_proj(x)))
        return self.o_proj(o), v0


class CudalmMLP(nn.Module):
    def __init__(self, D, Fi):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(D, Fi, bias=False), nn.Linear(D, Fi, bias=False)
        self.down_proj = nn.Linear(Fi, D, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CudalmMoE(nn.Module):
    """top-k over softmax(router); selection uses prob + bias[slot], the output weight is the raw prob."""

    def __init__(self, cfg, l):
        super().__init__()
        D, E, Fi = cfg.hidden_size, cfg.layer_num_experts[l], cfg.layer_intermediate_size[l]
        self.E, self.K, self.S = E, max(1, cfg.layer_top_k[l]), cfg.router_bias
        self.router = nn.Linear(D, E, bias=False)
        self.experts_gate = nn.Parameter(torch.empty(E, Fi, D))
        self.experts_up = nn.Parameter(torch.empty(E, Fi, D))
        self.experts_down = nn.Parameter(torch.empty(E, D, Fi))
        if self.S:
            self.register_buffer("router_bias", torch.zeros(self.S, E))

    def forward(self, x):
        B, n, D = x.shape
        xf = x.reshape(-1, D)
        p = torch.softmax(self.router(xf).float(), -1)
        taken = torch.zeros_like(p, dtype=torch.bool)
        out = torch.zeros_like(xf)
        for k in range(self.K):
            sc = p + self.router_bias[min(k, self.S - 1)].float() if self.S else p
            e = sc.masked_fill(taken, float("-inf")).argmax(-1)
            taken.scatter_(1, e[:, None], True)
            w = p.gather(1, e[:, None])
            for j in range(self.E):
                idx = (e == j).nonzero()[:, 0]
                if idx.numel():
                    h = xf[idx]
                    y = F.linear(F.silu(F.linear(h, self.experts_gate[j])) * F.linear(h, self.experts_up[j]),
                                 self.experts_down[j])
                    out.index_add_(0, idx, y * w[idx].to(y.dtype))
        return out.view(B, n, D)


class CudalmNGram(nn.Module):
    """hashed n-gram memory: FNV-1a over the last `order` ids, h ^= h >> 15, row h % R; no row before order-1
    tokens of history. tri: the table is split in two halves, the second keyed by the (order+1)-gram, rows summed.
    Gate g = sigmoid(<rms(h), rms(k)> / sqrt(D)), h += g * v."""

    def __init__(self, cfg):
        super().__init__()
        g, D = cfg.ngram, cfg.hidden_size
        self.tri = bool(g.get("tri"))
        self.R, self.order, self.gated = g["rows"] // (2 if self.tri else 1), g["order"], g["gate"]
        self.table = nn.Embedding(g["rows"], g["dim"])
        self.proj = nn.Linear(g["dim"], D, bias=False)
        self.key_proj = nn.Linear(g["dim"], D, bias=False) if self.gated else None

    def rows(self, ids, pos, o):                          # ids [B,m] history incl. the n current ids, pos [B,n]
        n = pos.shape[1]
        ip = F.pad(ids, (o - 1, 0)).long()
        m = ip.shape[1]
        h = torch.full_like(pos, 2166136261, dtype=torch.long)
        for j in range(o - 1, -1, -1):
            h = ((h ^ ip[:, m - n - j:m - j]) * 16777619) & 0xFFFFFFFF
        h = h ^ (h >> 15)
        return h % self.R, pos >= o - 1

    def forward(self, h, ids, pos):
        dt = self.table.weight.dtype
        row, ok = self.rows(ids, pos, self.order)
        e = self.table(row) * ok[..., None].to(dt)
        if self.tri:
            row, ok = self.rows(ids, pos, self.order + 1)
            e = e + self.table(row + self.R) * ok[..., None].to(dt)
        v = self.proj(e)
        if not self.gated:
            return h + v
        k32, h32 = self.key_proj(e).float(), h.float()
        r1 = torch.rsqrt(h32.pow(2).mean(-1, keepdim=True) + 1e-6)
        r2 = torch.rsqrt(k32.pow(2).mean(-1, keepdim=True) + 1e-6)
        g = torch.sigmoid((h32 * r1 * k32 * r2).sum(-1, keepdim=True) / math.sqrt(h.shape[-1]))
        return h + (g * v.float()).to(h.dtype)


class CudalmLayer(nn.Module):
    def __init__(self, cfg, l):
        super().__init__()
        D = cfg.hidden_size
        self.input_layernorm = CudalmRMSNorm(D, cfg.rms_norm_eps)
        self.self_attn = CudalmAttention(cfg, l)
        self.post_attention_layernorm = CudalmRMSNorm(D, cfg.rms_norm_eps)
        self.mlp = CudalmMoE(cfg, l) if cfg.layer_ffn[l] == "moe" else CudalmMLP(D, cfg.layer_intermediate_size[l])

    def forward(self, h, pos, mask, cache, v0):
        a, v0 = self.self_attn(self.input_layernorm(h), pos, mask, cache, v0)
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h)), v0


class CudalmPreTrainedModel(PreTrainedModel):
    config_class = CudalmConfig
    base_model_prefix = "model"
    _no_split_modules = ["CudalmLayer"]
    _supports_sdpa = True

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, CudalmRMSNorm):
            nn.init.ones_(m.weight)
        elif isinstance(m, CudalmMoE):
            for p in (m.experts_gate, m.experts_up, m.experts_down):
                nn.init.normal_(p, std=0.02)


class CudalmModel(CudalmPreTrainedModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([CudalmLayer(cfg, l) for l in range(cfg.num_hidden_layers)])
        self.norm = CudalmRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ngram = CudalmNGram(cfg) if cfg.ngram else None
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, v):
        self.embed_tokens = v

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        B, n = input_ids.shape
        dev = input_ids.device
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        cache = past_key_values if use_cache else None
        past = cache.get_seq_length() if cache is not None else 0
        m = past + n
        ids = input_ids
        if cache is not None:                             # the n-gram rows need the earlier ids
            prev = getattr(cache, "_cudalm_ids", None)
            if past and (prev is None or prev.shape[1] != past):
                raise ValueError("cudalm: cache has no matching token history (n-gram)")
            ids = input_ids if not past else torch.cat([prev, input_ids], 1)
            cache._cudalm_ids = ids
        if attention_mask is not None and attention_mask.shape[1] == m and not bool(attention_mask.all()):
            km = attention_mask.bool()
            kpos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        else:
            km = torch.ones(B, m, dtype=torch.bool, device=dev)
            kpos = torch.arange(m, device=dev).expand(B, m)
        pos = position_ids if position_ids is not None else kpos[:, past:]
        qi = torch.arange(past, m, device=dev)[:, None]
        kj = torch.arange(m, device=dev)[None]
        base = (kj <= qi)[None] & km[:, None, :]
        diag = (kj == qi)[None]
        masks = {}
        for w in set(self.config.layer_window):
            mw = base if not w else base & ((kpos[:, past:, None] - kpos[:, None, :]) < w)
            masks[w] = (mw | diag)[:, None]
        h = self.embed_tokens(input_ids)
        v0 = None
        for l, layer in enumerate(self.layers):
            if self.ngram is not None and l == self.config.ngram["layer"]:
                h = self.ngram(h, ids, pos)
            h, v0 = layer(h, pos, masks[self.config.layer_window[l]], cache, v0)
        return self.norm(h), past_key_values


class CudalmForCausalLM(CudalmPreTrainedModel, GenerationMixin):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = CudalmModel(cfg)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, v):
        self.model.embed_tokens = v

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        if input_ids is None:
            raise ValueError("cudalm needs input_ids (the n-gram memory hashes token ids)")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        h, past_key_values = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        logits = F.linear(h, self.model.embed_tokens.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values if use_cache else None)
