# GLM-4.7 (GLM-4-MoE) — 밑바닥부터 재작성

> 모델: `zai-org/GLM-4.7` (GLM-4-100B-A10B)
> 목적: 프레임워크 없이 아키텍처를 완전히 이해하고 재작성할 수 있도록 정리

---

## 전체 구조

```
Input Tokens
    ↓
[Embedding] (151552 → 5120)
    ↓
[Layer 0~2]  ← Dense (일반 MLP)
[Layer 3~91] ← MoE (160 experts 중 8개 선택)
    ↓
[RMSNorm]
    ↓
[LM Head] (5120 → 151552)
    ↓
Output Logits
```

---

## 핵심 하이퍼파라미터 (zai-org/GLM-4.7)

| 파라미터 | 값 | 의미 |
|---|---|---|
| `hidden_size` | 5120 | 모델 차원 |
| `num_hidden_layers` | 92 | 총 레이어 수 |
| `num_attention_heads` | 96 | Q 헤드 수 |
| `num_key_value_heads` | 8 | KV 헤드 수 (GQA 12:1) |
| `head_dim` | 128 | 어텐션 헤드 차원 (명시적) |
| `intermediate_size` | 12288 | Dense MLP 중간 차원 |
| `moe_intermediate_size` | 1536 | 각 Expert의 중간 차원 |
| `n_routed_experts` | 160 | 라우팅 Expert 총 수 |
| `num_experts_per_tok` | 8 | 토큰당 선택 Expert 수 |
| `n_shared_experts` | 1 | 공유 Expert 수 |
| `routed_scaling_factor` | 2.5 | 라우팅 가중치 스케일링 |
| `first_k_dense_replace` | 3 | 처음 3개 레이어는 Dense |
| `partial_rotary_factor` | 0.5 | RoPE가 head_dim의 절반만 회전 |
| `use_qk_norm` | true | QK 정규화 사용 |
| `attention_bias` | true | Q,K,V projection에 bias 있음 |
| `n_group` / `topk_group` | 1 / 1 | 그룹 라우팅 비활성화 |

---

## 완전한 구현 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════

@dataclass
class GLM4MoeConfig:
    vocab_size: int = 151552
    hidden_size: int = 5120
    num_hidden_layers: int = 92
    rms_norm_eps: float = 1e-5

    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    attention_bias: bool = True
    use_qk_norm: bool = True

    rope_theta: float = 1_000_000
    partial_rotary_factor: float = 0.5

    intermediate_size: int = 12288
    hidden_act: type[nn.Module] = nn.SiLU

    first_k_dense_replace: int = 3
    n_routed_experts: int = 160
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1536
    n_shared_experts: int = 1
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True

    n_group: int = 1
    topk_group: int = 1


# ════════════════════════════════════════════════
# KV Cache
# ════════════════════════════════════════════════

class KVCache:
    """레이어별 K, V를 누적 저장. 학습 파라미터가 아닌 추론 중간 결과물."""
    def __init__(self):
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.cache):
            prev_k, prev_v = self.cache[layer_idx]
            k = torch.cat([prev_k, k], dim=2)  # seq 차원으로 concat
            v = torch.cat([prev_v, v], dim=2)
            self.cache[layer_idx] = (k, v)
        else:
            self.cache.append((k, v))
        return k, v

    @property
    def seq_length(self) -> int:
        return self.cache[0][0].shape[2] if self.cache else 0


# ════════════════════════════════════════════════
# RMSNorm
# ════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


# ════════════════════════════════════════════════
# Rotary Position Embedding
# ════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).to(torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        inv_freq = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].to(torch.float32)
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(input_dtype), emb.sin().to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


# ════════════════════════════════════════════════
# GQA Attention + QK Norm
# ════════════════════════════════════════════════

class Attention(nn.Module):
    def __init__(self, config: GLM4MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        mask: torch.Tensor | None = None, cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(q.dtype)

        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


# ════════════════════════════════════════════════
# SwiGLU MLP
# ════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, config: GLM4MoeConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, config.hidden_size, bias=False)
        self.act = config.hidden_act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


# ════════════════════════════════════════════════
# MoE: Router → Experts → Shared Expert
# ════════════════════════════════════════════════

class TopkRouter(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(torch.float32), self.weight.to(torch.float32))


class Experts(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        self.num_experts = config.n_routed_experts
        E, H, I = config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.empty(E, H, I))
        self.act = config.hidden_act()

    def forward(self, x: torch.Tensor, top_k_idx: torch.Tensor, top_k_w: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)

        with torch.no_grad():
            mask = F.one_hot(top_k_idx, self.num_experts).permute(2, 1, 0)
            active = mask.sum(dim=(-1, -2)).nonzero()[:, 0]

        for e in active:
            k_pos, tok = torch.where(mask[e])
            gate, up = F.linear(x[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            h = F.linear(self.act(gate) * up, self.down_proj[e])
            out.index_add_(0, tok, (h * top_k_w[tok, k_pos, None]).to(out.dtype))

        return out


class MoE(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        self.gate = TopkRouter(config)
        self.experts = Experts(config)
        self.shared_experts = MLP(config, config.moe_intermediate_size * config.n_shared_experts)

        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = logits.sigmoid()
        biased = scores + self.gate.e_score_correction_bias

        group_scores = (
            biased.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_scores.topk(self.topk_group, dim=-1)[1], 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        masked = biased.masked_fill(~score_mask.bool(), 0.0)
        idx = masked.topk(self.top_k, dim=-1)[1]
        weights = scores.gather(1, idx)

        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor

        return idx, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        logits = self.gate(x)
        idx, weights = self.route(logits)
        routed = self.experts(x.view(-1, shape[-1]), idx, weights).view(shape)
        return routed + self.shared_experts(x)


# ════════════════════════════════════════════════
# Decoder Layer
# ════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, config: GLM4MoeConfig, layer_idx: int):
        super().__init__()
        self.attn = Attention(config, layer_idx)
        self.mlp = MoE(config) if layer_idx >= config.first_k_dense_replace else MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        mask: torch.Tensor | None = None, cache: KVCache | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask, cache)
        x = x + self.mlp(self.norm2(x))
        return x


# ════════════════════════════════════════════════
# Full Model
# ════════════════════════════════════════════════

class GLM4MoeModel(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, S = input_ids.shape
        x = self.embed(input_ids)

        past = cache.seq_length if cache else 0
        pos = torch.arange(past, past + S, device=x.device).unsqueeze(0)
        cos, sin = self.rope(x, pos)

        mask = torch.full((S, past + S), float("-inf"), device=x.device).triu(past + 1)
        mask = mask[None, None, :, :]

        for layer in self.layers:
            x = layer(x, cos, sin, mask, cache)

        return self.norm(x)


class GLM4MoeForCausalLM(nn.Module):
    def __init__(self, config: GLM4MoeConfig):
        super().__init__()
        self.model = GLM4MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        logits = self.lm_head(self.model(input_ids, cache))

        if labels is not None:
            return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return logits
```

---

## CPU 테스트용 tiny config

```python
tiny = GLM4MoeConfig(
    vocab_size=256, hidden_size=64, num_hidden_layers=6,
    num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    intermediate_size=128, moe_intermediate_size=48,
    n_routed_experts=8, num_experts_per_tok=2, rope_theta=10000,
)

model = GLM4MoeForCausalLM(tiny)
ids = torch.randint(0, 256, (1, 16))

# forward
logits = model(ids)
print(logits.shape)       # (1, 16, 256)

# loss
loss = model(ids, labels=ids)
print(loss)               # scalar

# KV Cache 테스트
cache = KVCache()
logits = model(torch.randint(0, 256, (1, 8)), cache=cache)
print(logits.shape)       # (1, 8, 256)
print(cache.seq_length)   # 8

logits = model(torch.randint(0, 256, (1, 1)), cache=cache)
print(logits.shape)       # (1, 1, 256)
print(cache.seq_length)   # 9
```

---

## 설계 메모

- **matmul dtype 규칙**: 반드시 동일 dtype끼리만 가능 (float32×float32, float16×float16 등)
- **`.to(torch.float32)` vs `.float()`**: `.to(dtype=...)` 가 표준. 명시적이고 일관적
- **`input_dtype` 패턴**: fp32로 올려서 정밀 계산 후 원래 dtype으로 복원하는 관용구
- **RMSNorm은 config를 받지 않음**: `(dim, eps)` 두 인자만 받아서 QK Norm 등 다양한 곳에서 재사용
- **KV Cache는 `list`**: `nn.ModuleList` 아님. 학습 파라미터가 아닌 추론 중간 결과물
- **`hidden_act: type[nn.Module]`**: 문자열 lookup 대신 클래스 직접 전달. 프레임워크 없는 구현에 적합
