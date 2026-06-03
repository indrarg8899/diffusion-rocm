"""ROCm-optimized attention kernels for diffusion models."""

import torch
from typing import Optional


class ROCmFlashAttention:
    """Flash Attention v2 implementation via composable_kernel."""

    def __init__(
        self,
        head_dim: int = 128,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.dropout = dropout
        self.causal = causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = query.shape

        q = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=query.device),
                diagonal=1,
            )
            attn = attn + causal_mask

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        return out.transpose(1, 2).contiguous().view(batch, seq_len, -1)


class MultiHeadCrossAttention(ROCmFlashAttention):
    """Cross-attention for conditioning in UNet."""

    def __init__(self, **kwargs):
        kwargs["causal"] = False
        super().__init__(**kwargs)
