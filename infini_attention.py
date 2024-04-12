import torch
import einops
import torch.nn as nn
import torch.nn.functional as F


class InfiniAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, seq_len=100):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.in_linear = nn.Linear(dim, inner_dim)

        self.k_linear = nn.Linear(dim, inner_dim, bias=False)
        self.v_linear = nn.Linear(dim, inner_dim, bias=False)
        self.q_linear = nn.Linear(dim, inner_dim, bias=False)

        self.out_linear = nn.Linear(inner_dim, dim)

        self.long_term_memory = torch.zeros(1, heads, dim_head, dim_head)
        self.long_term_memory_norm = torch.zeros(1, heads, seq_len, 1)

        self.local_memory_scalar = nn.Parameter(torch.tensor(1.0))
        self.long_term_memory_scalar = nn.Parameter(torch.tensor(1.0))
        self.long_term_memory_gate = nn.Linear(dim, seq_len, bias=False)

    def _query_long_term_memory(self, q):
        memory = (F.elu(q) @ self.long_term_memory) / (
            F.elu(q) * self.long_term_memory_norm
        )
        return memory

    def _update_long_term_memory(self, k, v):
        v_term = v - (
            (F.elu(k) @ self.long_term_memory) / (F.elu(k) * self.long_term_memory_norm)
        )
        self.long_term_memory = (
            self.long_term_memory + F.elu(k).transpose(-2, -1) @ v_term
        )
        self.long_term_memory_norm = F.elu(k).sum(dim=-1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (q, k, v),
        )

        # Local attention
        sim = q @ k.transpose(-2, -1) * self.scale
        local_attn = sim.softmax(dim=-1)
        if mask is not None:
            local_attn = local_attn.masked_fill(mask == 0, 0)
        # Gating
        local_attn = (1 - F.sigmoid(self.local_memory_scalar)) * local_attn

        # Long-term memory attention
        long_term_memory = self._query_long_term_memory(q)
        long_term_memory = F.sigmoid(
            self.long_term_memory_scalar
        ) * self.long_term_memory_gate(long_term_memory)

        attn = local_attn + long_term_memory

        self._update_long_term_memory(k, v)

        out = attn @ v

        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.out_linear(out)


if __name__ == "__main__":
    net = InfiniAttention(64)
    q = torch.randn(1, 100, 64)
    k = torch.randn(1, 100, 64)
    v = torch.randn(1, 100, 64)
    out = net(q, k, v)
    print(out.shape)
