import math
import torch
from torch import nn as nn


def scaled_softmax_attention(query, key, value, mask=None):
    """
    Args:
        query: torch.Tensor (..., L, D)
        key: torch.Tensor (..., L, D)
        value: torch.Tensor (..., L, D)
    Returns:
        res: torch.Tensor (..., L, D), output of the attention layer (\softmax(Q K^T / d) V
        attention: torch.Tensor (..., L, L), attention weights (\softmax(Q K^T / d))

    L is the length of sequence, D is the embedding dimension
    """
    if mask is None:
        mask = torch.zeros(query.shape[-2], query.shape[-2])
    attention = torch.softmax(
        torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1]) + mask,
        dim=-1,
    )
    res = torch.matmul(attention, value)
    return res


# https://blog.eleuther.ai/rotary-embeddings/
class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :].to(x.device)
            self.sin_cached = emb.sin()[None, None, :, :].to(x.device)
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.d = d
        self.eps = eps

    def forward(self, x):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        rms_x = norm_x * self.d ** (-0.5)
        return x / (rms_x + self.eps)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, cos, sin, mask=None):
        """
        Args:
            x: torch.Tensor (B, L, D)
            return_attention: If specified, returns attention along with outputs
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        size = (
            x.shape[0],
            x.shape[1],
            self.num_heads,
            self.embed_dim // self.num_heads,
        )
        qs = torch.reshape(self.q_proj(x), size).transpose(1, 2)
        ks = torch.reshape(self.k_proj(x), size).transpose(1, 2)
        vs = torch.reshape(self.v_proj(x), size).transpose(1, 2)
        # apply rotary embeds to query and key
        qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
        outputs = nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True).transpose(1, 2)
        outputs = self.o_proj(torch.reshape(outputs, x.shape))
        return outputs


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.p = dropout

        self.masked_multihead = MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
        )

        self.dropout = nn.Dropout(p=self.p, inplace=False)
        self.dropout1 = nn.Dropout(p=self.p, inplace=False)
        self.dropout2 = nn.Dropout(p=self.p, inplace=False)

        self.norm1 = RMSNorm(self.hidden_dim)
        self.norm2 = RMSNorm(self.hidden_dim)
        self.activation = activation()

        self.linear1 = nn.Linear(self.hidden_dim, 4 * self.hidden_dim, bias=False)
        self.linear2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, x, cos, sin, attn_mask=None):
        x = self.norm1(x)
        res = self.dropout(self.masked_multihead(x, cos, sin, mask=attn_mask)) + x
        res = res + self.dropout2(
            self.linear2(self.activation(self.dropout1(self.linear1(self.norm2(res)))))
        )
        return res


class BoutiqueLM(nn.Module):
    def __init__(
        self,
        vocab_stories_size,
        num_layers,
        num_heads,
        hidden_dim,
        max_len,
        dropout=0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.vocab_stories_size = vocab_stories_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embeds = nn.Embedding(self.vocab_stories_size, self.hidden_dim)

        self.decoders = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.positional_encoding = Rotary(
            dim=self.hidden_dim // num_heads, base=max_len
        )
        self.classifier = nn.Linear(
            self.hidden_dim,
            self.vocab_stories_size,
            bias=False,
        )

    def forward(self, tokens_story, attention_mask=None):
        x = self.embeds(tokens_story)
        _, _ = self.positional_encoding(x)
        for i in range(self.num_layers):
            x = self.decoders[i](
                x,
                self.positional_encoding.cos_cached,
                self.positional_encoding.sin_cached,
                attn_mask=attention_mask,
            )
        return self.classifier(x)
