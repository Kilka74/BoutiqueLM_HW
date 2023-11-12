import math
import torch
from torch import nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        # TODO
        # here should be a tensor of size (1, max_len, embed_dim), dummy dimension is needed for proper addition
        pe = torch.empty(max_len, embed_dim)
        positions = torch.arange(0, max_len, dtype=torch.bfloat16).unsqueeze(1)
        indices = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.bfloat16).unsqueeze(0) * -math.log(max_len) / embed_dim
        ).unsqueeze(0)
        pe[:, ::2] = torch.sin(positions * indices)
        pe[:, 1::2] = torch.cos(positions * indices)
        pe = pe.to(torch.bfloat16).unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbreddings(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        inv_freq = torch.exp(
            -math.log(max_len) * torch.arange(0, embed_dim, 2, dtype=torch.bfloat16) / embed_dim
        )
        t = torch.arange(max_len, dtype=torch.bfloat16)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("sin", sin.unsqueeze(0))
        self.register_buffer("cos", cos.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        return (
            x * self.cos[:, : x.shape[1], :]
            + rotate_half(x) * self.sin[:, : x.shape[1], :]
        )


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.d = d
        self.eps = eps

    def forward(self, x):
        norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)
        rms_x = norm_x * self.d ** (-0.5)
        return x / (rms_x + self.eps)


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, activation):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.p = dropout

        self.masked_multihead = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.p,
            batch_first=True,
            dtype=torch.bfloat16,
        )

        # self.multihead = nn.MultiheadAttention(
        #     embed_dim=self.hidden_dim,
        #     num_heads=self.num_heads,
        #     dropout=self.p,
        #     batch_first=True,
        #     dtype=torch.bfloat16,
        # )

        self.dropout = nn.Dropout(p=self.p, inplace=False)
        self.dropout1 = nn.Dropout(p=self.p, inplace=False)
        self.dropout2 = nn.Dropout(p=self.p, inplace=False)
        # self.dropout3 = nn.Dropout(p=self.p, inplace=False)

        self.norm1 = RMSNorm(self.hidden_dim)
        self.norm2 = RMSNorm(self.hidden_dim)
        # self.norm3 = RMSNorm(self.hidden_dim)
        self.activation = activation()

        self.linear1 = nn.Linear(self.hidden_dim, 4 * self.hidden_dim, bias=True, dtype=torch.bfloat16)
        self.linear2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim, bias=True, dtype=torch.bfloat16)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, q, k, v, attn_mask):
        res = self.dropout(self.masked_multihead(q, k, v, attn_mask=attn_mask)[0])
        res = self.norm1(res + v)
        # res = self.norm2(res + self.dropout1(self.multihead(q, k, res)[0]))
        res = self.norm2(
            res
            + self.dropout2(
                self.linear2(self.activation(self.dropout1(self.linear1(res))))
            )
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
        activation=nn.ReLU,
    ):
        super().__init__()
        self.vocab_stories_size = vocab_stories_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.embeds = nn.Embedding(self.vocab_stories_size, self.hidden_dim, dtype=torch.bfloat16)

        self.positional_encoding = RotaryEmbreddings(
            embed_dim=self.hidden_dim, max_len=self.max_len
        )

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

        self.classifier = nn.Linear(
            self.hidden_dim, self.vocab_stories_size, bias=False, dtype=torch.bfloat16,
        )

    def forward(self, tokens_story, attention_mask=None):
        x = self.embeds(tokens_story)
        qk = self.positional_encoding(x)
        for i in range(self.num_layers):
            x = self.decoders[i](qk, qk, x, attn_mask=attention_mask)
        return self.classifier(x)
