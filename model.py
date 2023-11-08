import math
import torch
from torch import nn as nn
from torch.nn import functional as F


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
        positions = torch.arange(0, max_len).unsqueeze(1)
        indices = torch.exp(
            torch.arange(0, embed_dim, 2).unsqueeze(0) * -math.log(10000) / embed_dim
        ).unsqueeze(0)
        pe[:, ::2] = torch.sin(positions * indices)
        pe[:, 1::2] = torch.cos(positions * indices)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.shape[1], :]


class BoutiqueLM(nn.Module):
    def __init__(
        self,
        vocab_stories_size,
        num_layers,
        num_heads,
        hidden_dim,
        max_len,
        droupout=0.1,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.vocab_stories_size = vocab_stories_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = droupout
        self.max_len = max_len
        self.activation = activation()
        self.embeds = nn.Embedding(self.vocab_stories_size, self.hidden_dim)

        self.positional_encoding = PositionalEncoding(
            embed_dim=self.hidden_dim, max_len=self.max_len
        )

        self.decoder = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4*self.hidden_dim,
            dropout=self.dropout,
            activation=self.activation,
            norm_first=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            self.hidden_dim, self.vocab_stories_size, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, tokens_story, attention_mask=None):
        x = self.embeds(tokens_story)
        x = self.positional_encoding(x)
        res = self.decoder(tgt=x, memory=x, tgt_mask=attention_mask)
        return F.softmax(self.classifier(res), dim=1)
