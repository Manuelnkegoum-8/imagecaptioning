import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
import random,math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.tau = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim)))
        self.q_proj = nn.Linear(embed_dim, embed_dim,bias = False)
        self.k_proj = nn.Linear(embed_dim, embed_dim,bias = False)
        self.v_proj = nn.Linear(embed_dim, embed_dim,bias = False)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        
    def _masked_softmax(self, attention_scores, attention_mask):
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask==0, float('-inf'))
        return F.softmax(attention_scores, dim=-1)

    def forward(self,Q,K,V, attn_mask=None):


        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)

        v = rearrange(v, 'b n (h d) -> b h n d', h = self.num_heads)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h = self.num_heads)


        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.tau
        if attn_mask is not None:
            attention_scores = self._masked_softmax(attention_scores, attn_mask)
        
        attention_output = torch.matmul(attention_scores, v)
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        return self.o_proj(attention_output)

