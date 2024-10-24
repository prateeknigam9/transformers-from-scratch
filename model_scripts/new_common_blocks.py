import torch
import torch.nn as nn
import math
import pdb

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.d_model = d_model
        self.embedding_layer = nn.Embedding(vocab_size,d_model)
    def forward(self, x):
        return self.embedding_layer(x) * (-math.sqrt(self.d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len:int, d_model:int, dropout:float):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # postional encoding vector space
        pe = torch.zeros(seq_len,d_model)
        # position
        position = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1)
        # div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model)) 
        
        # assigning to even odd position
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        self.pe = pe.unsqueeze(0)
    
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1]+1, :]).requires_grad_(False)
        return self.dropout(x)        

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len:int, d_model:int, n_heads:int, dropout:float):
        super().__init__()
        self.w_k = nn.Linear(d_model,d_model)
        self.w_q = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.seq_len = seq_len   
        self.d_model = d_model   
        self.n_heads = n_heads   
        self.d_k = d_model // n_heads
    
    @staticmethod
    def attention(key, query, value, mask, dropout):
        attention_score = (query @ key.transpose(-2,-1)) * math.sqrt(key.shape[-1])
        if mask:
            # pdb.set_trace()
            mask = torch.triu(torch.ones(key.shape[-2], key.shape[-2]), diagonal=1)
            mask_bool = mask.bool()
            attention_score.masked_fill_(mask_bool, -torch.inf)
        if dropout:
            attention_score = dropout(attention_score)
        attention_score = torch.softmax(attention_score, dim=-1)        
        return attention_score @ value
            
    
    def forward(self, k,q,v, mask):
        key = self.w_k(k)
        query = self.w_q(q)
        value = self.w_v(v)
        
        # import pdb; pdb.set_trace()
        # splitting for heads - (b, seq_len, heads, dk) - (b, heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1,2)
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1,2)
        
        x = MultiHeadAttention.attention(key, query, value, mask, self.dropout)
        
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)
        
        return self.w_o(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps:float= 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.beta

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))        
        
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.l1 = nn.Linear(d_model, d_model*4)
        self.gelu = GELU()
        self.l2 = nn.Linear(d_model*4, d_model)
    
    def forward(self,x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return self.dropout(x)

class ResidualConnections(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



 
    