import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_dim)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #create a matrix of shape (seq_len,d_model)
        pe = torch.zeros(seq_len,d_model)
        # create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(1000.0) / d_model))
        # apply the sine to even positions
        pe[:,0::2] = torch.sin(position* div_term)
        pe[:,1::2] = torch.cos(position* div_term)
        
        pe = pe.unsqueeze(0) #(1,seq_len,d_model) for BATCH_SIZE
        
        self.register_buffer('pe',pe) # save this tensor
    
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1]+1,:]).requires_grad_(False)
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len:int, embedding_dim:int, h:int, dropout:float):
        super().__init__()
        
        self.w_k = nn.Linear(embedding_dim,embedding_dim)
        self.w_q = nn.Linear(embedding_dim,embedding_dim)
        self.w_v = nn.Linear(embedding_dim,embedding_dim)
        self.w_o = nn.Linear(embedding_dim,embedding_dim)
        
        self.h = h
        assert (embedding_dim % h) == 0, "embedding_dim is not dvisible by heads in attention model"
        
        self.d_k = embedding_dim // h
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(key, query, value, mask, dropout:nn.Dropout):
        attention_score = (query @ key.transpose(-2,-1)) * math.sqrt(key.shape[-1])
        if mask is not None:
            attention_score.masked_fill_(mask==0, 10**-6)
        if dropout is not None:
            attention_score = dropout(attention_score)
        attention_score = torch.softmax(attention_score, dim=-1)
        
        return (attention_score @ value) , attention_score 
            
    
    def forward(self, key, query, value, mask):
        key = self.w_k(key)    
        query = self.w_k(query)    
        value = self.w_k(value)    
        
        # (B, seq_len, embedding_dim) --> (B, seq_len, h, k) --> (B, h, seq_len, k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)        
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)        
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, attention_score = MultiHeadAttention.attention(key, query, value, mask, self.dropout) 
        
        # (B, h, seq_len, k) --> (B, seq_len, h, k) --> (B, seq_len, embedding_dim)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h * self.d_k) 
        
        return self.w_o(x)
    
class LayerNormalization(nn.Module):
    def __init__(self,eps:float = 10**-6 ):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim = True)
        return ((self.alpha * (x - mean)) / (std + self.eps)) + self.beta

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self,x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class FeedForwardBlock(nn.Module):
    def __init__(self, d_ff: int, embedding_dim:int, dropout:float):
        super().__init__()
        self.l1 = nn.Linear(embedding_dim,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.l2 = nn.Linear(d_ff, embedding_dim)
        
    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.dropout(x)
        x = self.l2(x)
        return x
        
class encoder_block(nn.Module):
    def __init__(self, self_attention_layer : MultiHeadAttention,
                 feed_forward_layer: FeedForwardBlock,
                 dropout:float):
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.norm = LayerNormalization()
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_layer(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_layer)
        return self.norm(x)

class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return x        

class decoder_blocks(nn.Module):
    def __init__(self,
                 self_attention_layer: MultiHeadAttention,
                 cross_attention_layer: MultiHeadAttention,
                 feed_forward_layer: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.dropout = nn.Dropout(dropout)
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_layer(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_attention_layer(encoder_output,x,encoder_output,src_mask))
        x = self.residual_connections[2](x, self.feed_forward_layer)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
    def forward(self,x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x 

class projection(nn.Module):
    def __init__(self, tgt_vocab_size:int, embedding_dim:int):
        super().__init__()
        self.output = nn.Linear(embedding_dim, tgt_vocab_size)
    def forward(self, x):
        x = torch.log_softmax(self.output(x), dim=-1)
        
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim:int, num_classes:int):
        super().__init__()
        self.classifier_layer = nn.Linear(embedding_dim,num_classes)
    def forward(self,x):
        return self.classifier_layer(x)
        
class EncoderOnlyTransformer(nn.Module):
    def __init__(self,
                 src_embed: InputEmbeddings, pos: PositionalEncoding,
                 encoder:Encoder, classifier_layer :ClassificationHead):
        super().__init__()
        self.src_embed = src_embed
        self.pos = pos
        self.encoder = encoder
        self.classifier_layer = classifier_layer
                
    def encode(self,x,src_mask=None):
        x = self.src_embed(x)
        x = self.pos(x)
        x = self.encoder(x, src_mask)
        x = x[:, 0, :] #(bs, embedded_dim) # First Token
        # x = x[:, -1, :] #(bs, embedded_dim) # Last Token
        # x = torch.mean(x,dim=-1) #(bs, embedded_dim) # mean Token
        x = self.classifier_layer(x)
        # x = torch.softmax(x,dim=-1) # Compare to check
        return x