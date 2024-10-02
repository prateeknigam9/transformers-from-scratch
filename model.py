# English to Hindi
import torch
import torch.nn as nn
import math

#Input Embedding
class InputEmbeddings(nn.Module):
    def __init__(self,d_model: int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
# Positional Encoding
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
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # added
        
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1,keepdim = True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int, d_ff: int, dropout:float ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
        
    def forward(self,x):
        #(Batch,seq_len,d_model)  --> (Batch,seq_len,d_ff) --> (Batch,seq_len, d_model)
        out = torch.relu(self.linear_1(x))
        out = self.dropout(out)
        out = self.linear_2(out)
        return out

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int,h:int, dropout:int) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod #we can call it, without class instance - like - MultiHeadAttentionBlock.attention()
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[1]
        
        #(Batch,h,seq_len,d_k) --> (Batch, h, seq_len, seq_len)
        attention_Scores = (query @ key.transpose(-2,-1)/math.sqrt(d_k))
        #if mask
        if mask is not None:
            attention_Scores.masked_fill_(mask==0,-1e9)
        
        attention_Scores = attention_Scores.softmax(dim=-1) # (BATCH, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_Scores = dropout(attention_Scores)
        
        return (attention_Scores @ value), attention_Scores
        
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len,d_model) --> (Batch, seq_len,d_model)
        key = self.w_q(k) # (Batch, seq_len,d_model) --> (Batch, seq_len,d_model)
        value = self.w_q(v) # (Batch, seq_len,d_model) --> (Batch, seq_len,d_model)
        
        #dividing the vectors
         # (Batch, seq_len,d_model) --> (Batch, seq_len,h,d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        # (Batch, h, seq_len,d_k) --> (Batch,seq_len, h, d_k) --> (Batch, seq_len,d_model)
        x = x.transpose(1,2).contigous().view(x.shape[0],-1, self.h * self.d_k) 
        
        return self.w_o(x)
    

#residual connection
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block:FeedForwardBlock,
                 dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x,self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # n layers
        self.norm = LayerNormalization()
    
    def forward(self,x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock,
                 cross_head_attention_block:MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout:float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_head_attention_block = cross_head_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, self.cross_head_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.projection_layer = nn.Linear(d_model,vocab_size)
    
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.projection_layer(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder, 
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = projection
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x):
        return self.projection(x)
    

def build_transformer(src_vocab_size:int, tgt_vocab_size:int,
                      src_seq_len:int, tgt_seq_len:int,
                      d_model:int = 512, N: int = 6,
                      h:int = 8, dropout:float = 0.1,
                      d_ff: int = 2048):
    
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model,tgt_vocab_size)
    
    #create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    
    #encoder
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(encoder_block)
    
    #decoder
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,
                                     decoder_cross_attention_block,
                                     feed_forward_block,
                                     dropout)
        decoder_blocks.append(decoder_block)
    
    #encoder-decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #projectionlayer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    #transformer
    transformer = Transformer(encoder, decoder,
                              src_embed, tgt_embed,
                              src_pos, tgt_pos,
                              projection_layer)
    
    #initialize the parameters - Xavier
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    
    return transformer
        