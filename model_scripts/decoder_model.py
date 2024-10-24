from model_scripts import new_common_blocks
import torch.nn as nn
import torch

class DecoderBlock(nn.Module):
    def __init__(self, 
                 attention_layer : new_common_blocks.MultiHeadAttention,
                 feed_forward_layer : new_common_blocks.FeedForwardLayer,
                 dropout: float):
        super().__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList([new_common_blocks.ResidualConnections(dropout) for _ in range(2)])
        self.norm = new_common_blocks.LayerNormalization()
    
    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.attention_layer(x,x,x,mask))
        x = self.residual_connections[1](x, self.feed_forward_layer)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class OutputLayer(nn.Module):
    def __init__(self, tgt_vocab_size:int, d_model:int):
        super().__init__()
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    def forward(self,x):
        return self.output_layer(x)
        

class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 embedding_layer: new_common_blocks.InputEmbeddings,
                 positional_layer: new_common_blocks.PositionalEncoding,
                 decoder: Decoder,
                 output_layer : OutputLayer
                 ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.positional_layer = positional_layer
        self.decoder = decoder
        self.output_layer = output_layer
    
    def forward(self, x, mask):
        x = self.embedding_layer(x)
        x = self.positional_layer(x)
        x = self.decoder(x, mask)
        x = self.output_layer(x)
        return x


def construct_decoder_model(vocab_size, d_model, seq_len,
                            n_decoder_blocks,n_heads,
                            dropout:float=0.01):
    
    embedding = new_common_blocks.InputEmbeddings(vocab_size, d_model)
    pos = new_common_blocks.PositionalEncoding(seq_len, d_model, dropout)
    
    decoder_blocks = []
    for _ in range(n_decoder_blocks):
        attention = new_common_blocks.MultiHeadAttention(seq_len, d_model, n_heads, dropout)
        feedforward = new_common_blocks.FeedForwardLayer(d_model, dropout)
        decoder_block = DecoderBlock(attention, feedforward, dropout)
        decoder_blocks.append(decoder_block)
    
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    output = OutputLayer(vocab_size, d_model)
    
    model = DecoderOnlyTransformer(embedding, pos, decoder, output)
    
    return model
                