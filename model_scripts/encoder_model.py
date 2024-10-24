
import torch.nn as nn
import torch
from model_scripts import new_common_blocks

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
    def __init__(self, self_attention_layer : new_common_blocks.MultiHeadAttention,
                 feed_forward_layer: FeedForwardBlock,
                 dropout:float):
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList([new_common_blocks.ResidualConnections(dropout) for _ in range(2)])
        self.norm = new_common_blocks.LayerNormalization()
    
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


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim:int, num_classes:int):
        super().__init__()
        self.classifier_layer = nn.Linear(embedding_dim,num_classes)
    def forward(self,x):
        return self.classifier_layer(x)
        
class EncoderOnlyTransformer(nn.Module):
    def __init__(self,
                 src_embed: new_common_blocks.InputEmbeddings, pos: new_common_blocks.PositionalEncoding,
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
        x = x[:, 0, :]
        x = self.classifier_layer(x)
        return x
    
def build_encoder_only_transformer(src_vocab_size, n_classes,
                                   seq_len, config):

    embedding_dim = config['model']['embedding_dim']
    N = config['model']['N']
    h = config['model']['h']
    dropout = config['model']['dropout']
    d_ff = config['model']['d_ff']
    
    src_embed = new_common_blocks.InputEmbeddings(src_vocab_size,embedding_dim)
    src_pos = new_common_blocks.PositionalEncoding(embedding_dim, seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = new_common_blocks.MultiHeadAttention(seq_len, embedding_dim, h, dropout)
        feed_forward_block = FeedForwardBlock(d_ff,embedding_dim, dropout)
        enc = encoder_block(self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(enc)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    classifier = ClassificationHead(embedding_dim,n_classes)
    transformer = EncoderOnlyTransformer(src_embed, src_pos, encoder, classifier)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer