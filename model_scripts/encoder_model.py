
import torch.nn as nn
from model_scripts import common_blocks


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim:int, num_classes:int):
        super().__init__()
        self.classifier_layer = nn.Linear(embedding_dim,num_classes)
    def forward(self,x):
        return self.classifier_layer(x)
        
class EncoderOnlyTransformer(nn.Module):
    def __init__(self,
                 src_embed: common_blocks.InputEmbeddings, pos: common_blocks.PositionalEncoding,
                 encoder:common_blocks.Encoder, classifier_layer :ClassificationHead):
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
    
    src_embed = common_blocks.InputEmbeddings(src_vocab_size,embedding_dim)
    src_pos = common_blocks.PositionalEncoding(embedding_dim, seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = common_blocks.MultiHeadAttention(seq_len, embedding_dim, h, dropout)
        feed_forward_block = common_blocks.FeedForwardBlock(d_ff,embedding_dim, dropout)
        enc = common_blocks.encoder_block(self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(enc)

    encoder = common_blocks.Encoder(nn.ModuleList(encoder_blocks))
    classifier = ClassificationHead(embedding_dim,n_classes)
    transformer = EncoderOnlyTransformer(src_embed, src_pos, encoder, classifier)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer