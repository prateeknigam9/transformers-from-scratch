
import torch.nn as nn
from model_scripts import common_blocks


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim:int, vocab_size:int):
        super().__init__()
        self.generation_layer = nn.Linear(embedding_dim,vocab_size)
    def forward(self,x):
        return self.generation_layer(x)
        # return torch.log_softmax(self.output(x), dim=-1)
        
        
class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 src_embed: common_blocks.InputEmbeddings, pos: common_blocks.PositionalEncoding,
                 decoder:common_blocks.Decoder, projection_layer :ProjectionHead):
        super().__init__()
        self.src_embed = src_embed
        self.pos = pos
        self.decoder = decoder
        self.projection_layer = projection_layer
                
    def decode(self,x, src_mask, tgt_mask):
        x = self.src_embed(x)
        x = self.pos(x)
        x = self.decoder(x, x, src_mask, tgt_mask)
        x = x[:, 0, :]
        x = self.projection_layer(x)
        return x
    
def build_decoder_only_transformer(src_vocab_size, tgt_vocab_size,
                                   seq_len, config):

    embedding_dim = config['model']['embedding_dim']
    N = config['model']['N']
    h = config['model']['h']
    dropout = config['model']['dropout']
    d_ff = config['model']['d_ff']
    
    src_embed = common_blocks.InputEmbeddings(src_vocab_size,embedding_dim)
    src_pos = common_blocks.PositionalEncoding(embedding_dim, seq_len, dropout)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = common_blocks.MultiHeadAttention(seq_len, embedding_dim, h, dropout)
        feed_forward_block = common_blocks.FeedForwardBlock(d_ff,embedding_dim, dropout)
        enc = common_blocks.decoder_blocks(self_attention_block,self_attention_block,feed_forward_block,dropout)
        decoder_blocks.append(enc)

    decoder = common_blocks.Decoder(nn.ModuleList(decoder_blocks))
    classifier = ProjectionHead(embedding_dim,tgt_vocab_size)
    transformer = DecoderOnlyTransformer(src_embed, src_pos, decoder, classifier)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer