import torch
import torch.nn as nn
from torch.utils.data import dataset
from typing import Any


class BilingualDataset(dataset):
    def __init__(self, ds,
                 tokenizer_src, tokenizer_tgt,
                 src_lang,tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # tensor containing only one number - coming from token_to_id
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype = torch.int64)        
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype = torch.int64)        
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype = torch.int64)        
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        #encode to id - as array
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # we need to pad the sequence, how many padding token do we need?
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # both <SOS> <EOS>
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # only <SOS>
        
        #make sure the padding does not crosses seq_len
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('sentence is too long')
        
        # add <SOS> and <EOS> to the source text        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        ) 
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1,seq_len, seq_len)
            "label": label,
            "src_text":src_text,
            "tgt_text":tgt_text
        }
        
def causal_mask(size):
    # each word can only look at previous words only
    #triu - give me all value above the diagonal
    mask  = torch.triu(torch.ones(1,size, size),diagonal=1).type(torch.int)
    return mask == 0  
    