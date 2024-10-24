import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

class promptDataset(Dataset):
    def __init__(self, tokenizer, ds, max_seq_len:int):
        self.ds = ds[["prompt","label"]]
        self.tokenizer = tokenizer
        # self.pad_token_id = tokenizer.eot_token + 1
        self.pad_token_id = torch.tensor(tokenizer.encode("[PAD]").ids, dtype=torch.int64)
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self,idx):
        prompt = self.ds['prompt'][idx]
        label = self.ds['label'][idx]
        
        # tokenized_ids = self.tokenizer.encode(prompt,allowed_special="all")
        tokenized_ids = self.tokenizer.encode(prompt).ids
        
        required_padding_count = self.max_seq_len - len(tokenized_ids) 
        
        padded_prompt_tokens = torch.cat(
            [
            torch.tensor(tokenized_ids,dtype=torch.int64),
            torch.tensor([self.pad_token_id] * required_padding_count, dtype=torch.int64)
            ]
        )
        
        intent_tag = torch.tensor(label,dtype=torch.int64)
        
        return padded_prompt_tokens, intent_tag
    

class ReadBookDataset(Dataset):
    def __init__(self, txt:str, tokenizer,  stride: int, max_len:int):
        self.input_tokens = []
        self.target_tokens = []
        
        all_tokens = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        
        for i in range(0,len(all_tokens)-max_len,stride):
            input_txt_chunk = all_tokens[i : i + max_len]
            tgt_txt_chunk = all_tokens[i + 1: i + max_len + 1]

            self.input_tokens.append(torch.tensor(input_txt_chunk))
            self.target_tokens.append(torch.tensor(tgt_txt_chunk))
    
    def __len__(self):
        return len(self.input_tokens)
    
    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]       


def create_data_loader(txt:str, stride: int,
                       max_length: int,batch_size,
                       shuffle=True, drop_last= False):
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    bookdataset = ReadBookDataset(txt, tokenizer, stride, max_length)
    
    dataloader = DataLoader(bookdataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
    
    return dataloader 