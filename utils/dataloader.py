import torch
from torch.utils.data import DataLoader, Dataset

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
    

