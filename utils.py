from torch.utils.data import Dataset
import torch
import torch.nn as nn
import model_blocks


import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds):
    # Generator to yield sentences from a specific language in the dataset
    for item in ds.iterrows():
        yield item[1]["prompt"]

def get_or_build_tokenizer(ds):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    sentences = list(get_all_sentences(ds))
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    return tokenizer


def load_config():
    return {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "n_epochs":10,
        "train_batch_size":10,
        "val_batch_size":10,
        "sample_size":5
    }
    
def data_loading(path):
    with open(os.path.join(path, 'is_train.json'), 'r') as file:
        train_data = json.load(file)
    with open(os.path.join(path, 'is_val.json'), 'r') as file:
        val_data = json.load(file)
    return train_data, val_data

def get_max_len(tokenizer, ds_raw):
    max_length = 0
    for prompt in ds_raw['prompt']:
        tokenized_length = len(tokenizer.encode(prompt).tokens)
        if tokenized_length > max_length:
            max_length = tokenized_length
    return max_length  

def create_label_mapping(data):
    class_to_id = {}
    id_to_class = {}
    for _,row in data[['label_str','label']].iterrows():
        class_to_id[row['label_str']] = row['label']
        id_to_class[row['label']] = row['label_str']
    return class_to_id,id_to_class

def data_preprocessing(ds):
    prompt = []
    label = []
    ds_upd = pd.DataFrame()
    
    for indiv_data in ds:
        prompt.append(indiv_data[0])
        label.append(indiv_data[1])
    ds_upd["prompt"] = prompt
    ds_upd["label_str"] = label
    
    unique_classes = ds_upd['label_str'].unique()
    class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
    ds_upd['label'] = ds_upd['label_str'].map(class_to_id)
    
    return ds_upd, unique_classes

def get_model(tokenizer,n_classes,device,max_seq_len):
    model = build_encoder_only_transformer(src_vocab_size=tokenizer.get_vocab_size(),
                                           n_classes=n_classes,
                                           seq_len=max_seq_len+1)
    return model.to(device)

   

def train_one_epoch(model,dataloader,device,optimizer,loss_fn):
    model.train()
    batch_iterator = tqdm(enumerate(dataloader),total=len(dataloader))
    epoch_acc = 0
    losses = []
    epoch_all_batches = 0
    for idx, batch in batch_iterator:
        tokenized_prompt = batch[0].to(device)
        actual_label = batch[1].to(device)
        
        optimizer.zero_grad()
        
        model_output = model.encode(tokenized_prompt,src_mask=None)
        # model_output = model_output[:, -1, :]
        predicted_label = torch.argmax(model_output, dim=-1)
        
        
        loss = loss_fn(model_output,actual_label).to(device)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        batch_correct = (predicted_label == actual_label).sum().item()
        batch_accuracy = batch_correct/len(actual_label)
        epoch_acc += batch_accuracy
        epoch_all_batches += len(actual_label)
        
        batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}",
                                     "batch_acc": f"{batch_accuracy:.4f}",
                                     "train_acc": f"{epoch_acc/epoch_all_batches:.4f}"})
    
    return np.mean(losses), epoch_acc/epoch_all_batches

def eval_one_epoch(model, dataloader,device,loss_fn):
    model.eval()
    epoch_acc = 0
    losses = []
    epoch_all_batches = 0
    with torch.no_grad():
        batch_iterator = tqdm(enumerate(dataloader),total=len(dataloader))
        for idx, batch in batch_iterator:
            tokenized_prompt = batch[0].to(device)
            actual_label = batch[1].to(device)
            
            model_output = model.encode(tokenized_prompt,src_mask=None)
            # model_output = model_output[:, -1, :]
            predicted_label = torch.argmax(model_output, dim=-1)
            loss = loss_fn(model_output,actual_label).to(device)
            losses.append(loss.item())
            
            batch_correct = (predicted_label == actual_label).sum().item()
            batch_accuracy = batch_correct/len(actual_label)
            epoch_acc += batch_accuracy
            epoch_all_batches += len(actual_label)
            
            batch_iterator.set_postfix({"val_loss": f"{loss.item():6.3f}",
                                        "val_batch_acc": f"{batch_accuracy:.4f}",
                                        "val_acc": f"{epoch_acc/epoch_all_batches:.4f}"})
        
    return np.mean(losses), epoch_acc/epoch_all_batches
            
def train(model, train_loader, val_loader, epochs, device, optimizer, loss_fn):
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,device,optimizer, loss_fn)
        val_loss, val_acc = eval_one_epoch(model, val_loader,device,loss_fn)
        print(f'ep {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
        print(f'ep {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')



def predict_on_sample(model,tokenizer,max_seq_len, train_ds, config, id_to_class, data,n):
    sampled_data = data.sample(n)
    prompts = sampled_data['prompt'].tolist()
    actual_labels = sampled_data['label_str']

    model.eval()
    with torch.no_grad():
        for prompt, act_label in zip(prompts, actual_labels):
            tokenized_prompt = tokenizer.encode(prompt).ids
            required_pads = max_seq_len - len(tokenized_prompt) +1
            padded_tokens = torch.cat(
                [
                torch.tensor(tokenized_prompt,dtype=torch.int64),
                torch.tensor([train_ds.pad_token_id] * required_pads,dtype=torch.int64)
                ]
                )
            padded_tokens = torch.tensor(padded_tokens,dtype=torch.int64).to(config["device"])
            model_output = model.encode(padded_tokens)
            predicted_class_id = torch.argmax(model_output, dim=-1).item()
            print("prompt :", prompt)
            print("act_label :", act_label)
            print("predicted :", id_to_class[predicted_class_id])
            print()
        
class selfTokenizer():
    def __init__(self, corpus , seperator=" "):
        self.seperator = seperator
        list_of_token_words = list(set(corpus.split(seperator)))
        self.txt_to_id = {txt:idx for idx,txt in enumerate(list_of_token_words)}
        self.id_to_txt = {idx:txt for idx,txt in enumerate(list_of_token_words)}
        self.vocab_size = len(list_of_token_words)
        
    def encode(self, seq):
        list_of_tokens = seq.split(self.seperator)
        return [self.txt_to_id[word] for word in list_of_tokens]
    
    def decode(self, list_of_ids:list):
        return self.seperator.join([self.id_to_txt[id] for id in list_of_ids])

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
    
def build_encoder_only_transformer(src_vocab_size, n_classes,
                                   seq_len, embedding_dim=512,
                                   N =6, h=8, dropout=0.01, d_ff=2048):

    src_embed = model_blocks.InputEmbeddings(src_vocab_size,embedding_dim)
    src_pos = model_blocks.PositionalEncoding(embedding_dim, seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attention_block = model_blocks.MultiHeadAttention(seq_len, embedding_dim, h, dropout)
        feed_forward_block = model_blocks.FeedForwardBlock(d_ff,embedding_dim, dropout)
        enc = model_blocks.encoder_block(self_attention_block,feed_forward_block,dropout)
        encoder_blocks.append(enc)

    encoder = model_blocks.Encoder(nn.ModuleList(encoder_blocks))
    classifier = model_blocks.ClassificationHead(embedding_dim,n_classes)
    transformer = model_blocks.EncoderOnlyTransformer(src_embed, src_pos, encoder, classifier)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer