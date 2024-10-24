import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utils import common_utils

def train_one_epoch(model,dataloader,device,optimizer,loss_fn):
    model.train()
    batch_iterator = tqdm(enumerate(dataloader),total=len(dataloader))
    losses = []
    accuracies = []
    for idx, batch in batch_iterator:
        tokenized_prompt = batch[0].to(device)
        tokenized_target = batch[1].to(device)
        
        last_token_targets = tokenized_target[:,-1]
        
        optimizer.zero_grad()
        
        logits = model(tokenized_prompt,mask=True)
        last_token_logits = logits[:, -1, :]
                
        loss = loss_fn(last_token_logits,last_token_targets).to(device)
        loss.backward()
        optimizer.step()        
        losses.append(loss.item())
        
        predicted_indices = torch.argmax(last_token_logits, dim=-1)
        batch_correct = (predicted_indices == last_token_targets).sum().item()        
        batch_accuracy = batch_correct/len(tokenized_target)
        accuracies.append(batch_accuracy)
        
        batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}",
                                     "batch_acc": f"{batch_accuracy:.4f}"})
    
    return np.mean(losses), np.mean(accuracies)


def eval_one_epoch(model, dataloader,device,loss_fn):
    model.eval()
    losses = []
    accuracies = []
    with torch.no_grad():
        batch_iterator = tqdm(enumerate(dataloader),total=len(dataloader))
        for idx, batch in batch_iterator:            
            tokenized_prompt = batch[0].to(device)
            tokenized_target = batch[1].to(device)
            
            last_token_targets = tokenized_target[:,-1]
            
            logits = model(tokenized_prompt,mask=True)
            last_token_logits = logits[:, -1, :]
                    
            loss = loss_fn(last_token_logits,last_token_targets).to(device)
            losses.append(loss.item())
            
            predicted_indices = torch.argmax(last_token_logits, dim=-1)
            batch_correct = (predicted_indices == last_token_targets).sum().item()        
            batch_accuracy = batch_correct/len(tokenized_target)
            accuracies.append(batch_accuracy)
            
            batch_iterator.set_postfix({"train_loss": f"{loss.item():6.3f}",
                                        "batch_acc": f"{batch_accuracy:.4f}"})
                   
        return np.mean(losses), np.mean(accuracies)

def train(model, train_loader, val_loader, device, optimizer, loss_fn,config):
    epochs = config["training"]["num_epochs"]
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,device,optimizer, loss_fn)
        val_loss, val_acc = eval_one_epoch(model, val_loader,device,loss_fn)
        print(f'ep {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
        print(f'ep {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        
        #Saving checkpoints        
        if ep % 2 == 0:
            checkpoints = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}
            common_utils.save_checkpoints(checkpoints, filename = config['training']['checkpoint'])
            
