import torch
import torch.nn as nn
import tqdm
import numpy as np

def train_one_epoch(model,dataloader,device,optimizer,loss_fn):
    model.train()
    batch_iterator = tqdm(enumerate(dataloader),total=len(dataloader))
    losses = []
    accuracies = []
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
            actual_label = batch[1].to(device)
            
            model_output = model.encode(tokenized_prompt,src_mask=None)
            # model_output = model_output[:, -1, :]
            predicted_label = torch.argmax(model_output, dim=-1)
            loss = loss_fn(model_output,actual_label).to(device)
            losses.append(loss.item())
            
            batch_correct = (predicted_label == actual_label).sum().item()
            batch_accuracy = batch_correct/len(actual_label)
            accuracies.append(batch_accuracy)
            
            batch_iterator.set_postfix({"val_loss": f"{loss.item():6.3f}",
                                        "val_batch_acc": f"{batch_accuracy:.4f}"})
        
    return np.mean(losses), np.mean(accuracies)

def train(model, train_loader, val_loader, epochs, device, optimizer, loss_fn):
    for ep in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader,device,optimizer, loss_fn)
        val_loss, val_acc = eval_one_epoch(model, val_loader,device,loss_fn)
        print(f'ep {ep}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}')
        print(f'ep {ep}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        
