#data libraries
import kagglehub

import torch
import torch.nn as nn
import torchinfo
import utils
import tiktoken

  

path = kagglehub.dataset_download("stefanlarson/outofscope-intent-classification-dataset")
print("Path to dataset files:", path)

def training_model():
    model = utils.get_model(tokenizer,len(unique_classes),config['device'],max_seq_len)
    print("\n The Model Summary: \n")
    print(torchinfo.summary(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=10**-4,eps= 1e-9)
    loss_fn = nn.CrossEntropyLoss()
    
    utils.train(model, train_loader, val_loader, config["n_epochs"], config["device"], optimizer, loss_fn)
    
    return model

if __name__ == "__main__":
    torch.manual_seed(42)
    config = utils.load_config()
    train_data, val_data = utils.data_loading(path)
    
    train_data, unique_classes = utils.data_preprocessing(train_data)
    val_data, unique_classes = utils.data_preprocessing(val_data)
    
    class_to_id,id_to_class = utils.create_label_mapping(train_data)
    
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = utils.get_or_build_tokenizer(train_data)
    
    max_seq_len = utils.get_max_len(tokenizer,train_data) 
    print(f"the max seq len: {max_seq_len} \n")  
    
    train_ds = utils.promptDataset(tokenizer,train_data,max_seq_len+1)
    val_ds = utils.promptDataset(tokenizer,val_data,max_seq_len+1)
    
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=config["train_batch_size"],shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=config["val_batch_size"],shuffle=False)
    
    model = training_model()
    
    utils.predict_on_sample(model,tokenizer,max_seq_len, train_ds, config, id_to_class, train_data,n=5)