import os
import yaml
import torch
import torchinfo
import torch.nn as nn

import data_scripts.fetch_data
import data_scripts.preprocess
import model_scripts
import model_scripts.encoder_model
import training_scripts
import training_scripts.train_encoder
import utils
import data_scripts
import utils.dataloader
import utils.metrics
import utils.tokenizer_script 
import utils.common_utils

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_encoder(device):
    # Load configuration
    config = load_config("configs/encoder_config.yaml")
    torch.manual_seed(42)
    
    # DATA HANDLING
    # loading the data
    data_path = data_scripts.fetch_data.fetch_data("stefanlarson/outofscope-intent-classification-dataset")
    # preprocessing 
    train_data, val_data = data_scripts.preprocess.load_data(data_path)
    train_data, val_data = data_scripts.preprocess.load_data(data_path)
    train_data, unique_classes, class_to_id, id_to_class = data_scripts.preprocess.data_preprocessing(train_data)
    val_data, _, _, _ = data_scripts.preprocess.data_preprocessing(val_data)
        
    # loading tokenizer
    tokenizer = utils.tokenizer_script.get_or_build_tokenizer(train_data)    
    max_seq_len = utils.tokenizer_script.get_max_len(tokenizer,train_data) 
    print(f"the max seq len: {max_seq_len} \n")  
    
    # data loader 
    train_ds = utils.dataloader.promptDataset(tokenizer,train_data,max_seq_len+1)
    val_ds = utils.dataloader.promptDataset(tokenizer,val_data,max_seq_len+1)
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=config["data"]["train_batch_size"],shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds,batch_size=config["data"]["val_batch_size"],shuffle=False)
    
    model = model_scripts.encoder_model.build_encoder_only_transformer(
                                            src_vocab_size=tokenizer.get_vocab_size(),
                                            n_classes=len(unique_classes),
                                            seq_len=max_seq_len+1, 
                                            config = config).to(device)
    print("\n The Model Summary: \n")
    print(torchinfo.summary(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]['lr'],eps= config["training"]['eps'])
    loss_fn = nn.CrossEntropyLoss()
    
    if config['training']['train_or_load'] == 'train':
        if config['training']['load_from_checkpoint']:
            utils.common_utils.load_checkpoints(torch.load(config['training']['checkpoint']),model,optimizer)
            
        training_scripts.train_encoder.train(model, train_loader, val_loader,
                                            device,optimizer, loss_fn, config)
        
        # torch.save(model, f'{config['meta_data']['model_name']}.pth')
        model_state = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict()}            
        utils.common_utils.save_checkpoints(model_state, filename = config['training']['model_saved_at'])
        
    
    if config['training']['train_or_load'] == 'load':
        model_path = config['training']['model_saved_at']
        if os.path.exists(model_path):
            utils.common_utils.load_checkpoints(torch.load(model_path),model,optimizer)
        else:
            raise FileNotFoundError("model file not found")
            
    
    utils.metrics.predict_on_sample(model,tokenizer,max_seq_len,
                                    train_ds, id_to_class, train_data,device,n=5)
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    train_encoder(device)
