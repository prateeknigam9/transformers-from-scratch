import os
import yaml
import torch
import torchinfo
import torch.nn as nn

import tiktoken

import data_scripts.fetch_data
import data_scripts.preprocess
import model_scripts
import model_scripts.decoder_model
import model_scripts.encoder_model
import training_scripts
import training_scripts.train_decoder
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

def train_decoder(device):
    # Load configuration
    config = load_config("configs/decoder_config.yaml")
    torch.manual_seed(42)
    
    # DATA HANDLING
    # loading the data
    raw_text_book_data = data_scripts.preprocess.load_data_for_decoder('pride_and_prejudice.txt')
    
    train_ratio = config["train_test_split"]["ratio"]
    split_idx = int(train_ratio * len(raw_text_book_data))
    train_data = raw_text_book_data[:split_idx]
    validation_data = raw_text_book_data[split_idx:]
    
    # tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    
    train_loader = utils.dataloader.create_data_loader(train_data,batch_size=config["data"]["train_batch_size"],
                                                       stride=config["data"]["stride"],
                                                       max_length=config["data"]["max_length"],
                                                       shuffle=True, drop_last=True)
    val_loader = utils.dataloader.create_data_loader(validation_data,batch_size=config["data"]["val_batch_size"],
                                                       stride=config["data"]["stride"],
                                                       max_length=config["data"]["max_length"],
                                                       shuffle=False, drop_last=False)
    
    # setting up the model
    model = model_scripts.decoder_model.construct_decoder_model(vocab_size=tokenizer.max_token_value+1,
                                                                d_model = config['model']['d_model'],
                                                                seq_len=config['model']['seq_len'],
                                                                n_decoder_blocks = config['model']['N'],
                                                                n_heads = config['model']['h'],
                                                                dropout=config['model']['dropout']).to(device)
    print("\n The Model Summary: \n")
    print(torchinfo.summary(model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]['lr'],eps= config["training"]['eps'])
    loss_fn = nn.CrossEntropyLoss()
    
    if config['training']['train_or_load'] == 'train':
        if config['training']['load_from_checkpoint']:
            utils.common_utils.load_checkpoints(torch.load(config['training']['checkpoint']),model,optimizer)
            
        training_scripts.train_decoder.train(model, train_loader, val_loader,
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
            
    
    # utils.metrics.predict_on_sample(model,tokenizer,max_seq_len,
    #                                 train_ds, id_to_class, train_data,device,n=5)
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    train_decoder(device)
