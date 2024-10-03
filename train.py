import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

# Assuming these are your custom modules or functions
from dataset import BilingualDataset  # Check if this is your custom dataset class
from model import build_transformer  # Check if this is your model builder function
from config import get_config, get_weights_filepath  # Check if this is your configuration loader function

def get_all_sentences(ds, lang):
    # Generator to yield sentences from a specific language in the dataset
    for item in ds:
        yield item["translation"][lang]

def get_or_build_tokenizer(config, ds, lang):
    # Function to get or build tokenizer based on language and dataset
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        sentences = list(get_all_sentences(ds, lang))
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_datasets(config):
    # Function to load and preprocess datasets
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    return train_ds, val_ds, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    # Function to build your transformer model
    model = build_transformer(vocab_src_len, vocab_tgt_len,
                              config['seq_len'], config['seq_len'],
                              config['d_model'])
    return model

def train_model(config):
    # Function to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, tokenizer_src, tokenizer_tgt = get_datasets(config)

    # Get the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    # Cross entropy loss with padding token ignored
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]'), reduction='mean').to(device)

    writer = SummaryWriter(config['experiment_name'])
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_filepath(config, config['preload'])
        print(f"Pre-loading model from {model_filename}")
        checkpoint = torch.load(model_filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        epoch_loss = 0.0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (Batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (Batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)    # (Batch, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)    # (Batch, 1, seq_len, seq_len)
            label = batch['label'].to(device)                  # (Batch, seq_len)

            optimizer.zero_grad()

            # Run tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (Batch, seq_len, d_model)
            proj_output = model.project(decoder_output)  # (Batch, seq_len, tgt_vocab_size)

            # Calculate loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            batch_iterator.set_postfix({'loss': epoch_loss / len(train_dataloader)})

            # Log the loss to tensorboard
            writer.add_scalar('train_loss', loss.item(), global_step)

        # Save model checkpoint after each epoch
        model_filename = get_weights_filepath(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

    writer.close()

if __name__ == "__main__":
    config = get_config()
    train_model(config)
