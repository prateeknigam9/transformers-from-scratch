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

def get_max_len(tokenizer, ds_raw):
    max_length = 0
    for prompt in ds_raw['prompt']:
        tokenized_length = len(tokenizer.encode(prompt).tokens)
        if tokenized_length > max_length:
            max_length = tokenized_length
    return max_length  

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