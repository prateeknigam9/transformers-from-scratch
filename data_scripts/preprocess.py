import os
import json
import pandas as pd

  
def load_data(path):
    with open(os.path.join(path, 'is_train.json'), 'r') as file:
        train_data = json.load(file)
    with open(os.path.join(path, 'is_val.json'), 'r') as file:
        val_data = json.load(file)
    return train_data, val_data

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
    id_to_class = {idx: cls for idx, cls in enumerate(unique_classes)}
    ds_upd['label'] = ds_upd['label_str'].map(class_to_id)
    
    return ds_upd, unique_classes, class_to_id, id_to_class


