from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
import random
import os

def make_seq(df, non_cat, cat, target, seq_length=5):
    '''
    @args
    df: dataframe to transform
    non_cat: non_categorical cols -> LIST
    cat: categorical cols -> LIST
    target: target col -> str
    seq_length: how long seq to input
    '''
    non_cat_ = df[non_cat]
    cat_ = df[cat]
    target_ = df[target][seq_length:]
    enc = OneHotEncoder()
    enc.fit(cat_)
    arr_cat = enc.transform(cat_).toarray()
    
    data = list()
    for i in range(len(arr_cat)):
        data.append(np.concatenate((non_cat_.iloc[i], arr_cat[i])))
    data = np.array(data)

    temp = list()
    for i in range(len(data)-(seq_length-1)):
        temp.append(data[i:i+seq_length])
    data = np.array(temp)[:-1, :, :]
    print('shape of sequential input data: {}'.format(data.shape))
    return data, target_

class Custom_Dataset(Dataset):
    def __init__(self, input_arr, target_arr):
        self.input = input_arr
        self.target = target_arr
        print('get {:,} samples'.format(len(input_arr)))
        self.data = list()
        for i, t in zip(self.input, self.target):
            temp = {'inputs': i, 'labels': t}
            self.data.append(temp)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def Custom_Random_Split(dataset, val_ratio, test_ratio, random_seed):
    SEED = random_seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(SEED)

    num_val = int(len(dataset) * val_ratio)
    num_test = int(len(dataset) * test_ratio)
    num_train = len(dataset) - (num_val + num_test)
    train_set, val_set, test_set = random_split(dataset, [num_train, num_val, num_test], generator=torch.Generator().manual_seed(SEED))
    print('Number of datasets {:,} : {:,} : {:,}'.format(len(train_set), len(val_set), len(test_set)))
    return train_set, val_set, test_set

def Custom_Loader(train, val, test, batch_size):
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# for insert mode