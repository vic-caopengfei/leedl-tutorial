import os
import random
import pandas as pd
import numpy as np
# 导入pytorch
import torch
from torch.utils.data import Dataset
# 导入进度条
from tqdm import tqdm


# 定义导入feature函数
def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


# 将前后的特征联系在一起，如concat_n = 11 则前后都接5
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n 必须是奇数
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


# 数据预处理函数
def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41  # NOTE: 预先计算，不需要更改
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
        phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

        for line in phone_file:
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # 分割训练和验证数据
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
            label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
            y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
        y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
        print(y.shape)
        return X, y
    else:
        return X


# 返回的X代表数据的维度，如果不链接则为39 如果链接即为n*39 n为连接的特征总数,y为标签
class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


def same_seeds(seed): # 固定随机种子（CPU）
    torch.manual_seed(seed) # 固定随机种子（GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True # 固定网络结构