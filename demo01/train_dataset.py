import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    '''
    x: np.ndarray  特征矩阵.
    y: np.ndarray  目标标签, 如果为None,则是预测的数据集
    '''

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''
    特征选择
    选择较好的特征用来拟合回归模型
    '''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: 选择需要的特征 ，这部分可以自己调研一些特征选择的方法并完善.
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def formatDataSet(data):
    df = pd.DataFrame(data)
    # 使用 get_dummies 进行 One-Hot Encoding
    df_one_hot = pd.get_dummies(df, columns=[0])

    # 将编码后的列移到前面
    cols = df_one_hot.columns.tolist()
    brand_cols = [col for col in cols if col.startswith(0)]
    other_cols = [col for col in cols if not col.startswith(0)]
    new_order = brand_cols + other_cols
    df_one_hot = df_one_hot[new_order]
    return df_one_hot