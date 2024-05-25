import csv

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import same_seed, train_valid_split, My_Model, trainer, predict, save_pred
from train_dataset import select_feat, TrainDataset, formatDataSet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 12,  # 随机种子，可以自己填写. :)
    'select_all': True,  # 是否选择全部的特征
    'valid_ratio': 0.3,  # 验证集大小(validation_size) = 训练集大小(train_size) * 验证数据占比(valid_ratio)
    'n_epochs': 3000,  # 数据遍历训练次数
    'batch_size': 12,
    'learning_rate': 1e-5,
    'early_stop': 400,  # 如果early_stop轮损失没有下降就停止训练.
    'save_path': './models/model.ckpt'  # 模型存储的位置
}

if __name__ == '__main__':
    # 设置随机种子便于复现
    same_seed(config['seed'])

    pd.set_option('display.max_column', 200)  # 设置显示数据的列数
    train_df, test_df = pd.read_csv('./train.csv'), pd.read_csv('./test.csv')

    train_df['Asus'] = train_df['Asus'].map({True: 'True',  False: 'False'})
    train_df['Acer'] = train_df['Acer'].map({True: 'True',  False: 'False'})
    train_df['Lenovo'] = train_df['Lenovo'].map({True: 'True',  False: 'False'})
    train_df['HP'] = train_df['HP'].map({True: 'True',  False: 'False'})
    train_df['Dell'] = train_df['Dell'].map({True: 'True',  False: 'False'})

    test_df['Asus'] = test_df['Asus'].map({True: 'True',  False: 'False'})
    test_df['Acer'] = test_df['Acer'].map({True: 'True',  False: 'False'})
    test_df['Lenovo'] = test_df['Lenovo'].map({True: 'True',  False: 'False'})
    test_df['HP'] = test_df['HP'].map({True: 'True',  False: 'False'})
    test_df['Dell'] = test_df['Dell'].map({True: 'True',  False: 'False'})

    print(train_df.head(3))  # 显示前三行的样本
    train_data, test_data = train_df.values, test_df.values
    del train_df, test_df  # 删除数据减少内存占用
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # 打印数据的大小
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # 特征选择
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

    # 打印出特征数量.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)
    test_dataset = TrainDataset(x_test)

    # 使用Pytorch中Dataloader类按照Batch将数据集加载
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = My_Model(input_dim=x_train.shape[1]).to(device)  # 将模型和训练数据放在相同的存储位置(CPU/GPU)
    trainer(train_loader, valid_loader, model, config, device)

    # model = My_Model(input_dim=x_train.shape[1]).to(device)
    # model.load_state_dict(torch.load(config['save_path']))
    # preds = predict(test_loader, model, device)
    # save_pred(preds, 'pred.csv')
