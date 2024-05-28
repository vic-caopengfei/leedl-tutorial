# 数值、矩阵操作
import math
import numpy as np

# 数据读取与写入make_dot
import pandas as pd
import os
import csv

# 进度条
from tqdm import tqdm
# 如果是使用notebook 推荐使用以下（颜值更高 : ) ）
# from tqdm.notebook import tqdm

# Pytorch 深度学习张量操作框架
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
# 绘制pytorch的网络
from torchviz import make_dot

# 学习曲线绘制
from torch.utils.tensorboard import SummaryWriter

def same_seed(seed):
    '''
    设置随机种子(便于复现) 这里生成的就是伪随机数，只要给定一个固定的随机数的初始值，则每次生成的一系列随机数就是的固定的，便于复现结果
    '''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set Seed = {seed}')


def train_valid_split(data_set, valid_ratio, seed):
    '''
    数据集拆分成训练集（training set）和 验证集（validation set）
    '''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval() # 设置成eval模式.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改模型结构, 注意矩阵的维度（dimensions）
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # 损失函数的定义

    # 定义优化器
    # TODO: 可以查看学习更多的优化器 https://pytorch.org/docs/stable/optim.html
    # TODO: L2 正则( 可以使用optimizer(weight decay...) )或者 自己实现L2正则.
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # tensorboard 的记录器
    writer = SummaryWriter()

    if not os.path.isdir('./models'):
        # 创建文件夹-用于存储模型
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # 训练模式
        loss_record = []

        # tqdm可以帮助我们显示训练的进度
        train_pbar = tqdm(train_loader, position=0, leave=True)
        # 设置进度条的左边 ： 显示第几个Epoch了
        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        for x, y in train_pbar:
            optimizer.zero_grad()  # 将梯度置0.
            x, y = x.to(device), y.to(device)  # 将数据一到相应的存储位置(CPU/GPU)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # 反向传播 计算梯度.
            optimizer.step()  # 更新网络参数
            step += 1
            loss_record.append(loss.detach().item())

            # 训练完一个batch的数据，将loss 显示在进度条的右边
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        # 每个epoch,在tensorboard 中记录训练的损失（后面可以展示出来）
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # 将模型设置成 evaluation 模式.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # 每个epoch,在tensorboard 中记录验证的损失（后面可以展示出来）
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # 模型保存
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return

def save_pred(preds, file):
    ''' 将模型保存到指定位置'''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

