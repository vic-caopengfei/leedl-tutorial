import gc

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from HW02_classification.dataset import preprocess_data, LibriDataset, same_seeds
from HW02_classification.model import Classifier, train, predict

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    concat_nframes = 1  # 要连接的帧数,n必须为奇数（总共2k+1=n帧）
    train_ratio = 0.8  # 用于训练的数据比率，其余数据将用于验证
    # training parameters
    # 训练过程中的参数
    seed = 0  # 随机种子
    batch_size = 512  # 批次数目
    num_epoch = 5  # 训练epoch数
    learning_rate = 0.0001  # 学习率
    model_path = './model.ckpt'  # 选择保存检查点的路径（即下文调用保存模型函数的保存位置）
    # model parameters
    # 模型参数
    input_dim = 39 * concat_nframes  # 模型的输入维度，不应更改该值，这个值由上面的拼接函数决定
    hidden_layers = 1  # hidden_layer层的数量
    hidden_dim = 256
    data_set_path = './dataset/libriphone'
    # 预处理数据
    train_X, train_y = preprocess_data(split='train', feat_dir=data_set_path + '/feat', phone_path=data_set_path,
                                       concat_nframes=concat_nframes, train_ratio=train_ratio)
    val_X, val_y = preprocess_data(split='val', feat_dir=data_set_path + '/feat', phone_path=data_set_path,
                                   concat_nframes=concat_nframes, train_ratio=train_ratio)

    # 将数据导入
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # 删除原始数据以节省内存
    del train_X, train_y, val_X, val_y
    gc.collect()

    # 利用dataloader加载数据
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 固定随机种子
    same_seeds(seed)

    # 创建模型、定义损失函数和优化器
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train(num_epoch, model, train_loader, device, optimizer, criterion, val_set, val_loader, train_set, model_path)
    predict(concat_nframes, batch_size, input_dim, hidden_layers, hidden_dim, device, model_path, data_set_path)
