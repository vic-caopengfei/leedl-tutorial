import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from HW02_classification.dataset import preprocess_data, LibriDataset


# 建立神经网络
class BasicBlock(nn.Module):  # 继承 torch 的 Module
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],  # *[]将循环得到的解压
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def train(num_epoch, model, train_loader, device, optimizer, criterion, val_set, val_loader, train_set, model_path):
    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # 训练部分
        model.train()  # 设定模型到训练模式
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)  # 获得概率最高的类的索引
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # 验证部分
        if len(val_set) > 0:
            model.eval()  # 设定模型到评估模式
            with torch.no_grad():
                for i, batch in enumerate(tqdm(val_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)

                    loss = criterion(outputs, labels)

                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item()  # 获得概率最高的类的索引
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                    val_acc / len(val_set), val_loss / len(val_loader)
                ))

                # 如果模型获得提升，在此阶段保存模型
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
            ))

    # 如果结束验证，则保存最后一个epoch得到的模型
    if len(val_set) == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')


def predict(concat_nframes, batch_size, input_dim, hidden_layers, hidden_dim, device, model_path,data_set_path):
    test_X = preprocess_data(split='test',
                             feat_dir=data_set_path+'/feat',
                             phone_path=data_set_path,
                             concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    test_acc = 0.0
    test_lengths = 0
    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 1)  # 获得概率最高的类的索引
            pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))
