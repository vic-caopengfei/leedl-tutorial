from HW03_CNN.dataset import FoodDataset
from HW03_CNN.model import Classifier
from HW03_CNN.util import quick_observe, model_plot, all_seed, trainer
import torch
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    train_dir_root = './dataset/food11/training'
    # quick_observe(train_dir_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 6666,
        'dataset_dir': "./dataset/food11",
        'n_epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.0003,
        'weight_decay': 1e-5,
        'early_stop': 300,
        'clip_flag': True,
        'save_path': './models/model.ckpt',
        'resnet_save_path': './models/resnet_model.ckpt'
    }

    test_tfm = transforms.Compose([

        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # 当然，我们也可以再测试集中对数据进行扩增（对同样本的不同装换）
    #  - 用训练数据的装化方法（train_tfm）去对测试集数据进行转化，产出扩增样本
    #  - 对同个照片的不同样本分别进行预测
    #  - 最后可以用soft vote / hard vote 等集成方法输出最后的预测
    train_tfm = transforms.Compose([
        # 图片裁剪 (height = width = 128)
        transforms.Resize((128, 128)),
        # TODO:在这部分还可以增加一些图片处理的操作
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        # ToTensor() 放在所有处理的最后
        transforms.ToTensor(),
    ])

    print(device)
    all_seed(config['seed'])

    x = torch.randn(1, 3, 128, 128).requires_grad_(True)
    model_plot(Classifier, x)

    _dataset_dir = config['dataset_dir']

    train_set = FoodDataset(os.path.join(_dataset_dir, "training"), tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    valid_set = FoodDataset(os.path.join(_dataset_dir, "validation"), tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

    # 测试级保证输出顺序一致
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # 测试集数据扩增
    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=train_tfm)
    test_loader_extra1 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                    pin_memory=True)

    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=train_tfm)
    test_loader_extra2 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                    pin_memory=True)

    test_set = FoodDataset(os.path.join(_dataset_dir, "test"), tfm=train_tfm)
    test_loader_extra3 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0,
                                    pin_memory=True)

    model = Classifier().to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # 模型测试
    # model_best = Classifier().to(device)
    # model_best.load_state_dict(torch.load(config['save_path']))
    # model_best.eval()
    # prediction = []
    # with torch.no_grad():
    #     for data, _ in test_loader:
    #         test_pred = model_best(data.to(device))
    #         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
    #         prediction += test_label.squeeze().tolist()
    #
    # test_loaders = [test_loader_extra1, test_loader_extra2, test_loader_extra3, test_loader]
    # loader_nums = len(test_loaders)
    # # 存储每个dataloader预测结果，一个dataloader一个数组
    # loader_pred_list = []
    # for idx, d_loader in enumerate(test_loaders):
    #     # 存储一个dataloader的预测结果,  一个batch一个是数组
    #     pred_arr_list = []
    #     with torch.no_grad():
    #         tq_bar = tqdm(d_loader)
    #         tq_bar.set_description(f"[ DataLoader {idx + 1}/{loader_nums} ]")
    #         for data, _ in tq_bar:
    #             test_pred = model_best(data.to(device))
    #             logit_pred = test_pred.cpu().data.numpy()
    #             pred_arr_list.append(logit_pred)
    #         # 将每个batch的预测结果合并成一个数组
    #         loader_pred_list.append(np.concatenate(pred_arr_list, axis=0))
    #
    # # 将预测结果合并
    # pred_arr = np.zeros(loader_pred_list[0].shape)
    # for pred_arr_t in loader_pred_list:
    #     pred_arr += pred_arr_t
    #
    # soft_vote_prediction = np.argmax(0.5 * pred_arr / len(loader_pred_list) + 0.5 * loader_pred_list[-1], axis=1)

