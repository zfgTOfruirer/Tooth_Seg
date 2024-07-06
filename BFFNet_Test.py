#zfg BFFNet测试代码
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from lib.BFFNet_Model import BFFNet
from utils.dataloader import test_dataset

# zfg 定义参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='../BFFNet-199.pth')

# zfg 循环遍历数据集
for _data_name in ['test']:
    data_path = '../BFFNet-master/data/{}/'.format(_data_name)
    save_path = './results/BFFNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = BFFNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    # zfg 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    # zfg 遍历数据集中的图像
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        # zfg 获取模型输出
        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # zfg 将归一化后的图像转换为 8 位整数
        res_8bit = (res * 255).astype(np.uint8)

        # zfg 将 numpy 数组转换为 PIL 图像
        img = Image.fromarray(res_8bit)

        # zfg 保存图像
        img.save(save_path + name)
