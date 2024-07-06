#zfg BFFNet 训练代码
import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.BFFNet_Model import BFFNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter


#zfg 定义结构损失函数
def structure_loss(pred, mask):
   weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
   wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
   wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
   pred = torch.sigmoid(pred)
   inter = ((pred * mask) * weit).sum(dim=(2, 3))
   union = ((pred + mask) * weit).sum(dim=(2, 3))
   wiou = 1 - (inter + 1) / (union - inter + 1)
   return (wbce + wiou).mean()

#zfg 定义训练函数
def train(train_loader, model, optimizer, epoch):
   min_loss = 10
   model.train()
   # ---- multi-scale training ----
   size_rates = [0.75, 1, 1.25]
   loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
   for i, pack in enumerate(train_loader, start=1):
       for rate in size_rates:
           optimizer.zero_grad()
           # ---- data prepare ----
           images, gts = pack
           images = Variable(images).cuda()
           gts = Variable(gts).cuda()
           # ---- rescale ----
           trainsize = int(round(opt.trainsize * rate / 32) * 32)
           if rate != 1:
               images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
               gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
           # ---- forward ----
           lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images)
           # ---- loss function ----
           loss5 = structure_loss(lateral_map_5, gts)
           loss4 = structure_loss(lateral_map_4, gts)
           loss3 = structure_loss(lateral_map_3, gts)
           loss2 = structure_loss(lateral_map_2, gts)
           loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
           # ---- backward ----
           loss.backward()
           clip_gradient(optimizer, opt.clip)
           optimizer.step()
           # ---- recording loss ----
           if rate == 1:
               loss_record2.update(loss2.data, opt.batchsize)
               loss_record3.update(loss3.data, opt.batchsize)
               loss_record4.update(loss4.data, opt.batchsize)
               loss_record5.update(loss5.data, opt.batchsize)
       # ---- train visualization ----
       if i % 20 == 0 or i == total_step:
           print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                 '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                 format(datetime.now(), epoch, opt.epoch, i, total_step,
                        loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
   save_path = 'snapshots/{}/'.format(opt.train_save)
   os.makedirs(save_path, exist_ok=True)
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'BFFNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'BFFNet-%d.pth' % epoch)
    loss_record = loss_record2.show().item() + loss_record3.show().item() + loss_record4.show().item() + loss_record5.show().item()
    if epoch < 201:
        if loss_record <= min_loss:
            min_loss = loss_record
            best_model = model.state_dict()  # 当前模型的参数保存到best_model变量中
            torch.save(best_model, save_path + 'BFFNet-best.pth')
            print('[loss toatl:{}]'.format(loss_record))
            print('[Saving Snapshot:]', save_path + 'BFFNet-best_{}.pth'.format(epoch))


if __name__ == '__main__':  # zfg: 入口函数
    parser = argparse.ArgumentParser()  # zfg: 创建ArgumentParser对象以处理命令行参数
    parser.add_argument('--epoch', type=int,
                        default=20, help='epoch number')  # zfg: 添加epoch参数，默认值为20
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')  # zfg: 添加learning rate参数，默认值为0.0001
    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')  # zfg: 添加batch size参数，默认值为2
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')  # zfg: 添加training dataset size参数，默认值为352
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')  # zfg: 添加gradient clipping margin参数，默认值为0.5
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')  # zfg: 添加decay rate参数，默认值为0.1
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')  # zfg: 添加decay epoch参数，默认值为50
    parser.add_argument('--train_path', type=str,
                        default='../BFFNet-master/tooth/train',
                        help='path to train dataset')  # zfg: 添加train dataset path参数，默认路径为'../BFFNet-master/tooth/train'
    parser.add_argument('--train_save', type=str,
                        default='PraNet_Res2Net')  # zfg: 添加train save路径参数，默认值为'PraNet_Res2Net'
    opt = parser.parse_args()  # zfg: 解析命令行参数

    # ---- build models ----
    model = BFFNet().cuda()  # zfg: 实例化BFFNet模型并将其加载到GPU上
    # ---- flops and params ----
    params = model.parameters()  # zfg: 获取模型参数
    optimizer = torch.optim.Adam(params, opt.lr)  # zfg: 使用Adam优化器并设置学习率

    image_root = '{}/image/'.format(opt.train_path)  # zfg: 设置训练图像路径
    gt_root = '{}/mask/'.format(opt.train_path)  # zfg: 设置训练标签路径

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)  # zfg: 创建训练数据加载器
    total_step = len(train_loader)  # zfg: 获取总的训练步骤数

    print("#" * 20, "Start Training", "#" * 20)  # zfg: 打印开始训练的标志信息

    for epoch in range(1, opt.epoch):  # zfg: 训练循环，从1到epoch数
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)  # zfg: 调整学习率
        train(train_loader, model, optimizer, epoch)  # zfg: 调用训练函数进行训练

