import argparse
import gc
import logging
import os
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import gc
from dataset.data_aug import GaussianNoise
from dataset.data_aug import RandomChoice
from sklearn.metrics import roc_auc_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# 针对MCI
# import net_galm_3 as gfnet
# import net.resnet18_3d_dropout as resnet
from dataset.utils import AverageMeter,setup_seed,calc_eval

from dataset.data_label import set_labels,set_filenames,ADNIDataset,StratifiedSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('E:/CNN_FFT/AD_CN/Resnet18_0519/fold4_best model.pth', weights_only=True)
filenames = set_filenames()
labels = set_labels(filenames)

print(filenames)
# 划分后的测试集
test_dataset = ADNIDataset(filenames,labels)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
criterion = nn.CrossEntropyLoss()
model_state_dict = checkpoint['latent_encoder']
# model = gfnet.DenseNetWithGFNet(depth=6, num_classes=2)
# model=gfnet.GFNet(depth=8,num_classes=2)
model=resnet.resnet18_3d(num_classes=2,in_channels=1)
if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model).to(device)
opt_l = torch.optim.Adam(model.parameters(), lr=0.00045, weight_decay=1e-5)
model.load_state_dict(model_state_dict)
opt_l.load_state_dict(checkpoint['opt_l'])

model.eval()
test_loss=AverageMeter()
test_f1=AverageMeter()
test_acc=AverageMeter()
test_preds=[]
test_labels=[]
test_probs=[]
test_process = tqdm(test_loader, desc='testing')
with torch.no_grad():
    for idx, batch in enumerate(test_process):
        img, label = batch
        img = img.to(device)
        label = label.to(device)
        # label = label.squeeze()
        label=label.view(-1)
        output = model(img)

        softmax_output = torch.softmax(output, dim=1)



        eval_dict = calc_eval(output, label)
        loss = criterion(output, label)
        test_loss.update(loss.item(), img.size(0))
        test_acc.update(eval_dict['acc'], img.size(0))
        test_f1.update(eval_dict['f1'], img.size(0))
        predicted = output.argmax(dim=1)  # 预测标签

        test_probs.extend(softmax_output[:, 1].cpu().numpy())

        test_preds.extend(output.argmax(dim=1).cpu().numpy())
        test_labels.extend(label.cpu().numpy())

        # 计算AUC，使用每个样本的正类概率（通常是索引1）
        # auc = roc_auc_score(label.cpu().numpy(), softmax_output[:, 1].cpu().numpy())
        # test_auc = auc

    CM = confusion_matrix(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_probs)
    print("AUC", auc)
    print(' test_loss: {:.4f}'.format( test_loss.avg))
    print("Test Acc: {:.4f}, F1: {:.4f}".format(test_acc.avg, test_f1.avg))
    print("测试集混淆矩阵:\n", CM)