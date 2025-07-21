import argparse
import gc
import logging
import os
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import gc

from torchsummary import summary

from dataset.data_aug import GaussianNoise
from dataset.data_aug import RandomChoice


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# 针对MCI
# import dense_att_gfnet_ad_new as gfnet
import net.net_3 as gfnet
# import gfnet_3d_3 as gfnet
# import resnet18_3d_dropout as resnet

from dataset.utils import AverageMeter,setup_seed,calc_eval
from dataset.data_label import set_labels,set_filenames,ADNIDataset,StratifiedSampler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(args):
    path = args.result_path + 'CNN_FFT_{}'.format(args.date)
    if not os.path.exists(path):
        os.mkdir(path)
        print("make the dir")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                        filename=path + '/%s.log' % (args.date), filemode='a')

    logging.info(args)
    setup_seed(args.seed)

    # ============================================ dataset ========================================#


    # 获取文件路径
    ad_files, cn_files = set_filenames()

    # 交叉验证参数
    num_folds = 5
    ad_fold_size = len(ad_files) // num_folds
    cn_fold_size = len(cn_files) // num_folds

    # 进行 5 折交叉验证
    for fold in range(num_folds):
        print(f"\n===== Fold {fold} =====")

    # 计算 ad 和 cn 的验证集索引范围
        ad_val_start = fold * ad_fold_size
        ad_val_end = (fold + 1) * ad_fold_size if fold != num_folds - 1 else len(ad_files)

        cn_val_start = fold * cn_fold_size
        cn_val_end = (fold + 1) * cn_fold_size if fold != num_folds - 1 else len(cn_files)

        # 获取训练集和验证集索引
        ad_val_index = list(range(ad_val_start, ad_val_end))
        cn_val_index = list(range(cn_val_start, cn_val_end))

        ad_train_index = list(set(range(len(ad_files))) - set(ad_val_index))
        cn_train_index = list(set(range(len(cn_files))) - set(cn_val_index))

        # 获取训练集和验证集的文件名
        x_train = [ad_files[i] for i in ad_train_index] + [cn_files[i] for i in cn_train_index]
        y_train = [1] * len(ad_train_index) + [0] * len(cn_train_index)  # AD = 1, CN = 0

        x_val = [ad_files[i] for i in ad_val_index] + [cn_files[i] for i in cn_val_index]
        y_val = [1] * len(ad_val_index) + [0] * len(cn_val_index)
        print(x_val)




        augment_compose = RandomChoice(transforms=[
            GaussianNoise(mean=0, std=0.06)],
            p=0.4)
        # 创建训练集、验证集和测试集的数据集对象
        train_dataset = ADNIDataset(x_train, y_train,transform=augment_compose)
        val_dataset = ADNIDataset(x_val, y_val)
        # print("fold",selected_fold)
        print("train_dataset:",len(train_dataset))
        print("val_dataset:",len(val_dataset))
        class_vector = train_dataset.labels
        sampler = StratifiedSampler(class_vector=class_vector, batch_size=2)



        # 设置数据加载器
        # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=6, shuffle=False,sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=4, shuffle=True)

        # ========================================== model & optimizer ========================================#
        model=gfnet.DenseNetWithGFNet(depth=args.gf_depth,num_classes=args.output)
        # model = resnet.resnet18_3d(num_classes=2, in_channels=1)
        print("model:", model)
        if torch.cuda.device_count() > 0:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model).to(device)

        if args.optim == 'sgd':
            opt_l = torch.optim.SGD(model.parameters(), lr=args.l_lr, momentum=0.9,weight_decay=1e-5)
        else:
            opt_l = torch.optim.AdamW(model.parameters(), lr=args.l_lr, weight_decay=1e-5)
        warmup_epochs = 10
        initial_lr = 0.0001  # 预热的初始学习率
        target_lr = args.l_lr  # 最终学习率（例如0.0005）


        # step_scheduler = torch.optim.lr_scheduler.StepLR(opt_l, step_size=100, gamma=0.5)
        # step_scheduler = CosineAnnealingLR(opt_l, T_max=40*39, eta_min=0.0001)
        step_scheduler =torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_l, T_0=10*39, T_mult=2,eta_min=0.0001)


        criterion = nn.CrossEntropyLoss()

        max_val_acc = 0
        best_val_acc=0
        best_epoch = 0
        best_val_acc_epoch=0
        # ========================================= load latent vectors =======================================
        # 用于记录每个epoch的loss和acc
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for epoch in range(args.epochs):

            train_loss = AverageMeter()
            train_acc = AverageMeter()
            train_f1 = AverageMeter()
            training_process = tqdm(train_loader, desc='training')
            model.train()
            # 初始化存储每个批次预测和标签的列表
            train_preds = []
            train_labels = []
            if epoch < warmup_epochs:
                lr = (target_lr - initial_lr) * (epoch + 1) / warmup_epochs + initial_lr
                for param_group in opt_l.param_groups:
                    param_group['lr'] = lr
                print(f"Epoch {epoch}/{args.epochs}, Warmup Learning Rate: {lr}")

            elif epoch == warmup_epochs:
                for param_group in opt_l.param_groups:
                    param_group['lr'] = target_lr
                print(f"Warmup finished. Reset learning rate to {target_lr} and start CosineAnnealingLR.")

            for idx, batch in enumerate(training_process):
                img, label = batch
                label = label.to(device)
                img = img.to(device)
                model.zero_grad()
                y_pred = model(img)
                # print(y_pred.dtype)
                # print(label.dtype)

                # label = label.squeeze()
                label=label.view(-1)
                loss = criterion(y_pred, label)
                dict = calc_eval(y_pred, label)
                train_acc.update(dict['acc'], img.size(0))
                train_f1.update(dict['f1'], img.size(0))
                # loss.backward()
                train_loss.update(loss.item(), img.size(0))
                # 更新预测值和标签，用于计算混淆矩阵
                train_preds.extend(y_pred.argmax(dim=1).cpu().numpy())
                train_labels.extend(label.cpu().numpy())
                # 反向传播
                loss.backward()
                opt_l.step()
                if epoch >= warmup_epochs:
                    # print(f"Warmup phase finished. Switching to CosineAnnealingLR scheduler.")
                    step_scheduler.step()  # 余弦退火更新

            print(f"Epoch [{epoch}/{args.epochs}], Learning Rate now: {opt_l.param_groups[0]['lr']}")




            # lr_scheduler.step()
            # 记录训练集曲线
            train_losses.append(train_loss.avg)
            train_accuracies.append(train_acc.avg)

            CM = confusion_matrix(train_labels, train_preds)  # 计算混淆矩阵
            logging.info(
                'Fold {},Epoch {}, train_loss: {}'.format(fold,epoch, train_loss.avg))
            logging.info("train: acc: {} f1: {} ".format(train_acc.avg, train_f1.avg))
            print('Fold {},Epoch {}, train_loss: {:.4f}'.format(fold,epoch, train_loss.avg))
            print("train_acc: {:.4f} train_f1: {:.4f} ".format(train_acc.avg, train_f1.avg))
            print("训练集混淆矩阵：\n", CM)




            val_acc = AverageMeter()
            val_f1 = AverageMeter()
            val_loss = AverageMeter()
            val_preds = []
            val_labels = []

            model.eval()
            val_process = tqdm(val_loader, desc='valing')
            with torch.no_grad():
                for idx, batch in enumerate(val_process):
                    img, label = batch
                    img = img.to(device)
                    label = label.to(device)
                    # label = label.squeeze()
                    label=label.view(-1)
                    output = model(img)
                    eval_dict = calc_eval(output, label)
                    loss = criterion(output, label)
                    val_loss.update(loss.item(), img.size(0))
                    val_acc.update(eval_dict['acc'], img.size(0))
                    val_f1.update(eval_dict['f1'], img.size(0))


                    val_preds.extend(output.argmax(dim=1).cpu().numpy())
                    val_labels.extend(label.cpu().numpy())


                # 记录测试集的损失和准确率
                val_losses.append(val_loss.avg)
                val_accuracies.append(val_acc.avg)
                CM = confusion_matrix(val_labels, val_preds)  # 计算混淆矩阵
                logging.info(
                    'Fold {},Epoch {}, val_loss: {}'.format(fold,epoch, val_loss.avg))
                logging.info("val: acc: {} f1: {}".format(val_acc.avg, val_f1.avg))
                print('Fold {},Epoch {}, val_loss: {:.4f}'.format(fold,epoch, val_loss.avg))
                print("val_acc: {:.4f} f1: {:.4f}".format(val_acc.avg, val_f1.avg))
                print("验证集混淆矩阵:\n", CM)


            # 记录第一次取得最好acc的epoch
            if max_val_acc <= val_acc.avg and train_acc.avg >= 0.98:
                max_val_acc = val_acc.avg
                best_epoch = epoch
                state = {'latent_encoder': model.state_dict(),
                         'opt_l': opt_l.state_dict()}
                torch.save(state, path + f'/fold{fold}_best_val_stable model.pth')
            if best_val_acc < val_acc.avg:
                best_val_acc=val_acc.avg
                best_val_acc_epoch=epoch
                state = {'latent_encoder': model.state_dict(),
                         'opt_l': opt_l.state_dict()}
                torch.save(state, path + f'/fold{fold}_best model.pth')

        logging.info(
            'fold:{},best val acc epoch: {}, best val acc: {:.4f}'.format(fold, best_val_acc_epoch, best_val_acc)
        )
        logging.info(
            'fold:{},best epoch: {}, max val acc: {:.4f}'.format(fold,best_epoch, max_val_acc)
        )

        state = {'latent_encoder': model.state_dict(),
                 'opt_l': opt_l.state_dict()}
        torch.save(state, path  + f'/fold{fold}_last_val_model.pth')

        gc.collect()
        torch.cuda.empty_cache()


        # 绘制损失和准确率曲线
        plt.figure(figsize=(12, 6))

        # 绘制训练集和测试集损失
        plt.plot(range(args.epochs), train_losses, label='train_loss')
        plt.plot(range(args.epochs), val_losses, label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Train and Val Loss')
        y_tick = np.arange(0, 7, 0.3)
        plt.yticks(y_tick)
        loss_filename = os.path.join("E:/CNN_FFT/AD_CN/", f"Loss_result_fold{fold}.png")
        plt.savefig(loss_filename)
        plt.close()

        # 绘制训练集和测试集准确率曲线
        # plt.subplot(2, 1, 2)
        plt.figure(figsize=(12, 6))
        plt.plot(range(args.epochs), train_accuracies, label='train_acc')
        plt.plot(range(args.epochs), val_accuracies, label='val_acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train and Val Accuracy')
        y_tick = np.arange(0.5, 1.05, 0.05)
        plt.yticks(y_tick)
        loss_filename = os.path.join("E:/CNN_FFT/AD_CN/", f"ACC_result_fold{fold}.png")
        plt.savefig(loss_filename)
        plt.close()



def arg_parse():
    parser = argparse.ArgumentParser(description='ADNI three_classification')
    parser.add_argument('--seed', type=int, default=24,
                        help='random seed')
    parser.add_argument('--l_lr', type=float, default=0.00045,
                        help='latent encoder Learning rate.')
    parser.add_argument('--train_batch_size', type=int, default=18,
                        help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='val batch size')
    parser.add_argument('--test_batch_size', type=int, default=2,
                        help='test batch size')
    parser.add_argument('--reg', type=str, default='True',
                        help='regularization')
    parser.add_argument('--epochs', type=int, default=130,
                        help='Train Epochs')
    parser.add_argument('--input', type=int, default=1,
                        help='input channel of patch_emd')
    parser.add_argument('--gf_depth', type=int, default=6,
                        help='depth of gfnet')
    # sgd容易陷入local minimum
    parser.add_argument('--optim', type=str, default='AdamW',
                        help='type of optimizer')
    parser.add_argument('--result_path', type=str, default='E:/CNN_FFT/AD_CN/',
                        help='path to save')
    parser.add_argument('--date', type=str, default='0520',
                        help='date and num ')
    parser.add_argument('--output', type=int, default=2,
                        help='output channels of gfnet')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    run(args)

