"""
 @Time    : 2021/7/6 14:52
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : train.py
 @Function: Training
 
"""
import datetime
import time
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import cod_training_root
from config import backbone_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir
from PFNet import PFNet
from PFNet import PFNet
import loss

cudnn.benchmark = True

torch.manual_seed(2021)
device_ids = [0,1]

ckpt_path = '/disk/sdc/tanxin/network/CVPR2021_PFNet-main/checkpoint/ckpt2'
exp_name = 'PFNet_stan'

args = {
    'epoch_num': 2000,
    'train_batch_size': 4,
    'last_epoch': 301,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 320,
    'save_point': [],
    'poly_train': True,
    'optimizer': 'SGD',
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms.Normalize([0.485], [0.229])
])
target_transform = transforms.ToTensor()


##Loss
def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N

class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, activation='none'):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

    def forward(self, y_pr, y_gt):
        return 1 - diceCoeff(y_pr, y_gt, activation=self.activation)

# Prepare Data Set.
train_set = ImageFolder(cod_training_root, joint_transform, img_transform, target_transform)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = loss.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = loss.IOU().cuda(device_ids[0])

def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss

def main():
    print(args)
    print(exp_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #net = PFNet(backbone_path).cuda(device_ids[0]).train()
    net = PFNet(1,2)
    net.load_state_dict(torch.load('/disk/sdc/tanxin/network/CVPR2021_PFNet-main/checkpoint/ckpt2/PFNet_stan/300.pth'))


    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device=device)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()

def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()
    criterion = SoftDiceLoss()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_0_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        net.train()
        train_iterator = tqdm(train_loader, total=len(train_loader))
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_4, predict_3, predict_2, predict_1, predict_0= net(inputs)

            loss_4 = bce_iou_loss(predict_4, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_1 = structure_loss(predict_1, labels)
            loss_0 = structure_loss(predict_0, labels)

            loss = 1 * loss_4 + 1 * loss_3 + 1 * loss_2 + 2 * loss_1 + 4 * loss_0

            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_0_record.update(loss_0.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_0', loss_0, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                  (epoch, curr_iter, base_lr, loss_record.avg, loss_0_record.avg, loss_1_record.avg,
                   loss_2_record.avg, loss_3_record.avg, loss_4_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch > 0 and epoch % 5 == 0:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return

        # if epoch > 50 and epoch % 5 == 0:
        #     net.cpu()
        #     torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
        #     print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        #     print(exp_name)
        #     return

        # if epoch == args['epoch_num']:
        #     net.cpu()
        #     torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
        #     print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
        #     print(exp_name)
        #     print("Optimization Have Done!")
        #     return


if __name__ == '__main__':
    main()