"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration
 
"""
import os

backbone_path = '/disk/sdc/tanxin/network/CVPR2021_PFNet-main/backbone/resnet/ckpt/resnet50-19c8e357.pth'

datasets_root = '/disk/sdc/tanxin/network/unet/data/tumor_cut1'

cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')
