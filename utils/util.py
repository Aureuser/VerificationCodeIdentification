# -*- coding: utf-8 -*- 
# @Time : 2020/5/14 15:59 
# @Author : lxd 
# @File : util.py
import os, random

def image_train_test_split(root, p):
    imgs = list(os.listdir(root))
    index = int(len(imgs) * p)
    random.shuffle(imgs)
    return imgs[:index], imgs[index:]
