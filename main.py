# coding=utf-8
    
############## import 
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import random
import os
import time
import sys

import model_handlers, models, train, test, save
from data.dataset import DatasetFromFolder

############## arguments 
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--seed', type=int, default=1118, help='random seed to use.')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument("--gpus", type=str, default="", help="gpus to run")
parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
parser.add_argument('--model', type=str, default="", help='model to use')
parser.add_argument('--dataset', type=str, default='/data/zh/PSData/Dataset/GF-1')
parser.add_argument('--outpath', type=str, default='log and output dir for model')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument("--checkpoint", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--decay_step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--test_freq", type=int, default=10, help="frequency of epochs for testing model")
parser.add_argument("--save_freq", type=int, default=10, help="frequency of epochs for saveing model")
opt = parser.parse_args()

############## main
def main():
    def pure(model):
        # '-2' means training on full-resolution
        if '-2' in model:
            return model[:-2]
        return model

    for k,v in opt._get_kwargs():
        print (k, "=", v)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    if pure(opt.model) not in ['PGMAN']:
        print('[ERROR] invalid model name')
        return 

############## Loading datasets
    print('===> Loading datasets')
    if '-2' in opt.model:
        # train on full-resolution
        train_set = DatasetFromFolder( [os.path.join(opt.dataset, "train_full_res")] )
    else:
        # train on low-resolution
        train_set = DatasetFromFolder( [os.path.join(opt.dataset, "train_low_res")] )
        
    # test on low and full resolutions
    test_set        = DatasetFromFolder( [os.path.join(opt.dataset, "test_low_res")] )
    test_origin_set = DatasetFromFolder( [os.path.join(opt.dataset, "test_full_res")] )

    train_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, \
        batch_size=4, shuffle=False)
    test_origin_data_loader = DataLoader(dataset=test_origin_set, num_workers=opt.threads, \
        batch_size=4, shuffle=False)

############## Building model
    print("===> Building model")
    if opt.model in ['PGMAN']:
        handler = model_handlers.PGMAN_Handler(opt, train_data_loader, test_data_loader, test_origin_data_loader)
    
############## Setting GPU
    print("===> Setting GPU")
    if cuda:
        handler.cuda()

############## model weight initialization
    print("===> weight initializing")
    handler.init()

############## optionally resume from a checkpoint
    if opt.checkpoint:
        handler.load_checkpoint()
    else:
        print("=> no checkpoint found at '{}'".format(opt.checkpoint))

############## optionally copy weights from a checkpoint
    if opt.pretrained:
        handler.load_pretrained()
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

############## Setting Optimizer     
    print("===> Setting Optimizer")
    handler.set_optim()

############## Setting Scheduler     
    print("===> Setting Scheduler for learning_rate decay")
    handler.set_sched()

############## print params count
    print("===> total params:", handler.total_params())
    print("===> total trainable params:", handler.total_trainable_params())

############## Training
    print("===> Training")
    handler.train()      

############## Saving
    print("===> Saving")
    handler.save(opt.epochs)

############## Testing
    print("===> Testing")    
    handler.test("_{}".format(opt.epochs))
    
    handler.close_log()
    print("===> Finish")

if __name__ == "__main__":
    main()