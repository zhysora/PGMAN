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

import models, train, test, save
from functions import *

############## default handler for single model
class Handler():
    def __init__(self, opt, train_data_loader, test_data_loader, test_origin_data_loader, \
        model=None, criterion=nn.L1Loss()):

        self.opt = opt
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.test_origin_data_loader = test_origin_data_loader
        self.model = model
        self.criterion = criterion
        self.optim_class = 'Adam'

        # create log dir & out dir
        self.root_dir = '{}/model_out/{}'.format(self.opt.outpath, self.opt.model)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists('{}/log'.format(self.root_dir)):
            os.makedirs('{}/log'.format(self.root_dir))

        self.train_log = open('{}/log/train_{}.log'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1]), "w" )
        self.test_log = open('{}/log/test_{}_low_res.log'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1]), "w" )
        self.test_origin_log = open('{}/log/test_{}_full_res.log'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1]), "w" )

        if not os.path.exists("{}/out".format(self.root_dir)):
            os.makedirs("{}/out".format(self.root_dir))
        self.train_out = '{}/out/train_{}'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1])
        self.test_out = '{}/out/test_{}_low_res'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1])
        self.test_origin_out = '{}/out/test_{}_full_res'.format(self.root_dir, \
            self.opt.dataset.split('/')[-1])
    
    def init(self):
        self.model.apply(weight_init)

    def cuda(self):
        if self.opt.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.checkpoint)
        self.opt.start_epoch = checkpoint["epoch"] + 1
        self.model.load_state_dict(checkpoint["model"].state_dict())
    
    def load_pretrained(self):
        weights = torch.load(self.opt.pretrained)
        self.model.load_state_dict(weights["model"].state_dict())
    
    def set_optim(self):
        if self.optim_class == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.learning_rate)
        elif self.optim_class == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.opt.learning_rate)

    def set_sched(self):    
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.decay_step, gamma=0.9)

    def train(self):
        for epoch in range(self.opt.start_epoch, self.opt.epochs + 1):
            train.train(self.train_data_loader, self.opt, self.optimizer, self.model, \
                self.criterion, epoch, self.train_log)    
            if self.opt.save_freq != -1 and epoch % self.opt.save_freq == 0 and epoch != self.opt.epochs:
                self.save(epoch)
            if self.opt.test_freq != -1 and epoch % self.opt.test_freq == 0 and epoch != self.opt.epochs:
                self.test("_{}".format(epoch))
            self.scheduler.step()

    def save(self, epoch):
        save.save(self.model, epoch, self.train_out)

    def test(self, epoch=""):
        test.test(self.test_data_loader, self.opt, self.model, self.opt.epochs, \
            self.test_out + epoch, self.test_log)
        test.test(self.test_origin_data_loader, self.opt, self.model, self.opt.epochs, \
            self.test_origin_out + epoch, self.test_origin_log)
    
    def close_log(self):
        self.train_log.close()
        self.test_log.close()
        self.test_origin_log.close()

    def total_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def total_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

############## default handler for one generator with two discriminators
class G1D2_Handler(Handler):
    def __init__(self, opt, train_data_loader, test_data_loader, test_origin_data_loader, \
        model=None, criterion=nn.L1Loss()):
        super(G1D2_Handler, self).__init__(opt, train_data_loader, test_data_loader, \
            test_origin_data_loader, model, criterion)
    
    def init(self):
        self.G.apply(weight_init)
        self.D1.apply(weight_init)
        self.D2.apply(weight_init)

    def cuda(self):
        if self.opt.cuda:
            self.G = self.G.cuda()
            self.D1 = self.D1.cuda()
            self.D2 = self.D2.cuda()
            self.criterion = self.criterion.cuda()
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.checkpoint)
        self.opt.start_epoch = checkpoint["epoch"] + 1
        self.G.load_state_dict(checkpoint["G"].state_dict())
        self.D1.load_state_dict(checkpoint["D1"].state_dict())
        self.D2.load_state_dict(checkpoint["D2"].state_dict())

    def load_pretrained(self):
        weights = torch.load(self.opt.pretrained)
        self.G.load_state_dict(weights["G"].state_dict())
        self.D1.load_state_dict(weights["D1"].state_dict())
        self.D2.load_state_dict(weights["D2"].state_dict())

    def set_optim(self):
        if self.optim_class == 'Adam':
            self.optim_G = optim.Adam(self.G.parameters(), lr=self.opt.learning_rate)
            self.optim_D1 = optim.Adam(self.D1.parameters(), lr=self.opt.learning_rate)
            self.optim_D2 = optim.Adam(self.D2.parameters(), lr=self.opt.learning_rate)
        elif self.optim_class == 'RMSprop':
            self.optim_G = optim.RMSprop(self.G.parameters(), lr=self.opt.learning_rate)
            self.optim_D1 = optim.RMSprop(self.D1.parameters(), lr=self.opt.learning_rate)
            self.optim_D2 = optim.RMSprop(self.D2.parameters(), lr=self.opt.learning_rate)

    def set_sched(self):
        self.sched_G = optim.lr_scheduler.StepLR(self.optim_G, step_size=self.opt.decay_step, gamma=0.9)
        self.sched_D1 = optim.lr_scheduler.StepLR(self.optim_D1, step_size=self.opt.decay_step, gamma=0.9)
        self.sched_D2 = optim.lr_scheduler.StepLR(self.optim_D2, step_size=self.opt.decay_step, gamma=0.9)
        
    def save(self, epoch):
        save.save_G1D2(self.G, self.D1, self.D2, epoch, self.train_out)

    def test(self, epoch=""):
        test.test(self.test_data_loader, self.opt, self.G, self.opt.epochs, \
            self.test_out + epoch, self.test_log)
        test.test(self.test_origin_data_loader, self.opt, self.G, self.opt.epochs, \
            self.test_origin_out + epoch, self.test_origin_log)    

    def total_params(self):
        return sum(p.numel() for p in self.G.parameters()) + \
            sum(p.numel() for p in self.D1.parameters()) + \
            sum(p.numel() for p in self.D2.parameters())
    
    def total_trainable_params(self):
        return sum(p.numel() for p in self.G.parameters() if p.requires_grad) + \
            sum(p.numel() for p in self.D1.parameters() if p.requires_grad) + \
            sum(p.numel() for p in self.D2.parameters() if p.requires_grad)

############## PGMAN
class PGMAN_Handler(G1D2_Handler):
    def __init__(self, opt, train_data_loader, test_data_loader, test_origin_data_loader):
        super(PGMAN_Handler, self).__init__(opt, train_data_loader, test_data_loader, \
            test_origin_data_loader)

        self.G = models.PGMAN_Generator(withBN=True, high_pass=True)
        self.D1 = models.Patch_Discriminator(in_channels=4, n_layers=2, withBN=False)
        self.D2 = models.Patch_Discriminator(in_channels=1, n_layers=3, withBN=False)
    
    def set_optim(self):
        self.optim_G = optim.Adam(self.G.parameters(), self.opt.learning_rate, [0, 0.9])
        self.optim_D1 = optim.Adam(self.D1.parameters(), self.opt.learning_rate, [0, 0.9])
        self.optim_D2 = optim.Adam(self.D2.parameters(), self.opt.learning_rate, [0, 0.9])

    def train(self):
        for epoch in range(self.opt.start_epoch, self.opt.epochs + 1):
            train.train_PGMAN(self.train_data_loader, self.opt, \
                self.optim_G, self.optim_D1, self.optim_D2, self.G, self.D1, self.D2, \
                epoch, self.train_log) 
            if self.opt.save_freq != -1 and epoch % self.opt.save_freq == 0 and epoch != self.opt.epochs:
                self.save(epoch)
            if self.opt.test_freq != -1 and epoch % self.opt.test_freq == 0 and epoch != self.opt.epochs:
                self.test("_{}".format(epoch))
            self.sched_G.step()
            self.sched_D1.step()
            self.sched_D2.step()
