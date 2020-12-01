import time, os
import numpy as np
import torch
import torch.nn as nn
from functions import *
from loss import *

def train(data_loader, opt, optimizer, model, criterion, epoch, log):
    log.write("start on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M") ))

    print ("epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])
    model.train()

    for iteration, batch in enumerate(data_loader, 1):
        input_pan = batch[0]
        input_lr = batch[2]
        input_lr_u = batch[3]
        target = batch[4]

        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            target = target.cuda()
            
        output = model(input_pan, input_lr_u, input_lr)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log.write("{} {:.10f}\n".format((epoch-1)*len(data_loader)+iteration, loss.item()))
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(data_loader),
                                                                loss.item()))
    log.write("end on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M") ))

def train_PGMAN(data_loader, opt, optim_G, optim_D1, optim_D2, G, D1, D2, \
    epoch, log, params={'alpha': 2e-4, 'beta': 1e-4, 'lambda': 100}):
    log.write("start on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M") ))

    # arguments in paper
    alpha = params['alpha']
    beta = params['beta']
    lamda = params['lambda']
    l2loss = nn.MSELoss()
    l1loss = nn.L1Loss()
    
    if opt.cuda:
        l2loss = l2loss.cuda()
        l1loss = l1loss.cuda()

    print ("epoch =", epoch, "lr =", optim_G.param_groups[0]["lr"])
    G.train()
    D1.train()
    D2.train()

    for iteration, batch in enumerate(data_loader, 1):
        input_pan = batch[0]
        input_lr = batch[2]
        input_lr_u = batch[3]
        input_pan_l = batch[6]

        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            input_pan_l = input_pan_l.cuda()

        I_fake = G(input_pan, input_lr_u, input_lr)
        I_fake = trim_image(I_fake)
        I_fake_d = downsample(I_fake)

        D1_real = D1(input_lr)
        D1_fake = D1(I_fake_d)

        D2_real = D2(input_pan)
        D2_fake = D2(AP(I_fake))

        loss_adv1 = - torch.mean(D1_fake)
        D_lambda_val = D_lambda(I_fake, input_lr)

        loss_adv2 = - torch.mean(D2_fake)
        D_s_val = D_s(I_fake, input_lr, input_pan, input_pan_l)

        QNR_val = (1 - D_lambda_val) * (1 - D_s_val)
        loss_G = 1 - QNR_val + alpha * loss_adv1 + beta * loss_adv2   

        D1_grad_pen = gradient_penalty(opt.cuda, D1, input_lr, I_fake_d) 
        D2_grad_pen = gradient_penalty(opt.cuda, D2, input_pan, AP(I_fake))

        loss_D1 = - torch.mean(D1_real) + torch.mean(D1_fake) + lamda * D1_grad_pen
        loss_D2 = - torch.mean(D2_real) + torch.mean(D2_fake) + lamda * D2_grad_pen

        optim_D1.zero_grad()
        loss_D1.backward(retain_graph=True)
        optim_D1.step()

        optim_D2.zero_grad()
        loss_D2.backward(retain_graph=True)
        optim_D2.step()
            
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if iteration % 20 == 0:
            log.write("{} loss_G:{:.6f} loss_D1:{:.6f} loss_D2:{:.6f}\n".format(
                (epoch-1)*len(data_loader)+iteration, 
                loss_G.item(), loss_D1.item(), loss_D2.item()) )
            print("===> Epoch[{}]({}/{}): loss_G:{:.6f} loss_D1:{:.6f} loss_D2:{:.6f}".format(
                epoch, iteration, len(data_loader),
                loss_G.item(), loss_D1.item(), loss_D2.item()) )
            print("===> Debug: D_lambda: {:.6f} D_s: {:.6f} QNR: {:.6f}".format( \
                D_lambda_val.item(), D_s_val.item(),  QNR_val.item()) )
            print("===> Debug: loss_adv1: {:.6f} loss_adv2: {:.6f}".format( \
                loss_adv1.item(), loss_adv2.item()) )
            print("===> Debug: D1_grad_pen: {:.6f} D2_grad_pen: {:.6f}".format( \
                D1_grad_pen.item(), D2_grad_pen.item()) )
    log.write("end on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M")) )
