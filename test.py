import time, os
import numpy as np
import torch
from data.dataset import save_image
from functions import *
from loss import *

def test(data_loader, opt, model, epoch, path, log):
    model.eval()
    if not os.path.exists(path):
        os.makedirs(path)
    log.write("start on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M") ))
    
    timecost = 0
    QNR = []

    for index, batch in enumerate(data_loader):
        input_pan = batch[0]
        input_lr = batch[2]
        input_lr_u = batch[3]
        filename = batch[5]
        input_pan_l = batch[6]
        
        if opt.cuda:
            input_pan = input_pan.cuda()
            input_lr = input_lr.cuda()
            input_lr_u = input_lr_u.cuda()
            input_pan_l = input_pan_l.cuda()
            
        start_time = time.time()
        output = model(input_pan, input_lr_u, input_lr)
        timecost += (time.time() - start_time)

        output = trim_image(output)

        D_lambda_val = D_lambda(output, input_lr)
        D_s_val = D_s(output, input_lr, input_pan, input_pan_l)

        QNR_val = (1 - D_lambda_val) * (1 - D_s_val)
        QNR.append(QNR_val.cpu().detach().numpy())

        n = filename.size()[0]
        for i in range(n):
            save_image('%s/%d_mul_hat.tif' % (path, filename[i]), output[i].cpu().detach().numpy(), 4)

    log.write("Time Cost: {}s\n".format( timecost ))
    log.write("QNR:{}\n".format( np.mean(QNR) ))
    log.write("end on:{}\n".format( time.strftime("%Y-%m-%d::%H:%M") ))
