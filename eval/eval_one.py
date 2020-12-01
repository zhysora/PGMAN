#!/usr/bin/python
#coding=utf8

############## import ##############
import numpy as np
import argparse
import gdal, cv2
from scipy.ndimage import sobel
from numpy.linalg import norm
import os
import skimage.metrics
import torch
import torch.nn.functional as F

import sys
import metrics as mtc
sys.path.append("../data")
from dataset import save_image

############## arguments ##############
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", help="dataset path")
parser.add_argument("--model", help="model name")
parser.add_argument("--model_outpath", help="model path")
parser.add_argument("--save_path", help="save path")
parser.add_argument("--num", type=int, help="number of input image pairs")
parser.add_argument("--col", type=int, help="col")
parser.add_argument("--row", type=int, help="row")
parser.add_argument("--ref", type=int, help="reference metrics")
parser.add_argument("--save", type=int, help="save the full image")
parser.add_argument("--bit", type=int, help="bit depth")

a = parser.parse_args()

for k, v in a._get_kwargs():
    print (k, "=", v)

if __name__ == "__main__":
    col_sz = a.col
    row_sz = a.row
    blk = 100 # test patch size

    YSize = (blk // 2 * (a.row - 1) + blk) * 4
    XSize = (blk // 2 * (a.col - 1) + blk) * 4

    out = np.zeros(shape=[YSize, XSize, 4], dtype=np.float32)
    cnt = np.zeros(shape=out.shape, dtype=np.float32)
    
    print(out.shape)

    i = 0
    y = 0

    if a.ref == 1:
        Q4 = []
        ERGAS = []
        RASE = []
        SCC = [] 
        SAM = []
        CC = []
        UIQC = []
        MPSNR = []
        SSIM = []
    elif a.ref == 0:
        D_lambda = []
        D_s = []
        QNR = []


    for _ in range(a.row):
        x = 0
        for __ in range(a.col):
            img = '%s/%d_lr.tif' % (a.dataset_path, i)
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            
            ly = y
            ry = (y + blk)
            lx = x
            rx = (x + blk)

            blur = img
            
            img = '%s/%d_pan.tif' % (a.dataset_path, i)
            img = gdal.Open(img).ReadAsArray()
            img = np.array(img, dtype=np.float32)
            
            ly = y * 4
            ry = (y + blk) * 4
            lx = x * 4
            rx = (x + blk) * 4

            pan = img
            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1

            if a.ref != 0:
                img = '%s/%d_mul.tif' % (a.dataset_path, i)
                img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
                img = np.array(img, dtype=np.float32)
                
                mul = img

            img = '%s/%d_mul_hat.tif' % (a.model_outpath, i)
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            
            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img

            if a.ref != 0:
                Q4.append(mtc.Q4(mul, img))
                ERGAS.append(mtc.ERGAS(mul, img))
                SCC.append(mtc.sCC(mul, img))
                SAM.append(mtc.SAM(mul, img) * 180 / np.pi)
                CC.append(mtc.CC(mul, img)) 
                UIQC.append(mtc.UIQC(mul, img)) 
                SSIM.append(skimage.metrics.structural_similarity(mul, img, data_range=2**a.bit-1, multichannel=True) )
                tmp = []
                for c in range(4):
                    tmp.append( skimage.metrics.peak_signal_noise_ratio(mul[:,:,c], img[:,:,c], data_range=2**a.bit-1) )
                MPSNR.append( np.mean(tmp) )

                #mul = mul.transpose(2, 0, 1)
                #mul = mul[np.newaxis, :]
                #mul = torch.Tensor(mul)
                #img = img.transpose(2, 0, 1)
                #img = img[np.newaxis, :]
                #img = torch.Tensor(img)
                
                #SAM.append(mtc.SAM_torch(mul, img).numpy())

            else:
                D1 = mtc.D_lamda(img, blur)
                D2 = mtc.D_s(img, blur, pan)
                #img = img.transpose(2, 0, 1)
                #img = img[np.newaxis, :]
                #img = torch.Tensor(img)
                #blur = blur.transpose(2, 0, 1)
                #blur = blur[np.newaxis, :]
                #blur = torch.Tensor(blur)

                #pan_l = cv2.pyrDown(pan)
                #pan_l = cv2.pyrDown(pan_l)
                #pan_l = pan_l[np.newaxis, np.newaxis, :]
                #pan_l = torch.Tensor(pan_l)

                #pan = pan[np.newaxis, np.newaxis, :]
                #pan = torch.Tensor(pan)
                #D1 = mtc.D_lambda_torch(img, blur).numpy()
                #D2 = mtc.D_s_torch(img, blur, pan, pan_l).numpy()

                D_lambda.append( D1 )
                D_s.append( D2 )
                QNR.append( (1 - D1) * (1 - D2) )
            
            i = i + 1
            x = x + blk // 2 
        y = y + blk // 2 

    out = out / cnt

    if a.save == 1:
        save_image("{}/{}.tif".format(a.save_path, a.model), out.transpose(2, 0, 1), 4)

    if a.ref == 1:
        print ("SAM, CC, sCC, ERGAS, Q4, UIQC, SSIM, MPSNR")
        print ("%.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf " % ( \
                np.mean(SAM), np.std(SAM), np.mean(CC), np.std(CC), np.mean(SCC), np.std(SCC), np.mean(ERGAS), np.std(ERGAS), \
                np.mean(Q4), np.std(Q4), np.mean(UIQC), np.std(UIQC), np.mean(SSIM), np.std(SSIM), np.mean(MPSNR), np.std(MPSNR) \
                ) )

    elif a.ref == 0:
        print ("D_lambda, D_s, QNR")
        print ("%.4lf+-%.4lf, %.4lf+-%.4lf, %.4lf+-%.4lf " % ( \
                np.mean(D_lambda), np.std(D_lambda), np.mean(D_s), np.std(D_s), np.mean(QNR), np.std(QNR) \
                ) )

    print("done")
