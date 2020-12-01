############## import ##############
from __future__ import division
import gdal, ogr, os, osr
import numpy as np
import cv2

import sys
from dataset import save_image

############## arguments ##############
dataDir = '/data/zhouhuanyu/PSData3' # root of data directory
satellite = 'WV-3'           # name of dataset

dic = {'QB' : 5, 'GF-2' : 7, 'WV-3' : 3}
tot = dic[satellite]         # number of raw images

def downsample(img): 
    return cv2.pyrDown( cv2.pyrDown(img) )
    # h, w = img.shape[:2]
    # return cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)

def upsample(img): 
    return cv2.pyrUp( cv2.pyrUp(img) )
    # h, w = img.shape[:2]
    # return cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

if __name__ == "__main__":
    outDir = '%s/Dataset/%s' % (dataDir, satellite)
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # MUL -> mul(1), lr(1/4), lr_u(1/4*4)
    # PAN -> pan(1/4), pan_d(1/4*1/4*4)
    # origin 
    # MUL(crop 1/4) -> mul_o(1), mul_o_u(1*4)
    # PAN(crop 1/4) -> pan_o(1), pan_o_d(1/4*4)
    for i in range(0, tot): 
        newMul = '%s/Dataset/%s/%d_mul.tif' % (dataDir, satellite , i)
        newLR = '%s/Dataset/%s/%d_lr.tif' % (dataDir, satellite, i)
        newLR_u = '%s/Dataset/%s/%d_lr_u.tif' % (dataDir, satellite, i)
        newPan = '%s/Dataset/%s/%d_pan.tif' % (dataDir, satellite, i)
        newPan_d = '%s/Dataset/%s/%d_pan_d.tif' % (dataDir, satellite, i)

        newMul_o = '%s/Dataset/%s/%d_mul_o.tif' % (dataDir, satellite, i)
        newMul_o_u = '%s/Dataset/%s/%d_mul_o_u.tif' % (dataDir, satellite, i)
        newPan_o = '%s/Dataset/%s/%d_pan_o.tif' % (dataDir, satellite, i)
        newPan_o_d = '%s/Dataset/%s/%d_pan_o_d.tif' % (dataDir, satellite, i)

        rawPan = gdal.Open( '%s/Raw/%s/%d-PAN.tif' % (dataDir, satellite, i) ).ReadAsArray()
        rawMul = gdal.Open( '%s/Raw/%s/%d-MUL.tif' % (dataDir, satellite, i) ).ReadAsArray()
        print ("rawMul:", rawMul.shape, " rawPan:", rawPan.shape)

        rawMul = rawMul.transpose(1, 2, 0) # (h, w, c)
        
        h, w = rawMul.shape[:2]
        h -= 10
        w -= 10
        h = h // 4 * 4
        w = w // 4 * 4
        rawMul = rawMul[:h, :w, :]
        rawPan = rawPan[:h*4, :w*4]
        
        imgMul = rawMul                         # h * w * 4                     
        imgLR = downsample(imgMul)              # (h / 4) * (w / 4) * 4   
        imgLR_u = upsample(imgLR)               # h * w * 4
        imgPan = rawPan
        imgPan = downsample(imgPan)             # h * w
        imgPan_d = upsample(downsample(imgPan)) # h * w

        if i == 0:  # test scene, crop middle
            imgMul_o = rawMul[h//4*2:h//4*3, w//4*2:w//4*3, :]  # (h / 4) * (w / 4) * 4
            imgMul_o_u = upsample(imgMul_o)                     # h * w * 4
            imgPan_o = rawPan[h*2:h*3, w*2:w*3]                 # h * w 
            imgPan_o_d = upsample(downsample(imgPan_o))         # h * w
        else:
            imgMul_o = rawMul                                   # (h / 4) * (w / 4) * 4
            imgMul_o_u = upsample(imgMul_o)                     # h * w * 4
            imgPan_o = rawPan                                   # h * w 
            imgPan_o_d = upsample(downsample(imgPan_o))         # h * w

        imgMul = imgMul.transpose(2, 0, 1)
        imgLR = imgLR.transpose(2, 0, 1)
        imgLR_u = imgLR_u.transpose(2, 0, 1)
        imgMul_o = imgMul_o.transpose(2, 0, 1)
        imgMul_o_u = imgMul_o_u.transpose(2, 0, 1)

        if i == 0:
            a = h // 4
            b = w // 4
            a = a // 50 * 50
            b = b // 50 * 50
            save_image(newMul, imgMul[:,:a*4,:b*4], 4)  
            save_image(newLR, imgLR[:,:a,:b], 4)       
            save_image(newLR_u, imgLR_u[:,:a*4,:b*4], 4)     
            save_image(newPan, imgPan[:a*4,:b*4], 1)       
            save_image(newPan_d, imgPan_d[:a*4,:b*4], 1)   

            # a = h # keep raw size
            # b = w
            # a = a // 50 * 50
            # b = b // 50 * 50
            save_image(newMul_o, imgMul_o[:,:a,:b], 4)        
            save_image(newMul_o_u, imgMul_o_u[:,:a*4,:b*4], 4)     
            save_image(newPan_o, imgPan_o[:a*4,:b*4], 1)        
            save_image(newPan_o_d, imgPan_o_d[:a*4,:b*4], 1)  
        else:
            save_image(newMul, imgMul, 4)  
            save_image(newLR, imgLR, 4)       
            save_image(newLR_u, imgLR_u, 4)     
            save_image(newPan, imgPan, 1)       
            save_image(newPan_d, imgPan_d, 1) 

            save_image(newMul_o, imgMul_o, 4)        
            save_image(newMul_o_u, imgMul_o_u, 4)     
            save_image(newPan_o, imgPan_o, 1)        
            save_image(newPan_o_d, imgPan_o_d, 1)  

        
        print ('done%s' % i) 

    print ('finish') 
