############## import ##############
from __future__ import division
import cv2
import gdal, ogr, os, osr
import numpy as np
import random

import sys
from dataset import save_image

############## arguments ##############
dataDir = '/data/zhouhuanyu/PSData3' # root of data directory
satellite = 'WV-3'           # name of dataset
blk = 64                     # train patch size

dic = {'QB' : 5, 'GF-2' : 6, 'WV-3' : 3}
tot = dic[satellite]         # number of raw images
dic = {64: {'QB' : 1000, 'GF-2' : 5000, 'WV-3' : [0, 5000, 15000]} }
patch_num = dic[blk][satellite]   # number of patches from each TIF

train_mode = True
train_origin_mode = True
test_mode = True

if __name__ == "__main__":
    # train patch_size = 64
    if train_mode:
        ### train low-res
        trainCount = 0
        trainDir = "%s/Dataset/%s/train_low_res" % (dataDir, satellite)
        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        record = open('%s/record.txt' % trainDir, "w")
        for num in range(1, tot):
            mul = '%s/Dataset/%s/%d_mul.tif' % (dataDir, satellite, num)
            lr = '%s/Dataset/%s/%d_lr.tif' % (dataDir, satellite, num)
            lr_u = '%s/Dataset/%s/%d_lr_u.tif' % (dataDir, satellite, num)
            pan = '%s/Dataset/%s/%d_pan.tif' % (dataDir, satellite, num)
            pan_d = '%s/Dataset/%s/%d_pan_d.tif' % (dataDir, satellite, num)

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)
            dt_pan_d = gdal.Open(pan_d)
            
            img_mul = dt_mul.ReadAsArray() # (c, h, w)
            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_pan_d = dt_pan_d.ReadAsArray()    

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            for _ in range(patch_num[num]):
                x = random.randint(0, XSize - blk)
                y = random.randint(0, YSize - blk)

                save_image('%s/%d_mul.tif' % (trainDir, trainCount), 
                    img_mul[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                save_image('%s/%d_lr_u.tif' % (trainDir, trainCount), 
                    img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                save_image('%s/%d_lr.tif' % (trainDir, trainCount), 
                    img_lr[:, y:(y + blk), x:(x + blk)], 4)
                save_image('%s/%d_pan.tif' % (trainDir, trainCount), 
                    img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                save_image('%s/%d_pan_d.tif' % (trainDir, trainCount), 
                    img_pan_d[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)    
                trainCount += 1

            print ("done %d" % num)

        record.write("%d\n" % trainCount)
        record.close()  
    
    if train_origin_mode:
        ### train full-res
        originTrainCount = 0
        originTrainDir = "%s/Dataset/%s/train_full_res" % (dataDir, satellite)
        if not os.path.exists(originTrainDir):
            os.makedirs(originTrainDir)
        record = open('%s/record.txt' % originTrainDir, "w")
        for num in range(1, tot):
            lr = '%s/Dataset/%s/%d_mul_o.tif' % (dataDir, satellite, num)
            lr_u = '%s/Dataset/%s/%d_mul_o_u.tif' % (dataDir, satellite, num)
            pan = '%s/Dataset/%s/%d_pan_o.tif' % (dataDir, satellite, num)
            pan_d = '%s/Dataset/%s/%d_pan_o_d.tif' % (dataDir, satellite, num)

            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)
            dt_pan_d = gdal.Open(pan_d)
            
            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_pan_d = dt_pan_d.ReadAsArray()    

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            for _ in range(patch_num[num]):
                x = random.randint(0, XSize - blk)
                y = random.randint(0, YSize - blk)

                save_image('%s/%d_lr_u.tif' % (originTrainDir, originTrainCount), 
                    img_lr_u[:, y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 4)
                save_image('%s/%d_lr.tif' % (originTrainDir, originTrainCount), 
                    img_lr[:, y:(y + blk), x:(x + blk)], 4)
                save_image('%s/%d_pan.tif' % (originTrainDir, originTrainCount), 
                    img_pan[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)
                save_image('%s/%d_pan_d.tif' % (originTrainDir, originTrainCount), 
                    img_pan_d[y * 4:(y + blk) * 4, x * 4:(x + blk) * 4], 1)    
                originTrainCount += 1

            print ("done %d" % num)

        record.write("%d\n" % originTrainCount)
        record.close()  
    
    # test patch_size = 100
    if test_mode:
        ### test low-res
        testCount = 0
        testDir = "%s/Dataset/%s/test_low_res" % (dataDir, satellite)
        if not os.path.exists(testDir):
            os.makedirs(testDir)
        record = open('%s/record.txt' % testDir, "w")
        for num in range(1):
            mul = '%s/Dataset/%s/%s_mul.tif' % (dataDir, satellite, num)
            lr = '%s/Dataset/%s/%s_lr.tif' % (dataDir, satellite, num)
            lr_u = '%s/Dataset/%s/%s_lr_u.tif' % (dataDir, satellite, num)
            pan = '%s/Dataset/%s/%s_pan.tif' % (dataDir, satellite, num)
            pan_d = '%s/Dataset/%s/%s_pan_d.tif' % (dataDir, satellite, num)

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_pan_d = gdal.Open(pan_d)
            dt_lr_u = gdal.Open(lr_u)
            
            img_mul = dt_mul.ReadAsArray()
            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_pan_d = dt_pan_d.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            
            row = 0
            col = 0
            
            for y in range(0, YSize, 50): # 按顺序切(100, 100)小块 overlap 1/2
                if y + 100 > YSize:
                    continue
                col = 0
                
                for x in range(0, XSize, 50):
                    if x + 100 > XSize:
                        continue
                    save_image('%s/%d_mul.tif' % (testDir, testCount),
                        img_mul[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 4)
                    save_image('%s/%d_lr_u.tif' % (testDir, testCount), 
                        img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 4)
                    save_image('%s/%d_lr.tif' % (testDir, testCount), 
                        img_lr[:, y:(y + 100), x:(x + 100)], 4)
                    save_image('%s/%d_pan.tif' % (testDir, testCount), 
                        img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 1)
                    save_image('%s/%d_pan_d.tif' % (testDir, testCount), 
                        img_pan_d[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 1)
                    
                    testCount += 1      
                    col += 1
                
                row += 1
                print (num, row)
            
            record.write("%d: %d * %d\n" % (num, row, col))
            
        record.write("%d\n" % testCount)
        record.close()
        print ("done") 

        ### test full-res
        originTestCount = 0
        originTestDir = "%s/Dataset/%s/test_full_res" % (dataDir, satellite)
        if not os.path.exists(originTestDir):
            os.makedirs(originTestDir)
        record = open('%s/record.txt' % originTestDir, "w")
        for num in range(1):
            lr = '%s/Dataset/%s/%s_mul_o.tif' % (dataDir, satellite, num)
            lr_u = '%s/Dataset/%s/%s_mul_o_u.tif' % (dataDir, satellite, num)
            pan = '%s/Dataset/%s/%s_pan_o.tif' % (dataDir, satellite, num)
            pan_d = '%s/Dataset/%s/%s_pan_o_d.tif' % (dataDir, satellite, num)

            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_pan_d = gdal.Open(pan_d)
            dt_lr_u = gdal.Open(lr_u)
            
            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_pan_d = dt_pan_d.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            
            row = 0
            col = 0
            
            for y in range(0, YSize, 50): 
                if y + 100 > YSize:
                    continue
                col = 0
                
                for x in range(0, XSize, 50):
                    if x + 100 > XSize:
                        continue

                    save_image('%s/%d_lr_u.tif' % (originTestDir, originTestCount), 
                        img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 4)
                    save_image('%s/%d_lr.tif' % (originTestDir, originTestCount), 
                        img_lr[:, y:(y + 100), x:(x + 100)], 4)
                    save_image('%s/%d_pan.tif' % (originTestDir, originTestCount), 
                        img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 1)
                    save_image('%s/%d_pan_d.tif' % (originTestDir, originTestCount), 
                        img_pan_d[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4], 1)
                    
                    originTestCount += 1      
                    col += 1
                
                row += 1
                print (num, row)

            record.write("%d: %d * %d\n" % (num, row, col))
            
        record.write("%d\n" % originTestCount)
        record.close()
        print ("done") 

    print ("finish")
