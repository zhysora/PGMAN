import torch
import torch.utils.data as data
import os
from os import listdir
from os.path import join
import numpy as np
import gdal, osr, cv2

def is_pan_image(filename):
    return filename.endswith("pan.tif")

def load_image(path):
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)

    #if img.ndim != 3:
    #    img = img[np.newaxis, :, :]

    # img = torch.from_numpy( img ).float()
    return img

def save_image(path, array, bandSize):
    rasterOrigin = (-123.25745,45.43013)
    pixelWidth = 2.4
    pixelHeight = 2.4
    
    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]
        #print (path, cols, rows)

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(path, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]
        #print (path, cols, rows)
        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array[:, :])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_dirs = image_dirs
        self.image_filenames = []
        for y in image_dirs:
            for x in listdir(y):
                if is_pan_image(x):
                    self.image_filenames.append( join(y, x.split('_')[0]) )
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_pan = load_image('%s_pan.tif' % self.image_filenames[index])
        input_pan_d = load_image('%s_pan_d.tif' % self.image_filenames[index])
        input_lr = load_image('%s_lr.tif' % self.image_filenames[index])
        input_lr_u = load_image('%s_lr_u.tif' % self.image_filenames[index])
        input_pan_l = cv2.pyrDown(cv2.pyrDown(input_pan))

        input_pan = torch.from_numpy(input_pan[np.newaxis, :]).float()
        input_pan_d = torch.from_numpy(input_pan_d[np.newaxis, :]).float()
        input_lr = torch.from_numpy(input_lr).float()
        input_lr_u = torch.from_numpy(input_lr_u).float()
        input_pan_l = torch.from_numpy(input_pan_l[np.newaxis, :]).float()

        if os.path.exists('%s_mul.tif' % self.image_filenames[index]):
            target = load_image('%s_mul.tif' % self.image_filenames[index])
            target = torch.from_numpy(target).float()
        else:
            target = torch.from_numpy( np.zeros( input_lr_u.size(), dtype=np.double) ).float()

        filename = int(self.image_filenames[index].split('/')[-1])
        if self.input_transform:
            input_pan = self.input_transform(input_pan)
            input_pan_d = self.input_transform(input_pan_d)
            input_lr = self.input_transform(input_lr)
            input_lr_u = self.input_transform(input_lr_u)
            input_pan_l = self.input_transform(input_pan_l)
        if self.target_transform:
            target = self.target_transform(target)

        return input_pan, input_pan_d, input_lr, input_lr_u, target, filename, input_pan_l

    def __len__(self):
        return len(self.image_filenames)