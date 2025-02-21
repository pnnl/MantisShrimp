#this script looks to see if an index exists or not and saves a numpy binary mask as a result.
import os
from einops import rearrange
import numpy as np
from astropy.io import fits
import cv2
import sys
import warnings

def NAN_interpolation2(img):
    '''
    #Uses opencv2 library inpaint method to perform interpolation instead.
    #the reult is something that more naturally looks like the image.
    '''
    for i in range(len(img)):
        data = img[i]
        mask = np.isnan(data).astype(np.uint8)
        data = cv2.inpaint(data.astype(np.float32), mask, 3, cv2.INPAINT_TELEA)
        img[i]=data
    return img

def process_galex(path):
    '''
    take in fits path, returns image of galex data
    '''
    with fits.open(path, ignore_missing_simple=True) as hdu:
        img = hdu[0].data
        img = np.array(img) #2,50,50 = [NUV, FUV]

    NAN_FLAG = np.any(np.isnan(img))
    assert not(NAN_FLAG)

    return img

def process_unwise(path):
    '''
    take in fits path, returns image of unwise data
    '''
    with fits.open(path, ignore_missing_simple=True) as hdu:
        img = hdu[0].data # = [W1, W2]
        img = np.array(img) 

    NAN_FLAG = np.any(np.isnan(img))
    assert not(NAN_FLAG)
    
    return img

def process_panstarrs(list_of_paths):
    '''
    take in fits path, returns image of panstarrs data
    '''
    FILTERS = ['g','r','i','z','y']
    img_filters = []
    for path in list_of_paths:
        with fits.open(path, ignore_missing_simple=True) as hdu:
            img = hdu[0].data
            img_filters.append(img)

    img = np.stack(img_filters)
    NAN_FLAG = np.any(np.isnan(img))

    if NAN_FLAG:

        shape_per_band = img.shape[1] * img.shape[2]
        percent_nan_in_band = 100 * np.sum(np.isnan(img),axis=(1,2)) / shape_per_band

        if np.any(percent_nan_in_band) > 1.0:
            warnings.warn("Found a Panstarrs filter with >1% NaN, our original pipeline removed these", RuntimeWarning)
            #raise Exception #reject if 1% is NaN in any band
        else:
            #use cv2 since it looked good on bright sources.
            img = NAN_interpolation2(img)
    
    return img