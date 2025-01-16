import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from photutils import EllipticalAperture, EllipticalAnnulus

import sep
import numpy as np
from einops import rearrange
import torch

from mantis_shrimp import datasets
from tqdm import tqdm

import sys
import os

###TODO: 

#Record MAD (done)
#Matched Aperatures
#Galex Bands -- how to handle missing data?
#Learn the Photometric systems for
    # WISE
    # Galex

def mask_center_source(x_galex,x_ps,x_unwise,INDEX):
    
    FILTERNAMES = ['FUV','NUV','PS:g','PS:r','PS:i','PS:z','PS:y','W1','W2']
    
    k=INDEX #alias
    
    if k in [0,1]: #switch UV because they are actually stored backwards.
        img = x_galex[k]
        center = 16
        r_size = 2.0
    elif k in [2,3,4,5,6]:
        img = x_ps[k-2]
        center = 85
        r_size = 4.0
    else:
        img = x_unwise[k-7]
        center = 16
        r_size = 2.0
       
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
        
    
    bkg = sep.Background(img)
    bkg_rms = bkg.rms()
    data_sub = img - bkg
    objects = sep.extract(data_sub, 2.0, err=bkg.globalrms)
    
    
    m, s = np.mean(data_sub), np.std(data_sub)
    
    x = np.array([objects[i][7] for i in range(len(objects))])
    y = np.array([objects[i][8] for i in range(len(objects))])
    a = np.array([objects[i][15] for i in range(len(objects))])
    b = np.array([objects[i][16] for i in range(len(objects))])
    theta = np.array([objects[i][17] for i in range(len(objects))])
    flux = np.array([objects[i][21] for i in range(len(objects))])

    distance_to_center = np.sqrt(np.square(x-center)+np.square(y-center))
    if np.any(distance_to_center < 5):
        
        index_closest_to_center = np.argmin(distance_to_center)
        
        mask_one = np.zeros(data_sub.shape, dtype=bool)
        mask_all = np.zeros(data_sub.shape, dtype=bool)
        
        sep.mask_ellipse(mask_one,
                         x[index_closest_to_center:index_closest_to_center+1],
                         y[index_closest_to_center:index_closest_to_center+1],
                         a[index_closest_to_center:index_closest_to_center+1],
                         b[index_closest_to_center:index_closest_to_center+1],
                         theta[index_closest_to_center:index_closest_to_center+1],
                         r=r_size)
        
        sep.mask_ellipse(mask_all,
                 x[:],
                 y[:],
                 a[:],
                 b[:],
                 theta[:],
                 r=r_size)
        
    else:
        mask_one = np.zeros(data_sub.shape, dtype=bool)
        mask_all = np.zeros(data_sub.shape, dtype=bool)
        
        sep.mask_ellipse(mask_one,
                         data_sub.shape[0]/2,
                         data_sub.shape[1]/2,
                         1,
                         1,
                         0,
                         r=r_size) 
        
        sep.mask_ellipse(mask_all,
                 x[:],
                 y[:],
                 a[:],
                 b[:],
                 theta[:],
                 r=r_size)
    
    MAD = 1.4826 * np.median(np.abs(np.median(data_sub[~mask_all]) - data_sub[~mask_all]))
    data_sub[mask_one] = np.random.normal(np.median(data_sub[~mask_all]),MAD,size=mask_one.sum())
    
    return data_sub + bkg #finally add the background back in.

def keep_center_source(x_galex,x_ps,x_unwise,INDEX,set_zero=True):
    
    FILTERNAMES = ['FUV','NUV','PS:g','PS:r','PS:i','PS:z','PS:y','W1','W2']
    
    k=INDEX #alias
    
    if k in [0,1]:
        img = x_galex[k]
        center = 16
        r_size = 2.0
    elif k in [2,3,4,5,6]:
        img = x_ps[k-2]
        center = 85
        r_size = 4.0
    else:
        img = x_unwise[k-7]
        center = 16
        r_size = 2.0
       
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
        
    
    bkg = sep.Background(img)
    bkg_rms = bkg.rms()
    data_sub = img - bkg
    objects = sep.extract(data_sub, 2.0, err=bkg.globalrms)
    
    m, s = np.mean(data_sub), np.std(data_sub)
    
    x = np.array([objects[i][7] for i in range(len(objects))])
    y = np.array([objects[i][8] for i in range(len(objects))])
    a = np.array([objects[i][15] for i in range(len(objects))])
    b = np.array([objects[i][16] for i in range(len(objects))])
    theta = np.array([objects[i][17] for i in range(len(objects))])
    flux = np.array([objects[i][21] for i in range(len(objects))])

    distance_to_center = np.sqrt(np.square(x-center)+np.square(y-center))
    if np.any(distance_to_center < 5):
        
        index_closest_to_center = np.argmin(distance_to_center)
        
        mask_one = np.zeros(data_sub.shape, dtype=bool)
        
        sep.mask_ellipse(mask_one,
                         x[index_closest_to_center:index_closest_to_center+1],
                         y[index_closest_to_center:index_closest_to_center+1],
                         a[index_closest_to_center:index_closest_to_center+1],
                         b[index_closest_to_center:index_closest_to_center+1],
                         theta[index_closest_to_center:index_closest_to_center+1],
                         r=r_size)
        
    else:
        mask_one = np.zeros(data_sub.shape, dtype=bool)
        
        sep.mask_ellipse(mask_one,
                         data_sub.shape[0]/2,
                         data_sub.shape[1]/2,
                         1,
                         1,
                         0,
                         r=r_size) 
        

    if set_zero:
        data_sub[~mask_one] = 0.0
    else:
        MAD = 1.4826 * np.median(np.abs(np.median(data_sub[~mask_one]) - data_sub[~mask_one]))
        data_sub[~mask_one] = np.random.normal(np.median(data_sub[~mask_one]),MAD,size=(~mask_one).sum())
    
    return data_sub #finally add the background back in.


def extract_magnitude(x_galex,x_ps,x_unwise,INDEX):
    
    FILTERNAMES = ['FUV','NUV','PS:g','PS:r','PS:i','PS:z','PS:y','W1','W2']
    
    k=INDEX #alias
    
    if k in [0,1]: #switch UV because they are actually stored backwards.
        img = x_galex[k]
        center = 16
        r_size = 2.0
    elif k in [2,3,4,5,6]:
        img = x_ps[k-2]
        center = 85
        r_size = 4.0
    else:
        img = x_unwise[k-7]
        center = 16
        r_size = 2.0
       
    if isinstance(img,torch.Tensor):
        img = img.cpu().numpy()
        
    
    if np.all(img==0):
        #we want to recognize when we don't have a flux value.
        return 0.0, True
        
    
    bkg = sep.Background(img)
    bkg_rms = bkg.rms()
    data_sub = img - bkg
    objects = sep.extract(data_sub, 2.0, err=bkg.globalrms)
    MAD = 1.4826 * np.median(np.abs(data_sub - np.median(data_sub)))
    
    
    m, s = np.mean(data_sub), np.std(data_sub)
    
    x = np.array([objects[i][7] for i in range(len(objects))])
    y = np.array([objects[i][8] for i in range(len(objects))])
    a = np.array([objects[i][15] for i in range(len(objects))])
    b = np.array([objects[i][16] for i in range(len(objects))])
    theta = np.array([objects[i][17] for i in range(len(objects))])
    flux = np.array([objects[i][21] for i in range(len(objects))])

    distance_to_center = np.sqrt(np.square(x-center)+np.square(y-center))


    if np.any(distance_to_center < 5):
        index_closest_to_center = np.argmin(distance_to_center)
        Forced_Photo=False

        #sometimes this returns NaN. Unknown Currently when/why
        kronrad, krflag = sep.kron_radius(data_sub,
                                          x[index_closest_to_center:index_closest_to_center+1],
                                          y[index_closest_to_center:index_closest_to_center+1],
                                          a[index_closest_to_center:index_closest_to_center+1],
                                          b[index_closest_to_center:index_closest_to_center+1],
                                          theta[index_closest_to_center:index_closest_to_center+1],
                                          6.0)

        if np.isnan(kronrad) or kronrad==0:
            Forced_photo=True
            flux, fluxerr, flag = sep.sum_circle(data_sub, np.array([center,]), np.array([center,]), 4.0)
            flux = flux[0]
            return flux, Forced_Photo, MAD
        else:
            #compute KronRad
            try:
                #https://github.com/kbarbary/sep/issues/110
                THETA = theta[index_closest_to_center:index_closest_to_center+1]
                if THETA<-1*np.pi/2:
                    THETA = -1*np.pi/2
                if THETA>np.pi/2:
                    THETA = np.pi/2
                
                flux, fluxerr, flag = sep.sum_ellipse(data_sub,
                                                      x[index_closest_to_center:index_closest_to_center+1],
                                                      y[index_closest_to_center:index_closest_to_center+1],
                                                      a[index_closest_to_center:index_closest_to_center+1],
                                                      b[index_closest_to_center:index_closest_to_center+1],
                                                      THETA,
                                                      2.5*kronrad,
                                                      subpix=5)
            except Exception as e:
                #sometimes this fails, I want to trigger an assert outside this function if I do.
                return np.nan, True, MAD
            
            flux = flux[0]
            return flux, Forced_Photo
            #plt.plot(x[index_closest_to_center],y[index_closest_to_center],'c+')
    else:
        Forced_Photo=True
        #we do not detect any sources, so do forced photometry with a 3.0 pixel radius
        flux, fluxerr, flag = sep.sum_circle(data_sub, np.array([center,]), np.array([center,]), 4.0)
        flux = flux[0]
        return flux, Forced_Photo, MAD

def compute_morphology_and_flux(img,r_size=4.0):

    bkg = sep.Background(img)
    bkg_rms = bkg.rms()
    data_sub = img - bkg
    objects = sep.extract(data_sub, 2.0, err=bkg.globalrms)
    
    m, s = np.mean(data_sub), np.std(data_sub)
    
    x = np.array([objects[i][7] for i in range(len(objects))])
    y = np.array([objects[i][8] for i in range(len(objects))])
    a = np.array([objects[i][15] for i in range(len(objects))])
    b = np.array([objects[i][16] for i in range(len(objects))])
    theta = np.array([objects[i][17] for i in range(len(objects))])
    flux = np.array([objects[i][21] for i in range(len(objects))])

    center_x = img.shape[0]/2
    center_y = img.shape[1]/2
    distance_to_center = np.sqrt(np.square(x-center_x)+np.square(y-center_y))

    MAD = 1.4826 * np.median(np.abs(data_sub - np.median(data_sub)))
    
    if np.any(distance_to_center<5.0):
    
        Forced_photo=False
        index_closest_to_center = np.argmin(distance_to_center)
        
        mask_center = np.zeros(data_sub.shape, dtype=bool)
        mask_all = np.zeros(data_sub.shape, dtype=bool)
        
        #we can use all the detected sources to mask all sources
        sep.mask_ellipse(mask_all,
                 x[:],
                 y[:],
                 a[:],
                 b[:],
                 theta[:],
                 r=r_size)
        
        #we can use the center most detection, or define a forced aperature, to mask the center most source
        sep.mask_ellipse(mask_center,
                         x[index_closest_to_center:index_closest_to_center+1],
                         y[index_closest_to_center:index_closest_to_center+1],
                         a[index_closest_to_center:index_closest_to_center+1],
                         b[index_closest_to_center:index_closest_to_center+1],
                         theta[index_closest_to_center:index_closest_to_center+1],
                         r=r_size)

        #compute kronradius and kronmag
        kronrad, krflag = sep.kron_radius(data_sub,
                                          x[index_closest_to_center:index_closest_to_center+1],
                                          y[index_closest_to_center:index_closest_to_center+1],
                                          a[index_closest_to_center:index_closest_to_center+1],
                                          b[index_closest_to_center:index_closest_to_center+1],
                                          theta[index_closest_to_center:index_closest_to_center+1],
                                          6.0)


        
        #this would mask the center, but we want to mask everything but the center, so invert it.
        mask_center = ~mask_center
    
        # we need flux, axis ratio, half-light radius, and chi^2 p thing
        
        #okay so we get axis_ratio for free from sep
        axis_ratio = b[index_closest_to_center] / a[index_closest_to_center]
        #semi-minor / semi-major axis will always be less than 1, so good for ML
        
        #half-light radius we can use sep to compute
        # compute half-light radius with a maximum radius of 2*the semi-major axis of the ellipse.
        
        #we could do this inside the center most object
        half_light_radius, flags = sep.flux_radius(data_sub,
                                                   x[index_closest_to_center:index_closest_to_center+1],
                                                   y[index_closest_to_center:index_closest_to_center+1],
                                                   2*a[index_closest_to_center:index_closest_to_center+1],
                                                   0.5)
        
        #MAD = 1.4826 * np.median(np.abs(data_sub - np.median(data_sub)))
        #we will set MAD to be the background RMS
    
        #define ellipticity using the previous
        ellip = 1-b[index_closest_to_center]/a[index_closest_to_center]
        
        chi_square_Exp = calculate_Exp_chi_square(data_sub,
                                 mask_center,
                                 x[index_closest_to_center],
                                 y[index_closest_to_center],
                                 half_light_radius[0],
                                 ellip,
                                 theta[index_closest_to_center],
                                 MAD)
        
        chi_square_DeV = calculate_DeV_chi_square(data_sub,
                                 mask_center,
                                 x[index_closest_to_center],
                                 y[index_closest_to_center],
                                 half_light_radius[0],
                                 ellip,
                                 theta[index_closest_to_center],
                                 MAD)
    
        p = chi_square_DeV / (chi_square_DeV + chi_square_Exp)

        if np.isnan(kronrad) or kronrad==0:
            print('KronRad was nan or 0!')
            Forced_photo=True
            flux, fluxerr, flag = sep.sum_circle(data_sub, np.array([center_x,]), np.array([center_y,]), r_size)
            flux = flux[0]
            #return flux, Forced_Photo
        else:
            #compute KronRad
            try:
                #https://github.com/kbarbary/sep/issues/110
                THETA = theta[index_closest_to_center:index_closest_to_center+1]
                if THETA<-1*np.pi/2:
                    THETA = -1*np.pi/2
                if THETA>np.pi/2:
                    THETA = np.pi/2
                
                flux, fluxerr, flag = sep.sum_ellipse(data_sub,
                                                      x[index_closest_to_center:index_closest_to_center+1],
                                                      y[index_closest_to_center:index_closest_to_center+1],
                                                      a[index_closest_to_center:index_closest_to_center+1],
                                                      b[index_closest_to_center:index_closest_to_center+1],
                                                      THETA,
                                                      2.5*kronrad,
                                                      subpix=5)
                flux = flux[0]
                
            except Exception as e:
                #sometimes getting Kron fails-- if that is true then we can fall back onto flux detected from object
                flux = flux[index_closest_to_center]
                Forced_photo = True
        
        return flux, axis_ratio, half_light_radius[0], p, Forced_photo, MAD

    
    else: #performing forced photometry; didn't detect anything.
        Forced_photo=True
        flux, fluxerr, flag = sep.sum_circle(data_sub, np.array([center_x,]), np.array([center_y,]), r_size)
        flux = flux[0]

        #in this case, there is no morphology. so we should make some indication of that.

        #okay, well lets assume I'll add in a categorical variable of Forced_photo.

        #axis ratio from my flux calculation is 1.
        axis_ratio = 1

        #force this calculation. see what happens.
        half_light_radius, flags = sep.flux_radius(data_sub, np.array([center_x,]), np.array([center_y,]), np.array([20.0,]), 0.5)
        
        #since I can't detect anything, meaningless to try to calculate a morphological p. lets just skip
        p = 0.5 #if they are both very large, then assume large / (large + large) = 1/2.

        return flux, axis_ratio, half_light_radius, p, Forced_photo, MAD

# get_default_sersic_bounds returns:


# mkae the above into a function
def calculate_DeV_chi_square(img,mask,x,y,half_light_radius,ellip,theta,background_std,verbose=False):

    #first divide by the maximum value in my data.
    #I don't actually care to calculate a model flux so we can just do this.
    #the reason I want to is because computers don't like giant numbers while fitting data.
    
    amplitude = img[~mask].max()/2 #*np.exp(-7.67) #/2 #

   # if verbose:
        # img[mask] = 0
        # plt.imshow(img)
        # plt.title('masked_img')
        # plt.show()
        # raise Exception
    bounds = {
        'amplitude': (1e-3, img[~mask].max()),
        'r_eff': (1e-1, half_light_radius*3),
        'ellip': (0, 1),
        'n': (1.0,1.1),
        'theta': (-2 * np.pi, 2 * np.pi),
        'x_0': (x-half_light_radius, x+half_light_radius),
        'y_0': (y-half_light_radius, y+half_light_radius)
    }
        
    #somtimes I get very bright stars that are in my mask. this is not good.
    #my fit will fail, and it probably has other implifcations. for now impose ceiling.
    #if amplitude > 1000:
    #    amplitude = 1000
        
    # Define a de Vaucouleurs profile model; n=4
    initial_guess_Vauc = models.Sersic2D(amplitude=amplitude, r_eff=half_light_radius, n=4, 
                                 x_0=x, y_0=y, ellip=ellip, theta=theta, bounds=bounds, fixed={'n':True})
    
    # Use the astropy fitting routines
    #fit_p = fitting.LevMarLSQFitter()
    #fit_p = fitting.DogBoxLSQFitter(use_min_max_bounds=True)
    fit_p = fitting.LevMarLSQFitter()

    #define a meshgrid from my image
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]

    #while I can not directly mask some aspects of my image to the fit, I can
    #attempt to weigh the pixels that I do not care about down to effectively mask them
    #from the fit. start with a ones array, then everything I do not want included in the 
    #fit lets update to have a weight zero; i.e., will be ignored in the fit.
    weights = np.ones(img.shape)
    weights[mask] = 0
    weights[~mask] = 1/background_std

    #we are going to mask out brightest 5% of pixels in each fit.
    bright_pixel_mask = img>np.quantile(img[~mask],[0.95])
    weights[bright_pixel_mask] = 0

    # Make my fits
    try:
        de_vaucouleurs_fit = fit_p(initial_guess_Vauc, x, y, img, weights=weights, filter_non_finite=True)
        fitted_data = de_vaucouleurs_fit(x, y)

        if verbose:
            pretty_print(initial_guess_Vauc.parameters,initial_guess_Vauc.param_names)
            pretty_print(de_vaucouleurs_fit.parameters,de_vaucouleurs_fit.param_names)
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,3,1)
            plt.imshow(img,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))
            plt.subplot(1,3,2)
            plt.title('Fit DeV')
            plt.imshow(fitted_data,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))

        residual = img-fitted_data
        residual[mask] = 0
        residual[bright_pixel_mask] = 0

        #I have an issue where the central buldges are not well fit by a DeV profile. Therefore,
        #I need to mask out the central region. We will do this by instituting a 95% filter
        #from the MAD of the reisudals, i.e., 2*NMAD chi^2. 
        chi = residual/background_std
        
        if verbose:
            plt.subplot(1,3,3)
            plt.title('Chi DeV')
            plt.imshow(chi)
            plt.colorbar()
            plt.show()

        failed_fit=False
    except Exception as E:
        fitted_data = initial_guess_Vauc(x, y)
        if verbose:
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,3,1)
            plt.imshow(img,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))
            plt.subplot(1,3,2)            
            plt.imshow(fitted_data,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))

        residual = img-fitted_data
        residual[mask] = 0
        #I have an issue where the central buldges are not well fit by a DeV profile. Therefore,
        #I need to mask out the central region. We will do this by instituting a 95% filter
        #from the MAD of the reisudals, i.e., 2*NMAD chi^2. 
        chi = residual/background_std

        if verbose:
            plt.subplot(1,3,3)
            plt.title('failed fit residuals DeV')
            plt.imshow(chi)
            plt.colorbar()
            plt.show()

            print(E)
        failed_fit=True
        #return 10_000, None #return a large chi^2 since the fit failed

    #reconstuct the galaxy
    #Reconstruct the galaxy
    if failed_fit:
        fitted_data = initial_guess_Vauc(x, y)
        total_mask = ~mask
    else:
        fitted_data = de_vaucouleurs_fit(x, y)
        total_mask = np.logical_and(~bright_pixel_mask,~mask)
    

    #compute chi-square only including the img that was part of the source, and use my background std definition
    #just in case there are some very cluttered fields.
    chi_square = np.sum(np.square((fitted_data[total_mask] - img[total_mask]) / background_std))
    DOF = np.sum(total_mask) - 6 #n_parameters = 7

    #Now I need to mask out the highest reisuals. Let
    
    return chi_square/DOF

def calculate_Exp_chi_square(img,mask,x,y,half_light_radius,ellip,theta,background_std,verbose=False):
    #guess that the amplitude is the maximum flux value within my band
    
    amplitude = img[~mask].max()/6 #/ 6
    #somtimes I get very bright stars that are in my mask. this is not good.
    #my fit will fail, and it probably has other implifcations. for now impose ceiling.
    #if amplitude > 1000:
    #    amplitude = 1000

    bounds = {
        'amplitude': (1e-3, img[~mask].max()),
        'r_eff': (1e-1, half_light_radius*3),
        'ellip': (0, 1),
        'n': (1.0,1.1),
        'theta': (-2 * np.pi, 2 * np.pi),
        'x_0': (x-half_light_radius, x+half_light_radius),
        'y_0': (y-half_light_radius, y+half_light_radius)
    }
    
    # Define an Exponential profile model (Same as a Sersic model with n=1)
    initial_guess_Exp = models.Sersic2D(amplitude=amplitude, r_eff=half_light_radius, n=1, 
                                 x_0=x, y_0=y, ellip=ellip, theta=theta, bounds=bounds, fixed={'n':True})

    # Use the astropy fitting routines
    fit_p = fitting.LevMarLSQFitter()

    #define a meshgrid from my image
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]

    #while I can not directly mask some aspects of my image to the fit, I can
    #attempt to weigh the pixels that I do not care about down to effectively mask them
    #from the fit. start with a ones array, then everything I do not want included in the 
    #fit lets update to an absurdly high number, like 1e9. one is an okay choice for the
    #remainder of the pixels since i am not modeling the pixel-to-pixel uncertainty.
    weights = np.ones(img.shape)
    weights[mask] = 0
    weights[~mask] = 1/background_std
    
    # Make the fits
    #we are going to mask out brightest 5% of pixels in each fit.
    bright_pixel_mask = img>np.quantile(img[~mask],[0.95])
    weights[bright_pixel_mask] = 0
    
    try:
        exponential_fit = fit_p(initial_guess_Exp, x, y, img, weights=weights, filter_non_finite=True)
        fitted_data = exponential_fit(x, y)
        
        if verbose:
            pretty_print(initial_guess_Exp.parameters,initial_guess_Exp.param_names)
            pretty_print(exponential_fit.parameters,exponential_fit.param_names)
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,3,1)
            plt.imshow(img,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))
            plt.subplot(1,3,2)
            plt.title('Exp Fit')
            plt.imshow(fitted_data,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))


        residual = img-fitted_data
        residual[mask] = 0
        residual[bright_pixel_mask] = 0
        #I have an issue where the central buldges are not well fit by a DeV profile. Therefore,
        #I need to mask out the central region. We will do this by instituting a 95% filter
        #from the MAD of the reisudals, i.e., 2*NMAD chi^2. 
        chi = residual/background_std
        
        if verbose:
            plt.subplot(1,3,3)
            plt.title('Chi Exp')
            plt.imshow(chi)
            plt.colorbar()
            plt.show()

        failed_fit=False
    except Exception as E:
        fitted_data = initial_guess_Exp(x, y)
        
        if verbose:
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,3,1)
            plt.imshow(img,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))
            plt.subplot(1,3,2)        
            plt.imshow(fitted_data,vmin=np.quantile(img,0.01),vmax=np.quantile(img,0.95))

        residual = img-fitted_data
        residual[mask] = 0
        #I have an issue where the central buldges are not well fit by a DeV profile. Therefore,
        #I need to mask out the central region. We will do this by instituting a 95% filter
        #from the MAD of the reisudals, i.e., 2*NMAD chi^2. 
        chi = residual/background_std

        if verbose:
            plt.subplot(1,3,3)
            plt.title('failed fit residuals Exp')
            plt.imshow(chi)
            plt.colorbar()
            plt.show()

            print(E)
        failed_fit=True
        #return 10_000, None #return a very large chi^2 since the fit failed.
        
    #Reconstruct the galaxy
    if failed_fit:
        fitted_data = initial_guess_Exp(x, y)
        total_mask = ~mask
    else:
        fitted_data = exponential_fit(x, y)
        total_mask = np.logical_and(~mask,~bright_pixel_mask)

    #compute chi-square only including the img that was part of the source, and use my background std definition
    #just in case there are some very cluttered fields.
    
    chi_square = np.sum(np.square((fitted_data[total_mask] - img[total_mask]) / background_std))
    DOF = np.sum(total_mask) - 6 #n_parameters = 7

    return chi_square/DOF

if __name__ == '__main__':
    kind_ = str(sys.argv[1])
    chunk = int(sys.argv[2])

    fluxes = []
    sigmas = []
    fixed_photos = []
    axis_ratios = []
    half_light_radii = []
    ps = []
        
    for world_rank_ in [chunk,]:
        #we do not apply transform, so should be linear flux scaled...?
        MSD = datasets.MantisShrimpDataset(kind=kind_,
                                           WORLD_RANK=world_rank_,
                                           ZMAX=1.6,
                                           mmap=True,
                                           transform=False,
                                           sep =False,
                                           to_torch=False)
    
        
        for i in tqdm(range(len(MSD))):
            x_galex,x_ps,x_unwise,y,ebvs,pz_MGS,pz_WPS = MSD[i]
            
            x_ps[np.isnan(x_ps)] = 0.0
    
            inner_fluxes = []
            inner_fixed_photos = []
            inner_sigmas = []
            for j in range(9):
                if j!=4:
                    flux, fixed_photo, sigma = extract_magnitude(x_galex[0],x_ps[0],x_unwise[0],j)
                else: 
                    #if i band ps, then lets compute morphology stuff
                    img = x_ps[0][2]
                    flux, axis_ratio, half_light_radius, p, fixed_photo, sigma = compute_morphology_and_flux(img,r_size=4.0)
                    if isinstance(half_light_radius,np.ndarray):
                        half_light_radius = half_light_radius[0]
                    axis_ratios.append(axis_ratio)
                    half_light_radii.append(half_light_radius)
                    ps.append(p)
                    
                #Always append the total flux
                inner_fluxes.append(flux)
                inner_fixed_photos.append(fixed_photo)
                inner_sigmas.append(sigma)
            
            fluxes.append(inner_fluxes)
            fixed_photos.append(inner_fixed_photos)
            sigmas.append(inner_sigmas)
                
            assert np.all(~np.isnan(inner_fluxes))

    result = {}
    result['fluxes'] = np.stack(fluxes)
    result['fixed_photos'] = np.concatenate(fixed_photos)
    result['axis_ratio'] = np.array(axis_ratios)
    result['half_light_radii'] = half_light_radii #this possibly has something weird
    result['ps'] = np.array(ps)
    result['sigmas'] = np.stack(sigmas)

    BASE_PATH = './../data/my_photometry_extracted/'
    F_NAME = f'{kind_}_chunk{chunk}_photodict.npy'
    np.save(os.path.join(BASE_PATH,F_NAME),result)
    print('all done!')
