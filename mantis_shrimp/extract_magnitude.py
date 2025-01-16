import sep_pjw as sep
import numpy as np
from einops import rearrange
import torch

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
            return flux, Forced_Photo
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
                return np.nan, True
            
            flux = flux[0]
            return flux, Forced_Photo
            #plt.plot(x[index_closest_to_center],y[index_closest_to_center],'c+')
    else:
        Forced_Photo=True
        #we do not detect any sources, so do forced photometry with a 3.0 pixel radius
        flux, fluxerr, flag = sep.sum_circle(data_sub, np.array([center,]), np.array([center,]), 4.0)
        flux = flux[0]
        return flux, Forced_Photo





        