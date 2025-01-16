from torchvision import transforms
import numpy as np

augment_ps = transforms.CenterCrop((120,120))
augment_unwise = transforms.CenterCrop((22,22))
augment_galex = transforms.CenterCrop((22,22))
upsample = transforms.Resize((120,120),antialias=True)

class JitterCrop:
    '''takes in image of size (npix, npix, nchannel), 
    jitters by uniformly drawn (-jitter_lim, jitter_lim),
    and returns (outdim, outdim, nchannel) central pixels'''

    def __init__(self, outdim=96, jitter_lim=7):
        self.outdim = outdim
        self.jitter_lim = jitter_lim
        
    def __call__(self, image):                            
        if self.jitter_lim:
            center_x = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
        else:
            center_x = image.shape[0]//2
            center_y = image.shape[0]//2
        offset = self.outdim//2

        return image[(center_x-offset):(center_x+offset), (center_y-offset):(center_y+offset)]

jitter_ps = JitterCrop(60,4)
jitter_unwise = JitterCrop(11,1)
jitter_galex = JitterCrop(11,1)

random_resize_crop = transforms.RandomResizedCrop(120,(0.95,1.05),(0.95,1.05))

#early_fusion augmentation
def augment_fn(_galex,_ps,_unwise):

    angle = np.random.random() * 360.
    _ps = transforms.functional.rotate(_ps,angle)
    _unwise = transforms.functional.rotate(_unwise,angle)
    _galex = transforms.functional.rotate(_galex,angle)
    
    if np.random.random() > 0.5:
        _ps = transforms.functional.hflip(_ps)
        _unwise = transforms.functional.hflip(_unwise)
        _galex = transforms.functional.hflip(_galex)
    if np.random.random() > 0.5:
        _ps = transforms.functional.vflip(_ps)
        _unwise = transforms.functional.vflip(_unwise)
        _galex = transforms.functional.vflip(_galex)

    if True:
        ###crop each to remove the black parts from the rotation.
        #use jitter crop instead to see what happens
        #_ps = jitter_ps(_ps) #augment_ps(_ps)
        #_unwise = jitter_unwise(_unwise) #augment_unwise(_unwise)
        #_galex = jitter_galex(_galex) #augment_galex(_galex)
        _ps = augment_ps(_ps)
        _unwise = augment_unwise(_unwise)
        _galex = augment_galex(_galex)
        
        #finally, we upsample
        _unwise = upsample(_unwise)
        _galex = upsample(_galex)
    else:
        _ps = random_resize_crop(_ps)
        _galex = random_resize_crop(_galex)
        _unwise = random_resize_crop(_unwise)
    
    return _galex,_ps,_unwise

def augment_fn_latefusion(_galex,_ps,_unwise):
    angle = np.random.random() * 360.
    _ps = transforms.functional.rotate(_ps,angle)
    _unwise = transforms.functional.rotate(_unwise,angle)
    _galex = transforms.functional.rotate(_galex,angle)
    
    if np.random.random() > 0.5:
        _ps = transforms.functional.hflip(_ps)
        _unwise = transforms.functional.hflip(_unwise)
        _galex = transforms.functional.hflip(_galex)
    if np.random.random() > 0.5:
        _ps = transforms.functional.vflip(_ps)
        _unwise = transforms.functional.vflip(_unwise)
        _galex = transforms.functional.vflip(_galex)
       
    ###crop each to remove the black parts from the rotation. 
    _ps = augment_ps(_ps)

    #--- I dont think this actually matters, and for our latefusion model it seems better to use the pre-trained weights.
    #_unwise = augment_unwise(_unwise)
    #_galex = augment_galex(_galex)
    
    return _galex,_ps,_unwise
