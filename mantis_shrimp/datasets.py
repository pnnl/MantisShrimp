import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from einops import rearrange
from mantis_shrimp.utils import galex_transform, new_transform
from mantis_shrimp import extract_magnitude

class MantisShrimpDataset(Dataset):
    def __init__(self, kind: str, WORLD_RANK:int =0, ZMAX: float=1.0, mmap: bool=True,
                 loc='vast', transform=True, sep=False, set_zero=False, to_torch=True):
        
        assert kind in ['train','val','test']
        assert loc in ['rcfs','vast']

        self.to_torch = to_torch
        
        self.max = np.array([2.0336876,8.301209523105817,6.198499090528679])
        self.transform = transform
        self.set_zero = set_zero
        
        csv_path1 = '/rcfs/projects/mantis_shrimp/mantis_shrimp/data/spectroscopy/redshifts_withextinction.pkl'
        csv_path2 = '/rcfs/projects/mantis_shrimp/mantis_shrimp/data/redshifts_broken_beck/SDSS_MGS/MGS_qwsaz123.csv'
        csv_path3 = '/rcfs/projects/mantis_shrimp/mantis_shrimp/data/redshifts_broken_beck/WISE_PS1_STRM.csv'
        
        DF = pd.read_pickle(csv_path1,)
        DF_pas = pd.read_csv(csv_path2,usecols=['bestObjID','zphot'])
        
        DF_pas.drop_duplicates('bestObjID',inplace=True)
        DF_pas_comb = pd.merge(DF,DF_pas,'left',left_on='photoObjID_survey',right_on='bestObjID')
        DF_pas_comb = DF_pas_comb.drop('bestObjID',axis=1)
        DF = DF_pas_comb
        DF_wps = pd.read_csv(csv_path3,
                             usecols=['dstArcSec',
                                      'cellDistance_Photoz',
                                      'z_phot0',
                                      'z_photErr',
                                      'prob_Galaxy',
                                      'photoObjID_survey']
                             )
        self.sep = sep
        self.ZMAX = ZMAX
        
        DF = pd.merge(DF,DF_wps,how='left',on='photoObjID_survey')
        
        indices = np.load(f'/rcfs/projects/mantis_shrimp/mantis_shrimp/data/npy_blocks/{kind}_indices.npy')
        exists_mask = np.load('/rcfs/projects/mantis_shrimp/mantis_shrimp/data/exists_mask.npy')
        
        #get the correct chunk's indices-- these match what is in img.
        indices = indices[WORLD_RANK]
        
        
        z = DF['z'].values
        ebv_csfd = DF['ebv_csfd'].values
        ebv_planck = DF['ebv_planck'].values
        zphot_MGS = DF['zphot'].values
        zphot_WPS = DF['z_phot0'].values
        
        #now apply indices to find the correct values for this chunk
        z = z[indices]
        ebv_csfd = ebv_csfd[indices]
        ebv_planck = ebv_planck[indices]
        zphot_MGS = zphot_MGS[indices]
        zphot_WPS = zphot_WPS[indices]
        exists_mask = exists_mask[indices]
        
        #Now we use a mixture of whether the data exists + whether it satisfies our ZMAX constraint to create a mask.
        #unfortunately, we cannot mask img b/c it is a mmap array. So we need to create a dictionary mapping from indices
        #the user would supply to the existing data in img.
        
        #really, a max and min mask. I think there are some stars with z<0.0 in the dataset. remove them.
        zmax_mask = np.logical_and(z<ZMAX,z>0.0)
        
        total_mask = np.logical_and(exists_mask,zmax_mask) #both must be True to accept.
        
        self.z = z[total_mask]
        self.ebv_csfd = ebv_csfd[total_mask]
        self.ebv_planck = ebv_planck[total_mask]
        self.zphot_MGS = zphot_MGS[total_mask]
        self.zphot_WPS = zphot_WPS[total_mask]
        self._indices = indices[total_mask]
        
        self.DF = DF
        
        self.idx_to_imgidx = dict(zip(np.arange(total_mask.sum()),np.where(total_mask)[0]))
        
        if mmap:
            self.img = np.load(f'/{loc}/projects/mantis_shrimp/mantis_shrimp/data/npy_blocks/{kind}/mantis_shrimp_{WORLD_RANK}.npy',
                mmap_mode='r')
        if not(mmap):
            #WARNING: these are typically giant files. good luck.
            self.img = np.load(f'/{loc}/projects/mantis_shrimp/mantis_shrimp/data/npy_blocks/{kind}/mantis_shrimp_{WORLD_RANK}.npy')                 
        
    def __len__(self):
        return len(self.idx_to_imgidx)

    def __getitem__(self, idx):
        img_indices = np.array([self.idx_to_imgidx[idx],])
        idx = np.array([idx,])
        img = self.img[img_indices]
        if len(img.shape) == 1:
            img = img[None,] #add leading batch dimension
        
        galex = img[:,0:2048]
        panstarrs = img[:,2048:146548]
        unwise = img[:,146548::]
        
        galex = rearrange(galex,'b (f h w) -> b f h w',f=2,h=32,w=32)
        panstarrs = rearrange(panstarrs,'b (f h w) -> b f h w',f=5,h=170,w=170)
        unwise = rearrange(unwise,'b (f h w) -> b f h w',f=2,h=32,w=32)
        
        panstarrs[np.isnan(panstarrs)] = 0.0
        
        if self.sep:
            for k in range(len(galex)):
                for i in range(9):  
                    if i in [0,1]:
                        galex[k,i] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
                    elif i in [2,3,4,5,6]:
                        panstarrs[k,i-2] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
                    else:
                        unwise[k,i-7] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
        
        #these Nones will add a leading batch dimension.
        z = self.z[idx][None,]
        ebv_csfd = self.ebv_csfd[idx][None,]
        ebv_planck = self.ebv_planck[idx][None,]
        zphot_MGS = self.zphot_MGS[idx][None,]
        zphot_WPS = self.zphot_WPS[idx][None,]
        
        if self.transform:
            galex = galex_transform(galex,0.05)
            panstarrs =  new_transform(panstarrs,0.01,8.0)
            unwise = new_transform(unwise,0.1,6.0)
        
        ebvs = np.concatenate([ebv_csfd,ebv_planck]).T

        galex = galex.astype('float32')
        panstarrs = panstarrs.astype('float32')
        unwise = unwise.astype('float32')
        z = z.astype('float32')
        ebvs = ebvs.astype('float32')
        zphot_MGS = zphot_MGS.astype('float32')
        zphot_WPS = zphot_WPS.astype('float32')
        
        if self.to_torch:
            galex = torch.from_numpy(galex)
            panstarrs = torch.from_numpy(panstarrs)
            unwise = torch.from_numpy(unwise)
            z = torch.from_numpy(z).squeeze()
            ebvs = torch.from_numpy(ebvs)
            zphot_MGS = torch.from_numpy(zphot_MGS)
            zphot_WPS = torch.from_numpy(zphot_WPS)
        return galex, panstarrs, unwise, z, ebvs, zphot_MGS, zphot_WPS
    

    def __getitems__(self, idx):
        img_indices = np.array([self.idx_to_imgidx[idx_] for idx_ in idx])
        img = self.img[img_indices]
        if len(img.shape) == 1:
            img = img[None,] #add leading batch dimension
        
        galex = img[:,0:2048]
        panstarrs = img[:,2048:146548]
        unwise = img[:,146548::]
        
        galex = rearrange(galex,'b (f h w) -> b f h w',f=2,h=32,w=32)
        panstarrs = rearrange(panstarrs,'b (f h w) -> b f h w',f=5,h=170,w=170)
        unwise = rearrange(unwise,'b (f h w) -> b f h w',f=2,h=32,w=32)
        
        panstarrs[np.isnan(panstarrs)] = 0.0
        
        if self.sep:
            for k in range(len(galex)):
                for i in range(9):  
                    if i in [0,1]:
                        galex[k,i] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
                    elif i in [2,3,4,5,6]:
                        panstarrs[k,i-2] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
                    else:
                        unwise[k,i-7] = extract_magnitude.keep_center_source(galex[k],panstarrs[k],unwise[k],i,self.set_zero)
        
        #these Nones will add a leading batch dimension.
        z = self.z[idx][None,]
        ebv_csfd = self.ebv_csfd[idx][None,]
        ebv_planck = self.ebv_planck[idx][None,]
        zphot_MGS = self.zphot_MGS[idx][None,]
        zphot_WPS = self.zphot_WPS[idx][None,]
        
        if self.transform:
            #apply arcsinh scaling
            galex = galex_transform(galex,0.05)
            panstarrs =  new_transform(panstarrs,0.01,8.0)
            unwise = new_transform(unwise,0.1,6.0)
        
        ebvs = np.concatenate([ebv_csfd,ebv_planck]).T

        galex = galex.astype('float32')
        panstarrs = panstarrs.astype('float32')
        unwise = unwise.astype('float32')
        z = z.astype('float32')
        ebvs = ebvs.astype('float32')
        zphot_MGS = zphot_MGS.astype('float32')
        zphot_WPS = zphot_WPS.astype('float32')
        
        if self.to_torch:
            galex = torch.from_numpy(galex)
            panstarrs = torch.from_numpy(panstarrs)
            unwise = torch.from_numpy(unwise)
            z = torch.from_numpy(z).squeeze()
            ebvs = torch.from_numpy(ebvs)
            zphot_MGS = torch.from_numpy(zphot_MGS)
            zphot_WPS = torch.from_numpy(zphot_WPS)
        return galex, panstarrs, unwise, z, ebvs, zphot_MGS, zphot_WPS  
                                  
                                  
        
