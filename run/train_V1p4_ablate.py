from mantis_shrimp.utils import get_photopath, my_transform
from mantis_shrimp.datasets import MantisShrimpDataset
from mantis_shrimp.augmentation import augment_fn
from mantis_shrimp import models
from mantis_shrimp import datasets
from mantis_shrimp.ffcv_loader import WeightedRandomOrder
import socket
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ffcv.transforms import ToTensor, ToDevice
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder

from skimage.transform import rotate

from einops import rearrange
from astropy.visualization import make_lupton_rgb

from astropy.visualization import PercentileInterval, AsinhStretch
from scipy.ndimage import zoom

import gc #standard library garbage collector

from tqdm import tqdm
import pickle
import random
import argparse
import time
import datetime

import torch.distributed as dist
from torch.utils.data import DataLoader
from mantis_shrimp.training import train_epoch_ablate, val_epoch_ablate
import uuid

#I dont know if i can use summaryWriter on distributed?
from torch.utils.tensorboard import SummaryWriter
writer = None

import timm

#Just send it; this will fallback to eager mode
import torch._dynamo
torch._dynamo.config.suppress_errors = True

#tutorial #https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
#tutorial2 #https://leimao.github.io/blog/PyTorch-Distributed-Training/

# Environment variables set by torchrun

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
local_rank = LOCAL_RANK #alias.

class LateFusionModel(torch.nn.Module):
    def __init__(self,model_galex,model_ps,model_unwise,width:int=2048, num_classes:int= 200, USE_GALEX=True, USE_WISE=True):
        super().__init__()

        self.USE_GALEX = USE_GALEX
        self.USE_WISE = USE_WISE
        
        if self.USE_GALEX:
            self.model_galex = model_galex
        if self.USE_WISE:
            self.model_unwise = model_unwise   
            
        self.model_ps = model_ps

        if self.USE_GALEX and self.USE_WISE:
            self.fc0 = torch.nn.Linear(1532 + 770 + 770 + 2, width)
        elif self.USE_GALEX and not(self.USE_WISE):
            self.fc0 = torch.nn.Linear(1532 + 770 + 4, width)
        elif not(self.USE_GALEX) and self.USE_WISE:
            self.fc0 = torch.nn.Linear(1532 + 770 + 4, width)
        else:
            self.fc0 = torch.nn.Linear(1532+4, width)
        
        self.activation_fn = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(width,num_classes)
        
    def forward(self,x_galex,x_ps,x_unwise,ebv):
        #because I know 10% of the sample of galex are all zeros, I'm going to measure
        #the deviation of the contribution that learns galex from zero. If (x_galex - 0) = 0
        #then x_galex contributes nothing to the model, and the gradients should also be zero 
        #w.r.t. those samples.
        #b_galex = self.get_galex_zero(x_galex.device)

        if self.USE_GALEX:
            x_galex = self.model_galex(x_galex)
        if self.USE_WISE:
            x_unwise = self.model_unwise(x_unwise)
        x_ps = self.model_ps(x_ps)
        
        
        #now combine and do a pseudo average to make sure the features are well scaled.
        if self.USE_GALEX and self.USE_WISE:
            x = torch.concat([x_ps, x_unwise, x_galex],1) #assumes they are all resnet50
        elif self.USE_GALEX and not(self.USE_WISE):
            x = torch.concat([x_ps, x_galex],1)
        elif not(self.USE_GALEX) and self.USE_WISE:
            x = torch.concat([x_ps, x_unwise],1)
        else:
            x = x_ps
        
        x = self.activation_fn(x)
        x = torch.cat([x,ebv],1) #don't want to zero my ebv vector.
        x = self.fc0(x)
        x = self.activation_fn(x)
        x = self.fc1(x)
        
        return x

class ConvModel(torch.nn.Module):
    def __init__(self,Base,num_classes:int=300,hidden_width=2048):
        super().__init__()
        
        #642 if using nano
        #770 if using small
        self.fc0 = torch.nn.Linear(1538,hidden_width)
        self.fc1 = torch.nn.Linear(hidden_width,num_classes)
        self.activation = torch.nn.functional.gelu
        self.Base = Base
    
    def forward(self,x,ebv):
        x = self.Base(x)
        x = torch.cat([x,ebv],axis=1)
        x = self.fc0(x)
        x = self.activation(x)
        x = self.fc1(x)
        return x

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def collate_fn(data):
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="nccl", choices=['nccl', 'gloo'])
    parser.add_argument("--fusion_type", type=str, default="early", choices=['late','early'])
    parser.add_argument("--n_classes",type=int,default=300,help="number of classes, sets bin-width of redshift")
    parser.add_argument("--z_max",type=float,default=1.6,help="maximum redshift, will reject samples with target greater than value")
    parser.add_argument("--n_epochs",type=int,default=60)
    parser.add_argument("--use_amp",type=bool,default=False)
    parser.add_argument("--batchsize",type=int,default=16)
    parser.add_argument("--hidden_width",type=int,default=2048)
    parser.add_argument("--initial_lr",type=float,default=1e-3)
    parser.add_argument("--weight_decay",type=float,default=6.6e-6)
    parser.add_argument("--random_seed",type=int,default=0)
    parser.add_argument("--unwise",type=bool,default=False,help='If present, keep Wise')
    parser.add_argument("--galex",type=bool,default=False,help='If present, keep Galex')
    args = parser.parse_args()
    
    model_dir = '/rcfs/projects/mantis_shrimp/mantis_shrimp/MODELS/'
    UUID = str(uuid.uuid4()) #'c7d3ec57-f796-4583-8b74-85aee12fcd29' #
    print(UUID)
    model_filename = UUID+'model_V1p4.pt'
    #861ccf49-4900-4fd5-b3b6-75dd7a902678model_V1p3epoch_26.pt
    model_filepath = os.path.join(model_dir, model_filename)
    
    set_random_seeds(random_seed=args.random_seed)
   
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    #print('before: ',socket.gethostname(),' visible devices: ',visible_devices)
    dist.init_process_group(
        backend="nccl",timeout=datetime.timedelta(seconds=3600))
        #init_method='file:///rcfs/projects/mantis_shrimp/mantis_shrimp/torch_rdzv/')
        
    N_CLASSES=args.n_classes
    
    LATEFUSION = (args.fusion_type=='late')
    ###Uncomment to train from initialization
    #DEFINE MODEL
    if LATEFUSION:
        BaseModel_ps = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_large', pretrained=True, num_classes=0)
        BaseModel_galex = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_small', pretrained=True, num_classes=0)
        BaseModel_unwise = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_small', pretrained=True, num_classes=0)
        
        #PS
        first_layer_bias = BaseModel_ps.stem[0].bias.clone()
        first_layer_weight = BaseModel_ps.stem[0].weight.clone()
        first_layer_weight = (torch.mean(first_layer_weight,dim=1)[:,None,:,:]).repeat(1,5,1,1)
        BaseModel_ps.stem[0] = torch.nn.Conv2d(5,192,kernel_size=(4,4), stride=(4,4))
        BaseModel_ps.stem[0].bias.data = first_layer_bias
        BaseModel_ps.stem[0].weight.data = first_layer_weight
        
        #Galex
        first_layer_bias = BaseModel_galex.stem[0].bias.clone()
        first_layer_weight = BaseModel_galex.stem[0].weight.clone()
        first_layer_weight = (torch.mean(first_layer_weight,dim=1)[:,None,:,:]).repeat(1,2,1,1)
        BaseModel_galex.stem[0] = torch.nn.Conv2d(2,96,kernel_size=(4,4), stride=(4,4))
        BaseModel_galex.stem[0].bias.data = first_layer_bias
        BaseModel_galex.stem[0].weight.data = first_layer_weight
        
        #UnWISE
        first_layer_bias = BaseModel_unwise.stem[0].bias.clone()
        first_layer_weight = BaseModel_unwise.stem[0].weight.clone()
        first_layer_weight = (torch.mean(first_layer_weight,dim=1)[:,None,:,:]).repeat(1,2,1,1)
        BaseModel_unwise.stem[0] = torch.nn.Conv2d(2,96,kernel_size=(4,4), stride=(4,4))
        BaseModel_unwise.stem[0].bias.data = first_layer_bias
        BaseModel_unwise.stem[0].weight.data = first_layer_weight

        model = LateFusionModel(BaseModel_galex,BaseModel_ps,BaseModel_unwise,num_classes=N_CLASSES,USE_GALEX=args.galex,USE_WISE=args.unwise)
    else:
        BaseModel = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_large', pretrained=True, num_classes=0)
        
        first_layer_bias = BaseModel.stem[0].bias.clone()
        first_layer_weight = BaseModel.stem[0].weight.clone()
        first_layer_weight = (torch.mean(first_layer_weight,dim=1)[:,None,:,:]).repeat(1,9,1,1)

        num_channels = 5
        if args.unwise:
            num_channels+=2
        if args.galex:
            num_channels+=2

        first_layer_weight = (torch.mean(first_layer_weight,dim=1)[:,None,:,:]).repeat(1,num_channels,1,1)
        BaseModel.stem[0] = torch.nn.Conv2d(num_channels,192,kernel_size=(4,4), stride=(4,4))
        BaseModel.stem[0].bias.data = first_layer_bias
        BaseModel.stem[0].weight.data = first_layer_weight
        
        model = ConvModel(BaseModel,num_classes=args.n_classes)
    
    ###Uncomment to use model from previous training run.
#     model = models.resnet50()
#     model.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=3, padding=3, bias=False)
#     model.fc = torch.nn.Linear(2050,2048)
#     model.fc2 = torch.nn.Linear(2048,args.n_classes)

    #XXX here
    #best_filename = '861ccf49-4900-4fd5-b3b6-75dd7a902678model_V1p3epoch_28.pt'
    #best_filepath = os.path.join(model_dir, best_filename)
    #ckpt = torch.load(best_filepath)
    #model.load_state_dict(ckpt['state_dict'],strict=False)
    
    #ckpt = torch.load(best_filepath)
    #old_keys = list(ckpt.keys())
    #for key in old_keys:
    #    if 'module.' in key:
    #        new_key = key.split('module.')[1]
    #        ckpt[new_key] = ckpt.pop(key)
    #    else:
    #        continue
    
    #model.load_state_dict(ckpt,strict=True)    

    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    print('Visible device: ',visible_devices)
    
    #XXX
    device = "cuda:{}".format(dist.get_rank()%torch.cuda.device_count())
    model = model.to(device) #
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=device)
    
    model = torch.compile(model) #

    #DEFINE OPTIMIZER AND SIMILAR
    opt = torch.optim.NAdam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,verbose=True,min_lr=1e-6,patience=5)
    
    #XXX HERE
    #opt.load_state_dict(ckpt['opt_state'])
    #scheduler = ckpt['lr_scheduler_obj']
    
    #possible I need to delete ckpt bc I think its holding onto a copy of objects on GPUs?
    #del ckpt
    #gc.collect()

    torch.cuda.empty_cache()
                          
    #DEFINE LOSS FN
    BINS = np.linspace(0,args.z_max,N_CLASSES+1) #must be the same as CLASS_BINS below
    CLASS_BINS_npy = BINS #alias
    BINS = BINS.astype(np.float32)[0:-1]
    CLASS_BINS = torch.from_numpy(BINS).to(device)
    
    #we tried label smoothing with 0.1, was awful. abandoning.
    loss_fn = torch.nn.CrossEntropyLoss()
    
    now = time.time()
    #LOAD DATA
    print('pre-dataloader rank: ',WORLD_RANK)
    MSD = datasets.MantisShrimpDataset(kind='train',WORLD_RANK=WORLD_RANK,ZMAX=args.z_max,loc='vast')

    MSD_val = datasets.MantisShrimpDataset(kind='val',WORLD_RANK=WORLD_RANK,ZMAX=args.z_max,loc='vast')
    #print('past dataloader, time taken to load: ',(time.time()-now)/60,' min')    
    #okay so here is where we are going to be uncertain. 
    #MSD is local to each process. It potentially has differnt len on each node.
    #b/c its different on each node, we don't need to use DistributedSampler (b/c we're not
    #divying up a fixed dataset).
    
    #because MSD has a different length on each process, we will need
    #to communicate the number of batches in epoch by using dist.messaging protocols.
    
    #this assumes that len(MSD) has already been cut down to just training data.
    #length_MSD = torch.tensor([len(MSD)]).to(device) #we need to place this in a tensor to use distributed communication
    #dist.all_reduce(length_MSD, op=dist.ReduceOp.MIN,)#this operates in-place on length_MSD
    #meaning that length_MSD is now the minimum length of all MSDs
    #n_steps_per_epoch = int(length_MSD // args.batchsize)
    
    #n_steps_per_epoch = 1000#for some reason len is 2* this value
    def make_sampler(target_train):
        class_sample_count_train = np.array(
            [len(np.where(target_train == t)[0]) for t in np.unique(target_train)])
        weight_train = 1. / class_sample_count_train
        samples_weight_train = np.array([weight_train[t] for t in target_train])
    
        #samples_weight_train = torch.from_numpy(samples_weight_train)
        #samples_weight_train = samples_weight_train#.double()
        
        samples_weight_train = samples_weight_train / np.sum(samples_weight_train)
        sampler = WeightedRandomOrder(samples_weight_train,replacement=True)
        return sampler


    #not clear to me that multiple workers are neccessary since we are using an iterable style dataset.
    target_train = torch.argmin(abs(CLASS_BINS[None,:].cpu()-MSD.z[:,None]),axis=1)
    sampler_order = make_sampler(target_train) 

    PIPELINES = {
      'galex': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'panstarrs': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'unwise': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'z' : [FloatDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'ebvs' : [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'zphot_MGS': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
      'zphot_WPS': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda'), non_blocking=True)],
    }
    
    
    ORDERING = sampler_order #OrderOption.QUASI_RANDOM # #
    
    BATCH_SIZE = args.batchsize
    
    NUM_WORKERS = 8
    
    trainloader = Loader(f'/rcfs/projects/mantis_shrimp/Adam/mantis_shrimp_train_{WORLD_RANK}.beton',
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                order=ORDERING,
                pipelines=PIPELINES)
    
    valloader = Loader(f'/rcfs/projects/mantis_shrimp/Adam/mantis_shrimp_val_{WORLD_RANK}.beton',
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                order=OrderOption.SEQUENTIAL,
                pipelines=PIPELINES)

    
    
    #START MAIN TRAINING LOOP
    all_train_losses = []
    all_val_losses = []
    all_val_metrics = []
    lowest_val_loss = 1000 #some ridiculously high value.
    N_EPOCHS = args.n_epochs

    if WORLD_RANK==0:
        writer = SummaryWriter(log_dir='/rcfs/projects/mantis_shrimp/mantis_shrimp/tensorboardlog3/'+UUID)
    
    if WORLD_RANK==0:
        print('Starting Main Training Loop')
        
    #XXX
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    torch.set_float32_matmul_precision('high')
    
    for epoch in range(1,N_EPOCHS+1):
        #epoch,model,trainloader,loss_fn,scaler,opt,CLASS_BINS,device,WORLD_RANK,LATEFUSION=True,USE_AMP=False
        
        train_loss = train_epoch_ablate(epoch,
                                  model,
                                  opt,
                                  trainloader,
                                  loss_fn,
                                  CLASS_BINS,
                                  LATEFUSION,
                                  WORLD_RANK,
                                  device,
                                  use_amp=args.use_amp,
                                  galex=args.galex, 
                                  unwise=args.unwise,
                                  scaler=scaler)
      
        if np.isnan(train_loss):
            raise Exception('training went NaN')
        #TODO: should use torch distributed to find the average val_loss.
        #(epoch,model,valloader,loss_fn,opt,CLASS_BINS,WORLD_RANK,device,LATEFUSION=True)
        val_loss, MAD, BIAS, ETA = val_epoch_ablate(epoch,
                                              model,
                                              opt,
                                              valloader,
                                              loss_fn,
                                              CLASS_BINS,
                                              LATEFUSION,
                                              WORLD_RANK,
                                              device,
                                              use_amp=args.use_amp,
                                              galex = args.galex,
                                              unwise = args.unwise,
                                              scaler=scaler)

        #we need to communicate the value of loss, MAD, BIAS, and eta to each rank0 
        val_loss = torch.tensor([val_loss]).to(device)
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        val_loss = val_loss.detach().cpu().item()

        train_loss = torch.tensor([train_loss]).to(device)
        dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        train_loss = train_loss.detach().cpu().item()

        MAD = torch.tensor([MAD]).to(device)
        dist.all_reduce(MAD, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        MAD = MAD.detach().cpu().item()

        BIAS = torch.tensor([BIAS]).to(device)
        dist.all_reduce(BIAS, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        BIAS = BIAS.detach().cpu().item()

        ETA = torch.tensor([ETA]).to(device)
        dist.all_reduce(ETA, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        ETA = ETA.detach().cpu().item()

        all_val_losses.append(val_loss)
        all_train_losses.append(train_loss)
        all_val_metrics.append(np.array([MAD,BIAS,ETA]))
        
        scheduler.step(val_loss)
        
        if WORLD_RANK==0:
            print('VAL metrics:  TrainLoss={:.3f} ValLoss={:.3f}, MAD={:.4f}, Bias={:.4f}, Eta={:.3f}'.format(train_loss,
                                                                                                              val_loss,
                                                                                                              MAD,
                                                                                                              BIAS,
                                                                                                              ETA))
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Metrics/MAD', MAD, epoch)
            writer.add_scalar('Metrics/BIAS', BIAS, epoch)
            writer.add_scalar('Metrics/ETA', ETA, epoch)

        if WORLD_RANK==0 and val_loss < lowest_val_loss:
            print('Saving')
            ckpt = {}
            ckpt['state_dict'] = model.state_dict()
            ckpt['opt_state'] = opt.state_dict()
            ckpt['lr_scheduler_obj'] = scheduler 
            torch.save(ckpt,model_filepath.split('.pt')[0]+f'best.pt')
            lowest_val_loss = val_loss
        
        if WORLD_RANK==0:
            torch.save(model.state_dict(),model_filepath.split('.pt')[0]+f'epoch_{epoch}.pt')    


    if WORLD_RANK==0:
        torch.save(model.state_dict(),model_filepath.split('.pt')[0]+'_end.pt')

        writer.add_hparams({"lr":args.initial_lr,
                           "hidden_width": args.hidden_width,
                           "batch_size": args.batchsize,
                           "N_CLASSES": args.n_classes,
                           "weight_decay": args.weight_decay,
                           "random_seed": args.random_seed,
                           "fusion": args.fusion_type},
                           metric_dict={'val_loss':lowest_val_loss}
                            )
