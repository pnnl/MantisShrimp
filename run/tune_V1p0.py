from utils import get_photopath, my_transform
from augmentation import augment_fn
import models
import datasets
import socket
import torch
import os
import pandas as pd
import numpy as np
import tempfile

from skimage.transform import rotate

from einops import rearrange
from astropy.visualization import make_lupton_rgb

from astropy.visualization import PercentileInterval, AsinhStretch
from scipy.ndimage import zoom

from tqdm import tqdm
import pickle
import random
import argparse
import time
import datetime

import torch.distributed as dist
from torch.utils.data import DataLoader
import training
import torchvision

import ray
from ray import tune, train
from ray.air import session #no Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint, RunConfig
#ray.init()
#os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"]="0"

#I dont know if i can use summaryWriter on distributed?
from torch.utils.tensorboard import SummaryWriter
writer = None

#DDP tutorial
#tutorial #https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide
#tutorial2 #https://leimao.github.io/blog/PyTorch-Distributed-Training/

#Ray Tune tutorial
#https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html#tune-pytorch-cifar-ref

# Environment variables set by torchrun

#we're going to launch our hyperparamter tuning on a single Dl node using DataParallel
#but we inherit some stuff from our DistributedDataParallel script.
WORLD_RANK = 0
LOCAL_RANK = 0 
local_rank = LOCAL_RANK #alias.

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def collate_fn(data):
    return data

class LateFusionModel(torch.nn.Module):
    def __init__(self,model_galex,model_ps,model_unwise,width:int=2048,num_classes:int= 200):
        super().__init__()

        self.model_galex = model_galex
        self.model_ps = model_ps
        self.model_unwise = model_unwise
        

        self.fc0 = torch.nn.Linear(2048 + 512 + 512 + 2, width)
        self.activation_fn = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(width,num_classes)
        
        
    def get_galex_zero(self,device):
        for m in self.model_galex.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        
        b_galex = self.model_galex(torch.zeros((1,2,22,22),device=device))[0][None,:]
        
        for m in self.model_galex.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train()
                m.weight.requires_grad = True
                m.bias.requires_grad = True
        
        return b_galex 
        
    def forward(self,x_galex,x_ps,x_unwise,ebv):
        #because I know 10% of the sample of galex are all zeros, I'm going to measure
        #the deviation of the contribution that learns galex from zero. If (x_galex - 0) = 0
        #then x_galex contributes nothing to the model, and the gradients should also be zero 
        #w.r.t. those samples.
        #b_galex = self.get_galex_zero(x_galex.device)
        x_galex = self.model_galex(x_galex)
        x_ps = self.model_ps(x_ps)
        x_unwise = self.model_unwise(x_unwise)
        
        #now combine and do a pseudo average to make sure the features are well scaled.
        x = torch.concat([x_ps, x_unwise, x_galex],1) #assumes they are all resnet50
        x = torch.cat([x,ebv],1)
        x = self.fc0(x)
        x = self.activation_fn(x)
        x = self.fc1(x)
        
        return x
    

#by making these global variables, I'm testing if they can be shared between all processes spawned by Ray.
#MSD = datasets.MantisShrimpDataset('val',WORLD_RANK,mmap=False)
#MSD_val = datasets.MantisShrimpDataset('val',WORLD_RANK,mmap=False)
#print('Finished Loading Datasets!')

def TRAIN(config):
    model_dir = '/rcfs/projects/mantis_shrimp/mantis_shrimp/MODELS/'
    model_filename = 'model_V1p0_test.pt'
    model_filepath = os.path.join(model_dir, model_filename)
    
    set_random_seeds(random_seed=config['seed'])
   
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']

        
    N_CLASSES=300
    
    #DEFINE MODEL
    LATEFUSION = config['latefusion']
    
    if LATEFUSION:
        model_ps = models.resnet50_encoder(channels=5) #output = 2048
        model_unwise = models.resnet18_encoder(channels=3)
        model_galex = models.resnet18_encoder(channels=3) 
        
        #USE IMAGENET for ResNet18 weights
        wgts_resnet18 = torchvision.models.ResNet18_Weights.DEFAULT.get_state_dict(False)
        model_unwise.load_state_dict(wgts_resnet18,strict=False)
        model_galex.load_state_dict(wgts_resnet18,strict=False)

        model_unwise.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=3, padding=3, bias=False)
        model_galex.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=3, padding=3, bias=False)

        #load PS weights
        ckpt = torch.load('/rcfs/projects/mantis_shrimp/mantis_shrimp/PRETRAINED/resnet50_greyscale_224px.ckpt')



        model_ps.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=3, bias=False)
        old_keys = list(ckpt['state_dict'].keys())
        for key in old_keys:
            if 'encoder.' in key:
                new_key = key.split('encoder.')[1]
                ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(key)
            else:
                continue

        with torch.no_grad(): #we will need to check this actually updates.
            model_ps.load_state_dict(ckpt['state_dict'],strict=False)
            model_ps.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=3, padding=3, bias=False)
            for i in range(5):
                #I think the convolution model sums 
                model_ps.conv1.weight[:,i:i+1,:,:] = ckpt['state_dict']['conv1.weight'].clone()/5
        
        model = LateFusionModel(model_galex,model_ps,model_unwise,width=config['width'],num_classes=N_CLASSES)
        
    else:
        model = models.resnet50()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=3, bias=False)
        model.fc = torch.nn.Linear(2050,2048)
        model.fc2 = torch.nn.Linear(2048,300)

        #load a pre-trained Checkpoint to start training.
        ckpt = torch.load('/rcfs/projects/mantis_shrimp/mantis_shrimp/PRETRAINED/resnet50_greyscale_224px.ckpt')

        old_keys = list(ckpt['state_dict'].keys())
        for key in old_keys:
            if 'encoder.' in key:
                new_key = key.split('encoder.')[1]
                ckpt['state_dict'][new_key] = ckpt['state_dict'].pop(key)
            else:
                continue

        with torch.no_grad(): #we will need to check this actually updates.
            model.load_state_dict(ckpt['state_dict'],strict=False)
            model.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=3, padding=3, bias=False)
            for i in range(9):
                #I think the convolution model sums 
                model.conv1.weight[:,i:i+1,:,:] = ckpt['state_dict']['conv1.weight'].clone()/9
        
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    
    device = "cuda:{}".format(0)
    model = model.to(device) #
    model = torch.nn.DataParallel(model)
    
    #DEFINE OPTIMIZER AND SIMILAR
    opt = torch.optim.NAdam(model.parameters(),
                            lr=config['lr'],
                            weight_decay=config['weight_decay'])
    
    LRscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,verbose=True,min_lr=1e-6,patience=3)

    #DEFINE LOSS FN
    BINS = np.linspace(0,1.6,N_CLASSES+1) #must be the same as CLASS_BINS below
    CLASS_BINS_npy = BINS #alias
    BINS = BINS.astype(np.float32)[0:-1]
    CLASS_BINS = torch.from_numpy(BINS).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    

    #LOAD DATA
    #print('pre-dataloader rank: ',WORLD_RANK)

    #lets try having multiple dataloaders point to the same dataset object, that way maybe they can be shared?
    MSD = datasets.MantisShrimpDataset('train',WORLD_RANK,mmap=True)
    MSD_val = datasets.MantisShrimpDataset('val',WORLD_RANK,mmap=True)
  
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
    
    n_steps_per_epoch = 100
    def make_sampler(target_train):
        class_sample_count_train = np.array(
            [len(np.where(target_train == t)[0]) for t in np.unique(target_train)])
        weight_train = 1. / class_sample_count_train
        samples_weight_train = np.array([weight_train[t] for t in target_train])

        samples_weight_train = torch.from_numpy(samples_weight_train)
        samples_weight_train = samples_weight_train.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight_train,
                                                     num_samples=n_steps_per_epoch*config['batch_size'],
                                                     replacement=True)
        return sampler

    #not clear to me that multiple workers are neccessary since we are using an iterable style dataset.
    target_train = torch.argmin(abs(CLASS_BINS[None,:].cpu()-MSD.z[:,None]),axis=1)
    sampler_train = make_sampler(target_train)  


    trainloader = torch.utils.data.DataLoader(MSD,
                                         batch_size=config['batch_size'],
                                         sampler=sampler_train,
                                         pin_memory=True,
                                         num_workers = 8,
                                         prefetch_factor=1,
                                         persistent_workers=True,
                                         collate_fn=collate_fn,)


    val_sampler = torch.utils.data.RandomSampler(MSD_val,num_samples=100*config['batch_size'])
    valloader = torch.utils.data.DataLoader(MSD_val,
                                         batch_size=config['batch_size'],
                                         sampler=val_sampler,
                                         pin_memory=True,
                                         num_workers = 8,
                                         prefetch_factor=1,
                                         collate_fn=collate_fn,)
    

    
    
    #START MAIN TRAINING LOOP
    all_train_losses = []
    all_val_losses = []
    all_val_metrics = []
    lowest_val_loss = 1000 #some ridiculously high value.
    N_EPOCHS = 500

    if WORLD_RANK==0:
        writer = SummaryWriter(log_dir='/rcfs/projects/mantis_shrimp/mantis_shrimp/tensorboardlog')
    
    if WORLD_RANK==0:
        print('Starting Main Training Loop')
    for epoch in range(0,N_EPOCHS+1):
        #epoch,model,trainloader,loss_fn,scaler,opt,CLASS_BINS,device,WORLD_RANK,LATEFUSION=True,USE_AMP=False
        
        train_loss = training.train_epoch(epoch,
                                          model,
                                          opt,
                                          trainloader,
                                          loss_fn,
                                          CLASS_BINS,
                                          LATEFUSION,
                                          WORLD_RANK,
                                          device)
        #TODO: should use torch distributed to find the average val_loss.
        #(epoch,model,valloader,loss_fn,opt,CLASS_BINS,WORLD_RANK,device,LATEFUSION=True)
        val_loss, MAD, BIAS, ETA = training.val_epoch(epoch,
                                                      model,
                                                      opt,
                                                      valloader,
                                                      loss_fn,
                                                      CLASS_BINS,
                                                      LATEFUSION,
                                                      WORLD_RANK,
                                                      device)

        #we need to communicate the value of loss, MAD, BIAS, and eta to each rank0 
        #val_loss = torch.tensor([val_loss]).to(device)
        #dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        val_loss = val_loss.detach().cpu().item()

        #train_loss = torch.tensor([train_loss]).to(device)
        #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        train_loss = train_loss.detach().cpu().item()

        #MAD = torch.tensor([MAD]).to(device)
        #dist.all_reduce(MAD, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        MAD = MAD.detach().cpu().item()

        #BIAS = torch.tensor([BIAS]).to(device)
        #dist.all_reduce(BIAS, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        BIAS = BIAS.detach().cpu().item()

        #ETA = torch.tensor([ETA]).to(device)
        #dist.all_reduce(ETA, op=dist.ReduceOp.AVG)
        #extract val loss back to float to synchronize the scheduler.
        ETA = ETA.detach().cpu().item()
        
        LRscheduler.step(val_loss)
            
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), opt.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"loss": (val_loss), "MAD": (MAD), "Bias":(BIAS), "Eta":(ETA)},checkpoint=checkpoint)


if __name__ == "__main__":
    
    config = {
    "width": tune.sample_from(lambda _: 2**np.random.randint(8, 12)),
    "lr": tune.loguniform(1e-6, 1e-2),
    "batch_size": tune.choice([16, 16*2, 16*4, 16*8, 16*16]),
    "seed": tune.choice(list(range(2))), 
    "weight_decay": tune.loguniform(1e-7, 1e-2),
    "latefusion": tune.choice([True,])
       }
    

    scheduler = ASHAScheduler(
        max_t=100, #about
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(TRAIN), #for prototype...
            resources={"cpu": 8, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=30,
        ),
        run_config=RunConfig(storage_path="/rcfs/projects/mantis_shrimp/mantis_shrimp/rayresults", name="test_experiment"),
        param_space=config,
    )
    results = tuner.fit()
    Result = results.get_best_result()
    print('Best Result: ',Result)
                          
    
