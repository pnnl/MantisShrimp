from tqdm import tqdm
import numpy as np
import torch
from mantis_shrimp.augmentation import augment_fn, augment_fn_latefusion
from collections import deque

def train_epoch(epoch,model,opt,loader,loss_fn,CLASS_BINS,LATEFUSION,WORLD_RANK,device,use_amp,scaler):
    losses = []
    moving_average_loss = deque([])
    model.train()
    with tqdm(loader,disable=(WORLD_RANK!=0)) as tepoch:
        for i,(x_galex,x_ps,x_unwise,y,ebvs,__,__) in enumerate(tepoch):

            opt.zero_grad(set_to_none=True)            

            #much faster if transforms are done on GPU.
            x_galex = x_galex.to(device)
            x_ps = x_ps.to(device)
            x_unwise = x_unwise.to(device)
            y = y.to(device)
            ebvs = ebvs.to(device)
            
            x_ps[torch.isnan(x_ps)] = 0.0

            x_galex = x_galex[:,0,]
            x_ps = x_ps[:,0,]
            x_unwise = x_unwise[:,0,]
                
            #Take care of transforms as part of augmentation
            if LATEFUSION:
                x_galex, x_ps, x_unwise = augment_fn_latefusion(x_galex,x_ps,x_unwise)
            else:
                x_galex, x_ps, x_unwise = augment_fn(x_galex,x_ps,x_unwise)           
            
            if y.size() == torch.Size([]): #this shouldn't ever trigger using my normal pipeline
                continue
            
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                if LATEFUSION:
                    y_hat = model(x_galex,x_ps,x_unwise,ebvs)
                else:
                    x = torch.concatenate([x_galex,x_ps,x_unwise],1)
                    y_hat = model(x,ebvs)
                y_hat_log = torch.nn.functional.log_softmax(y_hat,dim=-1)
                Q = torch.argmin(abs(CLASS_BINS[None,:]-y[:]),dim=1)
                loss = loss_fn(y_hat_log,Q) 
            
            scaler.scale(loss).backward()
            scaler.step(opt)
    
            # Updates the scale for next iteration.
            scaler.update()
    
            opt.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance

            with torch.no_grad():
                point_z = torch.sum(CLASS_BINS * torch.nn.functional.softmax(y_hat,dim=-1),-1)
                residuals_scaled = (point_z.squeeze() - y.squeeze())/(1+y.squeeze())
                MAD = 1.4826*torch.median(abs(residuals_scaled - torch.median(residuals_scaled)))
                BIAS = torch.abs(torch.mean(residuals_scaled))
                ETA = torch.sum(abs(residuals_scaled)>=0.15)/len(residuals_scaled)

            losses.append(loss.item())
            moving_average_loss.append(loss.item())
            if len(moving_average_loss)>50:
                moving_average_loss.popleft()
            tepoch.set_postfix(Epoch=epoch,loss=loss.item(),avgloss=np.mean(moving_average_loss),)
            if i >= 1000:
                break
    return torch.mean(torch.tensor(losses))

def val_epoch(epoch,model,opt,loader,loss_fn,CLASS_BINS,LATEFUSION,WORLD_RANK,device,use_amp,scaler):
    losses = []
    moving_average_loss = deque([])
    residuals_scaled_all = []
    model.train(False)
    with torch.no_grad():
        with tqdm(loader,disable=(WORLD_RANK!=0)) as tepoch:
            for i,(x_galex,x_ps,x_unwise,y,ebvs,__,__) in enumerate(tepoch):

                #much faster if transforms are done on GPU.
                x_galex = x_galex.to(device)
                x_ps = x_ps.to(device)
                x_unwise = x_unwise.to(device)
                y = y.to(device)
                ebvs = ebvs.to(device)
                #ebv should be shape [batch_size, 2]

                
                x_ps[torch.isnan(x_ps)] = 0.0

                x_galex = x_galex[:,0,]
                x_ps = x_ps[:,0,]
                x_unwise = x_unwise[:,0,]
               
                if y.size() == torch.Size([]):
                    continue

                #Take care of transforms as part of augmentation
                if LATEFUSION:
                    x_galex, x_ps, x_unwise = augment_fn_latefusion(x_galex,x_ps,x_unwise)
                else:
                    x_galex, x_ps, x_unwise = augment_fn(x_galex,x_ps,x_unwise) 
            
                if LATEFUSION:
                    y_hat = model(x_galex,x_ps,x_unwise,ebvs)
                else:
                    x = torch.concatenate([x_galex,x_ps,x_unwise],1)
                    y_hat = model(x,ebvs)
                    
                y_hat_log = torch.nn.functional.log_softmax(y_hat,dim=-1)

                Q = torch.argmin(abs(CLASS_BINS[None,:]-y),dim=1)

                loss = loss_fn(y_hat_log,Q) #uncomment to use cross entropy
                #loss = CRPS(y_hat,y) #uncomment to use CRPS loss
                
                
                point_z = torch.sum(CLASS_BINS * torch.nn.functional.softmax(y_hat,dim=-1),-1)
                residuals_scaled = (point_z.squeeze() - y.squeeze())/(1+y.squeeze())
                residuals_scaled_all.append(residuals_scaled)

                losses.append(loss.item())
                moving_average_loss.append(loss.item())
                
                if len(moving_average_loss)>50:
                    moving_average_loss.popleft()
                    
                tepoch.set_postfix(Epoch=epoch,valavgloss=np.mean(moving_average_loss))
        model.train()
        residuals_scaled = torch.cat(residuals_scaled_all)
        MAD = 1.4826*torch.median(abs(residuals_scaled - torch.median(residuals_scaled)))
        BIAS = torch.mean(residuals_scaled)
        ETA = torch.sum(abs(residuals_scaled)>=0.15)/len(residuals_scaled)
        return torch.mean(torch.tensor(losses)), MAD, BIAS, ETA

def train_epoch_ablate(epoch,model,opt,loader,loss_fn,CLASS_BINS,LATEFUSION,WORLD_RANK,device,use_amp,galex,unwise,scaler):
    losses = []
    moving_average_loss = deque([])
    model.train()
    with tqdm(loader,disable=(WORLD_RANK!=0)) as tepoch:
        for i,(x_galex,x_ps,x_unwise,y,ebvs,__,__) in enumerate(tepoch):

            opt.zero_grad(set_to_none=True)            

            #much faster if transforms are done on GPU.
            x_galex = x_galex.to(device)
            x_ps = x_ps.to(device)
            x_unwise = x_unwise.to(device)
            y = y.to(device)
            ebvs = ebvs.to(device)
            
            x_ps[torch.isnan(x_ps)] = 0.0

            x_galex = x_galex[:,0,]
            x_ps = x_ps[:,0,]
            x_unwise = x_unwise[:,0,]
                
            #Take care of transforms as part of augmentation
            if LATEFUSION:
                x_galex, x_ps, x_unwise = augment_fn_latefusion(x_galex,x_ps,x_unwise)
            else:
                x_galex, x_ps, x_unwise = augment_fn(x_galex,x_ps,x_unwise)           
            
            if y.size() == torch.Size([]): #this shouldn't ever trigger using my normal pipeline
                continue
            
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                if LATEFUSION:
                    y_hat = model(x_galex,x_ps,x_unwise,ebvs)
                else:
                    if galex and unwise:
                        x = torch.concatenate([x_galex,x_ps,x_unwise],1)
                    elif galex and not(unwise):
                        x = torch.concatenate([x_galex,x_ps],1)
                    elif not(galex) and unwise:
                        x = torch.concatenate([x_ps,x_unwise],1)
                    elif not(galex) and not(unwise):
                        x = x_ps
                        
                    y_hat = model(x,ebvs)
                y_hat_log = torch.nn.functional.log_softmax(y_hat,dim=-1)
                Q = torch.argmin(abs(CLASS_BINS[None,:]-y[:]),dim=1)
                loss = loss_fn(y_hat_log,Q) 
            
            scaler.scale(loss).backward()
            scaler.step(opt)
    
            # Updates the scale for next iteration.
            scaler.update()
    
            opt.zero_grad(set_to_none=True) # set_to_none=True here can modestly improve performance

            with torch.no_grad():
                point_z = torch.sum(CLASS_BINS * torch.nn.functional.softmax(y_hat,dim=-1),-1)
                residuals_scaled = (point_z.squeeze() - y.squeeze())/(1+y.squeeze())
                MAD = 1.4826*torch.median(abs(residuals_scaled - torch.median(residuals_scaled)))
                BIAS = torch.abs(torch.mean(residuals_scaled))
                ETA = torch.sum(abs(residuals_scaled)>=0.15)/len(residuals_scaled)

            losses.append(loss.item())
            moving_average_loss.append(loss.item())
            if len(moving_average_loss)>50:
                moving_average_loss.popleft()
            tepoch.set_postfix(Epoch=epoch,loss=loss.item(),avgloss=np.mean(moving_average_loss),)
            if i >= 1000:
                break
    return torch.mean(torch.tensor(losses))

def val_epoch_ablate(epoch,model,opt,loader,loss_fn,CLASS_BINS,LATEFUSION,WORLD_RANK,device,galex,unwise,use_amp,scaler):
    losses = []
    moving_average_loss = deque([])
    residuals_scaled_all = []
    model.train(False)
    with torch.no_grad():
        with tqdm(loader,disable=(WORLD_RANK!=0)) as tepoch:
            for i,(x_galex,x_ps,x_unwise,y,ebvs,__,__) in enumerate(tepoch):

                #much faster if transforms are done on GPU.
                x_galex = x_galex.to(device)
                x_ps = x_ps.to(device)
                x_unwise = x_unwise.to(device)
                y = y.to(device)
                ebvs = ebvs.to(device)
                #ebv should be shape [batch_size, 2]

                
                x_ps[torch.isnan(x_ps)] = 0.0

                x_galex = x_galex[:,0,]
                x_ps = x_ps[:,0,]
                x_unwise = x_unwise[:,0,]
               
                if y.size() == torch.Size([]):
                    continue

                #Take care of transforms as part of augmentation
                if LATEFUSION:
                    x_galex, x_ps, x_unwise = augment_fn_latefusion(x_galex,x_ps,x_unwise)
                else:
                     x_galex, x_ps, x_unwise = augment_fn(x_galex,x_ps,x_unwise)
            
                if LATEFUSION:
                    y_hat = model(x_galex,x_ps,x_unwise,ebvs)
                else:
                    if galex and unwise:
                        x = torch.concatenate([x_galex,x_ps,x_unwise],1)
                    elif galex and not(unwise):
                        x = torch.concatenate([x_galex,x_ps],1)
                    elif not(galex) and unwise:
                        x = torch.concatenate([x_ps,x_unwise],1)
                    elif not(galex) and not(unwise):
                        x = x_ps

                    y_hat = model(x,ebvs)
                    
                y_hat_log = torch.nn.functional.log_softmax(y_hat,dim=-1)

                Q = torch.argmin(abs(CLASS_BINS[None,:]-y),dim=1)

                loss = loss_fn(y_hat_log,Q) #uncomment to use cross entropy
                #loss = CRPS(y_hat,y) #uncomment to use CRPS loss
                
                
                point_z = torch.sum(CLASS_BINS * torch.nn.functional.softmax(y_hat,dim=-1),-1)
                residuals_scaled = (point_z.squeeze() - y.squeeze())/(1+y.squeeze())
                residuals_scaled_all.append(residuals_scaled)

                losses.append(loss.item())
                moving_average_loss.append(loss.item())
                
                if len(moving_average_loss)>50:
                    moving_average_loss.popleft()
                    
                tepoch.set_postfix(Epoch=epoch,valavgloss=np.mean(moving_average_loss))
        model.train()
        residuals_scaled = torch.cat(residuals_scaled_all)
        MAD = 1.4826*torch.median(abs(residuals_scaled - torch.median(residuals_scaled)))
        BIAS = torch.mean(residuals_scaled)
        ETA = torch.sum(abs(residuals_scaled)>=0.15)/len(residuals_scaled)
        return torch.mean(torch.tensor(losses)), MAD, BIAS, ETA
   
