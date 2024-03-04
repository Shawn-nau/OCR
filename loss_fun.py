import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys


class QuantileLoss(nn.Module):
    def __init__(self, args,device):
        super(QuantileLoss, self).__init__()
        self.quantile = args.cs/(args.cs+args.ch)

    def forward(self, x,x1, target):
        errors = target[:,0] + target[:,1] - x[:,0]
        loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(loss)

class QuantileLoss_omni(nn.Module):
    def __init__(self, args,device):
        super(QuantileLoss_omni, self).__init__()
        self.quantile = args.cs/(args.cs+args.ch)

    def forward(self, x_k,x_m, target):
        errors_k = target[:,0] - x_k
        errors_m = target[:,1] - x_m
        loss_k = torch.max((self.quantile - 1) * errors_k, self.quantile * errors_k)
        loss_m = torch.max((self.quantile - 1) * errors_m, self.quantile * errors_m)
        return torch.mean(loss_k+loss_m)

class Omni_cost_loss(nn.Module):
    
    def __init__(self,args,device) -> None:
        super(Omni_cost_loss, self).__init__()
        self.cs = torch.FloatTensor([args.cs]).to(device)
        self.ch = torch.FloatTensor([args.ch]).to(device)
        self.cm2k = torch.FloatTensor([args.cm2k]).to(device)
        self.ck2m = torch.FloatTensor([args.ck2m]).to(device)
        pass
    
    
    def cost_func(self, x_k, x_m, y_k, y_m, p):
        cs = self.cs * p   # unit shortage cost
        ch = self.ch * p    # unit holding cost
        ck2m = self.ck2m    # unit transfer cost, from k to m
        cm2k = self.cm2k 
        
        
        lack_m = torch.max(y_m - x_m, torch.zeros_like(y_m))
        lack_k = torch.max(y_k - x_k, torch.zeros_like(y_k))
        over_k = torch.max(-y_k + x_k, torch.zeros_like(y_k))
        over_m = torch.max(-y_m + x_m, torch.zeros_like(y_m))
        
        sales_k = torch.min(y_k + lack_m,x_k)
        sales_m = torch.min(y_m + lack_k,x_m)
        
        
        shortage = cs * torch.max(y_k +y_m - sales_k -sales_m,torch.zeros_like(y_k) )
        
        holding_k  = ch * torch.max(x_k-y_k-lack_m, torch.zeros_like(y_k))
        picking = ck2m * torch.min(over_k,lack_m)
        
        
        holding_m = ch * torch.max(x_m-y_m-lack_k, torch.zeros_like(y_m))
        moving = cm2k * torch.min(over_m,lack_k)              
  
  
        cost = holding_k + shortage + picking + holding_m + moving

        return cost.mean() 
   
    def forward(self, pred_k, pred_m,real):
        # Calculate the cost using vectorized operations
        if pred_k.dim() ==1:
            pred_k = pred_k.unsqueeze(1)
            pred_m = pred_m.unsqueeze(1)
        total_cost = self.cost_func(pred_k, pred_m, real[:, 0:1], real[:, 1:2], real[:, 2:3])

        # Sum the costs along the batch dimension
        return total_cost     

class Likelyhood(nn.Module):
    def __init__(self,args,device):
        super(Likelyhood, self).__init__()

    def forward(self, r,p, target):
        
        dist_k = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,0:1],logits = p[:,0:1])   
        dist_m = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,1:2],logits = p[:,1:2]) 
    
        logprob_k = dist_k.log_prob(target[:,0:1]) 
        logprob_m = dist_m.log_prob(target[:,1:2])    
           

        return -torch.mean(logprob_k+logprob_m)    
    
class BiC_cost_detail():
    
    def __init__(self,args) -> None:
        self.cs = args.cs
        self.ch = args.ch
        self.cm2k = args.cm2k
        self.ck2m = args.ck2m 
        self.cm= 0
        pass
    
    
    def cost_func(self, x_k,x_m, y_k,y_m,p,):
        
        cs = self.cs * p   ## unit shortage cost
        ch = self.ch * p    ## unit holding cost
        ck2m = self.ck2m    ## unit transfer cost, from k to m
        cm2k = self.cm2k     ## unit transfer cost, from k to m, dependce on the batch
        cm = self.cm
        
        #if (x_m <= 5) & (x_m > 0):  
        #    cm2k = self.cm2k/x_m   
        #if (x_m <= 25) & (x_m > 5):
        #    cm2k = self.cm2k/6
        #if (x_m <= 50) & (x_m > 25):    
        #    cm2k = self.cm2k/12
        #if x_m > 50: 
        #    cm2k = self.cm2k/25      

        
        d_k = x_k - y_k   # front shortage
        d_m = x_m - y_m   # back shortage
        if (d_k<=0) & (d_m<=0):  
            #cost = -(d_k + d_m)*cs   # when both shortage, only shortage cost exist
            shortage_cost = -(d_k + d_m)*cs
            holding_cost  = 0
            trans_k2m = 0
            trans_m2k = 0
            pick_cost = cm *x_m         
        if (d_k>0) & (d_m>0):    
            #cost = (d_k + d_m)*ch    # when both overordered, only holding cost exist
            shortage_cost = 0
            holding_cost  = (d_k + d_m)*ch
            trans_k2m = 0
            trans_m2k = 0    
            pick_cost = cm *y_m     
               
        if (d_k<=0) & (d_m>0):  # only front shortage 
            d_k_m = d_m + d_k   # number left after  transfering from m to k
            if d_k_m>0:
                #cost = -cm2k*d_k + d_k_m*ch  # when enough in m, holding and transfer cost occur
                shortage_cost = 0
                holding_cost  = d_k_m*ch
                trans_k2m = 0
                trans_m2k = -cm2k*d_k   
                pick_cost = cm*y_m
            else:
                #cost = cm2k*d_m - d_k_m*cs #when not enough in m, shortage cost and transfer cost occur
                shortage_cost = - d_k_m*cs
                holding_cost  = 0
                trans_k2m = 0
                trans_m2k = cm2k*d_m
                pick_cost = cm*y_m
        if (d_k>0) & (d_m<=0): # only back shortage
            d_k_m = d_m + d_k    # left after transfer
            if d_k_m>0:     # enough in k
                #cost = -ck2m*d_m + d_k_m*ch  # holding and transfer cost
                shortage_cost = 0
                holding_cost  = d_k_m*ch
                trans_k2m = -ck2m*d_m
                trans_m2k = 0 
                pick_cost = cm*x_m
            else:
                #cost = ck2m*d_k - d_k_m*cs   #  shortage and transfer cost
                shortage_cost = - d_k_m*cs
                holding_cost  = 0
                trans_k2m = ck2m*d_k
                trans_m2k = 0 
                pick_cost = cm*x_m
        
        return np.array([shortage_cost,holding_cost,trans_k2m,trans_m2k,pick_cost])
    
    def df_loss(self, df):
        cost = df.apply(lambda x: self.cost_func(x.x_k,x.x_m, x.y_k,x.y_m,x.p),axis=1)
        return cost
    
    def np_loss(self, pred,y):
        cost = [self.cost_func(pred[i,0],pred[i,1], y[i,0],y[i,1],y[i,2]) for i in range(pred.shape[0])]
        return np.stack(cost)
        