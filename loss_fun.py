import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Normal, MultivariateNormal

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

class MseLoss_omni(nn.Module):
    def __init__(self, args,device):
        super(MseLoss_omni, self).__init__()        

    def forward(self, x_k,x_m, target):
        errors_k = target[:,0] - x_k
        errors_m = target[:,1] - x_m
        loss_k = errors_k**2
        loss_m = errors_m**2
        return torch.mean(loss_k+loss_m)

class Omni_cost_loss(nn.Module):
    
    def __init__(self,args,device) -> None:
        super(Omni_cost_loss, self).__init__()
        self.args = args
        self.cs = torch.FloatTensor([args.cs]).to(device)
        self.ch = torch.FloatTensor([args.ch]).to(device)
        self.cm2k = torch.FloatTensor([args.cm2k]).to(device)
        self.ck2m = torch.FloatTensor([args.ck2m]).to(device)
        self.shelf_penalty = args.shelf_penalty
        self.lambda_shelf = args.lambda_shelf
        self.cm2k_dict = {
                0.5: 0.05,
                0.8: 0.1,
                1.1: 0.15,
                1.4: 0.2,
                1.7: 0.25
            }
        self.ck2m_dict = {
                1: 0.05,
                1.3: 0.1,
                1.6: 0.15,
                1.9: 0.2,
                2.2: 0.25
            }
        pass
    
    
    def cost_func(self, x_k, x_m, y_k, y_m, p):
        cs = self.cs * p   # unit shortage cost
        ch = self.ch * p    # unit holding cost
        ck2m = self.ck2m #+ self.ck2m_dict[self.args.ck2m]*p   # unit transfer cost, from k to m
        cm2k = self.cm2k + 0.15*p
        
        
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
   
    def forward(self, pred_k, pred_m,real,integer_penalty=0):
        # Calculate the cost using vectorized operations
        if pred_k.dim() ==1:
            pred_k = pred_k.unsqueeze(1)
            pred_m = pred_m.unsqueeze(1)

        operational_cost = self.cost_func(pred_k, pred_m, real[:, 0:1], real[:, 1:2], real[:, 2:3])
        shelf_penalty = 0
        if self.shelf_penalty:
            shelf_capacities = real[:, 4:5]
            shelf_penalty = ((pred_k - shelf_capacities) ** 2).mean() * self.lambda_shelf

        # Sum the costs along the batch dimension
        return operational_cost #+ shelf_penalty +integer_penalty     

class Likelyhood(nn.Module):
    def __init__(self,args,device):
        super(Likelyhood, self).__init__()

    def forward(self, r,p, target):
        
        dist_k = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,0:1],logits = p[:,0:1])   
        dist_m = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,1:2],logits = p[:,1:2]) 
    
        logprob_k = dist_k.log_prob(target[:,0:1]) 
        logprob_m = dist_m.log_prob(target[:,1:2])    
           

        return -torch.mean(logprob_k+logprob_m)    
    

class JointLikelyhood_Clayton(nn.Module):
    def __init__(self, args, device):
        super(JointLikelyhood_Clayton, self).__init__()

    def nb_cdf_sum(self, y, r, p):
        """
        Compute NB CDF by brute‐force summing PMFs up to each y[i].
        y: (batch,) integer tensor
        r: (batch,) total_count
        p: (batch,) success_prob
        returns (batch,) tensor of CDFs
        """
        device = y.device
        max_y = int(y.max().item())
        k     = torch.arange(max_y + 1, device=device)          # shape (max_y+1,)

        # Expand params to (batch,1) so log_prob(k) → (batch, max_y+1)
        r2 = r.unsqueeze(1)   # (batch,1)
        p2 = p.unsqueeze(1)   # (batch,1)
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)

        # k.unsqueeze(0): (1, max_y+1) broadcasts with (batch,1) → (batch, max_y+1)
        logpmf = nb.log_prob(k.unsqueeze(0))  # (batch, max_y+1)
        pmf    = logpmf.exp()

        # mask out terms above y[i]
        mask = (k.unsqueeze(0) <= y.unsqueeze(1))  # (batch, max_y+1)
        cdf  = (pmf * mask).sum(dim=1)             # (batch,)

        return cdf

    def clayton_copula_log_density(self, u1, u2, theta):
        """
        Compute the log-density of the Clayton copula.
        u1, u2: (batch,) tensor of uniform marginals.
        theta: Clayton copula parameter.
        """
        theta = torch.clamp(theta, min=0.0001)  # Ensure theta is positive
        copula_density = torch.pow(u1 + u2 - 1, -theta - 1)  # Clayton copula density formula
        log_density = torch.log(copula_density)
        return log_density

    def bivariate_nb_loss_bruteforce(self, y1, y2, r1, p1, r2, p2, rho, eps=1e-6):
        # 1) marginals
        nb1 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r1, logits=p1)
        nb2 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)
        logp1 = nb1.log_prob(y1)
        logp2 = nb2.log_prob(y2)

        # 2) CDF by summation (fixed)
        u1 = self.nb_cdf_sum(y1, r1, p1).clamp(eps, 1 - eps)
        u2 = self.nb_cdf_sum(y2, r2, p2).clamp(eps, 1 - eps)

        # 3) Clayton copula log-density
        logc = self.clayton_copula_log_density(u1, u2, rho)

        # 4) joint log-likelihood
        ll = logp1 + logp2 + logc
        return -ll.mean()

    def forward(self, r, p, target):
        rho = F.relu(p[:, 1])  # Map parameter to [-1, 1] range
        r = torch.clamp(r, min=0.0001)  # Ensure r is positive
        return self.bivariate_nb_loss_bruteforce(
            y1=target[:, 0], y2=target[:, 1],
            r1=r[:, 0], p1=p[:, 0], r2=r[:, 1], p2=p[:, 0], rho=rho
        )


class JointLikelyhood_Gumbel(nn.Module):
    def __init__(self, args, device):
        super(JointLikelyhood_Gumbel, self).__init__()

    def nb_cdf_sum(self, y, r, p):
        """
        Compute NB CDF by brute‐force summing PMFs up to each y[i].
        y: (batch,) integer tensor
        r: (batch,) total_count
        p: (batch,) success_prob
        returns (batch,) tensor of CDFs
        """
        device = y.device
        max_y = int(y.max().item())
        k     = torch.arange(max_y + 1, device=device)          # shape (max_y+1,)

        # Expand params to (batch,1) so log_prob(k) → (batch, max_y+1)
        r2 = r.unsqueeze(1)   # (batch,1)
        p2 = p.unsqueeze(1)   # (batch,1)
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, probs=p2)

        # k.unsqueeze(0): (1, max_y+1) broadcasts with (batch,1) → (batch, max_y+1)
        logpmf = nb.log_prob(k.unsqueeze(0))  # (batch, max_y+1)
        pmf    = logpmf.exp()

        # mask out terms above y[i]
        mask = (k.unsqueeze(0) <= y.unsqueeze(1))  # (batch, max_y+1)
        cdf  = (pmf * mask).sum(dim=1)             # (batch,)

        return cdf

    def gumbel_copula_log_density(self, u1, u2, theta):
        """
        Compute the log-density of the Gumbel copula.
        u1, u2: (batch,) tensor of uniform marginals.
        theta: Gumbel copula parameter.
        """
        theta = torch.clamp(theta, min=1.00001)  # Ensure theta is greater than or equal to 1
        # Gumbel copula density formula
        term1 = (-torch.log(u1)) ** theta
        term2 = (-torch.log(u2)) ** theta
        copula_density = torch.exp(-(term1 + term2) ** (1 / theta))
        log_density = torch.log(copula_density)
        return log_density

    def bivariate_nb_loss_bruteforce(self, y1, y2, r1, p1, r2, p2, rho, eps=1e-6):
        # 1) marginals
        nb1 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r1, probs=p1)
        nb2 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, probs=p2)
        logp1 = nb1.log_prob(y1)
        logp2 = nb2.log_prob(y2)

        # 2) CDF by summation (fixed)
        u1 = self.nb_cdf_sum(y1, r1, p1).clamp(eps, 1 - eps)
        u2 = self.nb_cdf_sum(y2, r2, p2).clamp(eps, 1 - eps)

        # 3) Gumbel copula log-density
        logc = self.gumbel_copula_log_density(u1, u2, rho)

        # 4) joint log-likelihood
        ll = logp1 + logp2 + logc
        return -ll.mean()

    def forward(self, r, p, target):
        #rho = F.tanh(p[:, 1])  # Map parameter to [-1, 1] range
        rho = F.relu(p[:,1]) + 1
        r = torch.clamp(r, min=0.0001)  # Ensure r is positive
        p1 = F.tanh(p[:, 0])  # Ensure p is in the range (-1, 1)
        p1 = torch.clamp(p1, min=0.0001, max=0.9999)  # Ensure p is in the range (0, 1)
        return self.bivariate_nb_loss_bruteforce(
            y1=target[:, 0], y2=target[:, 1],
            r1=r[:, 0], p1=p1, r2=r[:, 1], p2=p1, rho=rho
        )


class JointLikelyhood(nn.Module):
    def __init__(self, args, device):
        super(JointLikelyhood, self).__init__()
        
    def nb_cdf_sum(self, y, r, p):
        """
        Compute NB CDF by brute‐force summing PMFs up to each y[i].
        y: (batch,) integer tensor
        r: (batch,) total_count
        p: (batch,) success_prob
        returns (batch,) tensor of CDFs
        """
        device = y.device
        max_y = 1000 # int(y.max().item())
        k = torch.arange(max_y + 1, device=device)  # shape (max_y+1,)

        # Expand params to (batch,1) so log_prob(k) → (batch, max_y+1)
        r2 = r.unsqueeze(1)   # (batch,1)
        p2 = p.unsqueeze(1)   # (batch,1)
        nb = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)

        # k.unsqueeze(0): (1, max_y+1) broadcasts with (batch,1) → (batch, max_y+1)
        logpmf = nb.log_prob(k.unsqueeze(0))  # (batch, max_y+1)
        pmf = torch.exp(logpmf)  # Using torch.exp for numerical stability

        # mask out terms above y[i]
        mask = (k.unsqueeze(0) <= y.unsqueeze(1))  # (batch, max_y+1)
        cdf = (pmf * mask).sum(dim=1)  # (batch,)

        return cdf

    def bivariate_nb_loss_bruteforce(self, y1, y2, r1, p1, r2, p2, rho, eps=1e-6):
        # 1) marginals
        nb1 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r1, logits=p1)
        nb2 = torch.distributions.negative_binomial.NegativeBinomial(total_count=r2, logits=p2)
        logp1 = nb1.log_prob(y1)
        logp2 = nb2.log_prob(y2)

        # 2) CDF by summation with proper clamping
        u1 = self.nb_cdf_sum(y1, r1, p1).clamp(eps, 1 - eps)
        u2 = self.nb_cdf_sum(y2, r2, p2).clamp(eps, 1 - eps)

        # 3) probit transform → Gaussian quantiles
        norm = Normal(0., 1.)
        z1 = norm.icdf(u1)
        z2 = norm.icdf(u2)
        
        # Properly constrain rho
        rho = torch.clamp(rho, min=-0.9999, max=0.9999)
        
        # 4) Gaussian copula log‑density with improved numerical stability
        Sigma = torch.stack([
            torch.stack([torch.ones_like(rho), rho], dim=-1),
            torch.stack([rho, torch.ones_like(rho)], dim=-1)
        ], dim=-2)  # (batch,2,2)

        # Ensure covariance matrix is positive definite
        batch_size = rho.size(0)
        eye = torch.eye(2, device=rho.device).unsqueeze(0).expand(batch_size, -1, -1)
        Sigma = Sigma + 1e-6 * eye  # Add small value to diagonal for stability

        mvn = MultivariateNormal(
            loc=torch.zeros(batch_size, 2, device=rho.device),
            covariance_matrix=Sigma
        )
        z = torch.stack([z1, z2], dim=-1)
        logc = mvn.log_prob(z) - norm.log_prob(z1) - norm.log_prob(z2)

        # 5) joint log‑likelihood - ensure we return a positive loss
        ll = logp1 + logp2 + 0.1*logc # + 1*torch.log(1-rho**2 + 1e-6)  # Adjusted for numerical stability
        
        # Take the negative mean to ensure positive loss (since log-likelihood is typically negative)
        return -ll.mean()
    
    def forward(self, r, p, target):
        # Fixed: Using p[:,2] for second distribution instead of p[:,0] again
        # Check if the shape of p allows this, otherwise adjust accordingly
        rho = F.tanh(p[:,1])  # Map parameter to [-1, 1] range
        
        if p.shape[1] >= 3:
            # If you have separate parameters for both distributions
            return self.bivariate_nb_loss_bruteforce(
                y1=target[:,0], y2=target[:,1], 
                r1=r[:,0], p1=p[:,0], 
                r2=r[:,1], p2=p[:,2],  # Changed from p[:,0] to p[:,2]
                rho=rho
            )
        else:
            # Fallback to original behavior if shape doesn't allow
            return self.bivariate_nb_loss_bruteforce(
                y1=target[:,0], y2=target[:,1], 
                r1=r[:,0], p1=p[:,0], 
                r2=r[:,1], p2=p[:,0],  # Using same parameter for both (check if this is intended)
                rho=rho
            )

    
class BiC_cost_detail():
    
    def __init__(self,args) -> None:
        self.args = args
        self.cs = args.cs
        self.ch = args.ch
        self.cm2k = args.cm2k
        self.ck2m = args.ck2m 
        self.cm= 0
        self.cm2k_dict = {
                0.5: 0.5,
                0.8: 0.1,
                1.1: 0.15,
                1.4: 0.2,
                1.7: 0.25
            }
        self.ck2m_dict = {
                1: 0.05,
                1.3: 0.1,
                1.6: 0.15,
                1.9: 0.2,
                2.2: 0.25
            }
        pass
        
    
    
    def cost_func(self, x_k,x_m, y_k,y_m,p,):
        
        cs = self.cs * p   ## unit shortage cost
        ch = self.ch * p    ## unit holding cost
        ck2m = self.ck2m #+ self.ck2m_dict[self.args.ck2m]*p    ## unit transfer cost, from k to m
        cm2k = self.cm2k + 0.15*p    ## unit transfer cost, from k to m, dependce on the batch
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
        