
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.optim import lr_scheduler 

from models import MLPRF,MLPQR,MLPSQR,MLPJoint
from data import data_provider
from loss_fun import QuantileLoss,QuantileLoss_omni,Omni_cost_loss,Likelyhood,JointLikelyhood,JointLikelyhood_Clayton,JointLikelyhood_Gumbel,MseLoss_omni
from tools import EarlyStopping

import scipy.stats  # Import SciPy stats
import torch.distributions as dist

# =============================================================================
# GPU-accelerated bivariate NB copula optimizer
# =============================================================================

def optimize_stock_allocation_torch_gpu_gaussian(
    r_offline, p_offline,
    r_online,  p_online,
    rho,
    C_h, C_s, C_f2b, C_b2f,
    max_k=100,
    device=None
):
    """
    Single-SKU solver on GPU. All inputs are 0-dim torch.Tensors on `device`.
    Returns a dict {x_f,x_b,F1,F2,error} with torch tensors.
    """
    # ensure device
    device = device or r_offline.device

    # --- 1) build grid of support indices ---
    K = max_k + 1
    idx = torch.arange(K, device=device)
    Y1 = idx.view(-1,1).expand(K,K)  # shape [K,K]
    Y2 = idx.view(1,-1).expand(K,K)

    # --- 2) compute marginal pmf and cdf via torch ---
    # pmf(k) = comb(k+r-1, k)*(1-p)^r * p^k
    def nb_pmf_cdf(r, p):
        # r,p are 0-dim tensors
        # pmf vector shape [K]
        lgamma = torch.lgamma
        log_comb = lgamma(idx + r) - lgamma(r) - lgamma(idx + 1)
        log_pmf = log_comb + r * torch.log1p(-p) + idx * torch.log(p)
        pmf = torch.exp(log_pmf)
        cdf = torch.cumsum(pmf, dim=0)
        # clip for numerical stability
        cdf = cdf.clamp(min=1e-6, max=1-1e-6)
        return pmf, cdf

    pmf_off, cdf_off = nb_pmf_cdf(r_offline, p_offline)
    pmf_on,  cdf_on  = nb_pmf_cdf(r_online,  p_online)

    # --- 3) broadcast to joint grid ---
    pmf1 = pmf_off.view(-1,1)  # [K,1]
    pmf2 = pmf_on .view(1,-1)  # [1,K]
    F1   = cdf_off.view(-1,1)
    F2   = cdf_on .view(1,-1)

    # --- 4) Gaussian copula density ---
    # norm.ppf approximation via erfinv:  ppf(u) = sqrt(2)*erfinv(2u-1)
    sqrt2 = torch.sqrt(torch.tensor(2., device=device))
    z1 = sqrt2 * torch.erfinv(2*F1-1)
    z2 = sqrt2 * torch.erfinv(2*F2-1)
    denom = torch.sqrt(1 - rho**2)
    exponent = - (z1**2 - 2*rho*z1*z2 + z2**2) / (2*(1-rho**2)) + (z1**2 + z2**2)/2
    copula = torch.exp(exponent) / denom

    # --- 5) joint pmf and normalization ---
    P_joint = pmf1 * pmf2 * copula
    P_joint = P_joint / P_joint.sum()

    # --- 6) precompute demand sum grid ---
    sum_dem = Y1 + Y2

    # --- 7) search bound from marginal quantiles ---
    quant = C_s / (C_s + C_h)
    max1 = (cdf_off >= quant).nonzero()[0].item()
    max2 = (cdf_on  >= quant).nonzero()[0].item()
    max_search = max1 + max2

    # --- 8) brute-force grid search on GPU ---
    best_err = torch.tensor(float('inf'), device=device)
    best = {'x_f': None, 'x_b': None, 'F1': None, 'F2': None, 'error': None}

    for xf in range(max_search+1):
        # broadcast xf, xb on GPU
        xf_val = torch.tensor(xf, device=device)
        for xb in range(max_search+1):
            xb_val = torch.tensor(xb, device=device)
            inv_mask = sum_dem < (xf_val + xb_val)
            P_gt = P_joint[inv_mask].sum()
            P_lt = 1 - P_gt  # since P_joint sums to 1

            # flow events
            A = (sum_dem > (xf_val+xb_val)) & (xf_val > Y1) & (xb_val < Y2)
            B = (sum_dem < (xf_val+xb_val)) & (xf_val < Y1) & (xb_val > Y2)
            C = (sum_dem < (xf_val+xb_val)) & (xf_val > Y1) & (xb_val < Y2)
            D = (sum_dem > (xf_val+xb_val)) & (xf_val < Y1) & (xb_val > Y2)

            P_A = P_joint[A].sum()
            P_B = P_joint[B].sum()
            P_C = P_joint[C].sum()
            P_D = P_joint[D].sum()

            F1_val = C_h*P_gt - C_s*P_lt + C_f2b*P_A - C_b2f*P_B
            F2_val = C_h*P_gt - C_s*P_lt - C_f2b*P_C + C_b2f*P_D
            err = F1_val*F1_val + F2_val*F2_val

            if err < best_err:
                best_err = err
                best = {
                    'x_f': xf_val,
                    'x_b': xb_val,
                    'F1': F1_val,
                    'F2': F2_val,
                    'error': err
                }

    return best


def batch_optimize_torch_gpu__gaussian(
    r_offline, p_offline,
    r_online,  p_online,
    rho,
    C_h, C_s, C_f2b, C_b2f,
    max_k=100
):
    """
    Batch solver on GPU. All inputs have shape [B] on same device.
    Returns two LongTensors x_f, x_b of shape [B].
    """
    device = r_offline.device
    B = r_offline.shape[0]
    x_f = torch.empty(B, dtype=torch.long, device=device)
    x_b = torch.empty(B, dtype=torch.long, device=device)

    for i in range(B):
        out = optimize_stock_allocation_torch_gpu_gaussian(
            r_offline[i], p_offline[i],
            r_online[i],  p_online[i],
            rho[i],
            C_h[i], C_s[i], C_f2b[i], C_b2f[i],
            max_k, device=device
        )
        x_f[i], x_b[i] = out['x_f'], out['x_b']

    return x_f, x_b

def batch_optimize_allocation_gpu_gaussian(
    r_offline: torch.Tensor,  # [B]
    p_offline: torch.Tensor,  # [B]
    r_online:  torch.Tensor,  # [B]
    p_online:  torch.Tensor,  # [B]
    rho:        torch.Tensor,  # [B]
    C_h:        torch.Tensor,  # [B]
    C_s:        torch.Tensor,  # [B]
    C_f2b:      torch.Tensor,  # [B]
    C_b2f:      torch.Tensor,  # [B]
    max_k: int = 100
) -> (torch.LongTensor, torch.LongTensor):
    """
    Batch‐vectorized GPU solver for omni‐channel allocation.
    Returns two LongTensors (x_f, x_b) each of shape [B].
    """
    device = r_offline.device
    B = r_offline.shape[0]
    K = max_k + 1

    # 1) support index [K]
    idx = torch.arange(K, device=device, dtype=r_offline.dtype)  # [K]

    # 2) marginal PMF and CDF for offline: [B,K]
    r_off = r_offline.unsqueeze(1)  # [B,1]
    p_off = p_offline.unsqueeze(1)
    log_comb_off = torch.lgamma(idx + r_off) - torch.lgamma(r_off) - torch.lgamma(idx + 1)
    log_pmf_off = log_comb_off + r_off * torch.log1p(-p_off) + idx * torch.log(p_off)
    pmf_off = torch.exp(log_pmf_off)          # [B,K]
    cdf_off = torch.cumsum(pmf_off, dim=1).clamp(1e-6, 1-1e-6)

    # 3) marginal PMF and CDF for online: [B,K]
    r_on = r_online.unsqueeze(1)
    p_on = p_online.unsqueeze(1)
    log_comb_on = torch.lgamma(idx + r_on) - torch.lgamma(r_on) - torch.lgamma(idx + 1)
    log_pmf_on = log_comb_on + r_on * torch.log1p(-p_on) + idx * torch.log(p_on)
    pmf_on = torch.exp(log_pmf_on)           # [B,K]
    cdf_on = torch.cumsum(pmf_on, dim=1).clamp(1e-6, 1-1e-6)

    # 4) Gaussian‐copula density grid per SKU: shape [B,K,K]
    #    Probit via erfinv
    sqrt2 = torch.sqrt(torch.tensor(2., device=device))
    z1 = sqrt2 * torch.erfinv(2*cdf_off - 1)    # [B,K]
    z2 = sqrt2 * torch.erfinv(2*cdf_on  - 1)    # [B,K]
    # broadcast to grid
    Z1 = z1.unsqueeze(2)  # [B,K,1]
    Z2 = z2.unsqueeze(1)  # [B,1,K]
    denom = torch.sqrt(1 - rho**2).unsqueeze(1).unsqueeze(2)  # [B,1,1]
    exponent = (
        - (Z1**2 - 2*rho.unsqueeze(1).unsqueeze(2)*Z1*Z2 + Z2**2)
          / (2*(1-rho.unsqueeze(1).unsqueeze(2)) )
        + (Z1**2 + Z2**2)/2
    )  # [B,K,K]
    copula = torch.exp(exponent) / denom  # [B,K,K]

    # 5) joint PMF per SKU: [B,K,K]
    P_joint = (pmf_off.unsqueeze(2) * pmf_on.unsqueeze(1)) * copula
    P_joint = P_joint / P_joint.sum(dim=(1,2), keepdim=True)

    # 6) demand‐sum grid [K,K]
    Y = idx
    D_off = Y.view(-1,1).expand(K,K)
    D_on  = Y.view(1,-1).expand(K,K)
    SUM = D_off + D_on  # [K,K]

    # 7) candidate xf/xb grid (static)
    M = K
    xf_vals = torch.arange(M, device=device)
    xb_vals = torch.arange(M, device=device)
    XF, XB = torch.meshgrid(xf_vals, xb_vals, indexing='ij')  # [M,M]
    INV = XF + XB  # [M,M]

    # 8) broadcast to [B,K,K,M,M]
    PJ = P_joint.unsqueeze(-1).unsqueeze(-1)   # [B,K,K,1,1]
    SD = SUM.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,K,K,1,1]
    XFb = XF.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]
    XBb = XB.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]

    # 9) event probabilities: [B,M,M]
    mask_gt = SD < INV[None,None]    # sums to 1 across all events
    P_gt = (PJ * mask_gt).sum(dim=(1,2))  # [B,M,M]
    P_lt = 1 - P_gt

    mask_A = (SD > INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_B = (SD < INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_C = (SD < INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_D = (SD > INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    P_A = (PJ * mask_A).sum(dim=(1,2))
    P_B = (PJ * mask_B).sum(dim=(1,2))
    P_C = (PJ * mask_C).sum(dim=(1,2))
    P_D = (PJ * mask_D).sum(dim=(1,2))

    # 10) compute F1/F2 and error grids: [B,M,M]
    F1 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         + C_f2b.unsqueeze(-1).unsqueeze(-1)*P_A \
         - C_b2f.unsqueeze(-1).unsqueeze(-1)*P_B

    F2 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         - C_f2b.unsqueeze(-1).unsqueeze(-1)*P_C \
         + C_b2f.unsqueeze(-1).unsqueeze(-1)*P_D

    err = F1**2 + F2**2  # [B,M,M]

    # 11) argmin over the last two dims
    err_flat = err.view(B, -1)               # [B, M*M]
    idx_flat = torch.argmin(err_flat, dim=1) # [B]
    x_f = xf_vals[idx_flat // M]             # [B]
    x_b = xb_vals[idx_flat %  M]             # [B]

    return x_f, x_b

def batch_optimize_allocation_gpu_gaussian_1(
    r_offline: torch.Tensor,  # [B]
    p_offline: torch.Tensor,  # [B]
    r_online:  torch.Tensor,  # [B]
    p_online:  torch.Tensor,  # [B]
    rho:        torch.Tensor,  # [B]
    C_h:        torch.Tensor,  # [B]
    C_s:        torch.Tensor,  # [B]
    C_f2b:      torch.Tensor,  # [B]
    C_b2f:      torch.Tensor,  # [B]
    max_k: int = 100
) -> (torch.LongTensor, torch.LongTensor):
    """
    Batch‐vectorized GPU solver with reduced memory usage using FP16 precision.
    Returns two LongTensors (x_f, x_b) each of shape [B].
    """
    device = r_offline.device
    B = r_offline.shape[0]
    K = max_k + 1

    # Convert inputs to float16 for reduced memory usage
    r_offline = r_offline.to(dtype=torch.float16)
    p_offline = p_offline.to(dtype=torch.float16)
    r_online = r_online.to(dtype=torch.float16)
    p_online = p_online.to(dtype=torch.float16)
    rho = rho.to(dtype=torch.float16)
    C_h = C_h.to(dtype=torch.float16)
    C_s = C_s.to(dtype=torch.float16)
    C_f2b = C_f2b.to(dtype=torch.float16)
    C_b2f = C_b2f.to(dtype=torch.float16)

    # 1) support index [K]
    idx = torch.arange(K, device=device, dtype=r_offline.dtype)  # [K]

    # 2) marginal PMF and CDF for offline: [B,K]
    r_off = r_offline.unsqueeze(1)  # [B,1]
    p_off = p_offline.unsqueeze(1)
    log_comb_off = torch.lgamma(idx + r_off) - torch.lgamma(r_off) - torch.lgamma(idx + 1)
    log_pmf_off = log_comb_off + r_off * torch.log1p(-p_off) + idx * torch.log(p_off)
    pmf_off = torch.exp(log_pmf_off)          # [B,K]
    cdf_off = torch.cumsum(pmf_off, dim=1).clamp(1e-6, 1-1e-6)

    # 3) marginal PMF and CDF for online: [B,K]
    r_on = r_online.unsqueeze(1)
    p_on = p_online.unsqueeze(1)
    log_comb_on = torch.lgamma(idx + r_on) - torch.lgamma(r_on) - torch.lgamma(idx + 1)
    log_pmf_on = log_comb_on + r_on * torch.log1p(-p_on) + idx * torch.log(p_on)
    pmf_on = torch.exp(log_pmf_on)           # [B,K]
    cdf_on = torch.cumsum(pmf_on, dim=1).clamp(1e-6, 1-1e-6)

    # 4) Gaussian‐copula density grid per SKU: shape [B,K,K]
    #    Probit via erfinv
    sqrt2 = torch.sqrt(torch.tensor(2., device=device))
    z1 = sqrt2 * torch.erfinv(2*cdf_off - 1)    # [B,K]
    z2 = sqrt2 * torch.erfinv(2*cdf_on  - 1)    # [B,K]
    # broadcast to grid
    Z1 = z1.unsqueeze(2)  # [B,K,1]
    Z2 = z2.unsqueeze(1)  # [B,1,K]
    denom = torch.sqrt(1 - rho**2).unsqueeze(1).unsqueeze(2)  # [B,1,1]
    exponent = (
        - (Z1**2 - 2*rho.unsqueeze(1).unsqueeze(2)*Z1*Z2 + Z2**2)
          / (2*(1-rho.unsqueeze(1).unsqueeze(2)) )
        + (Z1**2 + Z2**2)/2
    )  # [B,K,K]
    copula = torch.exp(exponent) / denom  # [B,K,K]

    # 5) joint PMF per SKU: [B,K,K]
    P_joint = (pmf_off.unsqueeze(2) * pmf_on.unsqueeze(1)) * copula
    P_joint = P_joint / P_joint.sum(dim=(1,2), keepdim=True)

    # 6) demand‐sum grid [K,K]
    Y = idx
    D_off = Y.view(-1,1).expand(K,K)
    D_on  = Y.view(1,-1).expand(K,K)
    SUM = D_off + D_on  # [K,K]

    # 7) candidate xf/xb grid (static)
    M = K
    xf_vals = torch.arange(M, device=device)
    xb_vals = torch.arange(M, device=device)
    XF, XB = torch.meshgrid(xf_vals, xb_vals, indexing='ij')  # [M,M]
    INV = XF + XB  # [M,M]

    # 8) broadcast to [B,K,K,M,M]
    PJ = P_joint.unsqueeze(-1).unsqueeze(-1)   # [B,K,K,1,1]
    SD = SUM.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,K,K,1,1]
    XFb = XF.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]
    XBb = XB.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]

    # 9) event probabilities: [B,M,M]
    mask_gt = SD < INV[None,None]    # sums to 1 across all events
    P_gt = (PJ * mask_gt).sum(dim=(1,2))  # [B,M,M]
    P_lt = 1 - P_gt

    mask_A = (SD > INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_B = (SD < INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_C = (SD < INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_D = (SD > INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    P_A = (PJ * mask_A).sum(dim=(1,2))
    P_B = (PJ * mask_B).sum(dim=(1,2))
    P_C = (PJ * mask_C).sum(dim=(1,2))
    P_D = (PJ * mask_D).sum(dim=(1,2))

    # 10) compute F1/F2 and error grids: [B,M,M]
    F1 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         + C_f2b.unsqueeze(-1).unsqueeze(-1)*P_A \
         - C_b2f.unsqueeze(-1).unsqueeze(-1)*P_B

    F2 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         - C_f2b.unsqueeze(-1).unsqueeze(-1)*P_C \
         + C_b2f.unsqueeze(-1).unsqueeze(-1)*P_D

    err = F1**2 + F2**2  # [B,M,M]

    # 11) argmin over the last two dims
    err_flat = err.view(B, -1)               # [B, M*M]
    idx_flat = torch.argmin(err_flat, dim=1) # [B]
    x_f = xf_vals[idx_flat // M]             # [B]
    x_b = xb_vals[idx_flat %  M]             # [B]

    return x_f, x_b


def batch_optimize_allocation_gpu(
    r_offline: torch.Tensor,  # [B]
    p_offline: torch.Tensor,  # [B]
    r_online:  torch.Tensor,  # [B]
    p_online:  torch.Tensor,  # [B]
    rho:        torch.Tensor,  # [B]
    C_h:        torch.Tensor,  # [B]
    C_s:        torch.Tensor,  # [B]
    C_f2b:      torch.Tensor,  # [B]
    C_b2f:      torch.Tensor,  # [B]
    max_k: int = 100
) -> (torch.LongTensor, torch.LongTensor):
    """
    Batch‐vectorized GPU solver for omni‐channel allocation.
    Returns two LongTensors (x_f, x_b) each of shape [B].
    """
    device = r_offline.device
    B = r_offline.shape[0]
    K = max_k + 1

    # 1) support index [K]
    idx = torch.arange(K, device=device, dtype=r_offline.dtype)  # [K]

    # 2) marginal PMF and CDF for offline: [B,K]
    r_off = r_offline.unsqueeze(1)  # [B,1]
    p_off = p_offline.unsqueeze(1)
    log_comb_off = torch.lgamma(idx + r_off) - torch.lgamma(r_off) - torch.lgamma(idx + 1)
    log_pmf_off = log_comb_off + r_off * torch.log1p(-p_off) + idx * torch.log(p_off)
    pmf_off = torch.exp(log_pmf_off)          # [B,K]
    cdf_off = torch.cumsum(pmf_off, dim=1).clamp(1e-6, 1-1e-6)

    # 3) marginal PMF and CDF for online: [B,K]
    r_on = r_online.unsqueeze(1)
    p_on = p_online.unsqueeze(1)
    log_comb_on = torch.lgamma(idx + r_on) - torch.lgamma(r_on) - torch.lgamma(idx + 1)
    log_pmf_on = log_comb_on + r_on * torch.log1p(-p_on) + idx * torch.log(p_on)
    pmf_on = torch.exp(log_pmf_on)           # [B,K]
    cdf_on = torch.cumsum(pmf_on, dim=1).clamp(1e-6, 1-1e-6)

    # 4) Gumbel-copula density grid per SKU: shape [B,K,K]
    # Gumbel copula: C(u1, u2) = exp( - ((-log(u1))^theta + (-log(u2))^theta)^(1/theta)) 
    theta = torch.clamp(rho, min=1.0001)  # Ensure theta >= 1
    term1 = (-torch.log(cdf_off)) ** theta
    term2 = (-torch.log(cdf_on)) ** theta
    copula_density = torch.exp(-(term1 + term2) ** (1 / theta))
    copula_density /= torch.sqrt(1 - theta**2)  # Normalize for Gumbel copula

    # 5) joint PMF per SKU: [B,K,K]
    P_joint = (pmf_off.unsqueeze(2) * pmf_on.unsqueeze(1)) * copula_density
    P_joint = P_joint / P_joint.sum(dim=(1,2), keepdim=True)

    # 6) demand‐sum grid [K,K]
    Y = idx
    D_off = Y.view(-1,1).expand(K,K)
    D_on  = Y.view(1,-1).expand(K,K)
    SUM = D_off + D_on  # [K,K]

    # 7) candidate xf/xb grid (static)
    M = K
    xf_vals = torch.arange(M, device=device)
    xb_vals = torch.arange(M, device=device)
    XF, XB = torch.meshgrid(xf_vals, xb_vals, indexing='ij')  # [M,M]
    INV = XF + XB  # [M,M]

    # 8) broadcast to [B,K,K,M,M]
    PJ = P_joint.unsqueeze(-1).unsqueeze(-1)   # [B,K,K,1,1]
    SD = SUM.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,K,K,1,1]
    XFb = XF.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]
    XBb = XB.unsqueeze(0).unsqueeze(1).unsqueeze(1)   # [1,1,1,M,M]

    # 9) event probabilities: [B,M,M]
    mask_gt = SD < INV[None,None]    # sums to 1 across all events
    P_gt = (PJ * mask_gt).sum(dim=(1,2))  # [B,M,M]
    P_lt = 1 - P_gt

    mask_A = (SD > INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_B = (SD < INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_C = (SD < INV[None,None]) & (XFb > D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb < D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
    mask_D = (SD > INV[None,None]) & (XFb < D_off.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) & (XBb > D_on.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))

    P_A = (PJ * mask_A).sum(dim=(1,2))
    P_B = (PJ * mask_B).sum(dim=(1,2))
    P_C = (PJ * mask_C).sum(dim=(1,2))
    P_D = (PJ * mask_D).sum(dim=(1,2))

    # 10) compute F1/F2 and error grids: [B,M,M]
    F1 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         + C_f2b.unsqueeze(-1).unsqueeze(-1)*P_A \
         - C_b2f.unsqueeze(-1).unsqueeze(-1)*P_B

    F2 = C_h.unsqueeze(-1).unsqueeze(-1)*P_gt \
         - C_s.unsqueeze(-1).unsqueeze(-1)*P_lt \
         - C_f2b.unsqueeze(-1).unsqueeze(-1)*P_C \
         + C_b2f.unsqueeze(-1).unsqueeze(-1)*P_D

    err = F1**2 + F2**2  # [B,M,M]

    # 11) argmin over the last two dims
    err_flat = err.view(B, -1)               # [B, M*M]
    idx_flat = torch.argmin(err_flat, dim=1) # [B]
    x_f = xf_vals[idx_flat // M]             # [B]
    x_b = xb_vals[idx_flat %  M]             # [B]

    return x_f, x_b



# --- (Keep other imports and functions as they were) ---

def generate_correlated_neg_binomial(
    nb_r_online, nb_p_online,
    nb_r_offline, nb_p_offline,
    correlation, n_samples, batch_size, device
):
    """
    Generates correlated samples from two Negative Binomial distributions
    using a Gaussian copula and SciPy for the inverse CDF (ppf).

    Args:
        nb_r_online (torch.Tensor): Tensor of 'r' parameters (total_count/number of failures) for online demand. Shape (batch_size,).
        nb_p_online (torch.Tensor): Tensor of 'p' parameters (probability of success) for online demand. Shape (batch_size,).
        nb_r_offline (torch.Tensor): Tensor of 'r' parameters for offline demand. Shape (batch_size,).
        nb_p_offline (torch.Tensor): Tensor of 'p' parameters for offline demand. Shape (batch_size,).
        correlation (torch.Tensor): Tensor of correlation coefficients rho between the underlying normals. Shape (batch_size,).
        n_samples (int): Number of demand samples to generate per product.
        batch_size (int): Number of products being processed.
        device (torch.device): The device (CPU or GPU) to perform calculations on.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of demand samples (D_online, D_offline),
                                          each of shape (n_samples, batch_size).
    """
    # Ensure parameter tensors are on CPU for potential NumPy conversion later
    # (Gaussian copula part can still run on GPU if specified)
    nb_r_online_cpu = nb_r_online.cpu()
    nb_p_online_cpu = nb_p_online.cpu()
    nb_r_offline_cpu = nb_r_offline.cpu()
    nb_p_offline_cpu = nb_p_offline.cpu()

    # Define standard normal distribution for transformations (can be on target device)
    std_normal = dist.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

    # Create covariance matrices for the Gaussian copula
    cov_matrices = torch.zeros(batch_size, 2, 2, device=device)
    cov_matrices[:, 0, 0] = 1.0
    cov_matrices[:, 1, 1] = 1.0
    cov_matrices[:, 0, 1] = correlation.to(device) # Ensure correlation is on device
    cov_matrices[:, 1, 0] = correlation.to(device) # Ensure correlation is on device

    # Define the multivariate normal distribution for the copula
    try:
        mvn = dist.MultivariateNormal(
            loc=torch.zeros(batch_size, 2, device=device),
            covariance_matrix=cov_matrices
        )
    except RuntimeError as e:
        if "cholesky" in str(e).lower():
            print("Warning: Cholesky decomposition failed. Adding jitter.")
            jitter = torch.eye(2, device=device) * 1e-6
            cov_matrices_jittered = cov_matrices + jitter.unsqueeze(0)
            mvn = dist.MultivariateNormal(
                loc=torch.zeros(batch_size, 2, device=device),
                covariance_matrix=cov_matrices_jittered
            )
        else: raise e

    # Generate samples from the multivariate normal distribution
    # Shape: (n_samples, batch_size, 2)
    normal_samples = mvn.sample((n_samples,))

    # Transform normal samples to uniform samples using the standard normal CDF
    # Shape: (n_samples, batch_size, 2)
    uniform_samples = std_normal.cdf(normal_samples)

    # Avoid exact 0 or 1 for numerical stability with ICDF/PPF
    uniform_samples = torch.clamp(uniform_samples, 1e-3, 1.0 - 1e-3)

    # --- Use SciPy for Inverse CDF ---
    # Move uniform samples to CPU and convert to NumPy
    uniform_samples_np = uniform_samples.cpu().numpy() # Shape (n_samples, batch_size, 2)

    # Convert NB parameters to NumPy arrays
    nb_r_online_np = nb_r_online_cpu.detach().numpy()
    nb_p_online_np = nb_p_online_cpu.detach().numpy()
    nb_r_offline_np = nb_r_offline_cpu.detach().numpy()
    nb_p_offline_np = nb_p_offline_cpu.detach().numpy()

    # Calculate Neg Binomial samples using scipy.stats.nbinom.ppf
    # scipy.stats.nbinom takes n (number of successes = torch total_count), p (prob of success)
    # Need to broadcast parameters correctly to match uniform_samples shape
    # Reshape parameters to (1, batch_size) for broadcasting with (n_samples, batch_size)
    d_online_np = scipy.stats.nbinom.ppf(uniform_samples_np[..., 0], # q values (n_samples, batch_size)
                                         nb_r_online_np[np.newaxis, :],   # n values (1, batch_size)
                                         nb_p_online_np[np.newaxis, :])   # p values (1, batch_size)

    d_offline_np = scipy.stats.nbinom.ppf(uniform_samples_np[..., 1], # q values (n_samples, batch_size)
                                          nb_r_offline_np[np.newaxis, :],  # n values (1, batch_size)
                                          nb_p_offline_np[np.newaxis, :])  # p values (1, batch_size)

    # Convert results back to PyTorch tensors and move to the original device
    d_online = torch.from_numpy(d_online_np).to(device).float() # Ensure float type
    d_offline = torch.from_numpy(d_offline_np).to(device).float() # Ensure float type
    # --------------------------------

    return d_online, d_offline


def calculate_cost(x_f, x_b, d_online, d_offline, C_h, C_s, C_f2b, C_b2f):
    """
    Calculates the cost for given inventory levels and demand samples.

    Args:
        x_f (torch.Tensor): Front store inventory levels. Shape (batch_size,).
        x_b (torch.Tensor): Backroom inventory levels. Shape (batch_size,).
        d_online (torch.Tensor): Online demand samples. Shape (n_samples, batch_size).
        d_offline (torch.Tensor): Offline demand samples. Shape (n_samples, batch_size).
        C_h (torch.Tensor): Unit holding cost. Shape (batch_size,).
        C_s (torch.Tensor): Unit shortage cost. Shape (batch_size,).
        C_f2b (torch.Tensor): Unit cost front-to-back. Shape (batch_size,).
        C_b2f (torch.Tensor): Unit cost back-to-front. Shape (batch_size,).

    Returns:
        torch.Tensor: Cost for each sample and product. Shape (n_samples, batch_size).
    """
    # Ensure inputs are on the same device
    x_f = x_f.to(d_online.device)
    x_b = x_b.to(d_online.device)
    C_h = C_h.to(d_online.device)
    C_s = C_s.to(d_online.device)
    C_f2b = C_f2b.to(d_online.device)
    C_b2f = C_b2f.to(d_online.device)

    # Unsqueeze inventory levels and costs to match demand sample dimensions for broadcasting
    # New shape: (1, batch_size)
    x_f_exp = x_f.unsqueeze(0)
    x_b_exp = x_b.unsqueeze(0)
    C_h_exp = C_h.unsqueeze(0)
    C_s_exp = C_s.unsqueeze(0)
    C_f2b_exp = C_f2b.unsqueeze(0)
    C_b2f_exp = C_b2f.unsqueeze(0)

    # Calculate total demand and total inventory
    total_demand = d_online + d_offline
    total_inventory = x_f_exp + x_b_exp

    # Calculate holding costs: C_h * max(0, total_inventory - total_demand)
    holding_cost = C_h_exp * torch.relu(total_inventory - total_demand)

    # Calculate shortage costs: C_s * max(0, total_demand - total_inventory)
    shortage_cost = C_s_exp * torch.relu(total_demand - total_inventory)

    # Calculate front-to-back transfer costs: C_f2b * min(max(0, x_f - d_offline), max(0, d_online - x_b))
    term1_f2b = torch.relu(x_f_exp - d_offline)
    term2_f2b = torch.relu(d_online - x_b_exp)
    cost_f2b = C_f2b_exp * torch.min(term1_f2b, term2_f2b)

    # Calculate back-to-front transfer costs: C_b2f * min(max(0, x_b - d_online), max(0, d_offline - x_f))
    term1_b2f = torch.relu(x_b_exp - d_online)
    term2_b2f = torch.relu(d_offline - x_f_exp)
    cost_b2f = C_b2f_exp * torch.min(term1_b2f, term2_b2f)

    # Calculate total cost per sample
    total_cost = holding_cost + shortage_cost + cost_f2b + cost_b2f

    return total_cost

# --- Optimization Function ---

def optimize_inventory_levels(
    # Demand parameters (batch_size,)
    nb_r_online, nb_p_online,
    nb_r_offline, nb_p_offline,
    correlation,
    # Cost parameters (batch_size,)
    C_h, C_s, C_f2b, C_b2f,
    # Optimization settings
    n_products,
    n_iterations=1000,
    learning_rate=1.0, # Often needs tuning
    n_samples_per_iter=10000, # Samples for expectation estimate per iteration
    n_samples_final_eval=10000, # Samples for final evaluation
    initial_guess_factor=1.0, # Factor to multiply mean demand for initial guess
    verbose=True,
    device=torch.device("cpu")
):
    """
    Optimizes inventory levels x_f and x_b for a batch of products.

    Args:
        nb_r_online, nb_p_online, nb_r_offline, nb_p_offline: Negative Binomial parameters.
        correlation: Correlation for Gaussian copula.
        C_h, C_s, C_f2b, C_b2f: Cost parameters.
        n_products (int): Number of products in the batch.
        n_iterations (int): Number of optimization steps.
        learning_rate (float): Learning rate for the optimizer.
        n_samples_per_iter (int): Number of demand samples per optimization step.
        n_samples_final_eval (int): Number of samples for final cost evaluation.
        initial_guess_factor (float): Multiplier for mean demand for initial x_f, x_b.
        verbose (bool): Whether to print progress.
        device (torch.device): Computation device.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal integer (x_f, x_b) and the estimated minimum expected cost.
    """
    start_time = time.time()

    # --- Initial Guess ---
    # Calculate mean demand for initial guess (Mean of NB is r*(1-p)/p)
    mean_online = nb_r_online * (1 - nb_p_online) / nb_p_online
    mean_offline = nb_r_offline * (1 - nb_p_offline) / nb_p_offline
    # Simple initial guess: allocate proportionally to mean demand, scaled by factor
    total_mean_demand = mean_online + mean_offline
    initial_x_f = (mean_offline / torch.clamp(total_mean_demand, min=1e-6)) * total_mean_demand * initial_guess_factor
    initial_x_b = (mean_online / torch.clamp(total_mean_demand, min=1e-6)) * total_mean_demand * initial_guess_factor

    # Ensure initial guesses are non-negative and move to device
    x_f = torch.relu(initial_x_f).clone().detach().to(device).requires_grad_(True)
    x_b = torch.relu(initial_x_b).clone().detach().to(device).requires_grad_(True)

    # --- Optimizer ---
    optimizer = optim.Adam([x_f, x_b], lr=learning_rate)

    if verbose:
        print(f"Starting optimization for {n_products} products...")
        print(f"Initial guess (x_f, x_b): ({initial_x_f.mean():.2f}, {initial_x_b.mean():.2f}) (average)")
    # Generate a large set of samples for accurate final evaluation
    # and for use in the optimization loop.
    # If correlation is zero, generate independently. Otherwise, use copula.
    
    """
    if torch.all(correlation == 0):
        if verbose:
            print("Correlation is zero. Generating independent NB samples.")
        dist_online = torch.distributions.NegativeBinomial(total_count=nb_r_online, probs=nb_p_online)
        dist_offline = torch.distributions.NegativeBinomial(total_count=nb_r_offline, probs=nb_p_offline)
        # Sample shape: (n_samples, batch_size)
        d_online = dist_online.sample((n_samples_final_eval,))
        d_offline = dist_offline.sample((n_samples_final_eval,))
    else:
    """
    if verbose:
        print("Generating correlated NB samples using Gaussian copula.")
    d_online, d_offline = generate_correlated_neg_binomial(
        nb_r_online, nb_p_online, nb_r_offline, nb_p_offline,
        correlation, n_samples_final_eval, n_products, device
    )
    # --- Optimization Loop ---
    for i in range(n_iterations):
        optimizer.zero_grad()

        # Generate demand samples for this iteration
        #d_online, d_offline = generate_correlated_neg_binomial(
        #    nb_r_online, nb_p_online, nb_r_offline, nb_p_offline,
        #    correlation, n_samples_per_iter, n_products, device
        #)

        # Calculate cost for each sample
        # Shape: (n_samples_per_iter, n_products)
        costs = calculate_cost(x_f, x_b, d_online, d_offline, C_h, C_s, C_f2b, C_b2f)

        # Calculate the mean cost across samples (estimate of expected cost)
        # Shape: (n_products,)
        expected_cost_estimate = costs.mean(dim=0)

        # Calculate the overall loss (summing over products for backward pass)
        loss = expected_cost_estimate.sum()

        # Backpropagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Project x_f and x_b to be non-negative
        with torch.no_grad():
            x_f.clamp_(min=0)
            x_b.clamp_(min=0)

        #if verbose and (i + 1) % 100 == 0:
        print(f"Iteration {i+1}/{n_iterations}, Avg Loss: {loss.item() / n_products:.4f}, "
                f"Avg x_f: {x_f.mean().item():.2f}, Avg x_b: {x_b.mean().item():.2f}")

    # --- Get Continuous Solution ---
    x_f_continuous = x_f.detach().clone()
    x_b_continuous = x_b.detach().clone()

    if verbose:
        print(f"Optimization finished. Continuous solution (avg): x_f={x_f_continuous.mean():.2f}, x_b={x_b_continuous.mean():.2f}")
        print("Evaluating integer solutions around the continuous optimum...")

    # --- Find Best Integer Solution ---
    # Evaluate the 4 integer points around the continuous solution
    x_f_floor = torch.floor(x_f_continuous).long()
    x_f_ceil = torch.ceil(x_f_continuous).long()
    x_b_floor = torch.floor(x_b_continuous).long()
    x_b_ceil = torch.ceil(x_b_continuous).long()

    # Create candidate pairs (ensure non-negativity)
    candidates_xf = [torch.clamp(x_f_floor, min=0), torch.clamp(x_f_ceil, min=0)]
    candidates_xb = [torch.clamp(x_b_floor, min=0), torch.clamp(x_b_ceil, min=0)]

    best_x_f_int = torch.zeros_like(x_f_continuous, dtype=torch.long)
    best_x_b_int = torch.zeros_like(x_b_continuous, dtype=torch.long)
    min_expected_cost = torch.full((n_products,), float('inf'), device=device)

    # Generate a large set of samples for accurate final evaluation

    # Evaluate each candidate pair
    for xf_cand in candidates_xf:
        for xb_cand in candidates_xb:
            # Calculate cost using the large evaluation sample set
            costs_eval = calculate_cost(
                xf_cand.float(), xb_cand.float(), # Cost function expects float
                d_online, d_offline,
                C_h, C_s, C_f2b, C_b2f
            )
            # Estimate expected cost for this candidate pair
            expected_cost_eval = costs_eval.mean(dim=0) # Shape: (n_products,)

            # Update best solution if this candidate is better
            is_better = expected_cost_eval < min_expected_cost
            min_expected_cost[is_better] = expected_cost_eval[is_better]
            best_x_f_int[is_better] = xf_cand[is_better]
            best_x_b_int[is_better] = xb_cand[is_better]

    end_time = time.time()
    if verbose:
        print(f"Integer solution evaluation complete. Took {end_time - start_time:.2f} seconds.")
        print(f"Optimal integer solution (avg): x_f={best_x_f_int.float().mean():.2f}, x_b={best_x_b_int.float().mean():.2f}")
        print(f"Minimum expected cost (avg): {min_expected_cost.mean():.4f}")

    return best_x_f_int, best_x_b_int, min_expected_cost



def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))

class Exp_Main(object):
    def __init__(self, args):
        super(Exp_Main, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device    

    def _build_model(self):
        model_dict = {
            'MLPQR': MLPQR,
            'MLPRF': MLPRF,
            'MLPSQR': MLPSQR,
            'MLPJoint': MLPJoint
        }
        model = model_dict[self.args.model](self.args)#.float()
        return model    
    
    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_dict = {
            'QuantileLoss': QuantileLoss,
            'QuantileLoss_omni': QuantileLoss_omni,
            'Omni_cost_loss':Omni_cost_loss,
            'MseLoss_omni':MseLoss_omni,
            'Likelyhood_loss':Likelyhood,
            'JointLikelyhood':JointLikelyhood,
            'JointLikelyhood_Clayton':JointLikelyhood_Clayton,
            'JointLikelyhood_Gumbel':JointLikelyhood_Gumbel
        }
        criterion = loss_dict[self.args.loss](self.args,self.device)
        return criterion   
    
    def vali(self,vali_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num,x_cat, val_targets in vali_loader:

                r,p = self.model(x_num.to(self.device),x_cat.to(self.device))
                #int_penalty = 0 # self.model.get_integer_penalty()
                cost =criterion( r,p, val_targets.to(self.device))#,int_penalty)
            
                val_loss += cost.item()
            average_val_loss = val_loss / len(vali_loader)
        self.model.train()
        return average_val_loss     

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
       
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=12)
        #scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
        #                            steps_per_epoch = train_steps,
        #                            pct_start = self.args.pct_start,
        #                            epochs = self.args.train_epochs,
        #                            max_lr = 0.05)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= train_steps,eta_min=0.00001)
        epochs = self.args.train_epochs
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for x_num,x_cat, targets in train_loader:

                optimizer.zero_grad()
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device))
                #int_penalty = 0# self.model.get_integer_penalty()
                loss = criterion( r,p, targets.to(self.device))#,int_penalty)                
                loss.backward()
                optimizer.step()
                #scheduler.step()
                total_loss += loss.item()

            average_loss = total_loss / train_steps
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}")
            val_loss = self.vali(vali_loader, criterion)
            #test_loss = self.vali(test_loader, criterion)

            # Validation
            
            scheduler.step(val_loss)
            early_stopping(val_loss, self.model, path)
            #adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
            
            
        return self.model
    
    def predict(self,setting, load=False):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)
                
                prediction = torch.stack([r,p])
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy().T)
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return

    
    def evaluate(self,setting,metric):
        path = os.path.join(self.args.checkpoints, setting)
        folder_path = path+ '/results/' 
        pred = np.load(folder_path  + 'test_pred.npy')
        true = np.load(folder_path  + 'test_true.npy')
        costs = metric.np_loss(pred,true)
        np.save(folder_path + 'test_cost.npy', costs)
        print(setting, costs.sum(1).mean())
        return(costs.mean(0))
  
    
class Exp_likelyhood(Exp_Main):  
    
    def predict(self,setting, load=False):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)
                
                prediction = torch.concatenate([r,p],dim=-1)
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy())
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return  
    
    def optimize(self,setting,model_setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        print('loading model')
        path = os.path.join(self.args.checkpoints, setting)
        model_path = os.path.join(self.args.checkpoints, model_setting)
        if load:                
            best_model_path = os.path.join(model_path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        nsample = self.args.nsample    
        #with torch.no_grad():
        for x_num,x_cat, targets in pred_loader:

            # Forward pass to get predictions
            r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)                  
            p1 = F.sigmoid(p[:,0])
            p2 = F.sigmoid(p[:,1])
            rho = p1*0               
            price = targets[:,2] 
            C_sb = self.args.cs * price   # unit shortage cost
            C_hb = self.args.ch * price    # unit holding cost
            C_f2b = self.args.ck2m * torch.ones_like(price) #+ self.ck2m_dict[self.args.ck2m]*p   # unit transfer cost, from k to m
            C_b2f = self.args.cm2k + 0.15*price    # unit transfer cost, from m to k    
            
            
                # --- Run Optimization ---
            xf_gpu, xb_gpu, min_cost = optimize_inventory_levels(
                nb_r_online=r[:,1],
                nb_p_online= 1-p2,
                nb_r_offline=r[:,0],
                nb_p_offline= 1-p1,
                correlation=rho,
                C_h=C_hb.to(self.device),
                C_s=C_sb.to(self.device),
                C_f2b=C_f2b.to(self.device),
                C_b2f=C_b2f.to(self.device),
                n_products= len(r),
                n_iterations=100,           # Might need more iterations for complex problems
                learning_rate=0.3,           # Needs tuning based on cost scale and parameters
                n_samples_per_iter=2048,     # More samples reduce gradient noise but slow down iterations
                n_samples_final_eval=2048,  # More samples for accurate final cost
                initial_guess_factor=1.2,    # Adjust initial guess if needed
                verbose=True,
                device= self.device
            )
            
            #batch_optimize_allocation_gpu
            # batch_optimize_torch_gpu
            # xf_gpu, xb_gpu = batch_optimize_allocation_gpu_gaussian(
            #    r[:,0], p1, r[:,1], p2, 
            #    rho, C_hb.to(self.device), C_sb.to(self.device), C_f2b.to(self.device), C_b2f.to(self.device), max_k=200
            # )                
                        
            prediction = torch.concat([xf_gpu.unsqueeze(1),xb_gpu.unsqueeze(1)],-1)
            # Append predictions and true labels to the lists
            predictions.extend(prediction.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())            
        
        folder_path = path+ '/results/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path  + 'test_decision.npy', predictions)
        np.save(folder_path  + 'test_true.npy', true_labels)
        return 

    def optimize_quantile(self,setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        print('loading model')
        path = os.path.join(self.args.checkpoints, setting)
        if load:                
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        nsample = self.args.nsample    
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)               
                dist_k = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,0:1],logits = p[:,0:1])   
                dist_m = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,1:2],logits = p[:,1:2]) 
                prediction_k = [dist_k.sample() for i in range(nsample)]  
                prediction_m = [dist_m.sample() for i in range(nsample)] 
                quantile_k = self.args.cs/(self.args.cs+self.args.ch)
                quantile_m = quantile_k
                prediction_k = torch.stack(prediction_k).quantile(quantile_k,dim=0)
                prediction_m = torch.stack(prediction_m).quantile(quantile_m,dim=0)
                prediction = torch.concat([prediction_k,prediction_m],-1)
                # Append predictions and true labels to the lists
                predictions.extend(prediction.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())            
           
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_decision_q.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return
    
class Exp_Jointlikelyhood(Exp_Main):   
    
    def predict(self,setting, load=False):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)
                
                prediction = torch.concatenate([r,p],dim=-1)
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy())
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return     
   
    def optimize(self,setting,model_setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        print('loading model')
        path = os.path.join(self.args.checkpoints, setting)
        model_path = os.path.join(self.args.checkpoints, model_setting)
        if load:                
            best_model_path = os.path.join(model_path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        nsample = self.args.nsample    
        #with torch.no_grad():
        for x_num,x_cat, targets in pred_loader:

            # Forward pass to get predictions
            r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)                  
            p1 = F.sigmoid(p[:,0])
            if p.shape[1] >= 3:
                p2 = F.sigmoid(p[:,2])
            else: 
                p2 = F.sigmoid(p[:,0])
            rho = F.tanh(p[:,1])   
            rho = torch.clamp(rho, min=-0.9999, max=0.9999)    
            price = targets[:,2] 
            C_sb = self.args.cs * price   # unit shortage cost
            C_hb = self.args.ch * price    # unit holding cost
            C_f2b = self.args.ck2m * torch.ones_like(price) #+ self.ck2m_dict[self.args.ck2m]*p   # unit transfer cost, from k to m
            C_b2f = self.args.cm2k + 0.15*price    # unit transfer cost, from m to k    
            
            
                # --- Run Optimization ---
            xf_gpu, xb_gpu, min_cost = optimize_inventory_levels(
                nb_r_online=r[:,1],
                nb_p_online= 1-p2,
                nb_r_offline=r[:,0],
                nb_p_offline= 1-p1,
                correlation=rho,
                C_h=C_hb.to(self.device),
                C_s=C_sb.to(self.device),
                C_f2b=C_f2b.to(self.device),
                C_b2f=C_b2f.to(self.device),
                n_products= len(r),
                n_iterations=100,           # Might need more iterations for complex problems
                learning_rate=0.3,           # Needs tuning based on cost scale and parameters
                n_samples_per_iter=2048,     # More samples reduce gradient noise but slow down iterations
                n_samples_final_eval=2048,  # More samples for accurate final cost
                initial_guess_factor=1.2,    # Adjust initial guess if needed
                verbose=True,
                device= self.device
            )
            
            #batch_optimize_allocation_gpu
            # batch_optimize_torch_gpu
            # xf_gpu, xb_gpu = batch_optimize_allocation_gpu_gaussian(
            #    r[:,0], p1, r[:,1], p2, 
            #    rho, C_hb.to(self.device), C_sb.to(self.device), C_f2b.to(self.device), C_b2f.to(self.device), max_k=200
            # )                
                        
            prediction = torch.concat([xf_gpu.unsqueeze(1),xb_gpu.unsqueeze(1)],-1)
            # Append predictions and true labels to the lists
            predictions.extend(prediction.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())            
        
        folder_path = path+ '/results/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path  + 'test_decision.npy', predictions)
        np.save(folder_path  + 'test_true.npy', true_labels)
        return
    
    def optimize_quantile(self,setting,model_setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        print('loading model')
        path = os.path.join(self.args.checkpoints, setting)
        model_path = os.path.join(self.args.checkpoints, model_setting)
        if load:                
            best_model_path = os.path.join(model_path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        nsample = self.args.nsample    
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)      
                
                p1 = F.sigmoid(p[:,0:1])
                if p.shape[1] >= 3:
                    p2 = F.sigmoid(p[:,2:3])
                else: 
                    p2 = F.sigmoid(p[:,0:1])         
                dist_k = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,0:1],probs = p1)   
                dist_m = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,1:2],probs = p2) 
                prediction_k = [dist_k.sample() for i in range(nsample)]  
                prediction_m = [dist_m.sample() for i in range(nsample)] 
                quantile_k = self.args.cs/(self.args.cs+self.args.ch)
                quantile_m = quantile_k
                prediction_k = torch.stack(prediction_k).quantile(quantile_k,dim=0)
                prediction_m = torch.stack(prediction_m).quantile(quantile_m,dim=0)
                prediction = torch.concat([prediction_k,prediction_m],-1)
                # Append predictions and true labels to the lists
                predictions.extend(prediction.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())            
           
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_decision_q.npy', predictions)
            #np.save(folder_path  + 'test_true.npy', true_labels)
        return
    
    
class Exp_single(Exp_Main):
       
    def predict(self,setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_k,x_m, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_k.to(self.device),x_m.to(self.device),)
                
                prediction = torch.stack([r[:,0]*targets[:,3].to(self.device),r[:,0]*(1-targets[:,3].to(self.device))])
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy().T)
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return



