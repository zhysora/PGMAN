import torch
import torch.nn.functional as F
import numpy as np
from functions import *

def Q(a,  b): # N x H x W
    E_a = torch.mean(a, dim=(1,2))
    E_a2 = torch.mean(a * a, dim=(1,2))
    E_b = torch.mean(b, dim=(1,2))
    E_b2 = torch.mean(b * b, dim=(1,2))
    E_ab = torch.mean(a * b, dim=(1,2))

    var_a = E_a2 - E_a * E_a
    var_b = E_b2 - E_b * E_b
    cov_ab = E_ab - E_a * E_b

    return torch.mean(4 * cov_ab * E_a * E_b / (var_a + var_b) / (E_a**2 + E_b**2))

def D_lambda(ps, l_ms): # N x C x H x W
    L = ps.shape[1]
    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        for j in range(L):
            if j!=i:
                sum += torch.abs(Q(ps[:,i,:,:], ps[:,j,:,:]) - Q(l_ms[:,i,:,:], l_ms[:,j,:,:]))
    
    return sum/L/(L-1)

def D_s(ps, l_ms, pan, l_pan): # N x C x H x W
    L = ps.shape[1]

    sum = torch.Tensor([0]).to(ps.device, dtype=ps.dtype)

    for i in range(L):
        sum += torch.abs(Q(ps[:,i,:,:], pan[:,0,:,:]) - Q(l_ms[:,i,:,:], l_pan[:,0,:,:]))

    return sum/L
