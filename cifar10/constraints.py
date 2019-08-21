import torch
import numpy as np

def max_norm(parameters, max_norm):
    for p in parameters():
        w = p.data
        if w.norm() > max_norm:
            w = w.mul(max_norm/w.norm() + 1e-10) 

def SRIP(parameters, d_rate):
    # Adapted for Python from https://github.com/TAMU-VITA/Orthogonality-in-CNNs/blob/master/ResNet/resnet_cifar_new.py

    for p in parameters():
        w = p.data
        if w.dim() == 4:
            filter_size = w.size()
            row_dims = filter_size[1]*filter_size[2]*filter_size[3]     #input channel * height * width
            col_dims = filter_size[0]   # output channel

            W = w.view(row_dims, col_dims)
            Wt = torch.transpose(W, 0, 1)

            I = torch.from_numpy(np.eye(col_dims)).float64()
            WtW = torch.mm(Wt,W)
            norm = WtW - I          # to be used to power iteration to find spectral norm

            # random approx. dominant eigenvector 
            v = np.reshape( np.random.rand(norm.shape[1]), (norm.shape[1], 1) )
            v = torch.from_numpy(v).float()
            
            # One step power iteration (https://en.wikipedia.org/wiki/Power_iteration)
            b0 = torch.mm(norm, v)          
            norm_b0 = torch.sum(b0**2)**0.5   
            b1 = torch.div(b0, norm_b0)
            norm_b2 = d_rate * torch.sum( (torch.mm(norm, b1))**2 )**0.5

            return Variable(norm_b2)
        else:
            return 0