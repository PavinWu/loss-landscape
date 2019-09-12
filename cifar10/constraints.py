import torch
import numpy as np

def max_norm(model, max_norm):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm()
            if norm > max_norm:
                param.data.copy_(param * (max_norm / (1e-10 + norm)))   # won't work without copy_???
            
def SRIP(model, d_rate):
    # Adapted for Python from https://github.com/TAMU-VITA/Orthogonality-in-CNNs/blob/master/ResNet/resnet_cifar_new.py
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'bias' not in name and param.dim() == 4:
            filter_size = param.size()
            row_dims = filter_size[1]*filter_size[2]*filter_size[3]     #input channel * height * width
            col_dims = filter_size[0]   # output channel

            W = param.view(row_dims, col_dims)
            Wt = torch.transpose(W, 0, 1)
            
            I = torch.from_numpy(np.eye(col_dims)).float().cuda()   # assume using cuda
            WtW = torch.mm(Wt,W)
            norm = WtW - I          # to be used to power iteration to find spectral norm

            # random approx. dominant eigenvector 
            v = np.reshape( np.random.rand(norm.shape[1]), (norm.shape[1], 1) )
            v = torch.from_numpy(v).float().cuda()
            
            # One step power iteration (https://en.wikipedia.org/wiki/Power_iteration)
            b0 = torch.mm(norm, v)          
            norm_b0 = torch.sum(b0**2)**0.5   
            b1 = torch.div(b0, norm_b0)
            norm_b2 = torch.sum( (torch.mm(norm, b1))**2 )**0.5
            reg_loss += norm_b2
    return d_rate * reg_loss
