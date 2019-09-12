import torch
import numpy as np
import h5py

def x_coord(elem):
    return elem[1]

def boundary_list(projtraj_file, num_plane, num_w_per_pca):
    b_list = []

    f = h5py.File(projtraj_file, 'r')
    if not ('proj_xcoord' in f.keys()) or not ('proj_ycoord' in f.keys()):
        f.close()
        print("Cannot find projected x and/or y coordinates in " + projtraj_file)
        return None        
    if num_plane > num_w_per_pca + 1:
        print("Number of subplane must be less than or equal to number of weights per PCA")
        return None

    l = len(f['proj_xcoord'])
    nw_inplane = l//num_plane   # number of weights in a subplane
    nw_inplane_left = l%num_plane

    pjt_list = [[i,x,y] for i,x,y in zip(range(l), f['proj_xcoord'], f['proj_ycoord'])]
    
    # sort projected x coordinates
    x_end = pjt_list[-1][1]
    x_begin = pjt_list[-(num_w_per_pca+1)][1]   # actual num is + 1
    pjt_list_sorted = []
    if x_begin > x_end:     # always make i of x_end comes later
        pjt_list_sorted = sorted(ptj_list, reverse=True, key=x_coord)
    else:
        pjt_list_sorted = sorted(ptj_list, key=x_coord)

    # determine boundaries
    nw_extra = 1
    i = 1
    i_max = 0      # i of sorted list, containing max index in subplane of original list 
    for ip in range(0, num_plane):
        if nw_inplane_left <= 0:
            nw_extra = 0
        else:
            nw_inplane_left -= 1

        for iw in range(0, nw_inplane+nw_extra):
            if pjt_list_sorted[i-1][0] > pjt_list_sorted[i][0]:
                i_max = i-1     
            elif iw == 0:
                b_list.append(pjt_list_sorted[i-1])
            i += 1
        b_list.append(pjt_list_sorted[i_max])
        if ip == num_plane - 1:
            b_list.append(pjt_list_sorted[i-1])             

    return b_list