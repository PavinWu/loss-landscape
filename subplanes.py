import h5py

def get_b_list(num_plane, nw_inplane, nw_inplane_left, pjt_list_sorted, print_lists):
    """
    Get a list of x-axis boundary of each subplane, for all subplanes
    The boundary is the first, last x of the subplane
        AND the x value which has the Max index in the trajectory list
        i.e. latest weight in trajectory
        IF there exists non-monotonic increment of (original trajectory list) index 
            in the sorted trajectory list.
        This is so there is a subplane which intersect the later weight in case the trajectory wraps around in x-axis,
            rather than just showing the plane for the old weights.
    """
    b_list = []

    nw_extra = 0
    i_max = None    # i of sorted list, containing max index in subplane of original list 
    i = 0
    for ip in range(0, num_plane):
        if ip > num_plane - nw_inplane_left - 1:    # add extra to last
            nw_extra = 1
            
        mono_ascend = True
        for iw in range(0, nw_inplane+nw_extra):
            if iw == 0:
                if ip == 0:
                    i_max = 0
                if pjt_list_sorted[i_max][0] > pjt_list_sorted[i][0]:   #when non-ascending only after compare to current bound
                    b_list.append(pjt_list_sorted[i_max])   
                    i_max = i       
                b_list.append(pjt_list_sorted[i])   # current bound
            elif pjt_list_sorted[i_max][0] <= pjt_list_sorted[i][0]:
                i_max = i

            if pjt_list_sorted[i_max][0] > pjt_list_sorted[i][0]:
                mono_ascend = False
            i += 1

        if (not mono_ascend) and (b_list[-1][0] != pjt_list_sorted[i_max][0]):
            b_list.append(pjt_list_sorted[i_max])   
            if ip < num_plane - 1:
                i_max = i   # don't want i_max lingering once it's added to b_list
        if (ip == num_plane - 1) and (b_list[-1][0] != pjt_list_sorted[i-1][0]):    # last bound
            b_list.append(pjt_list_sorted[i-1])       

        if print_lists:
            for j, elem in enumerate(b_list):
                print("i=", i, "b_list index=", j, elem)
            print()

    return b_list

def x_coord(elem):
    return elem[1]

def boundary_list(projtraj_file, num_plane, num_w_per_pca):
    
    projtraj_file = '/home/uvin/trained_nets/resnet56_sgd_lr=0.1_bs=64_wd=0.0005_mom=0.9_save_epoch=3/PCA_weights_save_epoch=3/directions_iter_0.h5_iter_0_proj_cos.h5'
    #num_plane = 14
    num_w_per_pca = 14
    
    print_lists = False

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
        pjt_list_sorted = sorted(pjt_list, reverse=True, key=x_coord)
    else:
        pjt_list_sorted = sorted(pjt_list, key=x_coord)
    
    if print_lists:
        print('pjt_list_sorted:')
        for i, elem in enumerate(pjt_list_sorted):
            print(i, elem)

    # determine boundaries
    return get_b_list(num_plane, nw_inplane, nw_inplane_left, pjt_list_sorted, print_lists)