"""
    Calculate and visualize the loss surface.
    Usage example:
    >>  python plot_surface.py --x=-1:1:101 --y=-1:1:101 --model resnet56 --cuda
"""
import argparse
import copy
import h5py
import torch
import time
import socket
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
import dataloader
import evaluation
import projection as proj
import net_plotter
import plot_2D
import plot_1D
import model_loader
import scheduler
import mpi4pytorch as mpi
import cifar10.constraints as constraints
import subplanes

def name_surface_file(args, dir_file, index=0):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))
    surf_file += '_Index=' + str(index)

    if args.subplanes:
        surf_file += ('_subplanes_nw=' + str(args.nw_subplane))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    return surf_file + ".h5"


def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file


def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, comm, rank, args):
    """
        Calculate the loss values and accuracies of modified models in parallel
        using MPI reduce.
    """

    f = h5py.File(surf_file, 'r+' if rank == 0 else 'r')
    losses, accuracies, inbound = [], [], []
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None

    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        inbound = np.ones(shape=shape)
        if rank == 0:
            f[loss_key] = losses
            f[acc_key] = accuracies
            f['inbound'] = np.ones(shape=shape)
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]
        inbound = f['inbound'][:]

    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates, comm)

    print('Computing %d values for rank %d'% (len(inds), rank))
    start_time = time.time()
    total_sync = 0.0

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    # Loop over all uncalculated loss values
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net

        # TODO get net with weights of the two boundary (or just pass b_list?)
        # TODO pass b_list (need x value of boundary)
        boundary = None
        if args.subplanes:
                # get x and weights at the boundaries 
                xl, xr, yl, yr, iwl, iwr = subplanes.locate_boundary(coord[0], args.b_list, args.save_epoch)
                print('x:', xl, xr, ', y:', yl, yr, ', iw:', iwl, iwr)
                # Load nets with weights of the boundaries
                netl = model_loader.load(args.dataset, args.model, net_plotter.get_model_name(args.model_folder, iwl))
                netr = model_loader.load(args.dataset, args.model, net_plotter.get_model_name(args.model_folder, iwr))
                wl, wr = net_plotter.get_weights(netl), net_plotter.get_weights(netr)
                boundary = {'wl': wl, 'wr': wr, 
                            'xl': xl, 'xr': xr,
                            'yl': yl, 'yr': yr}
                # Put this to changes

        if args.dir_type == 'weights':
            net_plotter.set_weights(net.module if args.ngpu > 1 else net, w, d, coord, args.ipca, boundary)  # return whether in constraints if msx norm
        elif args.dir_type == 'states':
            net_plotter.set_states(net.module if args.ngpu > 1 else net, s, d, coord, args.ipca, boundary)
        # constrain the weights to constr_param, or mark if out of bound? both!
        if args.constraint == 'max_norm':
            if args.modify_plane:
                constraints.max_norm(net.module if args.ngpu > 1 else net, args.max_norm_val)
            else:
                # mark whether point fall within constraints
                for name, param in net.named_parameters():
                    if 'bias' not in name:
                        norm = param.norm()
                        if norm > args.max_norm_val:
                            inbound.ravel()[ind] = 0
                            break            
                                
        # Record the time to compute the loss value
        loss_start = time.time()
        loss, acc = evaluation.eval_loss(net, criterion, dataloader, args.cuda, args.constraint, args.reg_rate)
        loss_compute_time = time.time() - loss_start

        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc

        # Send updated plot data to the master node
        syc_start = time.time()
        losses     = mpi.reduce_max(comm, losses)   # reduce between initialised -1, and loss values (get loss values)
        accuracies = mpi.reduce_max(comm, accuracies)
        inbound = mpi.reduce_min(comm, inbound)     # initialised to 1, only change in 0
        syc_time = time.time() - syc_start
        total_sync += syc_time

        # Only the master node writes to the file - this avoids write conflicts
        if rank == 0:
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f['inbound'][:] = inbound
            f.flush()

        print('Evaluating rank %d  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                rank, count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))

    # This is only needed to make MPI run smoothly. If this process has less work than
    # the rank0 process, then we need to keep calling reduce so the rank0 process doesn't block
    for i in range(max(inds_nums) - len(inds)):
        losses = mpi.reduce_max(comm, losses)
        accuracies = mpi.reduce_max(comm, accuracies)
        inbound = mpi.reduce_min(comm, inbound)

    total_time = time.time() - start_time
    print('Rank %d done!  Total time: %.2f Sync: %.2f' % (rank, total_time, total_sync))

    f.close()

###############################################################
#                          MAIN
###############################################################
def main(args):

    torch.manual_seed(123)
    #--------------------------------------------------------------------------
    # Environment setup
    #--------------------------------------------------------------------------
    if args.mpi:
        comm = mpi.setup_MPI()
        rank, nproc = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, nproc = None, 0, 1

    # in case of multiple GPUs per node, set the GPU to use for each rank
    if args.cuda:
        if not torch.cuda.is_available():
            raise Exception('User selected cuda option, but cuda is not available on this machine')
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        print('Rank %d use GPU %d of %d GPUs on %s' %
              (rank, torch.cuda.current_device(), gpu_count, socket.gethostname()))

    #--------------------------------------------------------------------------
    # Check plotting resolution
    #--------------------------------------------------------------------------
    try:
        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
        args.ymin, args.ymax, args.ynum = (None, None, None)
        if args.y:
            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
            assert args.ymin and args.ymax and args.ynum, \
            'You specified some arguments for the y axis, but not all'
    except:
        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

    #--------------------------------------------------------------------------
    # Load models and extract parameters
    #--------------------------------------------------------------------------
    net = model_loader.load(args.dataset, args.model, args.model_file)
    w = net_plotter.get_weights(net) # initial parameters
    s = copy.deepcopy(net.state_dict()) # deepcopy since state_dict are references
    if args.ngpu > 1:
        # data parallel with multiple GPUs on a single node
        net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    #--------------------------------------------------------------------------
    # Setup the direction file and the surface file
    #--------------------------------------------------------------------------
    dir_file = net_plotter.name_direction_file(args) # name the direction file
    if rank == 0:
        net_plotter.setup_direction(args, dir_file, net)

    surf_file = name_surface_file(args, dir_file)
    if rank == 0:
        setup_surface_file(args, surf_file, dir_file)

    # wait until master has setup the direction file and surface file
    mpi.barrier(comm)

    # load directions
    d = net_plotter.load_directions(dir_file)
    # calculate the consine similarity of the two directions
    if len(d) == 2 and rank == 0:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    #--------------------------------------------------------------------------
    # Setup dataloader
    #--------------------------------------------------------------------------
    # download CIFAR10 if it does not exit
    if rank == 0 and args.dataset == 'cifar10':
        torchvision.datasets.CIFAR10(root=args.dataset + '/data', train=True, download=True)

    mpi.barrier(comm)

    trainloader, testloader = dataloader.load_dataset(args.dataset, args.datapath,
                                args.batch_size, args.threads, args.raw_data,
                                args.data_split, args.split_idx,
                                args.trainloader, args.testloader)
    
    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    crunch(surf_file, net, w, s, d, trainloader, 'train_loss', 'train_acc', comm, rank, args)
    # crunch(surf_file, net, w, s, d, testloader, 'test_loss', 'test_acc', comm, rank, args)

    #--------------------------------------------------------------------------
    # Plot figures (may be just save and not show (use param))
    #--------------------------------------------------------------------------
    if args.plot and rank == 0:
        if args.y and args.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, args.proj_file, 'train_loss', args.show)
        elif args.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', args.vmin, args.vmax, args.vlevel, args.show)
        else:
            plot_1D.plot_1d_loss_err(surf_file, args.xmin, args.xmax, args.loss_max, args.log, args.show)
    
if __name__ == '__main__':
    # use 
    # mpirun -n 1 python plot_surface.py --mpi --cuda --model resnet56 --ipca 7 --xignore biasbn --yignore biasbn --model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3/model_300.t7  --dir_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3/PCA_weights_save_epoch=3/directions
    # mpirun -n 1 python plot_surface.py --mpi --cuda --model resnet56 --ipca 7 --model_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model_300.t7 --dir_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/PCA_weights_save_epoch=3/directions --ptj_prefix cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/PCA_weights_save_epoch=3/directions_iter_ --ptj_midfix .h5_iter_ --ptj_suffix _proj_cos.h5 --subplanes --nw_subplane 3 --constraint max_norm --max_norm_val 4 --save_epoch 3 --model_folder cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/
    # mpirun -n 1 python plot_surface.py --mpi --cuda --model resnet56_noshort --ipca 7 --model_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3_ngpu=2/model_300.t7 --dir_file cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3_ngpu=2/PCA_weights_save_epoch=3/directions --ptj_prefix cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3_ngpu=2/PCA_weights_save_epoch=3/directions_iter_ --ptj_midfix .h5_iter_ --ptj_suffix _proj_cos.h5 --subplanes --nw_subplane 2 --save_epoch 3 --model_folder cifar10/trained_nets/resnet56_noshort_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3_ngpu=2
    # mpirun -n 1 python plot_surface.py --mpi --cuda --model resnet56 --ipca 7 --model_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/model_300.t7 --dir_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/PCA_weights_save_epoch\=3/directions --ptj_prefix cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/PCA_weights_save_epoch\=3/directions_iter_ --ptj_midfix .h5_iter_ --ptj_suffix _proj_cos.h5 --subplanes --nw_subplane 2 --save_epoch 3 --model_folder cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/



    parser = argparse.ArgumentParser(description='plotting loss surface')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use for each rank, useful for data parallel evaluation')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')

    # data parameters
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='cifar10/data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='no data preprocessing')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # model parameters
    parser.add_argument('--model', default='resnet56', help='model name')
    parser.add_argument('--model_folder', default='', help='the common folder that contains model_file and model_file2')
    parser.add_argument('--model_file', default='', help='path to the trained model file')
    parser.add_argument('--model_file2', default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # direction parameters
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file. Use prefix of files instead for ipca > 0')
    parser.add_argument('--dir_type', default='weights', help='direction type: weights | states (including BN\'s running_mean/var)')
    parser.add_argument('--x', default='-1:1:51', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default=None, help='A string with format ymin:ymax:ynum')
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')

    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    # local PCA parameters
    parser.add_argument('--ipca', default=0, type=int, help='number of PCA directions to find the plot of')
    parser.add_argument('--icpca', default=0, type=int, help='number of PCA direction to continue from')
    parser.add_argument('--subplanes', action='store_true', default=False, help='whether to use subplanes offsets')
    parser.add_argument('--nw_subplane', default=0, type=int, help='number of weights per subplanes. ')
    parser.add_argument('--ptj_prefix', default='', help='prefix for projected trajectory files')
    parser.add_argument('--ptj_midfix', default='', help='middle text between iteration number')    # needed this because naming error when creating the projected files
    parser.add_argument('--ptj_suffix', default='', help='suffix for projected trajectory files')
    parser.add_argument('--num_w_per_pca', default=14, type=int, help='number of weight per PCA plot')  # for use with nw_subplane
    parser.add_argument('--save_epoch', default=3, type=int, help='interval where the net is saved during training')

    parser.add_argument('--only_find_boundaries', action='store_true', default=False, help='whether to not run usual crunch and just find the boundaries')

    # constraint parameters
    parser.add_argument('--constraint', default=None, help='constraint: max_norm | SRIP')
    parser.add_argument('--max_norm_val', default=3, type=float, help='max of weight norm to be used with max norm constraint')
    parser.add_argument('--bc', action='store_true', default=False, help='Whether to do only weights before constraint applied (use this arg), or only after')
    parser.add_argument('--reg_rate', default=0.01, type=float, help='regularizer constant to be used with SRIP regularizer')
    parser.add_argument('--modify_plane', default=False, help='Modify the weights to follow constraints')

    args = parser.parse_args()
    
    if args.constraint == 'max_norm':
        assert args.max_norm_val > 0, "max_norm_val must be greater than 0"
    if args.subplanes:
        assert args.nw_subplane > 0, "subplanes must be greater than 0"
    
    # pre-define range
    if args.ipca > 0:
        # cannot use 0 with y, or will assert false
        xdomains, ydomains = [], []
        if args.model == 'resnet56':
            if not args.constraint:
                xdomains = ['-10:40:25', 
                            '-13:14:25',
                            '-12:12:25',
                            '-7:16:25',
                            '1:17:25',
                            '-3:6:25',
                            '19:21:25']
                ydomains = ['-17:14:25', 
                            '-5:14:25',           # ends at 13.8
                            '-6:13:25',
                            '-5:16:25',
                            '-6:2:25',
                            '-1:6:25',            # no benefit (no in-between values)
                            '-0.3:0.12:25']
            elif args.constraint == 'max_norm' and args.max_norm_val == 4:
                if args.bc:
                    xdomains = ['-16.5:15:25', 
                                '-10:10:25',
                                '-9:9:25',
                                '-5:13:25',
                                '-3.5:10.5:25',
                                '-2:6:30',
                                '16.5:18.5:40']
                    ydomains = ['-15.5:8:25', 
                                '-4:11:25',         
                                '-4.5:10:25',
                                '-4.1:13:25',
                                '-3.2:4.1:25',
                                '-0.5:5.5:30',          
                                '-0.89:0.39:40']
                else:
                    xdomains = ['-16.5:15:25', 
                                '-10:10:25',
                                '-9:9:25',
                                '-5:13:25',
                                '-4:10.3:25',
                                '-2:5.5:30',
                                '16.5:18.5:40']
                    ydomains = ['-15.5:8:25', 
                                '-4:11:25',         
                                '-4.5:9.5:25',
                                '-4.5:13:25',
                                '-3.2:4.1:25',
                                '-0.5:5.5:30',          
                                '-0.89:-0.39:40']
            elif args.constraint == 'SRIP' and args.reg_rate == 0.01:
                xdomains = ['-10:40:25', 
                            '-13:14:25',
                            '-12:11:25',
                            '-7:16:25',
                            '1:17:25',
                            '-3:6:25',
                            '19:21:40']
                ydomains = ['-17:12:25', 
                            '-5:14:25',      
                            '-6:13:25',
                            '-5:16:25',
                            '-6:2:25',
                            '-1:6:25',    
                            '0.0001:0.8:40']
        elif args.model == 'resnet56_noshort':
            xdomains = ['-11:41:25', 
                        '-13.5:12:25',
                        '-1:0.001:25',		
                        '0:99:25',
                        '2:25:25',
                        '-1:5:30',
                        '8.9:17:40']
            ydomains = ['-9:19:25', 
                        '-12.5:2:25',     
                        '-9.5:9:25',
                        '-17:0.001:25',
                        '1.5:8.51:25',
                        '3:6:30',
                        '-3:0.001:40']	
            
        assert len(xdomains) >= args.ipca, 'number of x domains must be same or greater than ipca'
        assert len(ydomains) >= args.ipca, 'number of y domains must be same or greater than ipca'
        
        prefix = args.dir_file
        ptj_file = ''
        for i in range(args.icpca, args.ipca):
            args.x, args.y = xdomains[i], ydomains[i]
            args.dir_file = prefix + '_iter_' + str(i) + '.h5'
            
            if args.subplanes and args.nw_subplane > 0:
                ptj_file = args.ptj_prefix + str(i) + args.ptj_midfix + str(i) + args.ptj_suffix
                assert os.path.exists(ptj_file), "projected file %s does not exist" % ptj_file
                args.b_list = subplanes.boundary_list(ptj_file, args.nw_subplane, args.num_w_per_pca)
                assert args.b_list is not None, "error processing boundary list"

                if args.only_find_boundaries:
                    try:
                        args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
                        args.ymin, args.ymax, args.ynum = (None, None, None)
                        if args.y:
                            args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
                            assert args.ymin and args.ymax and args.ynum, \
                            'You specified some arguments for the y axis, but not all'
                    except:
                        raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')

                    surf_file = name_surface_file(args, args.dir_file)
                    bound_file = subplanes.boundaries_dict(args, args.b_list, surf_file)
                    d = net_plotter.load_directions(args.dir_file)
                    subplanes.similarity(args, bound_file, d)

            if not args.only_find_boundaries:
                main(args)
            print("Finished iter: " + str(i))
    else:
        main(args)
           
