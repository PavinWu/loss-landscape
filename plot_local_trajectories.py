"""
    Plot the number of optimization path in the space spanned by local principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader
import net_plotter
from projection import setup_PCA_directions, project_trajectory
#import plot_2D
import time


if __name__ == '__main__':
    # use python plot_local_trajectories.py --model_folder C:/Users/power/Downloads/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005_mom=0.9_save_epoch=3 --num_local_epoch 14
    
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--model', default='resnet56', help='trained models')
    parser.add_argument('--model_folder', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--prefix', default='model_', help='prefix for the checkpint model')
    parser.add_argument('--suffix', default='.t7', help='prefix for the checkpint model')
    parser.add_argument('--start_epoch', default=0, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=300, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=3, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')
    parser.add_argument('--num_local_epoch', default=5, type=int, 
        help='number of weight vectors to be used in local PCA direction, must be divisible by (max_epoch-start_epoch)/save_epoch')

    args = parser.parse_args()

    """
    assert ((args.max_epoch - args.start_epoch)/args.save_epoch ) % args.num_local_epoch == 0, 
                    'num_local_epoch must be divisible by (max_epoch-start_epoch)/save_epoch'     
    """ # # don't have to end at last!
    num_pca = int(((args.max_epoch - args.start_epoch)/args.save_epoch + 1) // args.num_local_epoch)

    model_files=[]
    for i_pca in range(num_pca):
        start_time = time.time()    
    
        offset_start = i_pca*args.num_local_epoch * args.save_epoch
        offset_end = (i_pca+1)*args.num_local_epoch * args.save_epoch
        
        #--------------------------------------------------------------------------
        # load the final model of each local PCA
        #--------------------------------------------------------------------------
        last_model_file = args.model_folder + '/' + args.prefix + str(args.start_epoch+offset_end) + args.suffix # NOTE 
        net = model_loader.load(args.dataset, args.model, last_model_file)
        w = net_plotter.get_weights(net)
        s = net.state_dict()

        #--------------------------------------------------------------------------
        # collect models to be projected
        #--------------------------------------------------------------------------
        current_model_files = []
        for epoch in range(args.start_epoch+offset_start, args.start_epoch+offset_end+1, args.save_epoch): # +1 to get to last one
            model_file = args.model_folder + '/' + args.prefix + str(epoch) + args.suffix
            assert os.path.exists(model_file), 'model %s does not exist' % model_file
            current_model_files.append(model_file)
            model_files.append(model_file)

        #--------------------------------------------------------------------------
        # load or create projection directions
        #--------------------------------------------------------------------------
        if args.dir_file:
            dir_file = args.dir_file
        else:
            dir_file = setup_PCA_directions(args, current_model_files, w, s, i_pca)

        #--------------------------------------------------------------------------
        # projection trajectory to given directions
        #--------------------------------------------------------------------------
        # need to include weights from previous iterations in this iteration, re-project onto new axes
        # need to save with different names each iteration
        proj_file = project_trajectory(dir_file, w, s, args.dataset, args.model,
                                    model_files, args.dir_type, 'cos', i_pca)  # TODO
        #plot_2D.plot_trajectory(proj_file, dir_file) 
        print("Finished iteration: " + str(i_pca) + ". Time: " + str(time.time()-start_time) + " s")
