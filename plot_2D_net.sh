#!/bin/bash

# input parent directory of the network as $1
# iteration number as $2
# whether to plot subplanes $3

# example:
# ./plot_2D_net.sh ../trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/PCA_weights_save_epoch\=3/ 2 

pushd $1 

surf_file=directions_iter_$2.h5_\[*_nw\=2.h5
if [ -z "$3" ]; then
    shopt -s extglob
    surf_file=directions_iter_$2.h5_\[!(*nw\=2).h5
#    shopt -u extglobz
else
proj_file=directions_iter_$2.h5_iter_$2_proj_cos.h5
dir_file=directions_iter_$2.h5

python plot_2D.py --pca_3d --surf_file $surf_file --proj_file $proj_file --dir_file $dir_file

popd
