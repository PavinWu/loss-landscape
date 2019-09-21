#!/bin/bash

# input parent directory of the network as $2
# iteration number as $3
# whether to plot subplanes $1

# example:
# ./plot_2D_net.sh ../trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3/PCA_weights_save_epoch\=3/ 2 

pushd $2 

shopt -s extglob    # need to be run outside!
surf_file="$2"/`ls directions_iter_$3.h5_\[!(*nw\=2).h5`     # assume no space
if [ $1 == "--subplanes" ]; then
     surf_file="$2"/`ls directions_iter_$3.h5_\[*_nw\=2.h5`
#    shopt -u extglobz
fi
echo "surf_file: "$surf_file
popd

proj_file="$2"/directions_iter_$3.h5_iter_$3_proj_cos.h5
dir_file="$2"/directions_iter_$3.h5

if [ $4 == "--show_constraint" ]; then
    python plot_2D.py --pca_3d --surf_file $surf_file --proj_file $proj_file --dir_file $dir_file --show_constraint
else
    python plot_2D.py --pca_3d --surf_file $surf_file --proj_file $proj_file --dir_file $dir_file
fi