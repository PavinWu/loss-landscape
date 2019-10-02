"""
    2D plotting funtions
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns
import subplanes


def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])
    else:
        print ('%s is not found in %s' % (surf_name, surf_file))

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
    print(Z)

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # --------------------------------------------------------------------
    # Plot 2D contours
    # --------------------------------------------------------------------
    fig = plt.figure()
    CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    fig.savefig(surf_file + '_' + surf_name + '_2dcontour' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    fig = plt.figure()
    print(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf')
    CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
    fig.savefig(surf_file + '_' + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    fig = plt.figure()
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + surf_name + '_2dheat.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    f.close()
    if show: plt.show()


def plot_trajectory(proj_file, dir_file, show=False):
    """ Plot optimization trajectory on the plane spanned by given directions."""

    assert exists(proj_file), 'Projection file does not exist.'
    f = h5py.File(proj_file, 'r')
    fig = plt.figure()
    plt.plot(f['proj_xcoord'], f['proj_ycoord'], marker='.')
    plt.tick_params('y', labelsize='x-large')
    plt.tick_params('x', labelsize='x-large')
    f.close()

    if exists(dir_file):
        f2 = h5py.File(dir_file,'r')
        if 'explained_variance_ratio_' in f2.keys():
            ratio_x = f2['explained_variance_ratio_'][0]
            ratio_y = f2['explained_variance_ratio_'][1]
            plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
            plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
        f2.close()

    fig.savefig(proj_file + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
    if show: plt.show()

def plot_3d_contour_trajectory(args, surf_file, dir_file, proj_file, surf_name='train_loss',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    # WARNING: not adapted to be used outside of this fyp. Further clean up is required.
    # TODO script to automate plotting of a network (bash script)
    
    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])
        # xTODO change NaN to max
        # helpful: np.nanmax, https://stackoverflow.com/questions/5124376/convert-nan-value-to-zero
        Z[np.isnan(Z)] = np.nanmax(Z) + 2
        Z = np.log(Z)
    elif surf_name == 'train_err' or surf_name == 'test_err' :
        Z = 100 - np.array(f[surf_name][:])

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)   # TODO colour issues
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.pdf', dpi=300,
                bbox_inches='tight', format='pdf')

    # xTODO mark constraints
    in_bound_list = []
    if args.show_constraints:       
        
        in_bound = np.array(f['inbound'][:])
        for j, row in enumerate(in_bound):
            for i, wb in enumerate(row):
                if wb == 0:
                    in_bound_list.append([x[i], y[j], Z[j][i]])
        in_bound_arr = np.transpose(np.array(in_bound_list))
        ax.scatter(in_bound_arr[0], in_bound_arr[1], in_bound_arr[2], marker='x', color='r')

    f.close()

    # plot trajectories
    # xTODO rainbow colour as epoch increases
    pf = h5py.File(proj_file, 'r')
    colours = np.arange(len(pf['proj_xcoord'][:]))
    pf_log_loss = np.log(pf['loss'][:])
    ax.scatter(pf['proj_xcoord'][:], pf['proj_ycoord'][:], pf_log_loss, marker='o', c=colours, cmap='rainbow')   # TODO check for loss attribute
    ax.plot(pf['proj_xcoord'][:], pf['proj_ycoord'][:], pf_log_loss, 'k-', alpha=0.1)  
    
    epoch_label_intv = 2
    len_pf = len(pf['proj_xcoord'][:])
    for iw in range(0, len_pf, epoch_label_intv):
        ax.text(pf['proj_xcoord'][iw], pf['proj_ycoord'][iw], pf_log_loss[iw], '%s' % (str(iw*3)), size=8+(iw*3)//50, zorder=1, color='k') 

    # plot red points when learning rate decays
    # xTODO e won't corespond to index
    for e in [150//3, 225//3, 276//3]:
        if len_pf > e:
            plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='v', color='r')

    # xTODO marker which weight used as boundary (TODO may be some bugs)
    if args.show_boundaries:        # not used/tested
        bf = h5py.File(args.bound_file, 'r')
        b_list = bf['b_list'][:]
        b_list = np.transpose(np.array(sorted(b_list)))    # sorted by index
        b_list_loss = pf_log_loss[b_list[0]]
        ax.scatter(b_list[1], b_list[2], b_list_loss, marker='|', color='k')
        bf.close()
    
    ax.set_xlim3d(np.min(x), np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    df.close()
    #plt.clabel(CS1, inline=1, fontsize=6)
    #plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pf.close()
    if show: plt.show()

def plot_contour_trajectory(surf_file, dir_file, proj_file, surf_name='loss_vals',
                            vmin=0.1, vmax=10, vlevel=0.5, show=False):
    """2D contour + trajectory"""

    assert exists(surf_file) and exists(proj_file) and exists(dir_file)

    # plot contours
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    if surf_name in f.keys():
        Z = np.array(f[surf_name][:])

    fig = plt.figure()
    CS1 = plt.contour(X, Y, Z)#, levels=np.arange(vmin, vmax, vlevel))
    CS2 = plt.contour(X, Y, Z)#, np.logspace(1, 8, num=8))

    # plot trajectories
    pf = h5py.File(proj_file, 'r')
    plt.plot(pf['proj_xcoord'], pf['proj_ycoord'], marker='.')

    # plot red points when learning rate decays
    # for e in [150, 225, 275]:
    #     plt.plot([pf['proj_xcoord'][e]], [pf['proj_ycoord'][e]], marker='.', color='r')

    # add PCA notes
    df = h5py.File(dir_file,'r')
    ratio_x = df['explained_variance_ratio_'][0]
    ratio_y = df['explained_variance_ratio_'][1]
    plt.xlabel('1st PC: %.2f %%' % (ratio_x*100), fontsize='xx-large')
    plt.ylabel('2nd PC: %.2f %%' % (ratio_y*100), fontsize='xx-large')
    df.close()
    plt.clabel(CS1, inline=1, fontsize=6)
    plt.clabel(CS2, inline=1, fontsize=6)
    fig.savefig(proj_file + '_' + surf_name + '_2dcontour_proj.pdf', dpi=300,
                bbox_inches='tight', format='pdf')
    pf.close()
    if show: plt.show()


def plot_2d_eig_ratio(surf_file, val_1='min_eig', val_2='max_eig', show=False):
    """ Plot the heatmap of eigenvalue ratios, i.e., |min_eig/max_eig| of hessian """

    print('------------------------------------------------------------------')
    print('plot_2d_eig_ratio')
    print('------------------------------------------------------------------')
    print("loading surface file: " + surf_file)
    f = h5py.File(surf_file,'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)

    Z1 = np.array(f[val_1][:])
    Z2 = np.array(f[val_2][:])

    # Plot 2D heatmaps with color bar using seaborn
    abs_ratio = np.absolute(np.divide(Z1, Z2))
    print(abs_ratio)

    fig = plt.figure()
    sns_plot = sns.heatmap(abs_ratio, cmap='viridis', vmin=0, vmax=.5, cbar=True,
                           xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_abs_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')

    # Plot 2D heatmaps with color bar using seaborn
    ratio = np.divide(Z1, Z2)
    print(ratio)
    fig = plt.figure()
    sns_plot = sns.heatmap(ratio, cmap='viridis', cbar=True, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure().savefig(surf_file + '_' + val_1 + '_' + val_2 + '_ratio_heat_sns.pdf',
                                  dpi=300, bbox_inches='tight', format='pdf')
    f.close()
    if show: plt.show()



if __name__ == '__main__':
    # python plot_2D.py --surf_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_0.h5_[-16.5,15.0,25]x[-15.5,8.0,25]_Index=0_subplanes_nw=2.h5 --proj_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_0.h5_iter_0_proj_cos.h5 --dir_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_0.h5

    #python plot_2D.py --surf_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_0.h5_[16.5,18.5,40]x[-0.89,-0.39,40]_Index\=0_subplanes_nw\=2.h5 --proj_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_6.h5_iter_6_proj_cos.h5 --dir_file cifar10/trained_nets/resnet56_sgd_lr\=0.1_bs\=128_wd\=0.0005_mom\=0.9_save_epoch\=3_constraint\=max_norm_max_norm_val\=4.0/model/PCA_weights_save_epoch\=3/directions_iter_6.h5


    parser = argparse.ArgumentParser(description='Plot 2D loss surface')
    parser.add_argument('--surf_file', '-f', default='', help='The h5 file that contains surface values')
    parser.add_argument('--dir_file', default='', help='The h5 file that contains directions')
    parser.add_argument('--proj_file', default='', help='The h5 file that contains the projected trajectories')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--zlim', default=10, type=float, help='Maximum loss value to show')
    parser.add_argument('--show', action='store_true', default=False, help='show plots')
    parser.add_argument('--pca_3d', action='store_true', default=False, help='Use 3D plot for the loss surface and trajectory')
    parser.add_argument('--show_constraints', action='store_true', default=False, help='Mark points not with in constraints')
    
    parser.add_argument('--show_boundaries', action='store_true', default=False, help='Mark trajectory weights used as boundaries')
    parser.add_argument('--bound_file', default='', help='File containing boundary dict')
    
    args = parser.parse_args()

    """
    args.vmin = 2.3
    args.vmax = 3
    args.vlevel = 0.001
    args.zlim = 100
    """
    args.show = True
    if exists(args.surf_file) and exists(args.proj_file) and exists(args.dir_file):
        if not args.pca_3d:
            plot_contour_trajectory(args.surf_file, args.dir_file, args.proj_file,
                                    args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
        else:
            plot_3d_contour_trajectory(args, args.surf_file, args.dir_file, args.proj_file, args.surf_name,
                            args.vmin, args.vmax, args.vlevel, args.show)
    elif exists(args.proj_file) and exists(args.dir_file):
        plot_trajectory(args.proj_file, args.dir_file, args.show)
    elif exists(args.surf_file):
        plot_2d_contour(args.surf_file, args.surf_name, args.vmin, args.vmax, args.vlevel, args.show)
