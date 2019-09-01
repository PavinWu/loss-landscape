from __future__ import print_function
import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel

import model_loader
import dataloader

import constraints

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def save_net(test_err, epoch, net, args, optimizer, save_folder, prefix_extra=''):
    acc = 100 - test_err
    state = {
        'acc': acc,
        'epoch': epoch,
        'state_dict': net.module.state_dict() if args.ngpu > 1 else net.state_dict(),
    }
    opt_state = {
        'optimizer': optimizer.state_dict()
    }

    torch.save(state, 'trained_nets/' + save_folder + '/model_' + prefix_extra + str(epoch) + '.t7')
    torch.save(opt_state, 'trained_nets/' + save_folder + '/opt_state_' + prefix_extra + str(epoch) + '.t7')
    
# Training
def train(trainloader, net, criterion, optimizer, use_cuda=True, constraint=None, constr_param=None, temp_net=None, temp_opt=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            
            loss = criterion(outputs, targets)
            if constraint == 'SRIP':
                reg_loss = Variable(constraints.SRIP(net, constr_param))
                loss += reg_loss
            loss.backward()
            optimizer.step()
            
            if constraint == 'max_norm':
                temp_net.load_state_dict(net.module.state_dict() if args.ngpu > 1 else net.state_dict())
                temp_opt.load_state_dict(optimizer.state_dict())
                constraints.max_norm(net, constr_param)
                
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):     # not used in this experiment
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return train_loss/total, 100 - 100.*correct/total


def test(testloader, net, criterion, use_cuda=True, constraint=None, constr_param=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if isinstance(criterion, nn.CrossEntropyLoss):
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if constraint == 'SRIP':
                reg_loss = Variable(constraints.SRIP(net, constr_param))
                loss += reg_loss
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().item()

    elif isinstance(criterion, nn.MSELoss):     # not used in this experiment
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_size = inputs.size(0)
            total += batch_size

            one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
            one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
            one_hot_targets = one_hot_targets.float()
            if use_cuda:
                inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
            inputs, one_hot_targets = Variable(inputs), Variable(one_hot_targets)
            outputs = F.softmax(net(inputs))
            loss = criterion(outputs, one_hot_targets)
            test_loss += loss.item()*batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.cpu().eq(targets).cpu().sum().item()

    return test_loss/total, 100 - 100.*correct/total

def name_save_folder(args):
    save_folder = args.model + '_' + str(args.optimizer) + '_lr=' + str(args.lr)
    if args.lr_decay != 0.1:
        save_folder += '_lr_decay=' + str(args.lr_decay)
    save_folder += '_bs=' + str(args.batch_size)
    save_folder += '_wd=' + str(args.weight_decay)
    save_folder += '_mom=' + str(args.momentum)
    save_folder += '_save_epoch=' + str(args.save_epoch)
    if args.loss_name != 'crossentropy':
        save_folder += '_loss=' + str(args.loss_name)
    if args.noaug:
        save_folder += '_noaug'
    if args.raw_data:
        save_folder += '_rawdata'
    if args.label_corrupt_prob > 0:
        save_folder += '_randlabel=' + str(args.label_corrupt_prob)
    if args.ngpu > 1:
        save_folder += '_ngpu=' + str(args.ngpu)
    if args.idx:
        save_folder += '_idx=' + str(args.idx)
    if args.constraint:
        save_folder += '_constraint=' + str(args.constraint)
        if args.constraint == 'max_norm':
            save_folder += '_max_norm_val=' + str(args.max_norm_val)
        elif args.constraint == 'SRIP':
            save_folder += '_reg_rate=' + str(args.reg_rate)

    return save_folder

    
if __name__ == '__main__':
    # e.g. parameter to use: --batch_size 128 --save_epoch 3 --model 'resnet56' --resume_model 'trained_nets\resnet56_sgd_lr=0.1_bs=8_wd=0.0005_mom=0.9_save_epoch=3\model_18.t7' --resume_opt 'trained_nets\resnet56_sgd_lr=0.1_bs=8_wd=0.0005_mom=0.9_save_epoch=3\opt_state_18.t7'

    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--save', default='trained_nets',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--resume_model', default='', help='resume model from checkpoint')
    parser.add_argument('--resume_opt', default='', help='resume optimizer from checkpoint')

    # model parameters
    parser.add_argument('--model', '-m', default='vgg9')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--noaug', default=False, action='store_true', help='no data augmentation')
    parser.add_argument('--label_corrupt_prob', type=float, default=0.0)
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    # constraint parameters
    parser.add_argument('--constraint', default=None, help='constraint: max_norm | SRIP')
    parser.add_argument('--max_norm_val', default=3, type=float, help='max of weight norm to be used with max norm constraint')
    parser.add_argument('--reg_rate', default=0.01, type=float, help='regularizer constant to be used with SRIP regularizer')

    parser.add_argument('--idx', default=0, type=int, help='the index for the repeated experiment')     # TODO what for?

    args = parser.parse_args()

    constraint_param = ""
    if args.constraint == 'max_norm':
        print('\nMaximum weight norm: %f' % args.max_norm_val)
        constraint_param = args.max_norm_val
    elif args.constraint == 'SRIP':
        print('\nRegularizer constant: %f' % args.reg_rate)
        constraint_param = args.reg_rate

    print('\nLearning Rate: %f' % args.lr)
    print('\nDecay Rate: %f' % args.lr_decay)

    use_cuda = torch.cuda.is_available()
    print('Current devices: ' + str(torch.cuda.current_device()))
    print('Device count: ' + str(torch.cuda.device_count()))

    # Set the seed for reproducing the results
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.rand_seed)
        cudnn.benchmark = True

    lr = args.lr  # current learning rate
    start_epoch = 1  # start from epoch 1 or last checkpoint epoch

    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    save_folder = name_save_folder(args)
    if not os.path.exists('trained_nets/' + save_folder):
        os.makedirs('trained_nets/' + save_folder)

    f = open('trained_nets/' + save_folder + '/log.out', 'a')

    trainloader, testloader = dataloader.get_data_loaders(args)

    if args.label_corrupt_prob and not args.resume_model:
        torch.save(trainloader, 'trained_nets/' + save_folder + '/trainloader.dat')
        torch.save(testloader, 'trained_nets/' + save_folder + '/testloader.dat')

    # Model
    if args.resume_model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume_model)
        net = model_loader.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        net = model_loader.load(args.model)
        print(net)
        init_params(net)

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    if args.loss_name == 'mse':
        criterion = nn.MSELoss()

    if use_cuda:
        net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        if args.constraint == 'max_norm':
            temp_opt = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.constraint == 'max_norm':
            temp_opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_opt:
        checkpoint_opt = torch.load(args.resume_opt)
        optimizer.load_state_dict(checkpoint_opt['optimizer'])
        
    # temp model to save intermediate states
    if args.constraint == 'max_norm':
        temp_net = model_loader.load(args.model)
        init_params(temp_net)
    else:
        temp_net = None
        temp_opt = None

    # record the performance of initial model
    if not args.resume_model:
        train_loss, train_err = test(trainloader, net, criterion, use_cuda, args.constraint, constraint_param)
        test_loss, test_err = test(testloader, net, criterion, use_cuda, args.constraint, constraint_param)
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        epoch = 0
        save_net(test_err, epoch, net, args, optimizer, save_folder, 'bc_')     # same network before constraint as after constraint applied
        save_net(test_err, epoch, net, args, optimizer, save_folder)

    for epoch in range(start_epoch, args.epochs + 1):
        loss, train_err = train(trainloader, net, criterion, optimizer, use_cuda, args.constraint, constraint_param, temp_net, temp_opt)
        test_loss, test_err = test(testloader, net, criterion, use_cuda, args.constraint, constraint_param)

        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        # Save checkpoint.
        if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150:
            save_net(test_err, epoch, temp_net, args, temp_opt, save_folder, 'bc_')
            save_net(test_err, epoch, net, args, optimizer, save_folder)

        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay

    f.close()
