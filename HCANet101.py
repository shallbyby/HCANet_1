#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8
# !/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from network.IGARSS.HCA101 import FCNRes101
import torch.optim as optim
from loader.load_postdam_test import vaihingenloader
from torch.utils.data import DataLoader
from metrics.metrics_potsdam import runningScore, averageMeter
import torch.backends.cudnn as cudnn
from utils.modeltools import netParams
from utils.set_logger import get_logger
import utils.utils

import warnings
warnings.filterwarnings('ignore')
# setup scheduler
def adjust_learning_rate(cur_epoch, max_epoch, curEpoch_iter, perEpoch_iter, baselr):
    """
    poly learning stategyt
    lr = baselr*(1-iter/max_iter)^power
    """
    cur_iter = cur_epoch*perEpoch_iter + curEpoch_iter
    max_iter = max_epoch*perEpoch_iter
    lr = baselr*pow((1 - 1.0*cur_iter/max_iter), 0.9)

    return lr


def test(args, testloader, model, criterion, epoch, logger ):
    '''
    args:
        test_loader: loaded for test dataset
        model: model
    return:
        mean IoU, IoU class
    '''
    model.eval()
    tloss = 0.
    # Setup Metrics
    running_Metrics = runningScore(args.num_classes)
    total_batches = len(testloader)
    print("=====> the number of iterations per epoch: ", total_batches)
    with torch.no_grad():
        for iter, batch in enumerate(testloader):
            # start_time = time.time()
            image, label, name = batch
            image = image[:, 0:3, :, :].cuda()
            label = label.cuda()
            output = model(image)
            loss = criterion(output, label)
            tloss += loss.item()
            # inter_time = time.time() - start_time
            output = output.cpu().detach()[0].numpy()
            gt = np.asarray(label[0].cpu().detach().numpy(), dtype=np.uint8)
            # print('gt size {}, output shape {}'.format(gt.shape, output.shape))
            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            running_Metrics.update(gt, output)


    print(f"test phase: Epoch [{epoch:d}/{args.max_epochs:d}] loss: {tloss/total_batches:.5f}")
    score, class_iou, class_F1 = running_Metrics.get_scores()


    running_Metrics.reset()

    return score, class_iou, class_F1


def train(args, trainloader, model, criterion, optimizer, epoch, logger):
    '''
    args:
        trainloader: loaded for traain dataset
        model: model
        criterion: loss function
        optimizer: optimizer algorithm, such as Adam or SGD
        epoch: epoch_number
    return:
        average loss
    '''
    model.train()
    total_batches = len(trainloader)
    for iter, batch in enumerate(trainloader, 0):
        lr = adjust_learning_rate(
                                cur_epoch=epoch,
                                max_epoch=args.max_epochs,
                                curEpoch_iter=iter,
                                perEpoch_iter=total_batches,
                                baselr=args.lr
                                )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        start_time = time.time()
        images, labels, name = batch
        images = images[:, 0:3, :, :].cuda()
        labels = labels.type(torch.long).cuda()
        output = model(images)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        interval_time = time.time() - start_time


        if iter+1 == total_batches:
            fmt_str = '======> epoch [{:d}/{:d}] cur_lr: {:.6f} loss: {:.5f} time: {:.2f}'
            print_str = fmt_str.format(
                                    epoch,
                                    args.max_epochs,
                                    lr,
                                    loss.item(),
                                    interval_time
                                    )
            print(print_str)
            logger.info(print_str)


def main(args, logger):

    cudnn.enabled = True     # Enables bencnmark mode in cudnn, to enable the inbuilt
    cudnn.benchmark = True   # cudnn auto-tuner to find the best algorithm to use for
                             # our hardware
    #Setup random seed
    # cudnn.deterministic = True # ensure consistent results
                                 # if benchmark = True, deterministic will be False.
    
    seed = random.randint(1, 10000)
    print('======>random seed {}'.format(seed))
    logger.info('======>random seed {}'.format(seed))
    
    random.seed(seed)  # python random seed
    np.random.seed(seed)  # set numpy random seed

    torch.manual_seed(seed)  # set random seed for cpu
    if torch.cuda.is_available():
        # torch.cuda.manual_seed(seed) # set random seed for GPU now
        torch.cuda.manual_seed_all(seed)  # set random seed for all GPU

    # Setup device
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


    # setup DatasetLoader
    train_set = vaihingenloader(root=args.root, split='train')
    test_set = vaihingenloader(root=args.root, split='test')

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=True, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False, **kwargs)

    # setup optimization criterion
    criterion = utils.utils.cross_entropy2d

    # setup model
    print('======> building network')
    logger.info('======> building network')

    #
    model = FCNRes101().to(device)
    # model = FCNRes34().cuda()
    #     model = UNet(n_channels=3, n_classes=6,).to(device)
    if torch.cuda.device_count() > 1:
        device_ids = list(map(int, args.gpu.split(',')))
        model = nn.DataParallel(model, device_ids=device_ids)

    print("======> computing network parameters")
    logger.info("======> computing network parameters")

    total_paramters = netParams(model)
    print("the number of parameters: " + str(total_paramters))
    logger.info("the number of parameters: " + str(total_paramters))

    # setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=5e-4)

    # setup savedir      
    args.savedir = (args.savedir + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu) + '/')
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    flag = True

    best_epoch = 0.
    best_overall = 0.
    best_mIoU = 0.
    best_F1 = 0. 

    while flag == True:
        for epoch in range(start_epoch, args.max_epochs):
            print('======> Epoch {} starting train.'.format(epoch))
            logger.info('======> Epoch {} starting train.'.format(epoch))

            train(args, train_loader, model, criterion, optimizer, epoch, logger)

            print('======> Epoch {} train finish.'.format(epoch))
            logger.info('======> Epoch {} train finish.'.format(epoch))

            if epoch % 1 == 0 or (epoch + 1) == args.max_epochs:
                print('Now Epoch {}, starting evaluate on Test dataset.'.format(epoch))
                logger.info('Now starting evaluate on Test dataset.')
                print('length of test set:', len(test_loader))
                logger.info('length of test set: {}'.format(len(test_loader)))

                score, class_iou, class_F1 = test(args, test_loader, model, criterion, epoch, logger)
        
                for k, v in score.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))
                
                print('Now print class iou')
                for k, v in class_iou.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))

                print('Now print class_F1')
                for k, v in class_F1.items():
                    print('{}: {:.5f}'.format(k, v))
                    logger.info('======>{0:^18} {1:^10}'.format(k, v))
                
                if score["Mean IoU : \t"] > best_mIoU:
                    best_mIoU = score["Mean IoU : \t"]
                
                if score["Overall Acc : \t"] > best_overall:
                    best_overall = score["Overall Acc : \t"]
                    # save model in best overall Acc
                    model_file_name = args.savedir + '/model.pth'
                    torch.save(model.state_dict(), model_file_name)
                    best_epoch = epoch

                if score["Mean F1 : \t"] > best_F1:
                    best_F1 = score["Mean F1 : \t"]

                print(f"best mean IoU: {best_mIoU}")
                print(f"best overall : {best_overall}")
                print(f"best F1: {best_F1}")
                print(f"best epoch: {best_epoch}")

        if (epoch + 1) == args.max_epochs:
            # print('the best pred mIoU: {}'.format(best_pred))
            flag = False
            break


if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()

    parser = argparse.ArgumentParser(description='Semantic Segmentation...')

    parser.add_argument('--model', default='HCA101', type=str)
    parser.add_argument('--root', default='./data/', help='data directory')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epochs', type=int, default=50, help='the number of epochs: default 100 ')
    parser.add_argument('--num_classes', default=6, type=int)
    parser.add_argument('--lr', default=0.009, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--workers', type=int, default=2, help=" the number of parallel threads")
    parser.add_argument('--show_interval', default=10, type=int)
    parser.add_argument('--show_val_interval', default=1000, type=int)
    parser.add_argument('--savedir', default="./runs_HCANet101_potsall/", help="directory to save the model snapshot")
    # parser.add_argument('--logFile', default= "log.txt", help = "storing the training and validation logs")
    # parser.add_argument('--gpu', type=str, default="4,5", help="default GPU devices (4,5)")
    parser.add_argument('--gpu', type=str, default="6,7", help="default GPU devices (6,7)")

    args = parser.parse_args()

    run_id = 'HCA101_1'
    print('Now run_id {}'.format(run_id))
    args.savedir = os.path.join(args.savedir, str(run_id))
    print(args.savedir)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    logger = get_logger(args.savedir)
    logger.info('just do it')
    
    print('Input arguments:')
    logger.info('======>Input arguments:')
    
    for key, val in vars(args).items():
        print('======>{:16} {}'.format(key, val))
        logger.info('======> {:16} {}'.format(key, val))
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.set_device(int(args.gpu.split(',')[0]))

    main(args, logger)
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)
    print('model save in {}.'.format(run_id))
