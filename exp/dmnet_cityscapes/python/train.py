import os
import sys
import ast
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from lib.dataset.cityscapes import cityscapes_video_dataset, cityscapes_video_dataset_PDA
from lib.dataset.utils import runningScore
from lib.model.scnet import SCNet_dmnet as SCNet


def get_arguments():
    parser = argparse.ArgumentParser(description="Train DMNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--root_data_path", type=str, help="root path to the dataset")
    parser.add_argument("--root_gt_path", type=str, help="root path to the ground truth")
    parser.add_argument("--root_mask_path", type=str, help="root path to the deeplab mask")
    parser.add_argument("--train_list_path", type=str, help="path to the list of train subset")
    parser.add_argument("--test_list_path", type=str, help="path to the list of test subset")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--resume", type=ast.literal_eval, default=False, help="resume or not")
    parser.add_argument("--resume_epoch", type=int, help="from which epoch for resume")
    parser.add_argument("--resume_load_path", type=str, help="resume model load path")
    parser.add_argument("--train_load_path", type=str, help="train model load path")
    parser.add_argument("--local_rank", type=int, help="index the replica")
    parser.add_argument("--dmnet_lr", type=float, help="learning rate")
    parser.add_argument("--random_seed", type=int, help="random seed")
    parser.add_argument("--train_flownet", type=ast.literal_eval, default=True, help="trian flownet or not")
    parser.add_argument("--train_power", type=float, help="power value for linear learning rate schedule")
    parser.add_argument("--final_lr", type=float, default=0.00001, help="learning rate in the second stage")
    parser.add_argument("--weight_decay", type=float, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, help="train batch size")
    parser.add_argument("--train_shuffle", type=ast.literal_eval, default=True, help="shuffle or not in training")
    parser.add_argument("--train_num_workers", type=int, default=8, help="num cpu use")
    parser.add_argument("--num_epoch", type=int, default=100, help="num of epoch in training")
    parser.add_argument("--snap_shot", type=int, default=1, help="save model every per snap_shot")
    parser.add_argument("--model_save_path", type=str, help="model save path")

    ###### testing setting ######
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch_size for validation")
    parser.add_argument("--test_shuffle", type=ast.literal_eval, default=False, help="shuffle or not in validation")
    parser.add_argument("--test_num_workers", type=int, default=4, help="num of used cpus in validation")

    ###### tensorboard setting ######
    parser.add_argument("--use_tensorboard", type=ast.literal_eval, default=True, help="use tensorboard or not")
    parser.add_argument("--tblog_dir", type=str, help="log save path")
    parser.add_argument("--tblog_interval", type=int, default=50, help="interval for tensorboard logging")

    return parser.parse_args()


def make_dirs(args):
    if args.use_tensorboard and not os.path.exists(args.tblog_dir):
        os.makedirs(args.tblog_dir)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)


def train():
    torch.distributed.init_process_group(backend="nccl")

    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    args = get_arguments()
    if local_rank == 0:
        print(args)
        make_dirs(args)

    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if local_rank == 0:
        print('random seed:{}'.format(random_seed))

    if local_rank == 0 and args.use_tensorboard:
        tblogger = SummaryWriter(args.tblog_dir)

    net = SCNet()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    if args.resume:
        old_weight = torch.load(args.resume_load_path, map_location=map_location)
        start_epoch = args.resume_epoch
    else:
        old_weight = torch.load(args.train_load_path, map_location=map_location)
        start_epoch = 0

    new_weight = {}
    for k, v in old_weight.items():
        k = k.replace('module.', '')
        new_weight[k] = v

    if args.resume:
        net.load_state_dict(new_weight, strict=True)
    else:
        net.load_state_dict(new_weight, strict=False)

    if local_rank == 0:
        print('Successful loading model!')

    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
                                              device_ids=[local_rank],
                                              output_device=local_rank,
                                              find_unused_parameters=True)

    train_data = cityscapes_video_dataset(args.root_data_path, args.root_gt_path, args.root_mask_path,
                                          args.train_list_path)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=args.train_batch_size,
                                                    shuffle=False,
                                                    pin_memory=False,
                                                    num_workers=args.train_num_workers,
                                                    drop_last=True,
                                                    sampler=DistributedSampler(train_data,
                                                                               num_replicas=world_size,
                                                                               rank=local_rank,
                                                                               shuffle=True))

    test_data = cityscapes_video_dataset_PDA(args.root_data_path, args.root_gt_path, args.test_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=args.test_batch_size,
                                                   shuffle=args.test_shuffle,
                                                   num_workers=args.test_num_workers)
    dmnet_params = []
    for m in net.module.dmnet.modules():
        for p in m.parameters():
            dmnet_params.append(p)
    dmnet_optimizer = optim.Adam(params=dmnet_params, lr=args.dmnet_lr, betas=(0.9, 0.999), weight_decay=0)

    running_loss = 0.0
    current_eval_loss = 100
    itr = start_epoch * len(train_data_loader)
    max_itr = args.num_epoch * len(train_data_loader)

    for epoch in range(start_epoch, args.num_epoch):
        net.module.deeplab.eval()
        net.module.flownet.eval()
        net.module.dmnet.train()
        train_data_loader.sampler.set_epoch(epoch)
        for i, data_batch in enumerate(train_data_loader):
            img_list, img_mask_list, gt_label = data_batch

            adjust_lr(args, dmnet_optimizer, itr, max_itr, args.dmnet_lr)

            dmnet_optimizer.zero_grad()
            loss_dmnet = net(img_list, img_mask_list)
            loss_dmnet = torch.mean(loss_dmnet)
            loss_dmnet.backward()
            dmnet_optimizer.step()

            if local_rank == 0:
                print('epoch:{}/{} batch:{}/{} iter:{} loss_dmnet:{:05f}'.format(epoch, args.num_epoch, i,
                                                                                 len(train_data_loader), itr,
                                                                                 loss_dmnet.item()))

                if args.use_tensorboard and itr % args.tblog_interval == 0:
                    tblogger.add_scalar('loss_dmnet', loss_dmnet.item(), itr)

            itr += 1

            # if i == 5:
            #     break

        dist.barrier()

        if (epoch+1) % args.snap_shot == 0:
            net.eval()
            distance_list = [1, 5, 9]
            eval_loss = []
            for d in distance_list:
                loss = 0.0
                with torch.no_grad():
                    for step, sample in enumerate(test_data_loader):
                        if local_rank == 0:
                            print(d, step)
                        img_list, gt_label = sample
                        gt_label = gt_label.squeeze().cpu().numpy()

                        img = img_list[9 - d].cuda()
                        feat = net.module.deeplab(img)
                        warp_im = F.upsample(img, scale_factor=0.25, mode='bilinear', align_corners=True)
                        for i in range(d):
                            img_1 = img_list[9 - d + i].cuda()
                            img_2 = img_list[10 - d + i].cuda()
                            flow = net.module.flownet(torch.cat([img_2, img_1], dim=1))
                            feat = net.module.warpnet(feat, flow)
                            warp_im = net.module.warpnet(warp_im, flow)

                        img_2_down = F.upsample(img_2, scale_factor=0.25, mode='bilinear', align_corners=True)
                        dm = net.module.dmnet(warp_im, img_2_down)
                        dm = F.interpolate(dm, scale_factor=4, mode='bilinear', align_corners=True)

                        feat = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=True)
                        out = torch.argmax(feat, dim=1, keepdim=True)
                        feat_cur = net.module.deeplab(img_2)
                        feat_cur = F.interpolate(feat_cur, scale_factor=4, mode='bilinear', align_corners=True)
                        out_cur = torch.argmax(feat_cur, dim=1, keepdim=True)
                        label = (out != out_cur).float()

                        loss += F.l1_loss(dm, label).squeeze().item()

                        # if step == 3:
                        #     break

                    loss /= len(test_data_loader)
                    eval_loss.append(loss)

                if local_rank == 0:
                    print('distance:{} eval_loss:{}'.format(d, loss))

            if local_rank == 0:
                if args.use_tensorboard:
                    for i, d in enumerate(distance_list):
                        tblogger.add_scalar('dmnet_eval_loss/distance_{}'.format(d), eval_loss[i], epoch)

                    dm = dm[0]
                    tblogger.add_image('dmnet', dm, epoch)
                    label = label[0]
                    tblogger.add_image('dmnet_GT', label, epoch)

                save_name = 'now.pth'
                save_path = os.path.join(args.model_save_path, save_name)
                torch.save(net.module.dmnet.state_dict(), save_path)

                mean_loss = np.mean(eval_loss)
                if mean_loss < current_eval_loss:
                    save_name = 'best.pth'
                    save_path = os.path.join(args.model_save_path, save_name)
                    torch.save(net.module.dmnet.state_dict(), save_path)
                    current_eval_loss = mean_loss

            dist.barrier()

    if local_rank == 0:
        save_name = 'final.pth'
        save_path = os.path.join(args.model_save_path, save_name)
        torch.save(net.module.dmnet.state_dict(), save_path)
        print('%s has been saved' % save_path)


def adjust_lr(args, optimizer, itr, max_itr, lr):
    if itr > max_itr / 2:
        now_lr = lr / 10
    else:
        now_lr = lr

    for group in optimizer.param_groups:
        group['lr'] = now_lr


if __name__ == '__main__':
    train()
    dist.destroy_process_group()