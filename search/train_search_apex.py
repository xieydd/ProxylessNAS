'''
@Description: Train Search ProxylessNAS  
@Author: xieydd
@Date: 2019-09-05 15:37:47
@LastEditTime: 2019-09-24 13:55:15
@LastEditors: Please set LastEditors
'''
import os
import sys
import time
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from architect import Architect
from operations import *
from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from latencyloss import LatencyLoss
from model_search import Network

import sys
sys.path.append("../")
from imagenet_dataloader import get_imagenet_iter_torch
import utils
from config import SearchConfig

config = SearchConfig()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)

def main():
  start = time.time()
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
    
  torch.cuda.set_device(config.local_rank % len(config.gpus))
  torch.distributed.init_process_group(backend='nccl',
                                    init_method='env://')
  config.world_size = torch.distributed.get_world_size()
  config.total_batch = config.world_size * config.batch_size
  
  np.random.seed(config.seed)
  torch.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)
  torch.backends.cudnn.benchmark = True

  CLASSES =  1000
  channels = SEARCH_SPACE['channel_size']
  strides =  SEARCH_SPACE['strides']

  ### Model
  model = Network(channels, strides,CLASSES)
  model = model.to(device)
  model.apply(utils.weights_init)
  #model = DDP(model, delay_allreduce=True)
  # For solve the custome loss can`t use model.parameters() in apex warpped model via https://github.com/NVIDIA/apex/issues/457 and https://github.com/NVIDIA/apex/issues/107 
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],output_device=config.local_rank)
  logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # Loss
  criterion = LatencyLoss(config.lambda1, channels,strides).cuda(config.gpus)

  alpha_weight = model.module.arch_parameters()
  weight = [param for param in model.parameters() if not utils.check_tensor_in_list(param, alpha_weight)]
  # Optimizer
  w_optimizer = torch.optim.SGD(
      weight,
      config.w_lr,
      momentum=config.w_momentum,
      weight_decay=config.w_weight_decay)

  alpha_optimizer = torch.optim.Adam(alpha_weight,
        lr=config.alpha_lr, betas=(0.5, 0.999), weight_decay=config.alpha_weight_decay)

  train_data = get_imagenet_iter_torch(
                type='train',
                # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
                # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/` 
                image_dir=config.data_path+config.dataset.lower(),
                batch_size = config.batch_size,
                num_threads = config.workers,
                world_size= config.world_size,
                local_rank= config.local_rank,
                crop=224, device_id=config.local_rank, num_gpus=config.gpus,portion=config.train_portion
            )
  valid_data = get_imagenet_iter_torch(
                type='train',
                # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
                # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/` 
                image_dir=config.data_path+"/"+config.dataset.lower(),
                batch_size = config.batch_size,
                num_threads = config.workers,
                world_size= config.world_size,
                local_rank= config.local_rank,
                crop=224, device_id=config.local_rank, num_gpus=config.gpus,portion=config.val_portion
            ) 

  w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optimizer, float(config.epochs), eta_min=config.w_lr_min)
  

  architect = Architect(model.module, alpha_optimizer,config)

  best_top1 = 0.
  for epoch in range(config.epochs):
    w_scheduler.step()
    lr = w_scheduler.get_lr()[0]
    logger.info('epoch %d lr %e', epoch, lr)

    # training
    train_top1, train_loss = train(train_data, valid_data, model, architect, criterion, w_optimizer,alpha_optimizer, lr,epoch, writer)
    logger.info('Train top1 %f', train_top1)

    # validation
    top1 = 0
    if epoch%10 == 0:
      top1, loss = infer(valid_data, model,epoch, criterion,writer)
      logger.info('valid top1 %f', top1)

    genotype = model.module.genotype()
    logger.info("genotype = {}".format(genotype))
    
    # save
    if best_top1 < top1:
        best_top1 = top1
        best_genotype = genotype
        is_best = True
    else:
        is_best = False
    utils.save_checkpoint(model, config.path, is_best)
    print("")

  utils.time(time.time() - start)
  logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
  logger.info("Best Genotype = {}".format(best_genotype))

def train(train_queue, valid_queue, model, architect, criterion, optimizer,alpha_optimizer, lr,epoch, writer):
  batch_time = utils.AverageMeters('Time', ':6.3f')
  data_time = utils.AverageMeters('Data', ':6.3f')
  losses = utils.AverageMeters('Loss', ':.4e')
  top1 = utils.AverageMeters('Acc@1', ':6.2f')
  top5 = utils.AverageMeters('Acc@5', ':6.2f')

  
  progress = utils.ProgressMeter(len(train_queue), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
  cur_step = epoch*len(train_queue)
  writer.add_scalar('train/lr', lr, cur_step)

  

  # Training weights firstly, after few epoch, train arch parameters
  if epoch > 15:
    for step, (input,target) in enumerate(train_queue):
      model.train()
      alpha_optimizer.zero_grad()
      input_search, target_search = next(iter(valid_queue))
      input_search = Variable(input_search, requires_grad=False).cuda()
      target_search = Variable(target_search, requires_grad=False).cuda()
      
      architect.step(target ,model(input_search), input_search,target_search, lr, alpha_optimizer, criterion,unrolled=config.unrolled)
  
  model.train()
  end = time.time()
  for step, (input, target) in enumerate(train_queue):
    # measure data loading time
    data_time.update(time.time() - end)
    
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    #target = Variable(target, requires_grad=False).cuda(async=True)
    target = Variable(target, requires_grad=False).cuda()
    optimizer.zero_grad()
    
    logits = model(input)
    loss = criterion(target, logits, model.module.selecteds, model.module.arch_parameters())
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), config.grad_clip)
    optimizer.step()

    acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
    reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
    acc1 = reduce_tensor(acc1,world_size=config.world_size)
    acc5 = reduce_tensor(acc5, world_size=config.world_size)

    losses.update(to_python_float(reduced_loss), n)
    top1.update(to_python_float(acc1), n)
    top5.update(to_python_float(acc5), n)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % config.print_freq == 0 or step == len(train_queue)-1:
      logger.info('train step:%03d %03d  loss:%e top1:%05f top5:%05f', step, len(train_queue),losses.avg,top1.avg, top5.avg)
      progress.print(step)
    writer.add_scalar('train/loss', losses.avg, cur_step)
    writer.add_scalar('train/top1', top1.avg, cur_step)
    writer.add_scalar('train/top5', top5.avg, cur_step)

  return top1.avg, losses.avg
  

def infer(valid_queue, model, epoch,criterion, writer):
  batch_time = utils.AverageMeters('Time', ':6.3f')
  losses = utils.AverageMeters('Loss', ':.4e')
  top1 = utils.AverageMeters('Acc@1', ':6.2f')
  top5 = utils.AverageMeters('Acc@5', ':6.2f')
  model.eval()

  progress = utils.ProgressMeter(len(valid_queue), batch_time, losses, top1, top5,
                             prefix='Test: ') 
  cur_step = epoch*len(valid_queue)

  end = time.time()
  for step, (input, target) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = Variable(input, volatile=True).cuda()
    #target = Variable(target, volatile=True).cuda(async=True)
    target = Variable(target, volatile=True).cuda()
    logits = model(input)
    loss = criterion(target, logits, model.module.selecteds, model.module.arch_parameters())
    acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
    acc1 = reduce_tensor(acc1,world_size=config.world_size)
    acc5 = reduce_tensor(acc5, world_size=config.world_size)
    losses.update(to_python_float(reduced_loss), n)
    top1.update(to_python_float(acc1), n)
    top5.update(to_python_float(acc5), n)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    
    if step % config.print_freq == 0:
      progress.print(step) 
      logger.info('valid %03d %e %f %f', step, losses.avg,top1.avg, top5.avg)

  writer.add_scalar('val/loss', losses.avg, cur_step)
  writer.add_scalar('val/top1', top1.avg, cur_step)
  writer.add_scalar('val/top5', top5.avg, cur_step)
  return top1.avg, losses.avg

def check_tensor_in_list(atensor, alist):
    if any([(atensor == t_).all() for t_ in alist if atensor.shape == t_.shape]):
        return True
    return False

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

if __name__ == '__main__':
  main() 

