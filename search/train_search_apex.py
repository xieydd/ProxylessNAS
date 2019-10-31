'''
@Description: Train Search ProxylessNAS
@Author: xieydd
@Date: 2019-09-05 15:37:47
@LastEditTime: 2019-10-19 18:57:18
@LastEditors: Please set LastEditors
'''
from imagenet_dataloader import get_imagenet_iter_torch
import utils
from config import SearchConfig
import os
import sys
import time
import math
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from operations import *
from apex.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from latencyloss import LatencyLoss
from model_search import Network

import sys
sys.path.append("../")

config = SearchConfig()
device = torch.device("cuda")

# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(
    config.path, "{}.log".format(config.name)))
arch_logger_path = os.path.join(
    config.path, "{}.log".format(config.name + '-arch')))
config.print_params(logger.info)

# ref values
ref_values={
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {},
}


def main():
    start = time.time()
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(config.local_rank % len(config.gpus))
    torch.distributed.init_process_group(backend='nccl',
                                         init_method = 'env://')
    config.world_size=torch.distributed.get_world_size()
    config.total_batch=config.world_size * config.batch_size

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark=True

    CLASSES=1000
    channels=SEARCH_SPACE['channel_size']
    strides=SEARCH_SPACE['strides']

    # Model
    model=Network(channels, strides, CLASSES)
    model=model.to(device)
    model.apply(utils.weights_init)
    model=DDP(model, delay_allreduce = True)
    # For solve the custome loss can`t use model.parameters() in apex warpped model via https://github.com/NVIDIA/apex/issues/457 and https://github.com/NVIDIA/apex/issues/107
    # model = torch.nn.parallel.DistributedDataParallel(
    #    model, device_ids=[config.local_rank], output_device=config.local_rank)
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if config.target_hardware is None:
        config.ref_value=None
    else:
        config.ref_value=ref_values[config.target_hardware]['%.2f' %
                                                              config.width_mult]

    # Loss
    criterion = LatencyLoss(config, channels, strides).cuda(config.gpus)
    normal_critersion = nn.CrossEntropyLoss()

    alpha_weight = model.module.arch_parameters()
    # weight = [param for param in model.parameters() if not utils.check_tensor_in_list(param, alpha_weight)]
    weight = model.weight_parameters()
    # Optimizer
    w_optimizer = torch.optim.SGD(
        weight,
        config.w_lr,
        momentum=config.w_momentum,
        weight_decay=config.w_weight_decay)

    alpha_optimizer = torch.optim.Adam(alpha_weight,
                                       lr=config.alpha_lr, betas=(config.arch_adam_beta1, config.arch_adam_beta2), eps=config.arch_adam_eps, weight_decay=config.alpha_weight_decay)

    train_data = get_imagenet_iter_torch(
        type='train',
        # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
        # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/`
        image_dir=config.data_path+config.dataset.lower(),
        batch_size=config.batch_size,
        num_threads=config.workers,
        world_size=config.world_size,
        local_rank=config.local_rank,
        crop=224, device_id=config.local_rank, num_gpus=config.gpus, portion=config.train_portion
    )
    valid_data = get_imagenet_iter_torch(
        type='val',
        # image_dir="/googol/atlas/public/cv/ILSVRC/Data/"
        # use soft link `mkdir ./data/imagenet && ln -s /googol/atlas/public/cv/ILSVRC/Data/CLS-LOC/* ./data/imagenet/`
        image_dir=config.data_path+"/"+config.dataset.lower(),
        batch_size=config.batch_size,
        num_threads=config.workers,
        world_size=config.world_size,
        local_rank=config.local_rank,
        crop=224, device_id=config.local_rank, num_gpus=config.gpus, portion=config.val_portion
    )

    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optimizer, float(config.epochs), eta_min=config.w_lr_min)

    best_top1 = 0.
    best_genotype = list()
    lr = 0


    ### Resume form warmup model or train model ###
    if config.resume:
        try:
            model, w_optimizer, alpha_optimizer =  load_model(model, config.path, w_optimizer, alpha_optimizer)
        except Exception:
            warmup_path = config.path +  '/warmup.pth.tar'
            if os.path.exists(warmup_path):
                print('load warmup weights')
                load_model(model,model_fname=warmup_path,w_optimizer,alpha_optimizer)
            else:
                print('fail to load models')

    if config.warmup:
        for epoch in range(config.warm_epoch, config.warmup_epochs):
            # warmup
            train_top1, train_loss = warm_up(train_data, valid_data, model,
                                            normal_critersion, criterion, w_optimizer,epoch, writer)
    update_schedule =  utils.get_update_schedule_grad(len(train_data), config)
    for epoch in range(config.start_epoch, config.epochs):
        if epoch > config.warmup_epochs:
            w_scheduler.step()
            lr = w_scheduler.get_lr()[0]
            logger.info('epoch %d lr %e', epoch, lr)
        # training
        train_top1, train_loss = train(train_data, valid_data, model,
                                           normal_critersion, criterion, w_optimizer, alpha_optimizer, lr, epoch, writer)
        logger.info('Train top1 %f', train_top1)

        # validation
        top1 = 0
        if epoch % 10 == 0:
            top1, loss = infer(valid_data, model, epoch, criterion, normal_critersion, writer)
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
        save_model(model, , is_best)

    utils.time(time.time() - start)
    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


def warm_up(train_queue, valid_queue, model, criterion, Latency, optimizer,epoch, writer):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    data_time = utils.AverageMeters('Data', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(len(train_queue), batch_time, data_time, losses, top1,
                                   top5, prefix="Epoch: [{}]".format(epoch))
    cur_step = epoch*len(train_queue)
    model.train()
    print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch), '-' * 30, '\n')
    end = time.time()
    lr = 0
    for step, (input, target) in enumerate(train_queue):
        # measure data loading time
        data_time.update(time.time() - end)
        # office warm up lr #l'r
        T_cur = epoch * len(train_queue) + step
        lr_max = 0.05
        T_total = config.warmup_epochs * len(train_queue)
        lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        writer.add_scalar('warm-up/lr', lr, cur_step+step)
        
        #### office warm up lr ####

        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        target = Variable(target, requires_grad=False).cuda()

        model.module.reset_binary_gates()
        model.module.unused_modules_off()

        logits = model(input)
        if config.label_smooth > 0 and epoch > config.warmup_epochs:
            loss = utils.cross_entropy_with_label_smoothing(
                logits, target, config.label_smooth)
        else:
            loss = criterion(logits, target)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
        acc1 = reduce_tensor(acc1, world_size=config.world_size)
        acc5 = reduce_tensor(acc5, world_size=config.world_size)

        losses.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(acc1), n)
        top5.update(to_python_float(acc5), n)

        # unused modules back
        model.module.unused_modules_back()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0 or step == len(train_queue)-1:
            logger.info('warmup train step:%03d %03d  loss:%e top1:%05f top5:%05f', step, len(
                train_queue), losses.avg, top1.avg, top5.avg)
            progress.print(step)
        writer.add_scalar('warmup-train/loss', losses.avg, cur_step)
        writer.add_scalar('warmup-train/top1', top1.avg, cur_step)
        writer.add_scalar('warmup-train/top5', top5.avg, cur_step)

    logger.info('warmup epoch %d lr %e', epoch, lr)
    # set chosen op active
    model.module.set_chosen_op_active()
    # remove unused modules
    model.module.unused_modules_off()
    valid_top1, valid_top5, valid_loss = validate_warmup(
        valid_queue, model, epoch, criterion, writer)
    shape = [1, 3, 224, 224]
    input_var = torch.zeros(shape, device=device)
    flops = model.module.get_flops(input_var)
    latency = 0
    if config.target_hardware in [None, 'flops']:
        latency = 0
    else:
        latency = Latency.predict_latency(model)
    # unused modules back
    logger.info('Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      '{top5:.3f}\tflops: {5:.1f}M {6:.3f}ms\t Train top-1 {top1:.3f}\ttop-5 '.
                format(epoch + 1, config.warmup_epochs, valid_loss, valid_top1, valid_top5, flops / 1e6, latency, top1=top1, top5=top5))
    model.module.unused_modules_back()

    config.warmup = epoch + 1 < config.warmup_epochs
    state_dict = model.state_dict()
    # rm architect params and binary getes
    for key in list(state_dict):
        if 'alpha' in key or 'path' in key:
            state_dict.pop(key)
    checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
    if config.warmup:
        checkpoint['warmup_epoch'] = epoch,
    save_model(model, checkpoint,model_name='warmup.pth.tar')
    return top1.avg, losses.avg


def validate_warmup(valid_queue, model, epoch, criterion, writer):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    model.train()

    progress = utils.ProgressMeter(len(valid_queue), batch_time, losses, top1, top5,
                                   prefix='Warmup-Test: ')
    cur_step = epoch*len(valid_queue)

    end = time.time()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)
            input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            target = Variable(target, volatile=True).cuda()
            logits = model(input)
            loss = criterion(logits, target)
            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            reduced_loss = reduce_tensor(
                loss.data, world_size=config.world_size)
            acc1 = reduce_tensor(acc1, world_size=config.world_size)
            acc5 = reduce_tensor(acc5, world_size=config.world_size)
            losses.update(to_python_float(reduced_loss), n)
            top1.update(to_python_float(acc1), n)
            top5.update(to_python_float(acc5), n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % config.print_freq == 0:
                progress.print(step)
                logger.info('warmup-valid %03d %e %f %f', step,
                            losses.avg, top1.avg, top5.avg)

    writer.add_scalar('warmup-val/loss', losses.avg, cur_step)
    writer.add_scalar('warmup-val/top1', top1.avg, cur_step)
    writer.add_scalar('warmup-val/top5', top5.avg, cur_step)
    return top1.avg, top5.avg, losses.avg


def train(train_queue, valid_queue, model, criterion, LatencyLoss, optimizer, alpha_optimizer, lr, epoch, writer, update_schedule):

    arch_param_num = np.sum(np.prod(params.size())
                            for i, params in model.module.arch_parameters())
    binary_gates_num = len(list(model.module.binary_gates()))
    weight_param_num = len(list(model.module.weight_parameters()))
    print(
        '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
        (arch_param_num, binary_gates_num, weight_param_num)
    )

    batch_time = utils.AverageMeters('Time', ':6.3f')
    data_time = utils.AverageMeters('Data', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')
    entropy = utils.AverageMeters('Entropy', ':.4e')

    progress = utils.ProgressMeter(len(train_queue), batch_time, data_time, losses, top1,
                                   top5, prefix="Epoch: [{}]".format(epoch))
    cur_step = epoch*len(train_queue)
    writer.add_scalar('train/lr', lr, cur_step)

    model.train()
    end = time.time()
    for step, (input, target) in enumerate(train_queue):

        # measure data loading time
        data_time.update(time.time() - end)

        net_entropy = model.module.entropy()
        entropy.update(net_entropy.data.item() / arch_param_num, 1)

        # sample random path
        model.module.reset_binary_gates()
        # close unused module
        model.module.unused_modules_off()

        n = input.size(0)
        input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(async=True)
        target = Variable(target, requires_grad=False).cuda()

        logits = model(input)
        if config.label_smooth > 0.0:
            loss = utils.cross_entropy_with_label_smoothing(
                logits, target, config.label_smooth)
        else:
            loss = criterion(logits, target)

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        reduced_loss = reduce_tensor(loss.data, world_size=config.world_size)
        acc1 = reduce_tensor(acc1, world_size=config.world_size)
        acc5 = reduce_tensor(acc5, world_size=config.world_size)

        losses.update(to_python_float(reduced_loss), n)
        top1.update(to_python_float(acc1), n)
        top5.update(to_python_float(acc5), n)
        model.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), config.grad_clip)
        optimizer.step()
        # unused module back
        model.module.unused_modules_back()

        # Training weights firstly, after few epoch, train arch parameters
        if epoch > 0:
            #### office warm up lr ####
            # T_cur = epoch * len(train_queue) + step
            # lr_max = 0.05
            # T_totol = config.warmup_eforhs * len(train_queue)
            # lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
            #### office warm up lr ####
            for j in range (update_schedule.get(step, 0)):
                model.train()
                latency_loss = 0
                expected_loss = 0
                
                valid_iter = iter(valid_queue)
                input_valid, target_valid = next(valid_iter)
                # alpha_optimizer.zero_grad()
                input_valid = Variable(input_valid, requires_grad=False).cuda()
                # target = Variable(target, requires_grad=False).cuda(async=True)
                target_valid = Variable(input_valid, requires_grad=False).cuda()
                model.module.reset_binary_gates()
                model.module.unused_modules_off()
                output_valid = model(input_valid)
                loss_ce = criterion(output_valid, target_valid)
                expected_loss = LatencyLoss.expected_latency(model)
                expected_loss_tensor = torch.cuda.FloatTensor([expected_loss])
                latency_loss = LatencyLoss(loss_ce, expected_loss_tensor, config)
                # compute gradient and do SGD step
                # zero grads of weight_param, arch_param & binary_param
                model.zero_grad()
                latency_loss.backward()
                # set architecture parameter gradients
                model.module.set_arch_param_grad()
                alpha_optimizer.step()
                model.module.rescale_updated_arch_param()
                model.module.unused_modules_back()
                log_str = 'Architecture [%d-%d]\t Loss %.4f\t %s LatencyLoss: %s' %
                        (epoch, step, latency_loss,
                        config.target_hardware, expected_loss)
                utils.write_log(arch_logger_path, log_str)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % config.print_freq == 0 or step == len(train_queue)-1:
            logger.info('train step:%03d %03d  loss:%e top1:%05f top5:%05f', step, len(
                train_queue), losses.avg, top1.avg, top5.avg)
            progress.print(step)
    writer.add_scalar('train/loss', losses.avg, cur_step)
    writer.add_scalar('train/top1', top1.avg, cur_step)
    writer.add_scalar('train/top5', top5.avg, cur_step)

    return top1.avg, losses.avg


def infer(valid_queue, model, epoch, Latency,criterion, writer):
    batch_time = utils.AverageMeters('Time', ':6.3f')
    losses = utils.AverageMeters('Loss', ':.4e')
    top1 = utils.AverageMeters('Acc@1', ':6.2f')
    top5 = utils.AverageMeters('Acc@5', ':6.2f')

    # set chosen op active
    model.module.set_chosen_op_active()
    model.module.unused_modules_off()

    model.eval()

    progress = utils.ProgressMeter(len(valid_queue), batch_time, losses, top1, top5,
                                   prefix='Test: ')
    cur_step = epoch*len(valid_queue)

    end = time.time()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)
            input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            target = Variable(target, volatile=True).cuda()
            logits = model(input)
            loss = criterion(logits, target)
            acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            reduced_loss = reduce_tensor(
                loss.data, world_size=config.world_size)
            acc1 = reduce_tensor(acc1, world_size=config.world_size)
            acc5 = reduce_tensor(acc5, world_size=config.world_size)
            losses.update(to_python_float(reduced_loss), n)
            top1.update(to_python_float(acc1), n)
            top5.update(to_python_float(acc5), n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            shape = [1, 3, 224, 224]
            input_var = torch.zeros(shape, device=device)
            flops = model.module.get_flops(input_var)
            if config.target_hardware in [None, 'flops']:
                latency = 0
            else:
                latency = Latency.predict_latency(model)

            model.module.unused_modules_back()

            if step % config.print_freq == 0:
                progress.print(step)
                logger.info('valid %03d\t loss: %e\t top1: %f\t top5: %f\t flops: %f\t latency: %f', step,
                            losses.avg, top1.avg, top5.avg, flops/1e6, latency)

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

def save_model(model, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': model.module.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = config.dataset # add `dataset` info to the checkpoint
        latest_fname = os.path.join(config.path, 'latest.txt')
        model_path = os.path.join(config.path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)



def load_model(model,model_fname=None, optimizer, arch_optimizer):
    latest_fname = os.path.join(config.path, 'latest.txt')
    if model_fname is None and os.path.exists(latest_fname):
        with open(latest_fname, 'r') as fin:
            model_fname = fin.readline()
            if model_fname[-1] == '\n':
                model_fname = model_fname[:-1]

    if model_fname is None or not os.path.exists(model_fname):
        model_fname = '%s/checkpoint.pth.tar' % config.save_path
        with open(latest_fname, 'w') as fout:
            fout.write(model_fname + '\n')
        print("=> loading checkpoint '{}'".format(model_fname))

    if torch.cuda.is_available():
        checkpoint = torch.load(model_fname)
    else:
        checkpoint = torch.load(model_fname, map_location='cpu')

    model_dict = model.state_dict()
    model_dict.update(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    if self.run_manager.out_log:
        print("=> loaded checkpoint '{}'".format(model_fname))

    # set new manual seed
    new_manual_seed = int(time.time())
    torch.manual_seed(new_manual_seed)
    torch.cuda.manual_seed_all(new_manual_seed)
    np.random.seed(new_manual_seed)

    if 'epoch' in checkpoint:
        config.start_epoch = checkpoint['epoch'] + 1
    if 'weight_optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['weight_optimizer'])
    if 'arch_optimizer' in checkpoint:
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
    if 'warmup' in checkpoint:
        config.warmup = checkpoint['warmup']
    if self.warmup and 'warmup_epoch' in checkpoint:
        config.warmup_epoch = checkpoint['warmup_epoch']
    
    return model, optimizer, arch_optimizer


if __name__ == '__main__':
    main()
