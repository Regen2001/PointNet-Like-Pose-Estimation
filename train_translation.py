import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import argparse

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelDataLoader import ModelDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_dir, exp_dir, checkpoints_dir = create_dir(args)

    logger = create_log(log_dir, args)

    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)

    trainDataLoader, testDataLoader = data_load(args, logger)

    model = importlib.import_module(args.model)

    classifier = model.get_model(num_category=args.num_category, mean_mlp=args.use_mean_mlp)
    criterion = model.get_loss(args.loss_function, args.loss_reduction)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger, 'Use pretrain model')
    except:
        log_string(logger, 'No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_epoch = 0
    min_loss = 1000

    log_string(logger, 'Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string(logger, 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        mean_loss = train(args, classifier, criterion, trainDataLoader, optimizer, scheduler, epoch, global_epoch, logger)

        train_instance_loss = np.mean(mean_loss)

        log_string(logger, 'Train mean Loss: %f' % train_instance_loss)

        with torch.no_grad():
            test_instance_loss = test(args, classifier, criterion, testDataLoader)
            min_loss = save_data(test_instance_loss, classifier, min_loss, optimizer, epoch, checkpoints_dir, logger)
        global_epoch += 1
    return 0

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='secify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='translation', help='model name [default: translation]')
    parser.add_argument('--num_category', default=7, type=int)
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='translation', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--loss_function', type=str, default='L2_loss', help='type of loss function')
    parser.add_argument('--loss_reduction', type=str, default='mean', help='type of loss function reduction')
    parser.add_argument('--use_mean_mlp', type=str, default=True, help='use the mean mlp')
    return parser.parse_args()

def train(args, classifier, criterion, trainDataLoader, optimizer, scheduler, epoch, global_epoch, logger):
    mean_loss = []

    classifier = classifier.train() 

    scheduler.step()

    for batch_id, (points, label, _, target, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()
        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        mean = np.mean(points[:,:3,:], axis=1)
        mean = torch.Tensor(mean)
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        # label = torch.Tensor(label)
        points = torch.Tensor(points)
        points = provider.splice_torch(points, label)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target, mean = points.cuda(), target.cuda(), mean.cuda()

        pred = classifier(points, mean)
        loss = criterion(pred.float(), target.float())

        loss.backward()

        loss = torch.abs(pred - target)
        mean_loss.append(torch.mean(loss, dim=0).detach().cpu().numpy())
        optimizer.step()

    mean_loss = np.asanyarray(mean_loss)
    mean_loss = np.mean(mean_loss)

    return mean_loss*100

def test(args, classifier, criterion, testDataLoader):
    mean_loss = []
    class_loss = np.zeros((args.num_category, 7))
    classifier = classifier.eval()

    for j, (points, label, _, target, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

        points = points.data.numpy()
        mean = np.mean(points[:,:3,:], axis=1)
        mean = torch.Tensor(mean)
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        # label = torch.Tensor(label)
        points = torch.Tensor(points)
        points = provider.splice_torch(points, label)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target, mean = points.cuda(), target.cuda(), mean.cuda()

        pred = classifier(points, mean)
        loss = criterion(pred.float(), target.float())

        for cat in np.unique(label.cpu()):
            temp_loss = torch.abs(pred[label==cat] - target[label==cat])
            class_loss[cat,1:4] += torch.mean(temp_loss, dim=0).detach().cpu().numpy()
            class_loss[cat,0] += 1

        loss = torch.abs(pred - target)
        mean_loss.append(torch.mean(loss, dim=0).detach().cpu().numpy())

    class_loss[:,4] = class_loss[:,1] / class_loss[:,0]
    class_loss[:,5] = class_loss[:,2] / class_loss[:,0]
    class_loss[:,6] = class_loss[:,3] / class_loss[:,0]
    mean_loss = np.asanyarray(mean_loss)
    mean_loss = np.mean(mean_loss)

    return mean_loss*100

def create_dir(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('pose')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    return log_dir, exp_dir, checkpoints_dir

def create_log(log_dir, args):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def data_load(args, logger):
    log_string(logger, 'Load dataset ...')

    data_path = 'data/data/'

    train_dataset = ModelDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelDataLoader(root=data_path, args=args, split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)

    return trainDataLoader, testDataLoader

def save_data(loss, classifier, min_loss, optimizer, epoch, checkpoints_dir, logger):
    if (min_loss >= loss):
        min_loss = loss
        best_epoch = epoch + 1
    log_string(logger, 'Test mean Loss: %f' % (loss))
    log_string(logger, 'Min Test mean Loss: %f' % (min_loss))

    if (min_loss >= loss):
        log_string(logger, 'Save model...')
        savepath = str(checkpoints_dir) + '/best_model.pth'
        log_string(logger, 'Saving at %s' % savepath)
        state = {
            'epoch': best_epoch,
            'instance_loss': loss,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
    return min_loss

def log_string(logger, str):
    logger.info(str)
    print(str)

if __name__ == '__main__':
    args = parse_args()
    main(args)