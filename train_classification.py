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

    classifier = model.get_model(args.num_category)
    criterion = model.get_loss()

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
    best_instance_accuracy = 0.0
    best_class_accuracy = 0.0

    log_string(logger, 'Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string(logger, 'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        mean_correct = train(args, classifier, criterion, trainDataLoader, optimizer, scheduler, epoch, global_epoch, logger)

        train_instance_accuracy = np.mean(mean_correct)

        log_string(logger, 'Train Instance Accuracy: %f' % train_instance_accuracy)

        with torch.no_grad():
            instance_accuracy, class_accuracy = test(args, classifier, testDataLoader)
            best_instance_accuracy, best_class_accuracy = save_data(instance_accuracy, class_accuracy, classifier, best_instance_accuracy, best_class_accuracy, optimizer, epoch, checkpoints_dir, logger)
        global_epoch += 1
    return 0

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='secify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=7, type=int)
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='pointnet_cls', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    return parser.parse_args()

def train(args, classifier, criterion, trainDataLoader, optimizer, scheduler, epoch, global_epoch, logger):
    mean_correct = []

    classifier = classifier.train()

    scheduler.step()

    for batch_id, (points, target, _, _, _) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()
        points = points.data.numpy()
        points = provider.random_point_dropout(points)
        points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat, pred_choice = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0])) 
        loss.backward()
        optimizer.step()
    return mean_correct

def test(args, model, testDataLoader):
    mean_correct = []
    class_accuracy = np.zeros((args.num_category, 3))
    classifier = model.eval()

    for j, (points, target, _, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

        points = points.data.numpy()
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        points = torch.Tensor(points)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat, pred_choice = classifier(points)

        for cat in np.unique(target.cpu()):
            temp_accuracy = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_accuracy[cat, 0] += temp_accuracy.item() / float(points[target == cat].size()[0])
            class_accuracy[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_accuracy[:, 2] = class_accuracy[:, 0] / class_accuracy[:, 1]
    class_accuracy = np.mean(class_accuracy[:, 2])
    instance_accuracy = np.mean(mean_correct)

    return instance_accuracy, class_accuracy

def create_dir(args):
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
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
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    return trainDataLoader, testDataLoader

def save_data(instance_accuracy, class_accuracy, classifier, best_instance_accuracy, best_class_accuracy, optimizer, epoch, checkpoints_dir, logger):
    if (instance_accuracy >= best_instance_accuracy):
        best_instance_accuracy = instance_accuracy
        best_epoch = epoch + 1

    if (class_accuracy >= best_class_accuracy):
        best_class_accuracy = class_accuracy
    log_string(logger, 'Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_accuracy, class_accuracy))
    log_string(logger, 'Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_accuracy, best_class_accuracy))

    if (instance_accuracy >= best_instance_accuracy):
        log_string(logger, 'Save model...')
        savepath = str(checkpoints_dir) + '/best_model.pth'
        log_string(logger, 'Saving at %s' % savepath)
        state = {
            'epoch': best_epoch,
            'instance_accuracy': instance_accuracy,
            'class_accuracy': class_accuracy,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(state, savepath)
    return best_instance_accuracy, best_class_accuracy

def log_string(logger, str):
    logger.info(str)
    print(str)

if __name__ == '__main__':
    args = parse_args()
    main(args)