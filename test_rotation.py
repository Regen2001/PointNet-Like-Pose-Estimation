import os
import sys
import torch
import numpy as np

import logging
import importlib
import argparse
import provider

from tqdm import tqdm
from data_utils.ModelDataLoader import ModelDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = 'log/pose/' + args.log_dir

    logger = create_log(experiment_dir)

    log_string(logger, 'PARAMETER ...')
    log_string(logger, args)
    
    cat = ['cube', 'cuboid', 'cylinder', 'h_structure', 'double_cube', 'double_cylinder', 'cube_cylinder']

    log_string(logger, 'Load dataset ...')
    data_path = 'data/data/'
    test_dataset = ModelDataLoader(root=data_path, args=args, split='test')
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_category=args.num_category)
    criterion = model.get_loss('L2_loss', 'mean')

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        mean_loss, class_loss = test(args, classifier, criterion, testDataLoader)
        log_string(logger, 'mean Loss for every axis is %f, the loss for x, y, and z-axis is %f, %f, %f' % (np.mean(mean_loss), mean_loss[0], mean_loss[1], mean_loss[2]))
        for i, loss in enumerate(class_loss):
            log_string(logger, 'the loss for %s on x, y, and z-axis is %f, %f, %f' % (cat[i], loss[0], loss[1], loss[2]))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=7, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--log_dir', type=str, default='rotation', help='Experiment root')
    return parser.parse_args()

def test(args, classifier, criterion, testDataLoader):
    mean_loss = []
    class_loss = np.zeros((args.num_category, 7))
    classifier = classifier.eval()

    for j, (points, label, target, _, _) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

        points = points.data.numpy()
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        # label = torch.Tensor(label)
        points = torch.Tensor(points)
        points = provider.splice_torch(points, label)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred = classifier(points)
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
    mean_loss = np.mean(mean_loss, axis=0)

    return mean_loss, class_loss[:,4:7]

def create_log(experiment_dir):
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def log_string(logger, str):
    logger.info(str)
    print(str)

if __name__ == '__main__':
    args = parse_args()
    main(args)