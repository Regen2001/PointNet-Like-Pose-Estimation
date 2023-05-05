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

    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_accuracy, class_accuracy = test(args, classifier, testDataLoader)
        log_string(logger, 'Test Instance Accuracy: %f' % (instance_accuracy))
        for i in range(0,args.num_category):
            log_string(logger, 'The accuracu of %s is %f' % (cat[i], class_accuracy[i]))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=7, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--log_dir', type=str, default='sign', help='Experiment root')
    return parser.parse_args()   

def test(args, model, testDataLoader):
    mean_correct = []
    class_accuracy = np.zeros((args.num_category, 3))
    classifier = model.eval()

    for j, (points, label, _, _, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):

        points = points.data.numpy()
        points[:, :, 0:3] = provider.normalization(points[:, :, 0:3])
        # label = torch.Tensor(label)
        points = torch.Tensor(points)
        points = provider.splice_torch(points, label)
        points = points.transpose(2, 1)

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        pred, sign = classifier(points)

        for cat in np.unique(label.cpu()):
            class_accuracy[cat, 0] += sign[label == cat].reshape(1,-1).eq(target[label == cat].long().data).cpu().sum()
            class_accuracy[cat, 1] += float(points[label == cat].size()[0])

        correct = sign.eq(target.float().reshape(-1,1).data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_accuracy[:, 2] = class_accuracy[:, 0] / class_accuracy[:, 1]
    # class_accuracy = np.mean(class_accuracy[:, 2])
    instance_accuracy = np.mean(mean_correct)

    return instance_accuracy, class_accuracy[:, 2]

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