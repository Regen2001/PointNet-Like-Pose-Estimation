import os
import numpy as np
import warnings
from random import shuffle

from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

def farthest_point_sample(point_cloud, number=1024):
    """
    Input:
        xyz: pointcloud data, [N, C]
        number: number of samples
    Return:
        centroids: sampled pointcloud index, [number, D]
    """
    N, C = point_cloud.shape
    xyz = point_cloud[:,:3]
    centroids = np.zeros((number,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(number):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point_cloud = point_cloud[centroids.astype(np.int32)]
    return point_cloud

def random_sample(point_cloud, number=1024):
    """
    Input:
        xyz: pointcloud data, [N, C]
        number: number of samples
    Return:
        centroids: sampled pointcloud index, [number, D]
    """
    N, C = point_cloud.shape
    if N <= number:
        return point_cloud
    sample_idx = np.random.choice(N, number, replace=False)
    sample = point_cloud[sample_idx]
    return sample

class ModelDataLoader(Dataset):
    def __init__(self, root, args, split='train'):
        self.root = root
        self.num_category = args.num_category
        self.cat = ['cube', 'cuboid', 'cylinder', 'h_structure', 'double_cube', 'double_cylinder', 'cube_cylinder']
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.datapath = list()

        temp_list = list(range(1,8001))
        train_list = temp_list[:6001]
        test_list = temp_list[6001:]

        if split == 'train':
            for item in self.cat:
                for i in train_list:
                    item_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '.txt'
                    rot_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '_rot.txt'
                    tran_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '_tran.txt'
                    self.datapath.append((item, item_path, rot_path, tran_path))
        if split == 'test':
            for item in self.cat:
                for i in test_list:
                    item_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '.txt'
                    rot_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '_rot.txt'
                    tran_path = root + item + '/' + item + '_' + '{:0>4d}'.format(i) + '_tran.txt'
                    self.datapath.append((item, item_path, rot_path, tran_path))

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        dataroot_tuple = self.datapath[index]
        label = self.classes[dataroot_tuple[0]]
        points_root = dataroot_tuple[1]
        rot_root = dataroot_tuple[2]
        tran_root = dataroot_tuple[3]

        points = np.loadtxt(points_root, delimiter =",")
        points = random_sample(points)
        rot = np.loadtxt(rot_root, delimiter =",")
        sign = np.sign(rot[2])
        rot[2] = np.absolute(rot[2])
        tran = np.loadtxt(tran_root, delimiter =",")
        return points, label, rot, tran, sign

if __name__ == '__main__':
    import torch
    import argparse

    def parse_args():
        '''PARAMETERS'''
        parser = argparse.ArgumentParser('training')
        parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
        parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
        return parser.parse_args()

    args = parse_args()
    data = ModelDataLoader(root='data/data/', args=args, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=24, shuffle=True)
    for point, label, rot, tran, sign in DataLoader:
        print(point.shape)
        print(label.shape)
        print(rot.shape)
        print(tran.shape)