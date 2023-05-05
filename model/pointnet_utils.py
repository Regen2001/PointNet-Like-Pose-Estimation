import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class TNet3d(nn.Module):
    def __init__(self, channel):
        super(TNet3d, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3*3)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = Variable(torch.from_numpy(np.eye(3).flatten().astype(np.float32))).view(1, 3*3).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class TNetkd(nn.Module):
    def __init__(self, channel):
        super(TNetkd, self).__init__()
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channel*channel)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.channel = channel

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.channel).flatten().astype(np.float32))).view(1, self.channel * self.channel).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.channel, self.channel)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, channel=3):
        super(PointNetEncoder, self).__init__()
        self.tnet = TNet3d(channel=channel)
        self.ftnet = TNetkd(channel=64)

        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feat = global_feat

    def forward(self, x):
        B, D, N = x.size()
        transform = self.tnet(x)
        # x = x.transpose(2, 1)
        
        # if D > 3:
        #     normal = x[:, :, 3:]
        #     x = x[:, :, :3]
        # x = torch.bmm(x, transform)
        
        # if D > 3:
        #     x = torch.cat([x, normal], dim=2)
        # x = x.transpose(2, 1)
        if D > 3:
            normal = x[:, 3:, :]
            x = x[:, :3, :]
        x = torch.bmm(transform, x)
        
        if D > 3:
            x = torch.cat([x, normal], dim=2)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.ftnet(x)
        # x = x.transpose(2, 1)
        # x = torch.bmm(x, trans_feat)
        # x = x.transpose(2, 1)
        x = torch.bmm(trans_feat, x)

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        if self.global_feat:
            return x, transform, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), transform, trans_feat

def feature_transform_reguliarzer(transform):
    d = transform.size()[1]
    I = torch.eye(d)[None, :, :]
    if transform.is_cuda:
        I = I.cuda()

    loss = torch.mean(torch.norm(torch.bmm(transform, transform.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

if __name__ == '__main__':
    x = torch.randn(2,3,5)
    print(x)
    tnet = TNet3d(channel=3)
    encoder = PointNetEncoder(global_feat=False ,channel=3)
    print(tnet(x))
    print(encoder(x))