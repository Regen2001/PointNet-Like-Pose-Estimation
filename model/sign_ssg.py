import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_category=7):
        super(get_model, self).__init__()
        channel = 3 + num_category

        self.sa1 = PointNetSetAbstraction(point_number=512, sample_number=32, radius=0.2, in_channel=channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(point_number=None, sample_number=None, radius=None, in_channel=128+channel, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.drop = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, points):
        B, _, _ = points.shape

        l1_points, l1_feature = self.sa1(points, None)

        l2_points, l2_feature = self.sa2(l1_points, l1_feature)

        x = l2_feature.view(B, 1024)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        x = torch.sigmoid(x)
        sign = torch.sign(x-0.5)
        return x, sign

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss
    
if __name__ == '__main__':
    import time
    import torch

    model = get_model()
    model.eval()
    loss = get_loss()
    x = torch.randn(2,10,1024)
    print(x.shape)
    start = time.time()
    pred, sign = model(x)
    end = time.time()
    print(pred, sign)
    print(pred.shape)
    target = torch.tensor([0,1]).float()
    print(loss(pred, target.reshape(-1,1)))
    print("The time used to calculated is ", end - start, 's')