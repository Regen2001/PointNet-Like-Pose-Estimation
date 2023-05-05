import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_category=7):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstraction(point_number=512, sample_number=32, radius=0.2, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(point_number=128, sample_number=64, radius=0.4, in_channel=128+3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(point_number=None, sample_number=None, radius=None, in_channel=256+3, mlp=[256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_category)

        self.drop = nn.Dropout(0.4)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, points):
        B, _, _ = points.shape

        l1_points, l1_feature = self.sa1(points, None)

        l2_points, l2_feature = self.sa2(l1_points, l1_feature)

        l3_points, l3_feature = self.sa3(l2_points, l2_feature)

        x = l3_feature.view(B, 1024)
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        x = F.log_softmax(x, -1)
        pred_choice = x.data.max(1)[1]
        return x, l3_feature, pred_choice

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss

if __name__ == '__main__':
    import torch

    x = torch.randn(2,3,1024)
    classifier = get_model(num_category=4)
    criterion = get_loss()
    classifier.train()
    pred, trans_feat, pred_choice = classifier(x)
    target = torch.tensor([2,1])
    loss = criterion(pred, target.long(), trans_feat)
    pred_choice = pred.data.max(1)[1]
    print(pred)
    print(trans_feat)
    print(loss)
    print(pred_choice)
