import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class get_model(nn.Module):
    def __init__(self, num_category=7):
        super(get_model, self).__init__()
        channel = 3

        self.feat = PointNetEncoder(channel=channel)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_category)

        self.dropout = nn.Dropout(p=0.4)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, _, trans_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)
        pred_choice = x.data.max(1)[1]
        return x, trans_feat, pred_choice

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

if __name__ == '__main__':
    x = torch.randn(2,3,10)
    classifier = get_model(num_category=3)
    criterion = get_loss()
    classifier.train()
    pred, trans_feat, pred_choice = classifier(x)
    target = torch.tensor([2,1])
    loss = criterion(pred, target.long(), trans_feat)
    # pred_choice = pred.data.max(1)[1]
    print(pred)
    # print(trans_feat)
    print(loss)
    print(pred_choice)
