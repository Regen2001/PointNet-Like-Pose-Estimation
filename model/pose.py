import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import TNet3d, TNetkd

class get_model(nn.Module):
    def __init__(self, mlp_list, linear_list, mean=False, classify=False, num_category=7, normal_channel=True, transform=False, feat_trans=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6 + num_category
        else:
            channel = 3 + num_category

        if transform:
            self.tnet = TNet3d(channel=channel)
        if feat_trans:
            self.ftnet = TNetkd(channel=64)

        self.conv = nn.ModuleList()
        self.bn_conv = nn.ModuleList()
        in_channel = channel
        for out_channel in mlp_list:
            self.conv.append(nn.Conv1d(in_channel, out_channel, 1))
            self.bn_conv.append(nn.BatchNorm1d(out_channel))
            in_channel = out_channel

        self.fc = nn.ModuleList()
        self.bn_fc = nn.ModuleList()
        for out_channel in linear_list:
            self.fc.append(nn.Linear(in_channel, out_channel))
            self.bn_fc.append(nn.BatchNorm1d(out_channel))
            in_channel = out_channel

        if mean:
            self.fc1 = nn.Linear(3,6)
            self.fc2 = nn.Linear(6,3)
            self.bn1 = nn.BatchNorm1d(6)

        self.dropout = nn.Dropout(p=0.4)

        self.mean = mean
        self.classify = classify
        self.transform = transform
        self.feat_trans = feat_trans

    def forward(self, x):
        B, D, N = x.size()

        if self.transform:
            transform = self.tnet(x)
            if D > 3:
                normal = x[:, 3:, :]
                x = x[:, :3, :]
            x = torch.bmm(transform, x)
            if D > 3:
                x = torch.cat([x, normal], dim=2)

        for i in range(2):
            x = F.relu(self.bn_conv[i](self.conv[i](x)))

        if self.feat_trans:        
            trans_feat = self.ftnet(x)
            x = torch.bmm(trans_feat, x)

        for i in range(2, len(self.conv)):
            x = F.relu(self.bn_conv[i](self.conv[i](x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        for i in range(len(self.fc)):
            if i < (len(self.fc)-1):
                x = F.relu(self.dropout(self.bn_fc[i](self.fc[i](x))))
            else:
                x = self.fc[i](x)
        
        if self.mean:
            mean = x[:,:3,:]
            mean = torch.mean(mean, dim=2)
            mean = F.relu(self.bn1(self.fc1(mean)))
            mean = self.fc2(mean)
            return mean + x
        
        if self.classify:
            x = F.log_softmax(x, dim=1)
            pred_choice = x.data.max(1)[1]
            sign = (-1) ** pred_choice
            return x, sign, pred_choice

        return x

class get_loss(torch.nn.Module):
    def __init__(self, L2_loss=False, classify=False):
        super(get_loss, self).__init__()
        if L2_loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
        if classify:
            self.loss = F.nll_loss()
    
    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss

if __name__ == '__main__':
    model = get_model(normal_channel=False)
    loss = get_loss(L2_loss=True)
    x = torch.randn(2,10,6)
    print(x)
    pred = model(x)
    print(pred)
    target = torch.randn(2,3)
    print(loss(pred, target))