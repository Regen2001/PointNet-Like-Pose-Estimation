import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class get_model(nn.Module):
    def __init__(self, mlp_list=[64,64,64,128,1024], linear_list=[512,256,3], num_category=5, normal_channel=True):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6 + num_category
        else:
            channel = 3 + num_category

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

        self.fc1 = nn.Linear(3,6)
        self.fc2 = nn.Linear(6,3)

        self.bn1 = nn.BatchNorm1d(6)

        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        mean = x[:,:3,:]
        mean = torch.mean(mean, dim=2)
        mean = F.relu(self.bn1(self.fc1(mean)))
        mean = self.fc2(mean)

        for i in range(len(self.conv)):
            x = F.relu(self.bn_conv[i](self.conv[i](x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        for i in range(len(self.fc)):
            if i < (len(self.fc)-1):
                x = F.relu(self.dropout(self.bn_fc[i](self.fc[i](x))))
            else:
                x = self.fc[i](x)
        return x + mean

class get_loss(torch.nn.Module):
    def __init__(self, L2_loss=False):
        super(get_loss, self).__init__()
        if L2_loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()
    
    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss

if __name__ == '__main__':
    import time

    model = get_model(normal_channel=False)
    loss = get_loss(L2_loss=True)
    x = torch.randn(2,8,1024*5)
    # print(x)
    start = time.time()
    pred = model(x)
    end = time.time()
    print(pred)
    target = torch.randn(2,3)
    print(loss(pred, target))
    print("The time used to calculated is ", end - start, 's')
    # The time used to calculated is  0.02100348472595215 s when point is 1024
    # The time used to calculated is  0.08001708984375 s  when point is 1024*5