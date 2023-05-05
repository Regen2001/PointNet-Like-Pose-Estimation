import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

class get_model(nn.Module):
    def __init__(self, mlp_list=[64,64,64,128,1024], linear_list=[512,256,2], num_category=5, normal_channel=True):
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

        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        for i in range(len(self.conv)):
            x = F.relu(self.bn_conv[i](self.conv[i](x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        for i in range(len(self.fc)):
            if i < (len(self.fc)-1):
                x = F.relu(self.dropout(self.bn_fc[i](self.fc[i](x))))
            else:
                x = self.fc[i](x)
        
        x = F.log_softmax(x, dim=1)
        pred_choice = x.data.max(1)[1]
        sign = (-1) ** pred_choice
        # return x, sign, pred_choice
        return sign

class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = F.nll_loss

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss

if __name__ == '__main__':
    import time

    model = get_model(normal_channel=False)
    loss = get_loss()
    x = torch.randn(2,8,1024*5)
    # print(x)
    start = time.time()
    pred, sign, pred_choice = model(x)
    end = time.time()
    print(pred, pred_choice, sign)
    target = torch.tensor([0,1])
    print(loss(pred, target.long()))
    print("The time used to calculated is ", end - start, 's')
    # The time used to calculated is  0.024004697799682617 s when point is 1024
    # The time used to calculated is  0.08401942253112793 s  when point is 1024*5