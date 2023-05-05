import torch
import torch.nn as nn
import torch.nn.functional as F

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm:
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(points, number):
    """
    Input:
        points: pointcloud data, [B, N, 3]
        number: number of samples
    Return:
        centroids_idx: sampled pointcloud index, [B, number]
    """
    device = points.device
    B, N, C = points.shape
    centroids_idx = torch.zeros(B, number, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(number):
        centroids_idx[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids_idx

def query_ball_point(radius, number, points, new_points):
    """
    Input:
        radius: local region radius
        number: max sample number in local region
        points: all points, [B, N, 3]
        new_points: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, number]
    """
    device = points.device
    B, N, C = points.shape
    _, S, _ = new_points.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_points, points)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :number]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, number])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(points, feature, point_number, sample_number, radius, returnfps=False):
    """
    Input:
        point_number:
        radius:
        sample_number:
        points: input points position data, [B, N, 3]
        feature: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, point_number, sample_number, 3]
        new_points: sampled points data, [B, point_number, sample_number, 3+D]
    """
    B, N, C = points.shape
    S = point_number
    fps_idx = farthest_point_sample(points, point_number)
    new_points = index_points(points, fps_idx)
    idx = query_ball_point(radius, sample_number, points, new_points)
    grouped_points = index_points(points, idx)
    grouped_points_norm = grouped_points - new_points.view(B, S, 1, C)

    if feature is not None:
        grouped_points = index_points(feature, idx)
        new_feature = torch.cat([grouped_points_norm, grouped_points], dim=-1)
    else:
        new_feature = grouped_points_norm
    if returnfps:
        return new_points, new_feature, grouped_points, fps_idx
    else:
        return new_points, new_feature

def sample_and_group_all(points, feature):
    """
    Input:
        points: input points position data, [B, N, 3]
        feature: input points data, [B, N, D]
    Return:
        new_points: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = points.device
    B, N, C = points.shape
    # new_xyz代表中心点，用原点表示
    new_points = torch.zeros(B, 1, C).to(device)
    grouped_points = points.view(B, 1, N, C)
    if feature is not None:
        # view(B, 1, N, -1)，-1代表自动计算，即结果等于view(B, 1, N, D)
        new_feature = torch.cat([grouped_points, feature.view(B, 1, N, -1)], dim=-1)
    else:
        new_feature = grouped_points
    return new_points, new_feature

class PointNetSetAbstraction(nn.Module):
    def __init__(self, point_number, sample_number, radius, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.point_number = point_number
        self.radius = radius
        self.sample_number = sample_number
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, points, feature):
        points = points.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        if self.group_all:
            new_points, new_feature = sample_and_group_all(points, feature)
        else:
            new_points, new_feature = sample_and_group(points, feature, self.point_number, self.sample_number, self.radius)
        new_feature = new_feature.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_feature =  F.relu(bn(conv(new_feature)))

        new_feature = torch.max(new_feature, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_points, new_feature

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, point_number, sample_number_list, radius_list, in_channel, mlp_list, num_category=0):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.point_number = point_number
        self.radius_list = radius_list
        self.sample_number_list = sample_number_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3 + num_category
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, points, feature):
        points = points.permute(0, 2, 1)
        if feature is not None:
            feature = feature.permute(0, 2, 1)

        B, N, C = points.shape
        new_points = index_points(points, farthest_point_sample(points, self.point_number))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            group_idx = query_ball_point(radius, self.sample_number_list[i], points, new_points)
            grouped_points = index_points(points, group_idx)
            grouped_points -= new_points.view(B, self.point_number, 1, C)
            if feature is not None:
                grouped_feature = index_points(feature, group_idx)
                grouped_feature = torch.cat([grouped_feature, grouped_points], dim=-1)
            else:
                grouped_feature = grouped_points

            grouped_feature = grouped_feature.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_feature =  F.relu(bn(conv(grouped_feature)))
            new_feature = torch.max(grouped_feature, 2)[0]
            new_points_list.append(new_feature)

        new_points = new_points.permute(0, 2, 1)
        new_feature_concat = torch.cat(new_points_list, dim=1)
        return new_points, new_feature_concat

if __name__ == '__main__':

    def normalization_torch(point_cloud):
        device = point_cloud.device
        B, N, C = point_cloud.size()
        normalize = torch.zeros(B, N, C).to(device)
        for i in range(B):
            pc = point_cloud[i]
            centroid = torch.mean(pc, dim=0)
            pc = pc - centroid
            m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
            pc = pc / m
            normalize[i] = pc
        return normalize

    x = torch.randn(2,3,100)
    y = torch.randn(2,3,100)
    x = normalization_torch(x)
    pointnetsetabstraction = PointNetSetAbstraction(point_number=512, sample_number=32, radius=0.2, in_channel=6, mlp=[64, 64, 128], group_all=False)
    pointnetsetabstraction2 = PointNetSetAbstraction(point_number=128, sample_number=64, radius=0.4, in_channel=128+3, mlp=[128, 64, 128], group_all=False)
    new_points, new_feature = pointnetsetabstraction(x, y)
    new_points, new_feature = pointnetsetabstraction2(new_points, new_feature)
    print(new_points.size(), new_feature.size())
    # pointnetsetabstractionmsg = PointNetSetAbstractionMsg(50, [5, 10,15], [0.5, 0.7,0.9], 0, [[16, 32], [32, 64], [64, 128]])
    # new_points, new_feature = pointnetsetabstractionmsg(x, y)
    # print(new_points.size(), new_feature.size())