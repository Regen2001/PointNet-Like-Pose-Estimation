import numpy as np
import torch
import open3d

def normalization(point_cloud):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = point_cloud.shape
    normalize = np.zeros((B, N, C))
    for i in range(B):
        pc = point_cloud[i]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        normalize[i] = pc
    return normalize

def normalization_torch(point_cloud):
    """ Normalize the batch data, use coordinates of the block centered at origin in torch
    Input:
        BxNxC tensor
    Output:
        BxNxC tensor
    """
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

def shuffle_data(point_cloud, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return point_cloud[idx, ...], labels[idx], idx

def shuffle_point(point_cloud):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(point_cloud.shape[1])
    np.random.shuffle(idx)
    return point_cloud[:,idx,:]

def rotate_point_cloud(point_cloud, angle=[0,0,0]):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based on X-Y-Z euler angles
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    angle = np.radians(angle)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle[0]), -np.sin(angle[0])],
                    [0, np.sin(angle[0]), np.cos(angle[0])]])
    R_y = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])],
                    [0, 1, 0],
                    [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    R_z = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                    [np.sin(angle[2]), np.cos(angle[2]), 0],
                    [0, 0, 1]])
    rotation_matrix = np.dot(np.dot(R_x, R_y), R_z)
    for i in range(point_cloud.shape[0]):
        pc = point_cloud[i, ...]
        rotated_data[i, ...] = np.dot(pc, rotation_matrix.T)
    return rotated_data

def rotate_point_cloud_with_normal(point_cloud, angle=[0,0,0]):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based on X-Y-Z euler angles
        Input:
          BxNx6 array, original batch of point clouds
        Return:
          BxNx6 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    angle = np.radians(angle)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle[0]), -np.sin(angle[0])],
                    [0, np.sin(angle[0]), np.cos(angle[0])]])
    R_y = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])],
                    [0, 1, 0],
                    [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    R_z = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                    [np.sin(angle[2]), np.cos(angle[2]), 0],
                    [0, 0, 1]])
    rotation_matrix = np.dot(np.dot(R_x, R_y), R_z)
    for i in range(point_cloud.shape[0]):
        shape_pc = point_cloud[i,:,3:6]
        shape_normal = point_cloud[i,:,3:6]
        rotated_data[i,:,0:3] = np.dot(shape_pc, rotation_matrix.T)
        rotated_data[i,:,3:6] = np.dot(shape_normal, rotation_matrix.T)
    return rotated_data

def jitter_point_cloud(point_cloud, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = point_cloud.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -clip, clip)
    jittered_data += point_cloud
    return jittered_data

def shift_point_cloud(point_cloud, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = point_cloud.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for i in range(B):
        point_cloud[i,:,:] += shifts[i,:]
    return point_cloud

def random_scale_point_cloud(point_cloud, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = point_cloud.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for i in range(B):
        point_cloud[i,:,:] *= scales[i]
    return point_cloud

def random_point_dropout(point_cloud, max_dropout_ratio=0.875):
    ''' point_cloud: BxNx3 '''
    for b in range(point_cloud.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((point_cloud.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            point_cloud[b,drop_idx,:] = point_cloud[b,0,:] # set to the first point
    return point_cloud

def splice_torch(point_cloud, category, num_category=7):
    """  splice the point cloud data and variety in torch
    Input:
        BxNxC tensor, point_class tensor, 
    Output:
        BxNx(C+class_number) tensor
    """
    device = point_cloud.device
    point_cloud = point_cloud.transpose(2,1)
    B, C, N = point_cloud.size()
    class_vector = torch.zeros(B, num_category, N).to(device)
    for i in range(B):
        class_vector[i,category[i].item(),:] = 1
    point_cloud = torch.cat([point_cloud, class_vector], 1).transpose(2,1)
    return point_cloud


def farthest_point_sample(point_cloud, number):
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

if __name__ == '__main__':
    x = np.random.rand(2,10,3)
    print(normalization(x))
    print(shuffle_data(x, np.array([[1,2,3], [4,5,6]])))
    print(shuffle_point(x))
    print(rotate_point_cloud(x, psi=True, theta=True, phi=True))
    print(jitter_point_cloud(x))
    print(shift_point_cloud(x))
    print(random_scale_point_cloud(x))
    print(random_point_dropout(x))
    x = np.random.rand(2,10,6)
    print(rotate_point_cloud_with_normal(x))
    x = torch.randn(3,10,3)
    print(normalization_torch(x))
    c = torch.tensor([0,1,2])
    print(splice_torch(x, c, 3))