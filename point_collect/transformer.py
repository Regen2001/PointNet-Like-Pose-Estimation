import numpy as np

def normalization(point_cloud):
    """ Normalize the data, use coordinates of the block centered at origin,
        Input:
            NxC array
        Output:
            NxC array
    """
    centroid = np.mean(point_cloud[:,0:3], axis=0)
    point_cloud[:,0:3] = point_cloud[:,0:3] - centroid
    m = np.max(np.sqrt(np.sum(point_cloud[:,0:3] ** 2, axis=1)))
    point_cloud[:,0:3] = point_cloud[:,0:3] / m
    return point_cloud

def translation(point_cloud, delta=[0,0,0]):
    """ Translate the point cloud
        Input:
            NxC array
        Output:
            NxC array
    """
    point_cloud[:,0:3] = point_cloud[:,0:3] + delta
    return point_cloud

def rotation(point_cloud, angle=[0,0,0]):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based on X-Y-Z euler angles
        angle = [psi theta phi]
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
        X-Y-Z Euler angle is
        [                             cos(psi)*cos(theta),                             -cos(theta)*sin(psi),           sin(theta)]
        [cos(phi)*sin(psi) + cos(psi)*sin(phi)*sin(theta), cos(phi)*cos(psi) - sin(phi)*sin(psi)*sin(theta), -cos(theta)*sin(phi)]
        [sin(phi)*sin(psi) - cos(phi)*cos(psi)*sin(theta), cos(psi)*sin(phi) + cos(phi)*sin(psi)*sin(theta),  cos(phi)*cos(theta)]
    """
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
    point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], rotation_matrix.T)
    return point_cloud

def rotation_normal(point_cloud, angle=[0,0,0]):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based on X-Y-Y euler angles
        angle = [psi theta phi]
        Input:
          Nx6 array, original batch of point clouds
        Return:
          Nx6 array, rotated batch of point clouds
    """
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
    point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], rotation_matrix.T)
    point_cloud[:,3:6] = np.dot(point_cloud[:,3:6], rotation_matrix.T)
    return point_cloud

def scale_point_cloud(point_cloud, scale_x=1, scale_y=1, scale_z=1):
    """ scale the point cloud
        Input:
            NxC array
        Output:
            NxC array
    """
    point_cloud[:,0] = point_cloud[:,0] * scale_x
    point_cloud[:,1] = point_cloud[:,1] * scale_y
    point_cloud[:,2] = point_cloud[:,2] * scale_z
    return point_cloud

def transform(point_cloud, angle=[0,0,0], delta=[0,0,0]):
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
    point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], rotation_matrix.T)
    point_cloud[:,0:3] = point_cloud[:,0:3] + delta
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

if __name__ == '__main__':
    x = np.eye(3)
    print(normalization(x))
    print(rotation(x))
    print(rotation(x, [0, 0, 90]))
    print(scale_point_cloud(x))
    print(transform(x, [30,30,30], [1,1,1]))
    x = np.random.rand(10,6)
    print(rotation_normal(x))
    print(farthest_point_sample(x,4))
    print(random_sample(x,4))