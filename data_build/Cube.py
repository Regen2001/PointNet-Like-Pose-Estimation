import numpy as np
from copy import deepcopy

class Cube:
    def __init__(self, length=0.05, width=0.05, height=0.05, clip=False, sample=True):
        self.points = np.empty((0,3))
        self.x_range = [-length/2, length/2]
        self.y_range = [-width/2, width/2]
        self.z_range = [-height/2, height/2]
        self.euler = np.asarray([0,0,0])
        self.delta = np.asarray([0,0,0])

        self.points = np.append(self.points, self.plane([self.x_range[0], self.x_range[0]], self.y_range), axis=0)
        self.points = np.append(self.points, self.plane([self.x_range[1], self.x_range[1]], self.y_range), axis=0)
        self.points = np.append(self.points, self.plane(self.x_range, [self.y_range[0], self.y_range[0]]), axis=0)
        self.points = np.append(self.points, self.plane(self.x_range, [self.y_range[1], self.y_range[1]]), axis=0)

        self.points = self.stack(self.points)

        plane_temp = self.plane(self.x_range, self.y_range)

        self.points = np.append(self.points, self.translate(plane_temp, self.z_range[0]), axis=0)
        self.points = np.append(self.points, self.translate(plane_temp, self.z_range[1]), axis=0)

        if sample:
            self.points = farthest_point_sample(self.points)

        if clip:
            self.clip_half()

    def number(self, range, delta = 1/500):
        l = range[1] - range[0]
        if l == 0:
            return 1
        return int(l/delta)

    def plane(self, x_range, y_range):
        plane = np.empty((0,3))
        for i in np.linspace(x_range[0], x_range[1], num=self.number(x_range)):
            for j in np.linspace(y_range[0], y_range[1], num=self.number(y_range)):
                plane = np.append(plane, np.array([i, j, 0]).reshape(1,3), axis=0)
        return plane

    def stack(self, plane):
        points = np.empty((0,3))
        for i in np.linspace(self.z_range[0], self.z_range[1], num=self.number(self.z_range)):
            points = np.append(points, self.translate(plane, i), axis=0)
        return points

    def translate(self, points, delta_z):
        point_cloud = deepcopy(points)
        point_cloud = point_cloud + [0, 0, delta_z]
        return point_cloud

    def translation(self, delta=[0,0,0]):
        self.points = self.points + delta
        self.delta = np.asarray(delta)

    def rotation(self, angle=[0,0,0]):
        self.euler = np.asarray(angle)
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
        self.points = np.dot(self.points, rotation_matrix.T)

    def clip_half(self):
        z_max = np.max(self.points[:,2])
        z_min = np.min(self.points[:,2])
        threshold = (z_max+z_min)/2
        idx = np.where(self.points[:,2]>=threshold)
        self.points = self.points[idx]

    def get_points(self):
        return self.points

    def get_translation(self):
        return self.delta

    def get_rotation(self):
        return self.euler

    def savePoints(self, name='cube', num=0):
        self.path(name, num)
        np.savetxt(self.points_path, self.points, fmt='%6f', delimiter =",")
        np.savetxt(self.translation_path, self.delta.reshape(1,3), fmt='%6f', delimiter =",")
        np.savetxt(self.rotation_path, self.euler.reshape(1,3), fmt='%6f', delimiter =",")

    def path(self, name, number):
        num = "{:0>4d}".format(number)
        self.points_path = 'data/'+ name + '/' + name + '_' + num + '.txt'
        self.translation_path = 'data/'+ name + '/' + name + '_' + num + '_tran.txt'
        self.rotation_path = 'data/'+ name + '/' + name + '_' + num + '_rot.txt'

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

# def visualization_point_cloud(point_cloud):
#     ''' visualize the point cloud via matplotlib
#     '''
#     points = deepcopy(point_cloud)
#     if points.shape[1] == 6:
#         color = points[:,3:]
#     else:
#         color = points[:, 2]
#     ax = plt.figure().add_subplot(projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, cmap='rainbow', marker=".")
#     ax.axis()
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.show()

# def visualization_point_cloud_open3d(point_cloud):
#     ''' visualize the point cloud via open3d
#     '''
#     points = deepcopy(point_cloud)
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(points[:,:3])
#     if points.shape[1] == 6:
#         pcd.colors = open3d.utility.Vector3dVector(points[:,3:])
#     else:
#         pcd.colors = open3d.utility.Vector3dVector(points)
#     open3d.visualization.draw_geometries([pcd], point_show_normal=False, width=800, height=600)

# def number(range, delta = 1/100):
#     l = range[1] - range[0]
#     if l == 0:
#         return 1
#     return int(l/delta)

# def plane(x_range, y_range):
#     plane = np.empty((0,3))
#     for i in np.linspace(x_range[0], x_range[1], num=number(x_range)):
#         for j in np.linspace(y_range[0], y_range[1], num=number(y_range)):
#             plane = np.append(plane, np.array([i, j, 0]).reshape(1,3), axis=0)
#     return plane

# def translation(points, delta=[0,0,0]):
#     point_cloud = deepcopy(points)
#     point_cloud = point_cloud + delta
#     return point_cloud

# def stack(plane, z_range):
#     points = np.empty((0,3))
#     for i in np.linspace(z_range[0], z_range[1], num=number(z_range)):
#         points = np.append(points, translation(plane, [0,0,i]), axis=0)
#     return points

# def cube(length, width, height):
#     points = np.empty((0,3))

#     x_range = [-length/2, length/2]
#     y_range = [-width/2, width/2]
#     z_range = [-height/2, height/2]

#     points = np.append(points, plane([x_range[0], x_range[0]], y_range), axis=0)
#     points = np.append(points, plane([x_range[1], x_range[1]], y_range), axis=0)
#     points = np.append(points, plane(x_range, [y_range[0], y_range[0]]), axis=0)
#     points = np.append(points, plane(x_range, [y_range[1], y_range[1]]), axis=0)

#     points = stack(points, z_range)

#     plane_temp = plane(x_range, y_range)

#     points = np.append(points, translation(plane_temp, [0,0,z_range[0]]), axis=0)
#     points = np.append(points, translation(plane_temp, [0,0,z_range[1]]), axis=0)

#     return points

# def clip_half(point_cloud):
#     ''' Keep the points in the range
#     '''
#     z_max = np.max(point_cloud[:,2])
#     z_min = np.min(point_cloud[:,2])
#     threshold = (z_max+z_min)/2
#     idx = np.where(point_cloud[:,2]>=threshold)
#     point_cloud = point_cloud[idx]
#     return point_cloud

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    def visualization_point_cloud(point_cloud):
        ''' visualize the point cloud via matplotlib
        '''
        points = deepcopy(point_cloud)
        if points.shape[1] == 6:
            color = points[:,3:]
        else:
            color = points[:, 2]
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, cmap='rainbow', marker=".")
        ax.axis()
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    cube = Cube(3/100, 7/100, 4/100)
    # cube.rotation([45,45,45])
    # cube.translation([1,1,1])
    # cube.clip_half()
    visualization_point_cloud(cube.get_points())
    # cube.savePoints(1)
    print(cube.get_points().shape)