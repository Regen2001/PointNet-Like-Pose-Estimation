import numpy as np
from copy import deepcopy

class Cylinder:
    def __init__(self, radius=0.015, height=0.1, clip=False, sample=True):
        self.z_range = [-height/2, height/2]
        self.radius = radius
        self.euler = np.asarray([0,0,0])
        self.delta = np.asarray([0,0,0])

        circle = self.circle_boundary()
        self.points = self.stack(circle)

        plane_temp = self.circle_plane()
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

    def circle_boundary(self):
        num_points = self.number([0, 2*self.radius*np.pi])
        theta = np.linspace(0, 2*np.pi, num=num_points)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        circle = np.zeros((3, num_points))
        circle[0,:] = x
        circle[1,:] = y
        return circle.T

    def circle_plane(self):
        plane = np.empty((0,3))
        for i in np.linspace(0, np.pi, num=self.number([0, self.radius*np.pi])):
            x_point = self.radius * np.cos(i)
            y_point = self.radius * np.sin(i)
            num_points = self.number([-y_point, y_point])
            y = np.linspace(-y_point, y_point, num=num_points)
            temp = np.zeros((3, np.size(y)))
            temp[0,:] = x_point
            temp[1,:] = y
            plane = np.append(plane, temp.T, axis=0)
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

    def savePoints(self, num=0):
        self.path(num)
        np.savetxt(self.points_path, self.points, fmt='%6f', delimiter =",")
        np.savetxt(self.translation_path, self.delta.reshape(1,3), fmt='%6f', delimiter =",")
        np.savetxt(self.rotation_path, self.euler.reshape(1,3), fmt='%6f', delimiter =",")

    def path(self, number=0):
        num = "{:0>4d}".format(number)
        self.points_path = 'data/cylinder/cylinder_' + num + '.txt'
        self.translation_path = 'data/cylinder/cylinder_' + num + '_tran.txt'
        self.rotation_path = 'data/cylinder/cylinder_' + num + '_rot.txt'

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

# def circle_plane(radius):
#     plane = np.empty((0,3))
#     for i in np.linspace(0, np.pi, num=number([0, radius*np.pi])):
#         x_point = radius * np.cos(i)
#         y_point = radius * np.sin(i)
#         num_points = number([-y_point, y_point])
#         y = np.linspace(-y_point, y_point, num=num_points)
#         temp = np.zeros((3, np.size(y)))
#         temp[0,:] = x_point
#         temp[1,:] = y
#         plane = np.append(plane, temp.T, axis=0)
#     return plane

# def circle_boundary(radius):
#     num_points = number([0, 2*radius*np.pi])
#     theta = np.linspace(0, 2*np.pi, num=num_points)
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     circle = np.zeros((3, num_points))
#     circle[0,:] = x
#     circle[1,:] = y
#     return circle.T

# def translation(points, delta=[0,0,0]):
#     point_cloud = deepcopy(points)
#     point_cloud = point_cloud + delta
#     return point_cloud

# def stack(plane, z_range):
#     points = np.empty((0,3))
#     for i in np.linspace(z_range[0], z_range[1], num=number(z_range)):
#         points = np.append(points, translation(plane, [0,0,i]), axis=0)
#     return points

# def cylinder(radius, height):
#     z_range = [-height/2, height/2]

#     circle = circle_boundary(radius)
#     points = stack(circle, z_range)

#     plane_temp = circle_plane(radius)
#     points = np.append(points, translation(plane_temp, [0,0,z_range[0]]), axis=0)
#     points = np.append(points, translation(plane_temp, [0,0,z_range[1]]), axis=0)

#     return points

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import Axes3D
    import open3d
    import matplotlib.pyplot as plt

    def visualization_point_cloud_open3d(point_cloud):
        ''' visualize the point cloud via open3d
        '''
        points = deepcopy(point_cloud)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points[:,:3])
        if points.shape[1] == 6:
            pcd.colors = open3d.utility.Vector3dVector(points[:,3:])
        else:
            pcd.colors = open3d.utility.Vector3dVector(points)
        open3d.visualization.draw_geometries([pcd], point_show_normal=False, width=800, height=600)

    cylinder = Cylinder(15/1000, 4/100)
    cylinder.rotation([10,20,30])
    # cylinder.translation([0.5,0.5,0.5])
    # points = circle_plane(1)
    visualization_point_cloud_open3d(cylinder.get_points())
    # cylinder.savePoints(1)
    print(cylinder.get_points().shape)