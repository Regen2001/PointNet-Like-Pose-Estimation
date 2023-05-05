import numpy as np
from copy import deepcopy

class H_structure:
    def __init__(self, H=0.1, B=0.1, t1=0.02, t2=0.02, height=0.1, clip=False):
        self.points = np.empty((0,3))
        self.x_range_max = [-B/2, B/2]
        self.x_range_min = [-t1/2, t1/2]
        self.y_range_max = [-H/2, H/2]
        self.y_range_min = [-(H/2-t2), H/2-t2]
        self.z_range = [-height/2, height/2]
        self.euler = np.asarray([0,0,0])
        self.delta = np.asarray([0,0,0])

        l1 = self.plane(self.x_range_max, [0, 0])
        self.points = self.copy_line(self.points, l1, delta_y=self.y_range_max+self.y_range_min)

        l2 = self.plane([0, 0], self.y_range_min)
        self.points = self.copy_line(self.points, l2, delta_x=self.x_range_min)

        l3 = self.plane([0, 0], [-t2/2, t2/2])
        self.points = self.copy_line(self.points, l3, self.x_range_max, [-(H-t2)/2, (H-t2)/2])

        self.points = self.delete(self.points, self.x_range_min, [self.y_range_min[0], self.y_range_min[0]])
        self.points = self.delete(self.points, self.x_range_min, [self.y_range_min[1], self.y_range_min[1]])

        self.points = self.stack(self.points)

        plane_small = self.plane(self.x_range_max, [-t2/2, t2/2])
        plane_temp = self.translate(plane_small, [0, -(H-t2)/2, 0])
        plane_temp = np.append(plane_temp, self.translate(plane_small, [0, (H-t2)/2, 0]), axis=0)
        plane_temp = np.append(plane_temp, self.plane(self.x_range_min, self.y_range_min), axis=0)

        self.points = np.append(self.points, self.translate(plane_temp, [0,0,self.z_range[0]]), axis=0)
        self.points = np.append(self.points, self.translate(plane_temp, [0,0,self.z_range[1]]), axis=0)

        self.points = farthest_point_sample(self.points)

        self.rotation([90,0,0])

        if clip:
            self.clip_half()

    def number(self, range, delta = 1/250):
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
            points = np.append(points, self.translate(plane, [0,0,i]), axis=0)
        return points

    def copy_line(self, point_cloud, line, delta_x=[0], delta_y=[0]):
        points = deepcopy(point_cloud)
        for i in delta_x:
            for j in delta_y:
                l_temp = line + np.array([i, j, 0])
                points = np.append(points, l_temp, axis=0)
        return points 

    def delete(self, point_cloud, x_range, y_range):
        points = np.empty((0,3))
        for i in point_cloud:
            if x_range[0]<=i[0] and i[0]<=x_range[1]:
                if y_range[0]<=i[1] and i[1]<=y_range[1]:
                    continue
            points = np.append(points, i.reshape(1,3), axis=0)
        return points

    def translate(self, points, delta=[0,0,0]):
        point_cloud = deepcopy(points)
        point_cloud = point_cloud + delta
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
        self.points_path = 'data/h_structure/h_structure_' + num + '.txt'
        self.translation_path = 'data/h_structure/h_structure_' + num + '_tran.txt'
        self.rotation_path = 'data/h_structure/h_structure_' + num + '_rot.txt'

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

# def translate(points, delta=[0,0,0]):
#     point_cloud = deepcopy(points)
#     point_cloud = point_cloud + delta
#     return point_cloud

# def stack(plane, z_range):
#     points = np.empty((0,3))
#     for i in np.linspace(z_range[0], z_range[1], num=number(z_range)):
#         points = np.append(points, translate(plane, [0,0,i]), axis=0)
#     return points

# def copy_line(point_cloud, line, delta_x=[0], delta_y=[0]):
#     points = deepcopy(point_cloud)
#     for i in delta_x:
#         for j in delta_y:
#             l_temp = line + np.array([i, j, 0])
#             points = np.append(points, l_temp, axis=0)
#     return points 

# def delete(point_cloud, x_range, y_range):
#     points = np.empty((0,3))
#     for i in point_cloud:
#         if x_range[0]<=i[0] and i[0]<=x_range[1]:
#             if y_range[0]<=i[1] and i[1]<=y_range[1]:
#                 continue
#         points = np.append(points, i.reshape(1,3), axis=0)
#     return points

# def H_structure(H, B, t1, t2, height):
#     points = np.empty((0,3))
#     x_range_max = [-B/2, B/2]
#     x_range_min = [-t1/2, t1/2]
#     y_range_max = [-H/2, H/2]
#     y_range_min = [-(H/2-t2), H/2-t2]
#     z_range = [-height/2, height/2]

#     l1 = plane(x_range_max, [0, 0])
#     points = copy_line(points, l1, delta_y=y_range_max+y_range_min)

#     l2 = plane([0, 0], y_range_min)
#     points = copy_line(points, l2, delta_x=x_range_min)

#     l3 = plane([0, 0], [-t2/2, t2/2])
#     points = copy_line(points, l3, x_range_max, [-(H-t2)/2, (H-t2)/2])

#     points = delete(points, x_range_min, [y_range_min[0], y_range_min[0]])
#     points = delete(points, x_range_min, [y_range_min[1], y_range_min[1]])

#     points = stack(points, z_range)

#     plane_small = plane(x_range_max, [-t2/2, t2/2])
#     plane_temp = translate(plane_small, [0, -(H-t2)/2, 0])
#     plane_temp = np.append(plane_temp, translate(plane_small, [0, (H-t2)/2, 0]), axis=0)
#     plane_temp = np.append(plane_temp, plane(x_range_min, y_range_min), axis=0)

#     points = np.append(points, translate(plane_temp, [0,0,z_range[0]]), axis=0)
#     points = np.append(points, translate(plane_temp, [0,0,z_range[1]]), axis=0)

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

    H = H_structure(H=8/100, B=8/100, t1=1/100, t2=1/100, height=5/100)
    # H.clip_half()
    # H.translation([0.5,0.5,0.5])
    # centroid = np.mean(H.get_points(), axis=0)
    # print(centroid)
    visualization_point_cloud(H.get_points())
    # H.savePoints(1)
    print(H.get_points().shape)