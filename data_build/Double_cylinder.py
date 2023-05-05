import numpy as np
from copy import deepcopy
from Cylinder import Cylinder

class Double_cylinder:
    def __init__(self, size_1=[0.03,0.03], size_2=[0.05,0.05], excursion=True, clip=False):
        self.euler = np.asarray([0,0,0])
        self.delta = np.asarray([0,0,0])
        self.excursion = excursion
        cylinder_1 = Cylinder(size_1[0], size_1[1], sample=False)
        cylinder_2 = Cylinder(size_2[0], size_2[1], sample=False)

        cylinder_1.translation([0, 0, (size_1[1]+size_2[1])/2])
        if self.excursion:
            cylinder_1.translation([(size_2[0]-size_1[0])/3, (size_2[0]-size_1[0])/2, 0])

        self.points = np.append(cylinder_1.get_points(), cylinder_2.get_points(), axis=0)

        self.points = farthest_point_sample(self.points)

        if clip:
            self.clip_half()

    def translation(self, delta=[0,0,0]):
        self.points = self.points + delta
        self.delta = np.array(delta)

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

    def path(self, number):
        num = "{:0>4d}".format(number)
        self.points_path = 'data/double_cylinder/double_cylinder_' + num + '.txt'
        self.translation_path = 'data/double_cylinder/double_cylinder_' + num + '_tran.txt'
        self.rotation_path = 'data/double_cylinder/double_cylinder_' + num + '_rot.txt'

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

    points = Double_cylinder(excursion=True)
    points.rotation([10,20,30])
    visualization_point_cloud_open3d(points.get_points())
    # points.savePoints(1)
    print(points.get_points().shape)