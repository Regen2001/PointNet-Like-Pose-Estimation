import numpy as np
import open3d
import transformer
from copy import deepcopy

def delete_plane(point_cloud, distance_threshold=0.006, ransac_n=20, num_iterations=1000, visualize=False):
    """
    Input:
        xyz: pointcloud data, [N, C]
    Return:
        point_cloud: the point cloud which remove the plance, plane: the point cloud for the plane
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud[:,:3])
    if point_cloud.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(point_cloud[:,3:])

    _, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

    if visualize:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        colors = np.array(pcd.colors)
        colors[inliers] = [0, 0, 1]
        pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
        open3d.visualization.draw_geometries([pcd], window_name="RANSAC Indicates plane segmentation", point_show_normal=False, width=800, height=600)
    
    point_cloud = np.delete(point_cloud, inliers, axis=0)
    return point_cloud

def cluster_point(point_cloud, eps=0.03, min_points=500, visualize=False):
    """
    Input:
        xyz: pointcloud data, [N, C]
    Return:
        point_cloud: the point cloud which the clustering and farthest point sample is done
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud[:,:3])
    if point_cloud.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(point_cloud[:,3:])

    label = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))

    number_label = np.max(label)
    if number_label == -1:
        print('Null point')
        return None
    
    if visualize:
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        colors = np.random.randint(255, size=(number_label+1, 3))/255
        colors = colors[label]
        colors[label < 0] = 0
        pcd.colors = open3d.utility.Vector3dVector(colors[:, :3])
        open3d.visualization.draw_geometries([pcd], window_name="DBSCAN cluster", point_show_normal=False, width=800, height=600)
    
    point_class = list()
    min_number = list()
    for i in range(number_label+1):
        temp = np.where(label==i)
        min_number.append(len(temp[0]))
        point_class.append(temp)
    
    min_number = min(min_number)
    cluster = np.zeros((number_label+1, min_number, point_cloud.shape[1]))
    for i in range(number_label+1):
        point = point_cloud[point_class[i],:].squeeze()
        cluster[i,:,:] = transformer.farthest_point_sample(point, min_number)
    return cluster

def clip_distance(point_cloud, dis=[0,2], axis=2):
    ''' Keep the points in the range
    '''
    idx = np.where(point_cloud[:,axis]>=dis[0])
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:,axis]<=dis[1])
    point_cloud = point_cloud[idx]
    return point_cloud

def delet_outlier_statistical(point_cloud, nb_neighbors=120, std_ratio=0.1):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud[:,:3])
    if point_cloud.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(point_cloud[:,3:])
    result = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    if point_cloud.shape[1] == 6:
        point_cloud = np.hstack((np.asarray(result[0].points), np.asarray(result[0].colors)))
    else:
        point_cloud = np.asarray(result[0].points)
    return point_cloud

def delet_outlier_radius(point_cloud, nb_points=200, radius=0.05):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud[:,:3])
    if point_cloud.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(point_cloud[:,3:])
    result = pcd.remove_radius_outlier(nb_points, radius)
    if point_cloud.shape[1] == 6:
        point_cloud = np.hstack((np.asarray(result[0].points), np.asarray(result[0].colors)))
    else:
        point_cloud = np.asarray(result[0].points)
    return point_cloud

def visualization_point_cloud(point_cloud):
    points = np.empty((0,6))
    if len(point_cloud.shape) == 3:
        for i in point_cloud:
            points = np.append(points, i, axis=0)
    else:
        points = deepcopy(point_cloud)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:,:3])
    if points.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(points[:,3:])
    else:
        pcd.colors = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([pcd], point_show_normal=False, width=800, height=600)

def load_ply(path):
    ply = open3d.io.read_triangle_mesh(path)
    points = np.append(np.asarray(ply.vertices), np.asarray(ply.vertex_colors), axis=1)
    return points


if __name__ == '__main__':
    x = np.random.randn(50, 6)
    delet_outlier_radius(x)
    delet_outlier_statistical(x)