from realsense import RealSense, visualization_point_cloud
import collect
import transformer

def main():
    camera = RealSense()
    camera.get_image([0,10])
    points = camera.calculate_point_cloud()
    visualization_point_cloud(points)
    # points = collect.clip_distance(points, [-1,-0.6])
    # camera.visualization_point_cloud_open3d(points)
    # points = collect.clip_distance(points, [-0.2,0.2], 0)
    # camera.visualization_point_cloud_open3d(points)
    # points = collect.delete_plane(points, visualize=True)
    # points = collect.delet_outlier_radius(points)
    # points = collect.cluster_point(points, visualize=True)

if __name__ == '__main__':
    main()