import pyrealsense2 as rs
import json
import numpy as np
import cv2
import open3d

def read_json(path='camera.json', model='read', depth_intr=None, color_intr=None):
    try:
        fp = open(path, 'r', encoding='utf-8')
        data = json.load(fp)
        fp.close
    except:
        data = {"width": 640, "height": 480, "fps": 30}
    
    if model == 'read':
        return data

    data["depth_ppx"] = depth_intr.ppx
    data["depth_ppy"] = depth_intr.ppy
    data["depth_fx"] = depth_intr.fx
    data["depth_fy"] = depth_intr.fy
    data["color_ppx"] = color_intr.ppx
    data["color_ppy"] = color_intr.ppy
    data["color_fx"] = color_intr.fx
    data["color_fy"] = color_intr.fy

    fp = open(path, 'w', encoding='utf-8')
    fp.write(json.dumps(data, indent=4, ensure_ascii=False))
    fp.close()
    return 0

def set_camera(state='open', pipeline=None, depth_intr=None, color_intr=None, path='camera.json'):
    if state == 'close':
        read_json(path, "save", depth_intr, color_intr)
        pipeline.stop()
        return 0

    data = read_json(path=path)

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, data["width"], data['height'], rs.format.z16, data["fps"])
    config.enable_stream(rs.stream.color, data["width"], data['height'], rs.format.bgr8, data["fps"])
    profile = pipeline.start(config)
    
    align_to = rs.stream.color
    align = rs.align(align_to)

    depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return profile, pipeline, align, depth_intr, color_intr

def get_image(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frame = align.process(frames)

    depth_frame = aligned_frame.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())

    color_frame = aligned_frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return depth_frame, color_frame, depth_image, color_image

def point_cloud(depth_frame, depth_intr):
    width = depth_intr.width
    height = depth_intr.height
    points = np.zeros((width*height, 3))
    z = 0
    for i in range(width):
        for j in range(height):
                dis = depth_frame.get_distance(i, j)
                points[z,:] = np.asarray(rs.rs2_deproject_pixel_to_point(depth_intr, [i, j], dis))
                z += 1
    return points

def visualization_RGBD(depth_image, color_image, depth_scale, clipping_distance_in_meters=[0.6, 6]):
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    bg_removed = np.where((depth_image_3d <= clipping_distance_in_meters[0]/depth_scale) | (depth_image_3d > clipping_distance_in_meters[1]/depth_scale), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB-D', images)
    return 0

def point_cloud_open3d(depth_image, color_image, depth_intr):
    camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(depth_intr.width, depth_intr.height, depth_intr.fx, depth_intr.fy, depth_intr.ppx, depth_intr.ppy)
    img_depth = open3d.geometry.Image(depth_image)
    img_color = open3d.geometry.Image(color_image)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
    return pcd

def image_realsense(point_type='array', clipping_distance_in_meters=[0.5, 2]):
    profile, pipeline, align, depth_intr, color_intr = set_camera('open')
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        while True:
            depth_frame, color_frame, depth_image, color_image = get_image(pipeline, align)
            if not depth_frame or not color_frame:
                continue
            visualization_RGBD(depth_image, color_image, depth_scale, clipping_distance_in_meters)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        set_camera('close', pipeline, depth_intr, color_intr)

    points = point_cloud_open3d(depth_image, color_image, depth_intr)

    if point_type == 'array':
        points = np.hstack((points.points, points.colors))
    return points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    def random_sample(point_cloud, number=10240):
        N, C = point_cloud.shape
        if N <= number:
            return point_cloud
        sample_idx = np.random.choice(N, number, replace=False)
        sample = point_cloud[sample_idx,:]
        return sample

    def clip_distance(point_cloud, dis=[0.2,2], axis=2):
        idx = np.where(point_cloud[:,axis]>=dis[0])
        point_cloud = point_cloud[idx]
        idx = np.where(point_cloud[:,axis]<=dis[1])
        point_cloud = point_cloud[idx]
        return point_cloud

    def visualization_point_cloud(points):
        N, C = points.shape
        if C == 6:
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

    def rotation(point_cloud, angle=[0,0,0]):   
        """ Randomly rotate the point clouds to augument the dataset
            rotation is per shape based on Z-Y-X euler angles
            angle = [psi theta phi]
            Input:
                Nx6 array, original batch of point clouds
            Return:
                Nx6 array, rotated batch of point clouds
        """
        angle = np.radians(angle)
        R_z = np.array([[np.cos(angle[0]), -np.sin(angle[0]), 0],
                        [np.sin(angle[0]), np.cos(angle[0]), 0],
                        [0, 0, 1]])
        R_y = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])],
                        [0, 1, 0],
                        [-np.sin(angle[1]), 0, np.cos(angle[1])]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(angle[2]), -np.sin(angle[2])],
                        [0, np.sin(angle[2]), np.cos(angle[2])]])
        rotation_matrix = np.dot(np.dot(R_z, R_y), R_x)
        point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], rotation_matrix.T)
        return point_cloud

    points =image_realsense()
    points = clip_distance(points, [0.5, 1])
    points = clip_distance(points, [-0.5, 0.5], axis=0)
    points = clip_distance(points, [-0.5, 0.5], axis=1)
    points = random_sample(points)
    visualization_point_cloud(points)
    points = rotation(points, [0,0,-90])
    visualization_point_cloud(points)