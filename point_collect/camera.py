import pyrealsense2 as rs
import json
import numpy as np
import cv2
import open3d
import transformer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy

def read_json(path='camera.json', model='read', depth_intr=None, color_intr=None):
    ''' load the configuration filw
        model = 'read' only return the cinfiguration data
                'save' save the interior parameter of camera in the json file
    '''
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
    ''' state = 'open' open the camera as the configuration file
                'close' close the camera and save the interior parameter of camera in the json file
    '''
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
    ''' from the camera get the depth and color image
        return the result
    '''
    frames = pipeline.wait_for_frames()
    aligned_frame = align.process(frames)

    depth_frame = aligned_frame.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data())

    color_frame = aligned_frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    return depth_frame, color_frame, depth_image, color_image

def visualization_RGBD(depth_image, color_image, depth_scale, clipping_distance_in_meters=[0.6, 6]):
    ''' visualize the rgb-d image
        clipping_distance_in_meters: only display the color image in the distance in this range
    '''
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    bg_removed = np.where((depth_image_3d <= clipping_distance_in_meters[0]/depth_scale) | (depth_image_3d > clipping_distance_in_meters[1]/depth_scale), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB-D', images)
    return 0

def point_cloud(depth_frame, depth_intr):
    ''' get the point cloud into numpy array
    '''
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

def point_cloud_open3d(depth_image, color_image, depth_intr):
    ''' get the point cloud into pcd type
    '''
    camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(depth_intr.width, depth_intr.height, depth_intr.fx, depth_intr.fy, depth_intr.ppx, depth_intr.ppy)
    img_depth = open3d.geometry.Image(depth_image)
    img_color = open3d.geometry.Image(color_image)
    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=True)
    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
    point = np.asarray(pcd.points)
    point[:,1:3] = -point[:,1:3]
    pcd.points = pcd.points = open3d.utility.Vector3dVector(point)
    return pcd

def image_realsense(point_type='array', clipping_distance_in_meters=[0.6, 2], circulation=True):
    ''' get the point cloud into numpy array from realsense
        clipping_distance_in_meters: only display the color image in the distance in this range
        circulation = True: only pause 'Esc' or 'q', then, return the point cloud
                      False: get the point cloud and return
        point_type = 'array': return the point cloud as numay array
                     'pcd': return the point cloud as pcd type
                     'view': do not return any data
    '''
    profile, pipeline, align, depth_intr, color_intr = set_camera('open')
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        if not circulation:
            i = 1
        while True:
            depth_frame, color_frame, depth_image, color_image = get_image(pipeline, align)
            if not depth_frame or not color_frame:
                continue
            visualization_RGBD(depth_image, color_image, depth_scale, clipping_distance_in_meters)
            if not circulation and i==10:
                break
            if not circulation:
                i += 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        set_camera('close', pipeline, depth_intr, color_intr)

    if point_type == 'view':
        return 0

    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    points = point_cloud_open3d(depth_image, color_image, depth_intr)

    if point_type == 'array':
        points = np.hstack((np.asarray(points.points), np.asarray(points.colors)))
    return points

def visualization_point_cloud(point_cloud, rotation=[0,0,90]):
    ''' visualize the point cloud via matplotlib
    '''
    points = np.empty((0,3))
    if point_cloud.shape[0] > 1:
        for i in points:
            points = np.append(points, i, axis=0)
    else:
        points = deepcopy(point_cloud)
    points = transformer.rotation(points, rotation)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='rainbow', marker=".")
    ax.axis()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def visualization_point_cloud_open3d(point_cloud, rotation=[0,0,0]):
    ''' visualize the point cloud via open3d
    '''
    points = np.empty((0,3))
    if point_cloud.shape[0] > 1:
        for i in points:
            points = np.append(points, i, axis=0)
    else:
        points = deepcopy(point_cloud)
    points = transformer.rotation(points, rotation)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points[:,:3])
    if points.shape[1] == 6:
        pcd.colors = open3d.utility.Vector3dVector(points[:,3:])
    else:
        pcd.colors = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([pcd], point_show_normal=False, width=800, height=600)

if __name__ == '__main__':
    points =image_realsense(clipping_distance_in_meters=[0,10])
    visualization_point_cloud_open3d(points)