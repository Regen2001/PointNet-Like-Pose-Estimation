import pyrealsense2 as rs
import json
import numpy as np
import cv2
import open3d
from copy import deepcopy

class RealSense:
    def __init__(self, path='camera.json'):
        self.__read_json(path=path)
        self.__pipeline = rs.pipeline()
        self.__config = rs.config()

        self.__config.enable_stream(rs.stream.depth, self.__data["width"], self.__data['height'], rs.format.z16, self.__data["fps"])
        self.__config.enable_stream(rs.stream.color, self.__data["width"], self.__data['height'], rs.format.bgr8, self.__data["fps"])
        self.__profile = self.__pipeline.start(self.__config)
    
        align_to = rs.stream.color
        self.__align = rs.align(align_to)

        self.__depth_intr = self.__profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.__color_intr = self.__profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.__depth_scale = self.__profile.get_device().first_depth_sensor().get_depth_scale()

        print('Opened the realsense camera')

    def __read_json(self, path):
            try:
                fp = open(path, 'r', encoding='utf-8')
                self.__data = json.load(fp)
                fp.close
            except:
                self.__data = {"width": 640, "height": 480, "fps": 30}
    
    def __load_json(self, path):
        self.__data["depth_ppx"] = self.__depth_intr.ppx
        self.__data["depth_ppy"] = self.__depth_intr.ppy
        self.__data["depth_fx"] = self.__depth_intr.fx
        self.__data["depth_fy"] = self.__depth_intr.fy
        self.__data["color_ppx"] = self.__color_intr.ppx
        self.__data["color_ppy"] = self.__color_intr.ppy
        self.__data["color_fx"] = self.__color_intr.fx
        self.__data["color_fy"] = self.__color_intr.fy

        fp = open(path, 'w', encoding='utf-8')
        fp.write(json.dumps(self.__data, indent=4, ensure_ascii=False))
        fp.close()

    def colse(self, path='camera.json'):
        self.__load_json(path=path)
        self.__pipeline.stop()
        print('Closed the realsense camera')
        return 0

    def get_image(self, clipping_distance_in_meters=[0,2], circulation=True):
        if not circulation:
            i = 1
        while True:
            frames = self.__pipeline.wait_for_frames()
            aligned_frame = self.__align.process(frames)

            self.__depth_frame = aligned_frame.get_depth_frame()
            self.__depth_image = np.asanyarray(self.__depth_frame.get_data())

            self.__color_frame = aligned_frame.get_color_frame()
            self.__color_image = np.asanyarray(self.__color_frame.get_data())
            if not self.__depth_frame or not self.__color_frame:
                continue
            visualization_RGBD(self.__depth_image, self.__color_image, self.__depth_scale, clipping_distance_in_meters=clipping_distance_in_meters)
            if not circulation and i==10:
                break
            if not circulation:
                i += 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        self.__color_image = cv2.cvtColor(self.__color_image, cv2.COLOR_BGR2RGB)
        return self.__color_image, self.__depth_image
    
    def calculate_point_cloud(self, type='array'):
        camera_intrinsic = open3d.camera.PinholeCameraIntrinsic(self.__depth_intr.width, self.__depth_intr.height, self.__depth_intr.fx, self.__depth_intr.fy, self.__depth_intr.ppx, self.__depth_intr.ppy)
        img_depth = open3d.geometry.Image(self.__depth_image)
        img_color = open3d.geometry.Image(self.__color_image)
        rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth, convert_rgb_to_intensity=False)
        self.__pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsic)
        self.__points = np.append(np.asarray(self.__pcd.points), np.asarray(self.__pcd.colors), axis=1)
        self.__points[:,1:3] = -self.__points[:,1:3]
        if type == 'array':
            return self.__points
        if type == 'pcd':
            self.__pcd.points = self.__pcd.points = open3d.utility.Vector3dVector(self.__points[:,:3])
            return self.__pcd

    def get_rgbd_image(self):
        return self.__color_image, self.__depth_image
    
    def get_point_cloud(self, type='array'):
        if type == 'array':
            return self.__points
        else:
            return self.__pcd
        
    def save_points_ply(self, path):
        ply = open3d.geometry.TriangleMesh()
        ply.vertices = open3d.utility.Vector3dVector(self.__points[:,:3])
        ply.vertex_colors = open3d.utility.Vector3dVector(self.__points[:,3:])
        open3d.io.write_triangle_mesh(path, ply)
        print('The point cloud is saved in ', path)

def visualization_RGBD(depth_image, color_image, depth_scale, clipping_distance_in_meters=[0,2]):
    grey_color = 153
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    bg_removed = np.where((depth_image_3d <= clipping_distance_in_meters[0]/depth_scale) | (depth_image_3d > clipping_distance_in_meters[1]/depth_scale), grey_color, color_image)

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
    images = np.hstack((bg_removed, depth_colormap))

    cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB-D', images)
    return 0

def visualization_point_cloud(point_cloud):
    points = np.empty((0,3))
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

if __name__ == '__main__':
    camera = RealSense()
    camera.get_image([0,10])
    points = camera.calculate_point_cloud()
    visualization_point_cloud(points)
    # camera.save_points_ply('test2.ply')
    camera.colse()