from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarSegPointCloud
import numpy as np
import torch
from pprint import pprint

class NuSceneData():
    def __init__(self, data_path, version='v1.0-trainval'):
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=False)
        self.sample_token = {}
        self.scene_token = {}
        self.lidarseg_token = {}
        self.scene_token_list = {}
        self.data_path = data_path
        self.version = version

        for i in self.nusc.sample:
            self.sample_token[i['token']] = i
        for i in self.nusc.scene:
            self.scene_token[i['token']] = i
        for i in self.nusc.lidarseg:
            self.lidarseg_token[i['token']] = i

        self.sample_token_keys = list(self.sample_token.keys())
        self.scene_token_keys = list(self.scene_token.keys())

        self.sample_token_idx = {}
        self.scene_token_idx = {}

        for i in range(len(self.sample_token_keys)):
            self.sample_token_idx[self.sample_token_keys[i]] = i
        for i in range(len(self.scene_token_keys)):
            self.scene_token_idx[self.scene_token_keys[i]] = i

        for i in self.nusc.scene:
            last_sample_token = i['last_sample_token']
            sample_tokens_in_i_scene = [last_sample_token]
            while self.sample_token[sample_tokens_in_i_scene[-1]]["prev"] != "":
                sample_tokens_in_i_scene.append(self.sample_token[sample_tokens_in_i_scene[-1]]["prev"])
            sample_tokens_in_i_scene.reverse()
            self.scene_token_list[i['token']] = sample_tokens_in_i_scene


    def __len__(self):
        return len(self.sample_token_keys)

    def get_lidar_point_cloud(self, sample_token=None, sample_token_idx=None):
        """
              LIDAR point cloud: (4, n_point);
              First 3 dimension is the cartesian coordinates of points
              The last dimension is the intensity of point
              Intensity of point: https://desktop.arcgis.com/en/arcmap/latest/manage-data/las-dataset/what-is-intensity-data-.htm
        """
        if sample_token == None:
            if sample_token_idx is not None:
                sample_token = self.sample_token_keys[sample_token_idx]
        sample = self.sample_token[sample_token]
        sample_data = sample['data']
        lidar_token = sample_data['LIDAR_TOP']
        file_name = self.nusc.get('sample_data', lidar_token)['filename']
        lidar_label_filename = self.lidarseg_token[lidar_token]['filename']
        point_cloud = LidarSegPointCloud(self.data_path + '/' + file_name, self.data_path + '/' + lidar_label_filename)
        return point_cloud.points, point_cloud.labels