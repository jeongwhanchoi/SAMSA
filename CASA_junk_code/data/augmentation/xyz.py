import numpy as np
import random

def RandomRotation_z(pointcloud):
    theta = random.random() * 2. * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta),      0],
                            [np.sin(theta),  np.cos(theta),      0],
                            [0,                               0,      1]])
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud

def RandomNoise(pointcloud):
    noise = np.random.normal(0, 0.02, (pointcloud.shape))
    noisy_pointcloud = pointcloud + noise
    return noisy_pointcloud


def ShufflePoints(pointcloud):
    np.random.shuffle(pointcloud)
    return pointcloud

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, shifted batch of point clouds
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    batch_data += shifts
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    batch_data = np.expand_dims(batch_data, axis=0)
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    jittered_data = np.squeeze(jittered_data, axis=0)
    return jittered_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    batch_data = np.expand_dims(batch_data, axis=0)
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=3) * 2 * np.pi
        Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))

        rotated_data[k, ...] = np.dot(batch_data[k, ...], R)

    rotated_data = np.squeeze(rotated_data, axis=0)
    return rotated_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high)
    batch_data *= scales
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''
    batch_pc = np.expand_dims(batch_pc, axis=0)
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((batch_pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        batch_pc[drop_idx,:] = batch_pc[0,:] # set to the first point
    batch_pc = np.squeeze(batch_pc, axis=0)
    return batch_pc