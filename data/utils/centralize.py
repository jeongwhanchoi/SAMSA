import numpy as np

def centralize(pc):
    """
    Centralize the point cloud
    Input:
        pc: pointcloud data, [N, D]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc