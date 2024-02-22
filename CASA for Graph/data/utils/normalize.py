import numpy as np

def pc_normalize(pc):
    """
    Normalize the point cloud
    Input:
        pc: pointcloud data, [N, D]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc