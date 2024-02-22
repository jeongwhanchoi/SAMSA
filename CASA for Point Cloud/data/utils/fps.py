import numpy as np

def farthest_point_sample(pc, n_sample):
    """
    Input:
        pc: pointcloud data, [N, D]
        n_sample: number of samples
    Return:
        centroids: sampled pointcloud index, [n_sample, D]
    """
    N, D = pc.shape
    xyz = pc[:, :3]
    centroids = np.zeros((n_sample,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc = pc[centroids.astype(np.int32)]
    return pc