import torch
import math

def scaled_dot(q, k):
    batch_size, head, length_k, d_tensor = k.size()
    k_t = k.transpose(2, 3)
    score = (q @ k_t) / math.sqrt(d_tensor)
    return score