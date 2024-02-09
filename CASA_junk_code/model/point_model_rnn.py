import random
from layer import RESC
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import RESC, RSTL, RandomColor, SoftplusPWP, SAM, GlobalNode, SSCL
# from layer import RESC, RandomColor, LocalAttention
from head import ClassificationHead, SegmentationHead

def create_layer(layer_type, hyperparams):
    if layer_type == "PCLL":
        return nn.LazyLinear(*hyperparams)
    if layer_type == "RC":
        return RandomColor()
    elif layer_type == "RESC":
        return RESC(*hyperparams)
    elif layer_type == "GLBN":
        return GlobalNode(*hyperparams)
    elif layer_type == "RSTL":
        return RSTL(*hyperparams)
    elif layer_type == "SSCL":
        return SSCL(*hyperparams)
    elif layer_type == "SPWP":
        return SoftplusPWP(*hyperparams)
    elif layer_type == "CLFH":
        return ClassificationHead(*hyperparams)
    elif layer_type == "SEGH":
        return SegmentationHead(*hyperparams)
    elif layer_type == "NORM":
        return nn.LayerNorm(*hyperparams)
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

def to_number(x):
    isfloat = False
    for i in range(len(x)):
        if x[i] == '.':
            isfloat = True
    if isfloat == True:
        x = float(x)
    else:
        x = int(x)
    return x

def to_var(x):
    if x[0].isdigit():
        return to_number(x)
    if x in ["True", "False"]:
        return x == "True"
    return x # String

def pairwise_distances(x, y):
    '''
    Input: x is a BxNxd matrix
           y is an optional BxMxd matirx
    Output: dist is a BxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(2).unsqueeze(2)
    y_t = torch.transpose(y, 1, 2)
    y_norm = (y**2).sum(2).unsqueeze(1)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist 

def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat

def canonicalize_point(S1, S2, S3): # S1, S2 to compute rotation/translation matrix, 
                                    # S1 is the sampled point cloud, 
                                    # S2 is the synthesized point cloud from neural network, 
                                    # S3 is the input point cloud
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        S3 = S3.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    # S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t
    S3_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S3) + t

    if transposed:
        # S1_hat = S1_hat.permute(0,2,1)
        S3_hat = S3_hat.permute(0,2,1)

    return S3_hat

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1], device=tensor.device).argsort(axis)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

class PointModelRNN(nn.Module):
    def __init__(self, layer_list, is_normal=False, is_rotation=True):
        super().__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.lin = self.layer_list[0]
        self.transf = self.layer_list[1:-1]
        self.classf = self.layer_list[-1]
        self.is_relative = False
        for i in range(len(self.layer_list)):
            if isinstance(self.layer_list[i], RSTL):
                self.is_relative = True
            elif isinstance(self.layer_list[i], SSCL):
                self.is_relative = True
        if self.is_relative is False:
            self.forward = self.forward_notrelative
        else:
            self.forward = self.forward_relative
        self.layer_len = len(self.layer_list)
        self.is_normal = is_normal
        self.is_rotation = is_rotation

    def forward(self, x, mask=None): # relative
        if self.is_normal == False:
            x_ref = x[:, :, :3] * 1
            x_ref = torch.cat([x_ref, torch.ones(x_ref.shape[0], x_ref.shape[1], 1, device=x_ref.device)], dim=2)
            if self.is_rotation is True:
                x[:, :, :3] = 1.0
        else:
            x_ref = x[:, :, :6] * 1
            x_ref = torch.cat([x_ref, torch.ones(x_ref.shape[0], x_ref.shape[1], 1, device=x_ref.device)], dim=2)
            if self.is_rotation is True:
                x[:, :, :6] = 1.0

        #Random features
        x = torch.cat([x, torch.randn_like(x)], dim=-1)
        
        x = self.lin(x)

        if self.training is True:
            n = random.randint(2, 4)
        else:
            n = 4

        for i in range(n):
            for j in range(len(self.transf)):
                if isinstance(self.layer_list[i], nn.LazyLinear) or isinstance(self.layer_list[i], nn.Linear):
                    x = self.layer_list[i](x)
                elif isinstance(self.layer_list[i], RSTL):
                    x = self.layer_list[i](x, x_ref, mask)
                elif isinstance(self.layer_list[i], SSCL):
                    x = self.layer_list[i](x, x_ref, mask)
                elif isinstance(self.layer_list[i], GlobalNode):
                    x, x_ref, mask = self.layer_list[i](x, x_ref, mask)
                else:
                    x = self.layer_list[i](x, mask)

        x = self.classf(x)
        return x

    def __len__(self):
        return len(self.layer_list)

    def __getitem__(self, idx):
        return self.layer_list[idx]

def parse_point_architecture(arch_string, is_normal=False, is_rotation=True):
    layer = arch_string.split('@')
    sequential_layer = []
    
    for layer_str in layer:
        layer_type, *hyperparams = layer_str.split(',')
        
        # Convert hyperparameters to the appropriate types
        hyperparams = [to_var(param) for param in hyperparams]
        
        # Convert to the desired format
        layer_str = f"{layer_type}({', '.join(map(str, hyperparams))})"
        
        # Wrap each hyperparameter individually with parentheses
        for i in range(len(hyperparams)):
            layer_str = layer_str.replace(str(hyperparams[i]), f"({hyperparams[i]})", 1)
        layer = create_layer(layer_type, hyperparams)
        sequential_layer.append(layer)
    
    # print("Model architecture consists of: \n", sequential_layer)
    return PointModel(sequential_layer, is_normal, is_rotation)