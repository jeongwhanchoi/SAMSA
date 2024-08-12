from ..name import module_name_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
from tqdm import tqdm

def indices_search_helper_func(index, x_flat, device):
    distance, indices = index.search(x_flat, 1)
    indices = torch.tensor(indices, device=device)
    distance = torch.tensor(distance, device=device)
    return distance, indices

indices_search_helper_func = torch.compiler.disable(indices_search_helper_func)
        
class FAISSAnchorKernelLayer(nn.Module):
    def __init__(self, n_anchors, n_dimensions_input, n_dimensions_output, key_initialization=None):
        super(FAISSAnchorKernelLayer, self).__init__()
        self.n_anchors = n_anchors
        self.n_dimensions_input = n_dimensions_input
        self.n_dimensions_output = n_dimensions_output

        self.Key = nn.Parameter(key_initialization, requires_grad=False)

        key_initialization = key_initialization.unsqueeze(0)
        att_m = torch.cdist(key_initialization, key_initialization)
        att_m = att_m.argsort(dim=2)
        att_m = att_m * -1
        att_m = 2 ** (att_m)
        att_m = att_m / torch.sum(att_m, dim=2)
        init_arr = att_m @ torch.sign(torch.randn(1, self.n_anchors, self.n_dimensions_output)).float()
        init_arr = init_arr.squeeze(0)

        self.init_mat = nn.Parameter(init_arr, requires_grad=False)
        self.Value = nn.Parameter(torch.randn_like(init_arr) * 0.01, requires_grad=True)
        self.meanK = nn.Parameter(torch.mean(key_initialization), requires_grad=False)
        
        self.coarse_quantizer = faiss.IndexFlatL2(n_dimensions_input)
        self.index = faiss.IndexIVFFlat(self.coarse_quantizer, n_dimensions_input, 24)
        self.index = faiss.index_cpu_to_all_gpus(self.index)
        self.index.train(self.Key.detach().cpu().numpy())
        self.index.add(self.Key.detach().cpu().numpy())  # Add Key tensor to FAISS index
        self.index.nprobe = 4  # set how many of nearest cells to search

        # Feature transformation for output
        self.feature_trans = nn.Sequential(
            nn.Linear(3, 4 * n_dimensions_output),
            nn.GELU(),
            nn.Linear(4 * n_dimensions_output, n_dimensions_output)
        )

    def forward(self, x):
        batch_size, n_tokens, _ = x.size()
        
        with torch.no_grad():
            # Reshape x for batch processing
            x_flat = x.view(-1, self.n_dimensions_input).cpu().detach().numpy()

            distance, indices = indices_search_helper_func(
                self.index, 
                x_flat,
                x.device,)

            indices = indices.view(batch_size, n_tokens, 1)
            del distance
            del x_flat


            # Gathered keys and value
            reshaped_indices = indices.reshape(batch_size * n_tokens, 1)
            ret_local = x - torch.take_along_dim(self.Key, reshaped_indices, dim=0).reshape(batch_size, n_tokens, 3)
        Value = self.init_mat + self.Value
        ret_global = torch.take_along_dim(Value, reshaped_indices, dim=0).reshape(batch_size, n_tokens, self.n_dimensions_output)
        del indices
        del reshaped_indices

        # Get global graphical features and local compact features
        ret = self.feature_trans(ret_local) + ret_global
        return ret

class NSFeaturizer(nn.Module):
    def __init__(self, n_dimensions_output, output_key: str = 'x'):
        super().__init__()
        self.FPS_NSCyl = torch.load('./layer_data/FPS_NuSceneCyl.pth', map_location='cpu')
        self.n_anchors = self.FPS_NSCyl.shape[0]
        self.FPS_NSCyl_AnchorLayer = FAISSAnchorKernelLayer(
            n_anchors = self.n_anchors,
            n_dimensions_input = 3,
            n_dimensions_output = n_dimensions_output - 1, 
            key_initialization = self.FPS_NSCyl
        )
        self.ref_scaler = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.output_key = output_key

    def forward(self, x, mask=None, **args):
        x_coords, x_ref = torch.split(x, dim=2, split_size_or_sections=[3, 1])
        ret_x = self.FPS_NSCyl_AnchorLayer(x_coords)
        return {self.output_key: torch.cat([ret_x, x_ref * self.ref_scaler], dim=2)}

module_name_dict["NSFEATURE"] = NSFeaturizer