# Many Data Modality Architecture Research Framework

## Description
The **Many Data Modality Architecture Research Framework** is a research-oriented project designed to facilitate efficient transformer-based research across a variety of data modalities. This framework is equipped with pre-implemented benchmarks for data types such as point clouds, graphs, and sequences, enabling researchers to focus on developing and testing new models.

## Installation

To get started with the Many Data Modality Architecture Research Framework, you'll need the following dependencies:

```bash
# Install PyTorch (version 2.4+ required)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention (For sliding window Attention if needed)
pip install flash-attn --no-build-isolation

# Warm-up, Timm (For DropPath), other common ML libraries
pip install pytorch-warmup timm numpy matplotlib

# Install Compress-Pickle to load the preprocessed Graph dataset (converting Pytorch-Geometric format to our format)
pip install compress_pickle[lz4]

# Install dataset-specific libraries (highly dependent on use-case)
# Example: For graph data
pip install torch-geometric
pip install nuscenes-devkit
```

For other data modalities, install relevant libraries as needed.
On the other hand, you can use our environment with a lot of packages in requirement.txt

## Usage
- We work on two levels: DL Python Module Implementations (implementations of control flows, layers, optimizers, schedulers, \dots) and Experiment files (flow of tensors, hyperparameters, \dots)
- The experiment file defines high level things: how the tensor flows from loaders to prediction results/loss computation and hyperparameters (what optimizers, schedulers, ...)
- The Python Implementations are meant to be small, consise, and reusable. (e.g. implementation of transformer layer, implementation of dataset modules, ...)
- The Python Implementations also includes specific experiments. We will work to simplify this into an unified parser working across all datasets, with more diverse optimization scheme

### Prepare Data
- Save the following dataset in data folder (the data folder outside src folder :p )
- Point Cloud datasets: ModelNet40 and ShapeNetPart
    - ModelNet40: Follow here to prepare your datasets: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master
    - Long Range Arena: Follow this drive link: https://drive.google.com/drive/folders/1x6bKtXvvteMofc3uKkXK6kC0f1uO8PfR?usp=share_link
    - Long Range Graph Benchmark (For Peptides dataset): Follow this drive link: https://drive.google.com/drive/folders/10ZuMoW8r2mMcdJblAyu_6GmT0SzodzYa?usp=sharing
- We want to support all graph datasets, however, the current torch-geometric does not support the newest version of PyTorch; will add more later.
- Here is we provide a code snippet that convert torch-geometric to our own form:
```python
import torch
import torch.nn.functional as F
import torch_geometric
from compress_pickle import dump, load

def preprocess_graph(graph, max_n_nodes, max_n_edges):
    x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
    node_features = torch.zeros(1, max_n_nodes, x.shape[1])
    node_features[:, :x.shape[0], :] = x.unsqueeze(0)
    mask_node = torch.ones(1, max_n_nodes, 1)
    mask_node[:, x.shape[0]:, :] = 0.0

    connection = torch.ones(1, max_n_edges, 2, dtype=torch.long) * (max_n_edges - 1)
    connection[:, :edge_index.shape[1], :] = edge_index.transpose(0, 1).unsqueeze(0)

    edge_features = torch.zeros(1, max_n_edges, edge_attr.shape[1])
    edge_features[:, :edge_attr.shape[0], :] = edge_attr.unsqueeze(0)
    mask_edge = torch.ones(1, max_n_edges, 1)
    mask_edge[:, edge_attr.shape[0]:, :] = 0.0

    return node_features, edge_features, mask_node, mask_edge, connection

def dump_dataset(your_pytorch_geometric_dataset, pe)
    self.dataset = your_pytorch_geometric_dataset
    self.dataset_ = []
    self.d_len = len(self.dataset)
    self.indx = [i for i in range(self.d_len)]
    random.shuffle(self.indx)
    self.pre_transformed_dataset = []
    self.max_n_nodes = 0
    self.max_n_edges = 0
    if pe == 'rwpe':
        PE_gen = torch_geometric.transforms.AddRandomWalkPE(walk_length=24, attr_name='PE')
    elif pe == 'lape':
        PE_gen = torch_geometric.transforms.AddLaplacianEigenvectorPE(24, attr_name="PE")

    with torch.no_grad():
        for i in tqdm(range(len(self.dataset))):
            dpoint = self.dataset[i]
            self.max_n_nodes = max(self.max_n_nodes, dpoint.x.shape[0])
            self.max_n_edges = max(self.max_n_edges, dpoint.edge_attr.shape[0])

        for i in tqdm(range(len(self.dataset))):
            dpoint = self.dataset[i]
            node_features, edge_features, mask_node, mask_edge, connection = preprocess_graph(dpoint, self.max_n_nodes, self.max_n_edges)
            if pe != "None":
                pe_feature = PE_gen(dpoint)['PE']
                pe_feature = pe_feature.unsqueeze(0)
                pe_feature = F.pad(pe_feature, (0, 0, 0, self.max_n_nodes - pe_feature.shape[1]))
                node_features = torch.cat([node_features, pe_feature], dim=-1)
            self.pre_transformed_dataset.append([node_features.squeeze(0), edge_features.squeeze(0), mask_node.squeeze(0), mask_edge.squeeze(0), connection.squeeze(0), dpoint.y])
    
    dump(self.pre_transformed_dataset, "lrgb_voc_" + str(split) + "_" + str(pe) + "_.pkl", compression="lzma")
```

### Module
- Modules receive a Python dictionary of tensors and output a Python dictionary of tensors

- Flow of tensors is implemented in Control.py:
    - Save: Save the Tensor with 'x' key to a designated key:
        - Init: (self, name='x_saved')
        - Forward: (self, x)
        - Return: Tensor of the same size
    - Merge: A learnable residual branch to merge Tensors in key 'x' and another given key:
        - Init: (self, name='x_saved')
        - Forward: (self, x)
        - Return: Tensor of the same size

- We implement SAMSA layer in HSTransformerLayer.py and the differentiable Multi Head Sampler in DSZRC.py:
    - SAMSA Layer (HSTransformerLayer): Differentiable Sampling Transformer Layer:
        - Init:
            - d_model: int, the tokens' number of hidden dimensions
            - d_attention: int, the q/k vector's number of hidden dimensions
            - d_feedforward: int, the feedforward MLP number of hidden dimensions
            - p_dropout_model: float, the dropout rate for latents
            - p_dropout_attention_map: float, the dropattention rate
            - p_droppath: float, the droppath rate
            - nhead: int, the number of attention head
            - n_sampled_token: int, the number of sampled tokens
            - temperature: float, the temperature of Gumbel Softmax/Sigmoid
            - hard: bool, whether to use hard/soft sampling
            - output_key: str = 'x', where in the dictionary the output tensor should be saved to, default: 'x'
        - Experiment: HSTRANS,d_model,d_attention,d_feedforward,p_dropout_model,p_dropout_attention_map,p_droppath,nhead,n_sampled_token,temperature,hard

    - DSZRC Layer (DSZRC.py): Differentiable Soft/Hard Token Sampler with Importance Scoring:
        - Init:
            - `nhead`: `int`, the number of attention heads. Each head operates independently and learns its own set of importance scores and token subsets.
            - `n_sampled_token`: `int`, the number of tokens to sample from the input based on their importance scores.
            - `n_dimension_x`: `int`, the dimension of the input feature vectors.
            - `n_dimension_qk`: `int`, the dimension of the query/key vectors used in the attention mechanism.
            - `temperature`: `float`, the temperature used in the Gumbel Softmax/Sigmoid for controlling the sharpness of the sampling process.
            - `hard`: `bool`, if set to `True`, the layer will use hard sampling (discrete choices with the `HardChooseDenseGrad` function). If `False`, it uses soft sampling (probabilistic choices with the `SoftChooseDenseGrad` function).
        - Forward:
            - `x`: `Tensor`, the input tensor with shape `(batch_size, sequence_length, n_dimension_x)`.
            - `q`: `Tensor`, the query tensor with shape `(batch_size, nhead, sequence_length, n_dimension_qk // nhead)`.
            - `mask`: `Tensor`, an optional mask tensor with shape `(batch_size, sequence_length)` that can be used to ignore certain tokens during the importance scoring.
            - Returns: A tensor of the same shape as the input `q` but with a subset of tokens selected based on their importance scores.
        - Backward:
            - Custom gradients are computed using either soft or hard choices, depending on the `hard` flag. The gradients with respect to the importance scores and token representations are carefully managed to ensure the model can learn effectively from sampled tokens.
    
    - To implement new modules:
        - Type: |from ..name import module_name_dict| in your .py file in the layer folder
        - Implement your module there, remember to add |**kwargs| in your forward function
        - Type: |module_name_dict["<your_module_name>"] = <YourModuleClassName>|
        - Go to __init__.py file in layer folder, type |from .<YourPythonFileName> import *|
        - Then, your newly class is registered

- The module is defined by a list of string, for example:
    ```
    LAZYLINEAR,128
    SAVE,e
    HSTRANS,128,128,512,0.1,0.1,0.2,16,256,0.1,True,z
    HSTRANS,128,128,512,0.1,0.1,0.2,16,256,0.1,True,y
    MERGE,z
    HSTRANS,128,128,512,0.1,0.1,0.2,16,256,0.1,True
    MERGE,y
    HSTRANS,128,128,512,0.1,0.1,0.2,16,256,0.1,True
    MERGE,e
    LAZYLINEAR,16
    ```
- The above architecture is equivalent to:
    - Input -Linear> x,
    - e <\SAVE- x,
    - x -HSTRANS> z
    - x -HSTRANS> y
    - x -MERGE> x + z -HSTRANS> x -MERGE> x + y -HSTRANS> x -MERGE> x + e -LINEAR> x

- In the future, we will implement in yaml format

### Run Experiments
- To create an experiment, create a folder with the name of the experiment, a settings.txt (this will be replaced with .yaml in the future), and an empty checkpoints folder
- To run experiment, create an experiment instance and use the method |.run_experiment()|
- To load experiment, create an experiment instance and use the method |.load_experiment()|. This loads the experiment to its latest checkpoint.
- Example:
```python
from src.train_eval.experiment import *

exp = LRAExperiment(your_experiment_name, your_device) # your device ~ 'cuda:0'
exp.run_experiment()
```
- The Experiments for our preprint is provided in experiments folder; you can run it or modify at will
- To load the latest state (in case your model crash), use create a new experiment object then use the `load experiment` method 

```python
from src.train_eval.experiment import *

exp = LRAExperiment(your_experiment_name, your_device)
exp.load_experiment()
exp.run_experiment()
```
- Validation and Test curves are objects in the Experiment data structure, |exp.val_curve|. They are list of Python dictionary of metrics.

### Experiment Results
- Check at the newest preprint version here: https://drive.google.com/file/d/19Nv2rp2yF59g259Q4oNscXkwsi73k_IQ/view?usp=sharing

## Contact
- If finding hard to use / Having an issue -> Contact, there will be some delay (hours).
- Friend & Message Me at Discord: flily914; mostly being active there; https://discord.gg/G4CH3HjtST
- Email: 
    - My advisor: thy@uab.edu
    - Me: mynlenhat@gmail.com

## Please cite our work if you find it useful

```bibtex
@misc{lenhat2024samsaefficienttransformerdata,
      title={SAMSA: Efficient Transformer for Many Data Modalities}, 
      author={Minh Lenhat and Viet Anh Nguyen and Khoa Nguyen and Duong Duc Hieu and Dao Huu Hung and Truong Son Hy},
      year={2024},
      eprint={2408.05391},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.05391}, 
}
```