import torch
try:
    import torch_geometric
except ModuleNotFoundError:
    TORCH = torch.__version__.split("+")[0]
    CUDA = "cu" + torch.version.cuda.replace(".","")
!pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-geometric
import torch_geometric