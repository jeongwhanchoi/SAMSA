import random
import numpy as np
import torch

def get_seed():
    # Recover seed used by random
    random_state = random.getstate()
    random_seed = random_state[1][0]

    # Recover seed used by numpy
    numpy_seed = np.random.get_state()[1][0]

    # Recover seed used by pytorch
    torch_seed = torch.initial_seed()

    return {
        "random": random_seed, 
        "numpy": numpy_seed, 
        "torch": torch_seed
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)