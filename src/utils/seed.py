# File to handle seed reproducibility

import random
import numpy as np
import torch


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed):
    """
    Set the random seed for reproducibility across random, numpy, and torch.
    Args:
    - seed: The random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)