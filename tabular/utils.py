import numpy as np
import random


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
