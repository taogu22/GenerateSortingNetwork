import numpy as np

def v_fwd(states: np.ndarray) -> dict:
    """
    store (key, val)
    key: comparators
    val: the number of fixed inversions by applying the comparator to the states
    """
    v = {}

    n = states.shape[1] # channel size
    for i in range(n):
        for j in range(i+1, n):
            count = np.sum(states[:, i] > states[:, j])
            v[(i, j)] = (j - i) * count
    
    return v
