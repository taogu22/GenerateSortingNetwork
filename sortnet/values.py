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

def greedy(v: dict, n:int) -> list:
    """
    Greedy algorithm: repeatedly pick the highest-value pair
    """
    # sort v by descending order on the number of inversions 
    pairs = sorted(v.items(), key=lambda x: -x[1])

    used = set() # track the channels alraedy used
    layer = [] # initialize the layer

    for (i, j), val in pairs:
        if val > 0 and i not in used and j not in used: 
            layer.append((i, j))
            used.add(i)
            used.add(j)
    return layer