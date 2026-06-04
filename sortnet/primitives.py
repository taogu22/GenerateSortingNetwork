import numpy as np

def all_binary_inputs(n: int) -> np.ndarray:
    dim = (2,) * n
    grid = np.indices(dim)
    # print(f"Grid looks like {grid}.")
    # print(f"Its shape is {grid.shape}")
    seq = np.stack(grid, axis=-1).reshape(-1, n).astype(np.int8) # type: ignore

    return seq

def apply_comparator(states: np.ndarray, i:int, j: int) -> np.ndarray:
    """
    apply a comparator to a list of states
    """
    s = states.copy()
    mask = s[:, i] > s[:, j]
    s[mask, i], s[mask, j] = s[mask, j].copy(), s[mask, i].copy()
    return s

def apply_layer(states: np.ndarray, layer: list) -> np.ndarray:
    """
    apply a layer to a list of sequences
    """
    s = states.copy()
    for (i, j) in layer:
        s = apply_comparator(s, i, j)
    return s

def is_sorted(seq: np.ndarray) -> bool:
    return all(seq[i] <= seq[i+1] for i in range(len(seq) - 1))

def sorted_fraction(states: np.ndarray) -> float:
    # fraction of sequences that are sorted
    sorted_num = sum(is_sorted(state) for state in states)
    return sorted_num / states.shape[0]

def total_inversions(states: np.ndarray) -> int:
    """
    total number of inversions over all sequences
    """
    n = states.shape[1] # channel size
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            total += int(np.sum(states[:, i] > states[:, j]))
    return total