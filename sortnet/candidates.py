import numpy as np
from values import v_fwd
from primitives import apply_layer, total_inversions

def greedy(v: dict) -> list:
    """
    Greedy algorithm: repeatedly pick the highest-value pair, to form a single layer
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

def generate_two_exchange(layer: list) -> set:
    """
    Generate all the 2-exchange results of a given layer.
    """
    result = set()

    n = len(layer)
    for ind1 in range(n):
        for ind2 in range(ind1+1, n):
            i1, j1 = layer[ind1]
            i2, j2 = layer[ind2]
            channels = sorted([i1, j1, i2, j2])
            x1, x2, x3, x4 = channels
            
            for p1, p2 in [((x1, x2), (x3, x4)), ((x1, x3), (x2, x4)), ((x1, x4), (x2, x3))]:
                new_layer = layer.copy()
                new_layer[ind1], new_layer[ind2] = p1, p2
                result.add(tuple(sorted(new_layer)))
        
    return result

def generate_candidate_layers(states: np.ndarray, K: int) -> list:
    """
    Generate K candidates as next layers.
    First, pick a `root_candidate` using greedy algorithm.
    Next, generate all small variations by applying 2-exchanges to `root_candidate`.
    Finally, pick the K best such variations.
    """
    v = v_fwd(states)
    root_candidate = greedy(v)
    
    candidates = generate_two_exchange(root_candidate) # The candidates are slight modifications of `root_candidate`. It is a set of TUPLES, so need to be turned into lists later.

    scores = [] # track the scores of the candidates, as a list of tuples (candidate, score)
    for candidate in candidates:
        candidate = list(candidate)
        # score cadidates by calculating the total inversion numbers
        result_states = apply_layer(states.copy(), candidate)
        score = total_inversions(result_states)
        scores.append((candidate, score))

    scores.sort(key = lambda x: x[1]) # sort by total number of inversions: less is better

    return [candidate for candidate, _ in scores[:K]]

def greedy_build(n: int, max_depth: int) -> tuple:
    """
    Build a network layer by layer using pure greedy algorithm, based on V_fwd.

    Args:
        n: number of channels
        max_depth: maximum number of layers

    Returns:
        (layers, history) where layers is the list of chosen layers and
        history is the total inversion count after each layer
    """
    from primitives import all_binary_inputs
    states = all_binary_inputs(n)
    layers = []
    history = [total_inversions(states)]

    for _ in range(max_depth):
        if total_inversions(states) == 0:
            break
        v = v_fwd(states)
        layer = greedy(v)
        if not layer:
            break
        states = apply_layer(states, layer)
        layers.append(layer)
        history.append(total_inversions(states))

    return layers, history


