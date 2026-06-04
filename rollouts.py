import numpy as np
from primitives import total_inversions, apply_layer
from values import v_fwd
from candidates import greedy, generate_candidate_layers

def greedy_rollout(states: np.ndarray, n: int, remaining: int, rand: bool = False, rand_para: float = 0.0, K: int = 8) -> tuple:
    """
    Args:
        states: current inputs
        n: numbers of channels
        remaining: number of layers to be added
        rand: control whether random or not
        rand_para: the random parameter in the rollout, which is only effective when `rand` is True.
            - temp < 0.0: no randomness at all, and always choose the top.
            - temp > 0.0: sample from the top-K candidates proportional to their scores

    Returns: 
        the states after the whole rollout
        the layers chosen in the whole rollout
    """
    s = states.copy()
    layers = []

    for depth in range(remaining):
        if total_inversions(s) == 0:
            break

        if rand is False:
            v = v_fwd(s)
            layer = greedy(v)
            layers.append(layer)
            s = apply_layer(s, layer)
        else: # randomness applies
            candidates = generate_candidate_layers(s, K)

            scores = []
            for candidate in candidates:
                result = apply_layer(s, candidate)
                scores.append(total_inversions(result))

            scores_arr = np.array(scores, dtype=float)
            logits = -scores_arr / rand_para
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

            next_id = np.random.choice(len(candidates), p=probs)
            l = list(candidates[next_id])
            layers.append(l)
            s = apply_layer(s, l)

    return s, layers
