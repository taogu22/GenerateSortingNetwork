from primitives import all_binary_inputs, apply_layer, is_sorted


def verify(network, n: int = None) -> bool:
    """
    Verify whether a given network is a valid sorting network using the 0-1 principle.
    Checks all 2^n binary inputs.

    Args:
        network: either a list of layers  [[(i,j), ...], [(i,j), ...], ...]
                 or a flat list of comparators  [(i,j), (i,j), ...]
        n: number of channels. If None, inferred from the comparators.

    Returns:
        True if the network sorts all binary inputs, False otherwise.
    """
    if not network:
        return False

    # Detect format
    if isinstance(network[0][0], (list, tuple)):
        # list of layers: [[...], [...], ...]
        layers = network
    else:
        # flat list of comparators: treat as a single layer
        layers = [network]

    # Infer n if not provided
    if n is None:
        n = max(max(i, j) for layer in layers for i, j in layer) + 1

    states = all_binary_inputs(n)

    for layer in layers:
        states = apply_layer(states, layer)

    return all(is_sorted(state) for state in states)
