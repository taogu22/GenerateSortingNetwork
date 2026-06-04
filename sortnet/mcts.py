import numpy as np
import math
from primitives import total_inversions, apply_layer, all_binary_inputs
from candidates import generate_candidate_layers
from rollouts import greedy_rollout

class MCTSNode:
    __slots__ = ['states', 'layers', 'depth', 'parent', 'children', 'untried', 'best_score', 'visits']

    def __init__(self, states, layers, depth, parent=None):
        """
        In our setting, each node is a layer. The path from the root to a node form a network.
        states: outputs of the layer
        layers: the established layers so far, as a list of layers
        depth: the current depth in the tree as well as in the network
        untried: the untried children
        best_score: the smallest inversions obtained from any simulation so far starting from some child of the current node
        """
        self.states = states
        self.layers = layers
        self.depth = depth
        self.parent = parent
        self.children = []
        self.untried = None
        self.best_score = float('inf')
        self.visits = 0

def _ucb(node: MCTSNode, c: float = 1.414) -> float:
    """
    Calculate UCB1.
    """
    # If the current node is never visited, then the score is +∞.
    if node.visits == 0:
        return float('inf')
    
    # exploitation: best result so far visited in rollouts from here
    # best_score is the minimal inversion count so far
    # for UCB maximisation, it needs to be negated
    exploitation = -node.best_score
    
    # exploration: favour less-visited nodes
    assert node.parent is not None
    exploration = c * math.sqrt(math.log(node.parent.visits) / node.visits)

    return exploitation + exploration

def _select(root: MCTSNode, c: float) -> MCTSNode:
    """
    SLELECTION stage: go down following highest UCB, until a node whose children are not fully expanded --- that is, a node that has children and has some children untried.
    """
    node = root
    while node.children and node.untried == []:
        node = max(node.children, key = lambda child: _ucb(child, c))
    
    return node

def _expansion(node: MCTSNode, n: int, K: int, max_depth: int) -> MCTSNode:
    """
    Expand a node by randomly trying its untried children.
    
    Args:
        node: current node, to be expanded
        n: number of channels
        K: best-K as candidates (children nodes)
        max_depth: maximum allowed depth of network

    Returns:
        if `node` is terminal, just return `node`
        otherwise, create a new child node
    """
    is_terminal = (node.depth >= max_depth or total_inversions(node.states) == 0)

    expand_node = node

    if is_terminal:
        return expand_node
    
    # when `node` has not been expanded at all, generate its `untried` first
    if node.untried is None:
        node.untried = generate_candidate_layers(node.states, K)

    if node.untried:
        expand_layer = node.untried.pop(0)
        expand_states = apply_layer(node.states, expand_layer)
        expand_node = MCTSNode(states=expand_states, layers = node.layers + [list(expand_layer)], depth=node.depth + 1, parent=node)

        node.children.append(expand_node)

    return expand_node

def _simulation(node: MCTSNode, n: int, max_layers: int):
    """
    After fixing the node to be expanded, simulate the remaining game randomly until the end

    Args:
        node: the node where the simulation starts
        n: nuber of channels
        max_layers: the maximal number of layers allowed

    Returns:

    """
    remaining = max_layers - node.depth
    result_states = greedy_rollout(node.states, n, remaining, rand=False, rand_para=0.5, K=8)
    score = total_inversions(result_states)

    return score


def _backpropogation(node: MCTSNode, score: int) -> None:
    """
    Traverse back from the current expanded node to the root, and update all the `vists` and `best_score` along the way
    Args:
        node: the currrent expanded node
        score: the score from the simulation
    Returns:
    """
    current_node = node
    while current_node is not None:
        current_node.visits += 1
        # prefer lower scores
        if current_node.best_score > score:
            current_node.best_score = score
        current_node = current_node.parent

    return 

def mcts_build(n: int,
               max_layers: int,
               budget: int,
               K: int,
               c: float,
               rollout_temp: float) -> tuple:
    
    inputs = all_binary_inputs(n)
    root = MCTSNode(states=inputs, layers=[],depth=0, parent=None)

    best_score = total_inversions(inputs)
    best_layers = []
    history = [total_inversions(inputs)]

    for _ in range(budget):

        select_node = _select(root, c)

        expand_node = _expansion(select_node, n, K, max_layers)

        # ---- Simulation (guided rollout) ----
        remaining = max_layers - expand_node.depth
        final_states, simulated_layers = greedy_rollout(expand_node.states.copy(), n, remaining, rand=False, rand_para=rollout_temp, K=K)
        score = total_inversions(final_states)

        if score < best_score:
            best_score = score
            best_layers = expand_node.layers + simulated_layers
            # best_layers = expand_node.layers  # partial; rollout adds the rest

        history.append(best_score)

        # if verbose and (iteration % 50 == 0 or score == 0):
        #     print(f"  iter {iteration:4d}  best_score={best_score}")

        if best_score == 0:
        #     if verbose:
        #         print(f"  ✓ Found sorting network at iteration {iteration}")
            break

        # ---- Backpropagation ----
        current_node = expand_node
        while current_node is not None:
            current_node.visits += 1
            if score < current_node.best_score:
                current_node.best_score = score
            current_node = current_node.parent

    return best_layers, best_score, history