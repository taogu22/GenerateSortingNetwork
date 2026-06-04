# import sys, os
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from primitives import all_binary_inputs, apply_layer, total_inversions
from candidates import greedy_build
from mcts import mcts_build


def inversion_profile(layers: list, n: int) -> list:
    """Track total inversion count after each layer."""
    states = all_binary_inputs(n)
    profile = [total_inversions(states)]
    for layer in layers:
        states = apply_layer(states, layer)
        profile.append(total_inversions(states))
    return profile


def run_experiment(n: int = 6, max_layers: int = 5,
                   mcts_budget: int = 300, K: int = 8,
                   c: float = 500.0, rollout_temp: float = 1.0):
    """
    Compare MCTS (single run) vs greedy baseline.
    """
    print("=" * 60)
    print(f"Experiment: N={n}, max_layers={max_layers}")
    print("=" * 60)

    # --- Pure greedy (deterministic, single run) ---
    print("\n[Greedy baseline]")
    t0 = time.time()
    g_layers, g_history = greedy_build(n, max_layers)
    g_time = time.time() - t0
    g_score = g_history[-1]
    print(f"  Inversion count per layer: {g_history}")
    print(f"  Final score: {g_score}  ({g_time*1000:.1f} ms)")

    # --- MCTS ---
    print(f"\n[MCTS  budget={mcts_budget}, K={K}]")
    t0 = time.time()
    m_layers, m_score, m_history = mcts_build(n, max_layers,
                                               budget=mcts_budget,
                                               K=K, c=c,
                                               rollout_temp=rollout_temp)
    m_time = time.time() - t0
    print(f"  Final score: {m_score}  ({m_time:.2f} s)")
    print(f"  Inversion count per layer: {inversion_profile(m_layers, n)}")
    print(f"  Best layers: {m_layers}")
    # --- Summary ---
    print("\n[Summary]")
    print(f"  Greedy : {g_score}")
    print(f"  MCTS   : {m_score}  ({'better' if m_score < g_score else 'same' if m_score == g_score else 'worse'})")

    return {
        'greedy_history': g_history,
        'greedy_score': g_score,
        'mcts_history': m_history,
        'mcts_score': m_score,
    }


if __name__ == "__main__":
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)

    run_experiment(n=6, max_layers=5, mcts_budget=300, K=8)
