# experiments.py

import numpy as np
from markov import hitting_times_for_digraph_and_undirected
from graph_data import random_wireless_digraph


def is_strongly_connected(W: np.ndarray) -> bool:
    """
    Simple strong-connectivity check via reachability.
    Returns True if every node can reach every other node.
    """
    n = W.shape[0]
    A = (W > 0).astype(float)
    # A + I, then (A + I)^n – if all entries > 0, graph is strongly connected
    reach = np.linalg.matrix_power(A + np.eye(n), n)
    return bool(np.all(reach > 0))


def relative_error_matrix(H_true: np.ndarray, H_approx: np.ndarray) -> np.ndarray:
    """
    Elementwise relative error |H_approx - H_true| / H_true for H_true > 0.
    """
    H_true = np.array(H_true, dtype=float)
    H_approx = np.array(H_approx, dtype=float)
    err = np.zeros_like(H_true)
    mask = H_true > 0
    err[mask] = np.abs(H_approx[mask] - H_true[mask]) / H_true[mask]
    return err


def run_section_vi_experiment(
    N: int = 30,
    L: float = 1.0,
    R: float = 0.3,
    trials: int = 100,
    p_min: float = 0.3,
    p_max: float = 1.0,
):
    max_rel_errors = []

    rng = np.random.default_rng()
    valid = 0
    skipped_disconnected = 0
    skipped_singular = 0

    print(f"Running experiment: N={N}, L={L}, R={R}, trials={trials}")
    print("-" * 55)

    for _ in range(trials):
        """
        Generating the digraph, should be the better performing one.
        W  is the asymmetric (meaning probability i -> j != j -> i) weight matrix of the digraph, where W[i, j] is the probability of successful transmission from node i to node j.
        If Euclidean distance between nodes i and j is less than or equal to R, then W[i, j] is drawn uniformly from [p_min, p_max]; otherwise, W[i, j] = 0.
        """
        positions, W = random_wireless_digraph(N=N, L=L, R=R, p_min=p_min, p_max=p_max, rng=rng)

        # 1) Skip non–strongly-connected graphs (Markov chain not irreducible)
        if not is_strongly_connected(W):
            skipped_disconnected += 1
            continue

        # 2) Try hitting time computation; skip if stationary distribution solve fails
        try:
            H_dg, H_sym = hitting_times_for_digraph_and_undirected(W)
        except np.linalg.LinAlgError:
            skipped_singular += 1
            continue

        rel_err = relative_error_matrix(H_dg, H_sym)
        max_rel_errors.append(rel_err.max())
        valid += 1

    print(f"Valid trials         : {valid} / {trials}")
    print(f"Skipped (disconnected): {skipped_disconnected}")
    print(f"Skipped (singular A)  : {skipped_singular}")

    if valid == 0:
        print("\nNo valid topologies. Increase R (e.g. R=0.5 or 0.6) or N.")
        return

    print("\nAverage max relative error over trials:", np.mean(max_rel_errors))
    print("Std of max relative error:", np.std(max_rel_errors))


if __name__ == "__main__":
    # Start with parameters that give dense-enough graphs
    run_section_vi_experiment(N=10, L=1.0, R=0.5, trials=20)