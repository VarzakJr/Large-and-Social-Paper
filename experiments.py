import numpy as np
import matplotlib.pyplot as plt

def is_strongly_connected(W):
    """
    Simple strong-connectivity check via reachability.
    Returns True if every node can reach every other node.
    """
    n = W.shape[0]
    A = (W > 0).astype(float)
    # A + I, then (A + I)^n – if all entries > 0, graph is strongly connected
    reach = np.linalg.matrix_power(A + np.eye(n), n)
    return bool(np.all(reach > 0))


def relative_error_matrix(H_true,H_approx):
    """
    Elementwise relative error |H_approx - H_true| / H_true for H_true > 0.
    """
    H_true = np.array(H_true, dtype=float)
    H_approx = np.array(H_approx, dtype=float)
    err = np.zeros_like(H_true)
    mask = H_true > 0
    err[mask] = np.abs(H_approx[mask] - H_true[mask]) / H_true[mask]
    return err

def monte_carlo_hitting_time(P, src, dst, num_trials=1000, max_steps=10000, rng=None):
    """
    Empirically estimate the hitting time from src to dst for the Markov chain P
    by simulating random walks.

    P: (n x n) transition matrix
    src, dst: integer node indices
    num_trials: number of simulated walks
    max_steps: safety cap on steps per walk (to avoid infinite loops)
    rng: np.random.Generator (optional)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = P.shape[0]
    P = np.asarray(P, dtype=float)

    # Precompute cumulative probs for efficient sampling
    cum_P = np.cumsum(P, axis=1)

    def one_walk():
        state = src
        for step in range(1, max_steps + 1):
            if state == dst:
                return step - 1  # already at dst at this step count
            u = rng.random()
            # sample next state according to row P[state]
            state = np.searchsorted(cum_P[state], u)
        # If max_steps reached without hitting dst, treat as censored
        return max_steps

    return np.mean([one_walk() for _ in range(num_trials)])

def sample_random_pairs(N, k, rng):
    all_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    k = min(k, len(all_pairs))
    idx = rng.choice(len(all_pairs), size=k, replace=False)
    return [all_pairs[t] for t in idx]

def run_section_vi_experiment(
    N: int = 30,
    L: float = 1.0,
    R: float = 0.3,
    trials: int = 100,
    p_min: float = 0.3,
    p_max: float = 1.0,
    mc_trials: int = 1000,
    mc_max_steps: int = 10000,
    sample_pairs_k: int = 10,
    pair_seed: int = 12345,
):
    rng = np.random.default_rng()
    pair_rng = np.random.default_rng(pair_seed)

    valid = 0
    skipped_disconnected = 0
    skipped_singular = 0

    graph_pair_avg_errors = []
    graph_pair_max_errors = []

    sampled_pairs = sample_random_pairs(N, sample_pairs_k, pair_rng)
    sampled_pair_stats = {pair: {"H_dg": [], "H_sym": [], "H_mc": [], "mc_rel": [], "sym_rel": []} for pair in sampled_pairs}

    print(f"Running experiment: N={N}, L={L}, R={R}, trials={trials}")
    print("-" * 55)

    for _ in range(trials):
        positions, W = random_wireless_digraph(N=N, L=L, R=R, p_min=p_min, p_max=p_max, rng=rng)

        #Ignoring not strongly connected graphs, as the paper does
        if not is_strongly_connected(W):
            skipped_disconnected += 1
            continue

        try:
            #Calling method for calculating tranmissions
            H_dg, H_sym = hitting_times_for_digraph_and_undirected(W)
        except np.linalg.LinAlgError:
            skipped_singular += 1
            continue

        valid += 1

        #Calculating error between a pair of nodes
        rel_err = relative_error_matrix(H_dg, H_sym)
        mask = ~np.eye(N, dtype=bool)
        pairwise = rel_err[mask]
        graph_pair_avg_errors.append(float(np.mean(pairwise)))
        graph_pair_max_errors.append(float(np.max(pairwise)))

        #Creating digraph transition matrix
        P_dg = make_transition_P_from_W(W)

        #Calculating theoretical transmissions
        for (src, dst) in sampled_pairs:
            H_theory = H_dg[src, dst]
            H_sym_sd = H_sym[src, dst]
            if not (np.isfinite(H_theory) and H_theory > 0):
                continue
            #Monte Carlo run for evaluating accuracy compared to theory
            H_mc = monte_carlo_hitting_time(
                P_dg, src, dst,
                num_trials=mc_trials,
                max_steps=mc_max_steps,
                rng=rng,
            )

            #Keeping stats to evaluate results
            stats = sampled_pair_stats[(src, dst)]
            stats["H_dg"].append(H_theory)
            stats["H_sym"].append(H_sym_sd)
            stats["H_mc"].append(H_mc)
            stats["mc_rel"].append(abs(H_mc - H_theory) / H_theory)
            stats["sym_rel"].append(abs(H_sym_sd - H_theory) / H_theory)

    print(f"Valid trials          : {valid} / {trials}")
    print(f"Skipped (disconnected): {skipped_disconnected}")
    print(f"Skipped (singular A)  : {skipped_singular}")

    #If no valid graphs, exit
    if valid == 0:
        print("\nNo valid topologies, increase R")
        return

    #Evaluation metrics for errors due to symmetric\assymetric links
    print("\nAll-pairs digraph vs symmetrized-undirected")
    print(f" Avg pairwise relative error : {np.mean(graph_pair_avg_errors):.4f}")
    print(f" Std pairwise relative error : {np.std(graph_pair_avg_errors):.4f}")
    print(f" Avg max relative error      : {np.mean(graph_pair_max_errors):.4f}")
    print(f" Std max relative error      : {np.std(graph_pair_max_errors):.4f}")
    
    #NOTE, NOT USED IN THE REPORT JUST A SANITY CHECK PRINT OF k pairs of nodes
    print(f"\n{sample_pairs_k} random pairs(fixed seed={pair_seed})")
    header = f"{'pair':>10} | {'avg H_dg':>10} | {'avg H_MC':>10} | {'avg H_sym':>10} | {'MC rel err':>10} | {'sym rel err':>11}"
    print(header)
    print("-" * len(header))

    all_dg = []
    all_mc = []
    all_sym = []
    all_mc_rel = []
    all_sym_rel = []

    for pair in sampled_pairs:
        stats = sampled_pair_stats[pair]
        #Unnecessary, here only for debug purposes to prit n/a if no valid nodes
        if len(stats['H_dg']) == 0:
            print(f"{str(pair):>10} | {'n/a':>10} | {'n/a':>10} | {'n/a':>10} | {'n/a':>10} | {'n/a':>11}")
            continue

        #Evaluation metrics for validating MC to digraph model
        avg_H_dg = np.mean(stats['H_dg'])
        avg_H_mc = np.mean(stats['H_mc'])
        avg_H_sym = np.mean(stats['H_sym'])
        avg_mc_rel = np.mean(stats['mc_rel'])
        avg_sym_rel = np.mean(stats['sym_rel'])

        print(f"{str(pair):>10} | {avg_H_dg:10.4f} | {avg_H_mc:10.4f} | {avg_H_sym:10.4f} | {avg_mc_rel:10.4f} | {avg_sym_rel:11.4f}")

        all_dg.extend(stats["H_dg"])
        all_mc.extend(stats["H_mc"])
        all_sym.extend(stats["H_sym"])
        all_mc_rel.extend(stats["mc_rel"])
        all_sym_rel.extend(stats["sym_rel"])

    if all_dg:
        print("\nOverall averages across sampled pairs")
        print(f" Avg H_dg    : {np.mean(all_dg):.4f}")
        print(f" Avg H_MC    : {np.mean(all_mc):.4f}")
        print(f" Avg H_sym   : {np.mean(all_sym):.4f}")
        print(f" Avg MC rel  : {np.mean(all_mc_rel):.4f}")
        print(f" Avg sym rel : {np.mean(all_sym_rel):.4f}")       
    return {
        "N": N,
        "L": L,
        "R": R,
        "density": N / (L ** 2),
        "valid": valid,
        "graph_pair_avg_errors": np.array(graph_pair_avg_errors),
        "graph_pair_max_errors": np.array(graph_pair_max_errors),
    }

def plot_sparse_dense_results(sparse_result, dense_result):
    """
    Produce two figures:
    1) average max relative error for sparse vs dense
    2) boxplot of per-trial max relative errors for sparse vs dense
    """
    labels = [
        f"Sparse\n(N={sparse_result['N']}, R={sparse_result['R']})",
        f"Dense\n(N={dense_result['N']}, R={dense_result['R']})",
    ]

    avg_max = [
        np.mean(sparse_result["graph_pair_max_errors"]),
        np.mean(dense_result["graph_pair_max_errors"]),
    ]
    std_max = [
        np.std(sparse_result["graph_pair_max_errors"]),
        np.std(dense_result["graph_pair_max_errors"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    #Figure 1: average max relative error
    x = np.arange(2)
    axes[0].bar(
        x,
        avg_max,
        yerr=std_max,
        capsize=6,
        color=["#d95f02", "#1b9e77"]
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Average max relative error")
    axes[0].set_title("Sparse vs dense: average max error")
    axes[0].grid(axis="y", alpha=0.3)

    for i, val in enumerate(avg_max):
        axes[0].text(i, val + 0.02, f"{val:.3f}", ha="center", va="bottom")

    # Figure 2: distribution of per-trial max relative errors
    axes[1].boxplot(
        [
            sparse_result["graph_pair_max_errors"],
            dense_result["graph_pair_max_errors"],
        ],
        labels=labels,
        showfliers=True
    )
    axes[1].set_ylabel("Per-trial max relative error")
    axes[1].set_title("Sparse vs dense: distribution of max errors")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Parameters for running the experiment. Modify here to validate
    sparse_result = run_section_vi_experiment(
        N=20,
        L=1.0,
        R=0.3,
        trials=20,
        mc_trials=2000,
        mc_max_steps=50000,
        sample_pairs_k=5,
        pair_seed=16
    )

    dense_result = run_section_vi_experiment(
        N=40,
        L=1.0,
        R=0.7,
        trials=20,
        mc_trials=2000,
        mc_max_steps=50000,
        sample_pairs_k=5,
        pair_seed=16
    )

    plot_sparse_dense_results(sparse_result, dense_result)
