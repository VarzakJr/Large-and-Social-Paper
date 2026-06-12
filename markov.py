import numpy as np
from numpy.linalg import eig, inv

def make_transition_P_from_W(W: np.ndarray) -> np.ndarray:
    """
    Turns matrix W(essentially link quality for each node),
    into a stochastic transition matrix P(each row sums to 1, expressing transition probabilities).
    If a row i has all zeros, make it an absorbing state at i.
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]
    row_sums = W.sum(axis=1, keepdims=True)
    P = np.zeros_like(W, dtype=float)

    # Normalize rows with positive total weight
    mask = row_sums[:, 0] > 0
    P[mask] = W[mask] / row_sums[mask]

    # For rows with sum == 0, create self-loops
    for i in range(n):
        if row_sums[i, 0] == 0.0:
            P[i, i] = 1.0

    return P

def stationary_distribution(P: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute stationary distribution pi of an irreducible Markov chain with
    transition matrix P by solving pi^T P = pi^T, sum(pi) = 1.
    """
    n = P.shape[0]
    # Solve (P^T - I)^T pi = 0 with normalization sum(pi) = 1
    A = P.T - np.eye(n)
    # Replace one equation by sum(pi)=1
    A[-1, :] = np.ones(n)
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    # Ensure non-negative / normalized
    pi = np.maximum(pi, 0)
    pi = pi / pi.sum()
    return pi

def fundamental_matrix(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    Compute fundamental matrix Z.
    """
    n = P.shape[0]
    I = np.eye(n)
    ones = np.ones((n, 1))
    Z = inv(I - P + ones @ pi.reshape(1, -1))
    return Z


def hitting_times(P: np.ndarray) -> np.ndarray:
    """
    Computes all-pairs hitting time matrix H for a Markov chain with
    transition matrix P using the fundamental matrix formula.
    H[i, j] = expected number of steps for walk starting at i to first hit j.
    """
    n = P.shape[0]
    pi = stationary_distribution(P)
    Z = fundamental_matrix(P, pi)
    H = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                H[i, j] = 0.0
            else:
                H[i, j] = (Z[j, j] - Z[i, j]) / pi[j]
    return H

def symmetrize_weights(W: np.ndarray) -> np.ndarray:
    """
    Symmetrize a directed weight matrix by averaging W and W^T.
    """
    return 0.5 * (W + W.T)


def hitting_times_for_digraph_and_undirected(W: np.ndarray):
    """
    - builds Markov chain on the asymmetric digraph
    - builds Markov chain on the symmetrized undirected graph
    - returns the two all-pairs hitting-time matrices.
    """
    #Asymmetric digraph model
    P_digraph = make_transition_P_from_W(W)
    H_digraph = hitting_times(P_digraph)

    #Symmetrized undirected model
    W_undirected = symmetrize_weights(W)
    P_undirected = make_transition_P_from_W(W_undirected)
    H_undirected = hitting_times(P_undirected)

    return H_digraph, H_undirected