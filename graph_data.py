import numpy as np

def random_wireless_digraph(N: int,L: float,R: float,p_min: float = 0.3,p_max: float = 1.0,rng: np.random.Generator | None = None,):
    """
    Generate a random wireless topology as in Section VI:

    - N nodes uniformly in [0, L] x [0, L]
    - directed edge i->j if distance(i,j) <= R
    - asymmetric link delivery probabilities p_ij ~ U[p_min, p_max]
    """

    if rng is None:
        rng = np.random.default_rng()

    # 1. Random positions (bivariate uniform)
    positions = rng.uniform(0.0, L, size=(N, 2))

    # 2. Distance matrix
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # 3. Weight matrix W: p_ij if within range, else 0
    W = np.zeros((N, N), dtype=float)
    within = (dists <= R) & (dists > 0)  # no self-links

    # Draw asymmetric probabilities independently
    W[within] = rng.uniform(p_min, p_max, size=within.sum())

    return positions, W