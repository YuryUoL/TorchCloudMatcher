import time
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment  # optional if you want exact Hungarian
# Note: you can avoid scipy if you only use greedy matching below

# ------------------------
# Cost / distance functions
# ------------------------
def pairwise_cost_batch_torch(X, Y, metric="sqeuclid", labels_X=None, labels_Y=None, large_cost=1e9):
    """
    X, Y: (B, N, D) torch tensors (on CPU or CUDA)
    Returns C: (B, N, N)
    """
    B, N, D = X.shape

    if metric in ("sqeuclid", "euclid", "rms"):
        XX = (X * X).sum(dim=2, keepdim=True)      # (B, N, 1)
        YY = (Y * Y).sum(dim=2, keepdim=True)      # (B, N, 1)
        XY = X @ Y.transpose(1, 2)                 # (B, N, N)
        Csq = XX - 2 * XY + YY.transpose(1, 2)
        Csq = torch.clamp(Csq, min=0.0)

        if metric == "sqeuclid":
            C = Csq
        elif metric == "euclid":
            C = torch.sqrt(Csq)
        else:  # metric == "rms"
            C = torch.sqrt(Csq / D)

    elif metric == "linf":
        C = (X[:, :, None, :] - Y[:, None, :, :]).abs().max(dim=3).values

    else:
        raise ValueError("Unsupported metric")

    # ---- Label masking ----
    if labels_X is not None and labels_Y is not None:
        Li = labels_X[:, :, None]   # (B, N, 1)
        Lj = labels_Y[:, None, :]   # (B, 1, N)
        mismatch = (Li != Lj)

        C = torch.where(mismatch, torch.full_like(C, large_cost), C)

    return C

# ------------------------
# stable log-sum-exp on GPU (torch)
# ------------------------
def logsumexp_axis(a, axis):
    return torch.logsumexp(a, dim=axis, keepdim=True)
# ------------------------
# Batched log-Sinkhorn (GPU)
# ------------------------
def batched_log_sinkhorn_torch(C, reg=1e-3, n_iters=100, a=None, b=None):
    """
    C: (B, N, N)
    Returns: G (B, N, N)
    """
    B, N, _ = C.shape
    device = C.device

    if a is None:
        a = torch.full((B, N, 1), 1.0 / N, device=device)
    else:
        a = a.unsqueeze(-1)

    if b is None:
        b = torch.full((B, N, 1), 1.0 / N, device=device)
    else:
        b = b.unsqueeze(-1)

    logK = -C / reg
    logu = torch.zeros((B, N, 1), device=device)
    logv = torch.zeros((B, N, 1), device=device)

    for _ in range(n_iters):
        logu = torch.log(a) - torch.logsumexp(logK + logv.transpose(1, 2), dim=2, keepdim=True)
        logv = torch.log(b) - torch.logsumexp(logK.transpose(1, 2) + logu.transpose(1, 2), dim=2, keepdim=True)

    logG = logu + logK + logv.transpose(1, 2)
    return torch.exp(logG)

# ------------------------
# Greedy bijection extraction (numpy)
# ------------------------
def greedy_bijection_from_G_np(G):
    N = G.shape[0]
    perm = -np.ones(N, dtype=int)
    used_rows = np.zeros(N, bool)
    used_cols = np.zeros(N, bool)

    flat_idx = np.argsort(-G.ravel())

    for idx in flat_idx:
        r = idx // N
        c = idx % N
        if not used_rows[r] and not used_cols[c]:
            perm[r] = c
            used_rows[r] = True
            used_cols[c] = True
            if used_rows.all():
                break

    # Fill remaining
    unused = np.where(~used_cols)[0]
    for r in np.where(perm == -1)[0]:
        perm[r] = unused[0]
        unused = unused[1:]

    return perm




#def run_sinkhorn_from_pairs_distanceMats(Matrices,):

#def launch_sinkhorn_from_distanceMat(X_cpu, Y_cpu, C_cpu, inv_perms=None,
#                            reg=1e-3, sinkhorn_iters=100, extract_bijection=True, use_greedy=True, starttime = None)




def run_sinkhorn_torch(
    X, Y,
    reg=1e-3,
    metric="sqeuclid",
    sinkhorn_iters=100,
    extract=False,
    use_greedy=True,
):
    """
    Debug version:
    - Ensures X/Y are torch tensors
    - Prints device info
    - Returns numpy arrays everywhere
    """

    # --------------------------
    # Ensure X, Y are Torch tensors
    # --------------------------
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.as_tensor(Y, dtype=torch.float32)

    # --------------------------
    # Decide device (prefer GPU)
    # --------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
      #  print(">> Using GPU:", device)
    else:
        device = torch.device("cpu")
      #  print(">> Using CPU")

    # Move data to device
    X = X.to(device)
    Y = Y.to(device)

    # --------------------------
    # Compute cost matrix C
    # --------------------------

    C = pairwise_cost_batch_torch(X, Y, metric=metric)

    return run_sinkhorn_distanceMat(C, reg = reg, sinkhorn_iters= sinkhorn_iters, extract= extract, use_greedy = use_greedy)



def run_sinkhorn_distanceMat(C, reg=1e-3,sinkhorn_iters=100,
    extract=False,
    use_greedy=True, ):

  #  if torch.cuda.is_available():
  #      device = torch.device("cuda")
  #      print(">> Using GPU:", device)
  #  else:
  #      device = torch.device("cpu")
  #      print(">> Using CPU")


    # --------------------------
    # Run Sinkhorn solver
    # --------------------------
    G = batched_log_sinkhorn_torch(C, reg=reg, n_iters=sinkhorn_iters)

    # Expected Wp cost
    expected = (G * C).sum(dim=(1, 2))

    # --------------------------
    # FAST RETURN (no bijection)
    # --------------------------
    if not extract:
        return {
            "C": C,
            "G": G,
            "expected": expected
        }

    # --------------------------
    # Extract bijection (CPU)
    # --------------------------
    G_cpu = G.detach().cpu().numpy()
    B, N = G_cpu.shape[:2]

    bij_perm = np.zeros((B, N), dtype=int)

    for i in range(B):
        Gi = G_cpu[i]

        if use_greedy:
            # Greedy mass-based selection
            perm = greedy_bijection_from_G_np(Gi)
        else:
            # Hungarian solver: maximize G <=> minimize -G
            cost = -Gi
            rows, cols = linear_sum_assignment(cost)
            perm = np.zeros(N, dtype=int)
            perm[rows] = cols

        bij_perm[i] = perm

    # --------------------------
    # Return all results as NumPy
    # --------------------------
    # .detach().cpu().numpy(),
    return {
        "C": C,
        "G": G,
        "expected": expected,
        "bij_perm": bij_perm
    }








def JustTest():
# 1 batch, 4 points in 2D
    X_cpu = np.array([[[0,0], [1,0], [0,1], [1,1]]], dtype=float)  # shape (1,4,2)

# Shuffle Y points (so matching is nontrivial)
    Y_cpu = np.array([[[1,1], [0,0], [1,0], [0,1]]], dtype=float)

# ground truth inverse permutation: where X[i] should go in Y
# i.e. X[0] = [0,0] is at Y[1], X[1]=[1,0] is at Y[2], etc.
    inv_perms = np.array([[1, 2, 3, 0]])

# Run Sinkhorn experiment
    results = run_sinkhorn_torch(X_cpu, Y_cpu, sinkhorn_iters=50, extract = True, use_greedy= False)

# Print
    print("Matrix G (coupling):\n", results["G"][0])
    print("Greedy bijection perm:", results["bij_perm"][0])
    #print("Pred argmax (row-wise max):", results["pred_argmax"][0])
    print("X -> Y mapping:")
    for i, j in enumerate(results["bij_perm"][0]):
        print(f"X[{i}] = {X_cpu[0,i]} matched to Y[{j}] = {Y_cpu[0,j]}")

    G = results["G"]
    row_sums = np.sum(G, axis=2)
    col_sums = np.sum(G, axis=1)
    print(row_sums)
    print(col_sums)

#JustTest()