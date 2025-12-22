import torch
import numpy as np
from TorchEMD import pairwise_cost_batch_torch, batched_log_sinkhorn_torch


# ------------------------
# SO(n) utilities
# ------------------------
def skew(A):
    return A - A.transpose(-1, -2)


def rotation_from_param(A):
    """
    A: (K, D, D) unconstrained
    Returns R: (K, D, D) in SO(D)
    """
    return torch.matrix_exp(skew(A))

def batched_weighted_procrustes(X, Y, G, eps=1e-8):
    """
    X, Y: (K, N, D)
    G:    (K, N, N) transport plans

    Returns:
        R: (K, D, D) optimal rotations
    """
    K, N, D = X.shape

    # row / col marginals
    wx = G.sum(dim=2)          # (K, N)
    wy = G.sum(dim=1)          # (K, N)

    wx = wx / (wx.sum(dim=1, keepdim=True) + eps)
    wy = wy / (wy.sum(dim=1, keepdim=True) + eps)

    # centroids
    muX = (wx[..., None] * X).sum(dim=1, keepdim=True)  # (K,1,D)
    muY = (wy[..., None] * Y).sum(dim=1, keepdim=True)

    Xc = X - muX
    Yc = Y - muY

    # cross-covariance
    M = torch.bmm(Xc.transpose(1, 2), torch.bmm(G, Yc))  # (K,D,D)

    # SVD
    U, _, Vt = torch.linalg.svd(M)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    # det correction
    det = torch.det(R)
    mask = det < 0
    if mask.any():
        Vt[mask, -1, :] *= -1
        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    return R

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
# Main routine
# ------------------------
def run_sinkhorn_torch_rot_hypotheses(
    X,
    Y,
    R_init,
    reg=1e-3,
    metric="sqeuclid",
    sinkhorn_iters=50,
    rot_iters=1,
    verbose=False,
):
    """
    OT-Procrustes alternating optimization with soft OT distance.

    X, Y: (N, D) point clouds (NOT batched)
    R_init: (K, D, D) initial rotations
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    R = torch.as_tensor(R_init, dtype=torch.float32, device=device)

    N, D = X.shape
    K = R.shape[0]

    Xb = X.unsqueeze(0).expand(K, -1, -1)
    Yb = Y.unsqueeze(0).expand(K, -1, -1)

    for it in range(rot_iters):

        # --- Apply rotations ---
        RX = torch.bmm(Xb, R.transpose(1, 2))

        # --- Sinkhorn (no gradients needed for OT-Procrustes) ---
        with torch.no_grad():
            C = pairwise_cost_batch_torch(RX, Yb, metric=metric)
            G = batched_log_sinkhorn_torch(C, reg=reg, n_iters=sinkhorn_iters)

        # --- Procrustes update ---
        R = batched_weighted_procrustes(Xb, Yb, G)

        if verbose:
            loss_per_hyp = (G * C).sum(dim=(1, 2))
            print(f"[OT-Procrustes] iter {it+1}/{rot_iters} | mean soft distance = {loss_per_hyp.mean():.6f}")

    # --- Compute final rotated points and soft OT distance ---
    RX_final = torch.bmm(Xb, R.transpose(1, 2))
    with torch.no_grad():
        C_final = pairwise_cost_batch_torch(RX_final, Yb, metric=metric)
        G_final = batched_log_sinkhorn_torch(C_final, reg=reg, n_iters=sinkhorn_iters)
        expected_distance = (G_final * C_final).sum(dim=(1, 2))

    return {
        "R": R,                       # (K, D, D) final rotations
        "G": G_final,                  # (K, N, N) final soft coupling
        "expected_distance": expected_distance  # (K,) soft distances
    }

