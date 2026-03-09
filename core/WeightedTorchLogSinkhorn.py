import torch

# ============================================================
# Utilities
# ============================================================

def normalize_weights_torch(w, eps=1e-12):
    """
    w: (B, N)
    """
    w = torch.clamp(w, min=0)
    s = torch.sum(w, dim=1, keepdim=True)
    s = torch.clamp(s, min=eps)
    return w / s


def logsumexp_axis(a, axis):
    return torch.logsumexp(a, dim=axis, keepdim=True)


# ============================================================
# Batched pairwise cost (metric-aware)
# ============================================================

def pairwise_cost_batch_torch(
    X,
    Y,
    *,
    metric="sqeuclid",
    labels_X=None,
    labels_Y=None,
    large_cost=1e9
):
    """
    Batched pairwise cost matrix.

    X, Y     : (B, N, D) torch tensors
    metric   : "sqeuclid" | "euclid" | "rms" | "linf" | "linf_memory"
    labels_* : optional (B, N)
    returns  : (B, N, N)
    """
    B, N, D = X.shape
    device = X.device
    dtype = X.dtype

    if metric in ("sqeuclid", "euclid", "rms"):
        # ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2
        XX = torch.sum(X * X, dim=2, keepdim=True)          # (B, N, 1)
        YY = torch.sum(Y * Y, dim=2, keepdim=True)          # (B, N, 1)
        XY = torch.matmul(X, Y.transpose(1, 2))             # (B, N, N)

        Csq = torch.clamp(
            XX - 2.0 * XY + YY.transpose(1, 2),
            min=0.0
        )

        if metric == "sqeuclid":
            C = Csq
        elif metric == "euclid":
            C = torch.sqrt(Csq)
        else:  # rms
            C = torch.sqrt(Csq / float(D))

    elif metric == "linf":
        # Full broadcast version (fast but memory heavy)
        Xb = X[:, :, None, :]        # (B, N, 1, D)
        Yb = Y[:, None, :, :]        # (B, 1, N, D)
        C = torch.max(torch.abs(Xb - Yb), dim=3).values

    elif metric == "linf_memory":
        # Memory-efficient version
        C = torch.empty((B, N, N), device=device, dtype=dtype)
        for b in range(B):
            for i in range(N):
                diff = torch.abs(X[b, i:i+1] - Y[b])  # (N, D)
                C[b, i] = torch.max(diff, dim=1).values

    else:
        raise ValueError(
            "metric must be one of "
            "{'sqeuclid', 'euclid', 'rms', 'linf', 'linf_memory'}"
        )

    # --------------------------------------------------------
    # Label masking
    # --------------------------------------------------------
    if labels_X is not None and labels_Y is not None:
        if not torch.is_tensor(labels_X):
            labels_X = torch.as_tensor(labels_X, device=device)
        if not torch.is_tensor(labels_Y):
            labels_Y = torch.as_tensor(labels_Y, device=device)

        mismatch = labels_X[:, :, None] != labels_Y[:, None, :]
        C = torch.where(
            mismatch,
            torch.full_like(C, large_cost),
            C
        )

    return C


# ============================================================
# Batched log-Sinkhorn
# ============================================================

def batched_log_sinkhorn_torch(
    C,
    *,
    reg=1e-3,
    n_iters=100,
    a=None,
    b=None
):
    """
    C : (B, N, N)
    a : optional (B, N)
    b : optional (B, N)

    returns:
        G : (B, N, N)
    """
    B, N, _ = C.shape
    device = C.device
    dtype = C.dtype

    if a is None:
        a = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    else:
        a = normalize_weights_torch(a)

    if b is None:
        b = torch.full((B, N), 1.0 / N, device=device, dtype=dtype)
    else:
        b = normalize_weights_torch(b)

    logK = -C / reg
    loga = torch.log(a).unsqueeze(-1)   # (B, N, 1)
    logb = torch.log(b).unsqueeze(-1)   # (B, N, 1)

    logu = torch.zeros((B, N, 1), device=device, dtype=dtype)
    logv = torch.zeros((B, N, 1), device=device, dtype=dtype)

    for _ in range(n_iters):
        logu = loga - logsumexp_axis(
            logK + logv.transpose(1, 2), axis=2
        )
        logv = logb - logsumexp_axis(
            logK.transpose(1, 2) + logu.transpose(1, 2), axis=2
        )

    logG = logu + logK + logv.transpose(1, 2)
    return torch.exp(logG)


# ============================================================
# High-level API
# ============================================================


def run_sinkhorn_from_pairs_torch(
    Matrices,
    Labels=None,
    P=None,
    Weights=None,
    **kwargs
):
    assert P is not None, "P (pair indices) must be provided."
    assert Matrices.ndim == 3, "Matrices should be (B, N, D)."

    X = Matrices[P[:, 0]]
    Y = Matrices[P[:, 1]]

    labels_X = Labels[P[:, 0]] if Labels is not None else None
    labels_Y = Labels[P[:, 1]] if Labels is not None else None

    weights_X = Weights[P[:, 0]] if Weights is not None else None
    weights_Y = Weights[P[:, 1]] if Weights is not None else None

    return run_log_sinkhorn_torch(
        X,
        Y,
        labels_X=labels_X,
        labels_Y=labels_Y,
        weights_X=weights_X,
        weights_Y=weights_Y,
        **kwargs
    )


def run_log_sinkhorn_torch(
    X,
    Y,
    labels_X=None,
    labels_Y=None,
    weights_X=None,
    weights_Y=None,
    metric="sqeuclid",
    reg=1e-3,
    sinkhorn_iters=100,
    return_cpu = True
):
    """
    Batched Sinkhorn with metric choice, labels, and coefficients.

    X, Y      : (B, N, D) torch tensors
    weights_* : optional (B, N)
    labels_*  : optional (B, N)

    returns:
        dict with keys:
            C
            G
            expected_costs
    """


    C = pairwise_cost_batch_torch(
        X,
        Y,
        metric=metric,
        labels_X=labels_X,
        labels_Y=labels_Y
    )

    G = batched_log_sinkhorn_torch(
        C,
        reg=reg,
        n_iters=sinkhorn_iters,
        a=weights_X,
        b=weights_Y
    )

    # Expected cost
    expected_costs = torch.sum(
        G * torch.where(torch.isfinite(C), C, torch.zeros_like(C)),
        dim=(1, 2)
    )

    result = {
        "C": C,
        "G": G,
        "expected_costs": expected_costs
    }

    if return_cpu:
        result = {k: v.detach().cpu() for k, v in result.items()}



    return result
