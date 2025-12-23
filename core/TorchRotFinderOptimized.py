import torch
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from core import TorchEMDRotOptimizedHypotheses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_point_cloud(X):
    """
    Center a point cloud at the origin.
    X: (N, n)
    """
    centroid = X.mean(dim=0, keepdim=True)
    return X - centroid


def max_subspace_volume(A, combos):
    """
    Find the combination of points that maximizes the (n-1)-volume.

    A: (N, n) tensor
    combos: (C, n-1) int tensor of indices into A

    Returns:
        best_idx: index of best combination
    """
    V = A[combos]  # (C, n-1, n)
    Vt = V.transpose(1, 2)  # (C, n, n-1)
    G = torch.matmul(Vt.transpose(1, 2), Vt)  # Gram matrix (C, n-1, n-1)
    detG = torch.linalg.det(G)
    vols = torch.sqrt(torch.clamp(detG, min=0))
    best_idx = torch.argmax(vols)
    return best_idx


def rotation_matrices_batch_ND(x, Y):
    """
    Compute rotation matrices aligning x to each cloud in Y.
    x: (n-1, n)
    Y: (m, n-1, n)
    Returns:
        R_batch: (m, n, n)
    """
    m, n_minus1, n = Y.shape
    R_batch = torch.zeros((m, n, n), device=Y.device, dtype=Y.dtype)

    for i in range(m):
        M = Y[i].T @ x  # (n, n-1)
        U, _, Vt = torch.linalg.svd(M, full_matrices=True)
        R = U @ Vt

        # Ensure proper rotation
        if torch.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        R_batch[i] = R

    return R_batch


def ComputeIsometryWithMatchingnD(A, B, maxchunksize=1000, sinkhorn_iters=50, rot_iters=1, reg=1e-3):
    """
    Compute approximate isometry between point clouds A and B, optimized for memory.
    Returns CPU numpy arrays/floats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    A = torch.as_tensor(A, dtype=torch.float32, device=device)
    B = torch.as_tensor(B, dtype=torch.float32, device=device)

    cloudSize, k = A.shape
    perms = np.array(list(itertools.permutations(range(cloudSize), k - 1)))
    perm_tensor = torch.as_tensor(perms, device=device, dtype=torch.long)

    # Center the clouds
    A = center_point_cloud(A)
    B = center_point_cloud(B)

    best_idx = max_subspace_volume(A, perm_tensor)
    total_pairs = perm_tensor.shape[0]

    # Select reference subcloud
    Aselect = A[perm_tensor[best_idx], :]

    # Initialize best result trackers
    best_val = float('inf')
    best_G = None
    best_rot = None

    for start in range(0, total_pairs, maxchunksize):
        end = min(start + maxchunksize, total_pairs)
        selected_indices = perm_tensor[start:end, :]

        # Compute candidate rotations
        rotMats = rotation_matrices_batch_ND(Aselect, B[selected_indices])

        # Run Sinkhorn + Procrustes
        dataBD = TorchEMDRotOptimizedHypotheses.run_sinkhorn_torch_rot_hypotheses(
            A, B, rotMats, reg=reg, sinkhorn_iters=sinkhorn_iters, rot_iters=rot_iters
        )

        # Find best in this chunk
        chunk_min_val, chunk_min_idx = torch.min(dataBD["expected_distance"], dim=0)

        if chunk_min_val < best_val:
            best_val = chunk_min_val
            best_G = dataBD["G"][chunk_min_idx]
            best_rot = dataBD["R"][chunk_min_idx]

        # Optional: free memory immediately
        del dataBD, rotMats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Convert to CPU NumPy arrays / float
    best_val = best_val.item()
    best_G = best_G.cpu().numpy()
    best_rot = best_rot.cpu().numpy()

    Af = A.cpu().numpy()
    Bf = B.cpu().numpy()

    cost = -best_G
    rows, cols = linear_sum_assignment(cost)
    perm = np.zeros(best_G.shape[0], dtype=int)
    perm[rows] = cols

    return Af,Bf, best_val, best_rot, best_G,perm

