import torch
import itertools
import numpy as np
from core import TorchEMD

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


def ComputeIsometryWithMatchingnD(A, B, maxchunksize=1000, sinkhorn_iters=50, reg=1e-3):
    """
    Compute approximate isometry between point clouds A and B.
    """
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)


    cloudSize, k = A.shape
    perms = np.array(list(itertools.permutations(range(cloudSize), k - 1)))
    perm_tensor = torch.tensor(perms, device=device, dtype=torch.long)

    A = center_point_cloud(A)
    B = center_point_cloud(B)

    best_idx = max_subspace_volume(A, perm_tensor)
    total_pairs = perm_tensor.shape[0]

    d_list = []
    Aselect = A[perm_tensor[best_idx], :]

    for start in range(0, total_pairs, maxchunksize):
        end = min(start + maxchunksize, total_pairs)
        selected_indices = perm_tensor[start:end, :]

        rotMats = rotation_matrices_batch_ND(Aselect, B[selected_indices])
        N = rotMats.shape[0]

        batchedX = A.unsqueeze(0).expand(N, -1, -1)
        rotatedX = torch.bmm(batchedX, rotMats.transpose(1, 2))
        batchedY = B.unsqueeze(0).expand(N, -1, -1)

        dataBD = TorchEMD.run_sinkhorn_torch(
            rotatedX, batchedY, reg=reg, sinkhorn_iters=sinkhorn_iters,
            extract=False, use_greedy=False
        )
        d_list.append(dataBD["expected"])

    d_all = torch.cat(d_list, dim=0)
    min_val, min_idx_r = torch.min(d_all, dim=0)

    Bdash = B[perm_tensor[min_idx_r], :].unsqueeze(0)
    rotF = rotation_matrices_batch_ND(Aselect, Bdash)

    # Compute final Sinkhorn
    batchedXone = A.unsqueeze(0)
    rotatedXone = torch.bmm(batchedXone, rotF.transpose(1, 2))
    batchedYtwo = B.unsqueeze(0)

    dataBDspecial = TorchEMD.run_sinkhorn_torch(
        rotatedXone, batchedYtwo, reg=reg, sinkhorn_iters=sinkhorn_iters,
        extract=True, use_greedy=False
    )

    G = dataBDspecial["G"][0]
    t = dataBDspecial["bij_perm"][0]

    Af = A.cpu().numpy()
    Bf = B.cpu().numpy()
    valF = min_val.item()
    rotR = rotF[0].cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()


    return Af, Bf, valF, rotR, G, t
