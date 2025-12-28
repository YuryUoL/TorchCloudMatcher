import numpy as np
from core.TorchRotFinderOptimized import ComputeIsometryWithMatchingnD
from core.TorchRotFinderOptimized import ComputeBD

def compute_isometry(A_np, B_np, maxchunksize=200, sinkhorn_iters=50, rot_iters=2, reg=1e-3):
    """
    Wrapper to compute isometry with matching and return sanitized outputs.
    """
    A2, B2, min_val, R, G, tmap = ComputeIsometryWithMatchingnD(
        A_np, B_np,
        maxchunksize=int(maxchunksize),
        sinkhorn_iters=int(sinkhorn_iters),
        rot_iters=int(rot_iters),
        reg=float(reg)
    )

    A2 = np.asarray(A2)
    B2 = np.asarray(B2)
    R = np.asarray(R)
    tmap = np.asarray(tmap).astype(int).flatten()

    # Final rotated A
    R_end = R.T
    theta_end = np.arctan2(R_end[1, 0], R_end[0, 0])
    A_final = A2 @ R_end

    bddist = ComputeBD(A_final, B2 , tmap,metric = 'euclidean')

    return dict(A2=A2, B2=B2, R=R, G=G, tmap=tmap, A_final=A_final, min_val=float(min_val), theta_end=theta_end, bddist = bddist)

def compute_isometry_nD(A_np, B_np, maxchunksize=200, sinkhorn_iters=50, rot_iters=2, reg=1e-3):
    """
    Wrapper to compute isometry with matching and return sanitized outputs.
    """
    A2, B2, min_val, R, G, tmap = ComputeIsometryWithMatchingnD(
        A_np, B_np,
        maxchunksize=int(maxchunksize),
        sinkhorn_iters=int(sinkhorn_iters),
        rot_iters=int(rot_iters),
        reg=float(reg)
    )

    A2 = np.asarray(A2)
    B2 = np.asarray(B2)
    R = np.asarray(R)
    tmap = np.asarray(tmap).astype(int).flatten()

    # Final rotated A
    R_end = R.T
    A_final = A2 @ R_end

    bddist = ComputeBD(A_final, B2, tmap,metric = 'euclidean')

    return dict(A2=A2, B2=B2, R=R, G=G, tmap=tmap, A_final=A_final, min_val=float(min_val), bddist = bddist)