import numpy as np


# -----------------------------
#  Utility: random rotation
# -----------------------------
def random_rotation_matrix():
    theta = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float32)

# -----------------------------
#  Utility: random reflection
#  reflection across random axis
# -----------------------------
def random_reflection_matrix():
    angle = np.random.uniform(0, 2*np.pi)
    c, s = np.cos(angle), np.sin(angle)
    # unit vector u = (c, s)
    u = np.array([c, s], dtype=np.float32)
    # R = 2 uu^T - I
    R = 2 * np.outer(u, u) - np.eye(2, dtype=np.float32)
    return R.astype(np.float32)

# -----------------------------
#  Generate separated points
# -----------------------------
def generate_separated_points_np(n, k, delta, max_tries=100000):
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        c = np.random.rand(k)
        if all(np.linalg.norm(c - p) >= delta for p in pts):
            pts.append(c)
        tries += 1
    if len(pts) < n:
        raise RuntimeError(f"Could not generate {n} separated points")
    return np.array(pts, dtype=np.float32)

# -----------------------------
#  Single cloud generation task
# -----------------------------
def generate_cloud_task(args):
    N, K, eps, delta = args

    # Base point cloud X
    X = generate_separated_points_np(N, K, delta)

    # Decide transformation type
    if np.random.rand() < 0.5:
        mat = random_rotation_matrix()
        ttype = "rotation"
    else:
        mat = random_reflection_matrix()
        ttype = "reflection"

    # Apply rigid transform BEFORE noise
    X_trans = X @ mat.T

    # Random permutation
    perm = np.random.permutation(N)
    Y = X_trans[perm] + np.random.uniform(-eps, eps, size=(N, K)).astype(np.float32)

    return X, Y, perm, mat, ttype

# -----------------------------
#  Parallel batch generator
# -----------------------------
def make_batch_cpu_parallel(B, N, K, eps, delta, n_workers=30):
    import multiprocessing

    print("Parameters entered:", B, N, K)

    Xs = np.empty((B, N, K), dtype=np.float32)
    Ys = np.empty((B, N, K), dtype=np.float32)
    perms = np.empty((B, N), dtype=np.int64)
    mats = np.empty((B, K, K), dtype=np.float32)
    types = np.empty((B,), dtype=object)

    task_args = [(N, K, eps, delta)] * B

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(generate_cloud_task, task_args)

    for b, (X, Y, perm, mat, ttype) in enumerate(results):
        Xs[b] = X
        Ys[b] = Y
        perms[b] = perm
        mats[b] = mat
        types[b] = ttype

    inv_perms = np.argsort(perms, axis=1)

    return Xs, Ys, inv_perms, mats, types

# -----------------------------
#  Simple CPU (non-parallel) version
# -----------------------------
def make_batch_cpu_old(B, N, K, eps, delta, mode="both"):
    """
    mode: "rotation"   -> only rotations
          "reflection" -> only reflections
          "both"       -> random choice per sample
          "plain"      -> no transformation
    """
    Xs = np.empty((B, N, K), dtype=np.float32)
    Ys = np.empty((B, N, K), dtype=np.float32)
    perms = np.empty((B, N), dtype=np.int64)
    mats = np.empty((B, K, K), dtype=np.float32)
    types = np.empty((B,), dtype=object)

    for b in range(B):
        X = generate_separated_points_np(N, K, delta)

        # Determine matrix type based on mode
        if mode == "rotation":
            mat = random_rotation_matrix()
            ttype = "rotation"
        elif mode == "reflection":
            mat = random_reflection_matrix()
            ttype = "reflection"
        elif mode == "both":
            if np.random.rand() < 0.5:
                mat = random_rotation_matrix()
                ttype = "rotation"
            else:
                mat = random_reflection_matrix()
                ttype = "reflection"
        elif mode == "plain":
            mat = np.eye(K, dtype=np.float32)  # Identity, no change
            ttype = "plain"
        else:
            raise ValueError("Invalid mode. Choose from 'rotation', 'reflection', 'both', or 'plain'.")

        X_trans = X @ mat.T  # Will be unchanged for "plain"

        perm = np.random.permutation(N)
        Y = X_trans[perm] + np.random.uniform(-eps, eps, size=(N, K)).astype(np.float32)

        Xs[b] = X
        Ys[b] = Y
        perms[b] = perm
        mats[b] = mat
        types[b] = ttype

    inv_perms = np.argsort(perms, axis=1)
    return Xs, Ys, inv_perms, mats, types

def random_rotation_matrix(K):
    """
    Generate a random KxK rotation matrix (uniform over SO(K)).
    """
    A = np.random.normal(size=(K, K))
    Q, R = np.linalg.qr(A)
    # Make Q a proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)

def random_reflection_matrix(K):
    """
    Generate a random KxK reflection matrix.
    """
    R = random_rotation_matrix(K)
    # Flip sign of exactly one axis → det becomes -1
    i = np.random.randint(0, K)
    R[:, i] *= -1
    return R.astype(np.float32)



def random_points_on_sphere(N, K, radius=1.0):
    X = np.random.normal(size=(N, K))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= radius
    return X

def make_batch_cpu(B, N, K, eps, delta, mode="both", generationmode = 'box'):
    """
    Produces Xs and Ys that are both centered (zero-mean per sample).
    Noise is isotropic uniform inside an L2-ball of radius eps.
    """
    Xs = np.empty((B, N, K), dtype=np.float32)
    Ys = np.empty((B, N, K), dtype=np.float32)
    perms = np.empty((B, N), dtype=np.int64)
    mats = np.empty((B, K, K), dtype=np.float32)
    types = np.empty((B,), dtype=object)

    for b in range(B):

        if generationmode == 'box':
            X = generate_separated_points_np(N, K, delta).astype(np.float32)
        # center X
            mu = X.mean(axis=0, keepdims=True).astype(np.float32)
            Xc = X - mu
        if generationmode == 'sphere':
            Xc = random_points_on_sphere(N, K).astype(np.float32)


        # choose matrix
        if mode == "rotation":
          #  mat = random_rotation_matrix().astype(np.float32)
            mat = random_rotation_matrix(K).astype(np.float32)
            ttype = "rotation"
        elif mode == "reflection":
            mat = random_reflection_matrix(K).astype(np.float32)
            ttype = "reflection"
        elif mode == "both":
            if np.random.rand() < 0.5:
                mat = random_rotation_matrix(K).astype(np.float32)
                ttype = "rotation"
            else:
                mat = random_reflection_matrix(K).astype(np.float32)
                ttype = "reflection"
        elif mode == "plain":
            mat = np.eye(K, dtype=np.float32)
            ttype = "plain"
        else:
            raise ValueError("Invalid mode.")

        # transform
        Xc_trans = Xc @ mat.T

        # permutation
        perm = np.random.permutation(N)

        # === isotropic uniform noise in L2 ball of radius eps ===
        # v uniformly distributed direction (normal)
        v = np.random.normal(size=(N, K)).astype(np.float32)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
        # radius scaling: U^(1/K)
        u = np.random.rand(N, 1).astype(np.float32) ** (1.0 / K)
        noise = eps * u * (v / v_norm)
        # ========================================================

        Yc = Xc_trans[perm] + noise

        # recenter Y for exact centering
        Yc -= Yc.mean(axis=0, keepdims=True)

        Xs[b] = Xc
        Ys[b] = Yc
        perms[b] = perm
        mats[b] = mat
        types[b] = ttype

    inv_perms = np.argsort(perms, axis=1)
    return Xs, Ys, inv_perms, mats, types





