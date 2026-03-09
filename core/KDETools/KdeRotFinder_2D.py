
from utils import Cloudgen_2D as CloudGen
import torch
import time

# GPU-friendly 2D polar KDE
# -------------------------------
def polar_kde_gpu(points, Nr=32, Ntheta=180, sigma_r=0.2, kappa_theta=8.0, device='cuda'):
    device = points.device if points.device.type != 'cpu' else device
    center = points.mean(0, keepdim=True)
    pts = points - center
    r = torch.norm(pts, dim=1)
    r_norm = r / (r.max() + 1e-6)  # normalize radius
    theta = torch.atan2(pts[:, 1], pts[:, 0])

    r_grid = torch.linspace(0, 1.0, Nr, device=device)
    theta_grid = torch.linspace(0, 2*torch.pi, Ntheta, device=device)

    r_diff = r_grid.view(Nr,1) - r_norm.view(1,-1)
    theta_diff = theta_grid.view(Ntheta,1) - theta.view(1,-1)

    radial_kernel = torch.exp(-0.5 * (r_diff / sigma_r)**2)
    angular_kernel = torch.exp(kappa_theta * torch.cos(theta_diff))

    kde = torch.einsum('in,jn->ij', radial_kernel, angular_kernel)
    kde /= points.shape[0]
    return kde

# -------------------------------
# Differentiable rotation (vectorized for batch)
# -------------------------------
def rotate_kde_interp_batch(kde, theta_batch):
    """
    kde: (Nr, Ntheta)
    theta_batch: (B,) tensor of rotation angles
    returns: (B, Nr, Ntheta) rotated KDEs
    """
    Nr, Ntheta = kde.shape
    B = theta_batch.shape[0]
    device = kde.device

    phi = torch.linspace(0, 2*torch.pi, Ntheta, device=device)
    phi = phi.view(1, 1, Ntheta)  # shape (1,1,Ntheta)
    theta_batch = theta_batch.view(B, 1, 1)  # shape (B,1,1)

    # Compute shifted positions in index space
    grid = ((phi + theta_batch) % (2*torch.pi)) / (2*torch.pi) * Ntheta  # (B,1,Ntheta)
    i0 = torch.floor(grid).long() % Ntheta
    i1 = (i0 + 1) % Ntheta
    w = (grid - torch.floor(grid))  # (B,1,Ntheta)

    kde_expanded = kde.view(1, Nr, Ntheta).expand(B, Nr, Ntheta)  # (B,Nr,Ntheta)
    rotated = (1 - w) * torch.gather(kde_expanded, 2, i0.expand(B, Nr, Ntheta)) + \
               w * torch.gather(kde_expanded, 2, i1.expand(B, Nr, Ntheta))
    return rotated  # (B, Nr, Ntheta)

# -------------------------------
# Multi-start gradient descent (fully batchable)
# -------------------------------
def optimize_rotation_multistart(kde_a, kde_b, num_starts=8, lr=0.05, steps=200):
    device = kde_a.device
    theta = torch.linspace(0, 2*torch.pi, num_starts, device=device, requires_grad=True)
    theta = theta.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        rotated_b = rotate_kde_interp_batch(kde_b, theta)  # (B,Nr,Ntheta)
        losses = ((rotated_b - kde_a.unsqueeze(0))**2).sum(dim=(1,2))  # (B,)
        total_loss = losses.sum()
        total_loss.backward()
        optimizer.step()
        theta.data = theta.data % (2*torch.pi)

    # Pick best theta
    with torch.no_grad():
        rotated_b_final = rotate_kde_interp_batch(kde_b, theta)
        final_losses = ((rotated_b_final - kde_a.unsqueeze(0))**2).sum(dim=(1,2))
        best_idx = torch.argmin(final_losses)
        best_theta = theta[best_idx].item()
        best_loss = final_losses[best_idx].item()

    return best_theta, best_loss

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Random point cloud
 #   N = 50000
   # points_a = torch.randn(N, 2, device=device)

    # Rotate by known angle
 #   theta_true = torch.tensor(torch.pi / 3, device=device)  # 60 degrees
 #   c, s = torch.cos(theta_true), torch.sin(theta_true)
   # R_true = torch.tensor([[c, -s], [s, c]], device=device)

    B = 1
    N = 100000
    err = 0.05

    Xss, Yss, inv_perms, mats, types = CloudGen.make_batch_cpu(B, N, 2, err, 0.0, mode='rotation',
                                                               generationmode='box')
    R_true = torch.from_numpy(mats[0]).float()
    points_a = torch.from_numpy(Xss).to("cuda").float()[0]
    points_b = torch.from_numpy(Yss).to("cuda").float()[0]

    print("Initial rotation matrix (R_true):\n", R_true)

    theta_rad = torch.atan2(R_true[1, 0], R_true[0, 0])  # in [-π, π]
    theta_rad_0_2pi = theta_rad % (2 * torch.pi)  # map to [0, 2π]

    theta_deg = torch.rad2deg(theta_rad_0_2pi)  # in [0, 360°]

    print("Rotation angle (deg):", theta_deg.item())

    print("Rotation angle initial (deg):", theta_deg.item())

   # points_b = points_a @ R_true.T



    # Shuffle points
    perm = torch.randperm(N, device=device)
    points_b = points_b[perm]

    # Compute polar KDEs
    # You can increase precision by increasing Nr (radial bins) and Ntheta (angular bins),
    # and using smaller sigma_r (radial bandwidth) and larger kappa_theta (angular concentration)
    start_time = time.time()

    # Compute KDEs with higher precision
    kde_a = polar_kde_gpu(points_a, Nr=256, Ntheta=720, sigma_r=0.1, kappa_theta=20.0)
    kde_b = polar_kde_gpu(points_b, Nr=256, Ntheta=720, sigma_r=0.1, kappa_theta=20.0)

    # Optimize rotation
    theta_opt, final_loss = optimize_rotation_multistart(
        kde_a, kde_b, num_starts=360, lr=0.1, steps=50
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Block execution time: {elapsed:.2f} seconds")

    # Compute final rotation matrix from estimated angle
    c_opt, s_opt = torch.cos(torch.tensor(theta_opt)), torch.sin(torch.tensor(theta_opt))
    R_opt = torch.tensor([[c_opt, -s_opt], [s_opt, c_opt]], device=device)
    print("Final rotation matrix (R_opt):\n", R_opt)

    # Print results
  #  print("True rotation (deg):", torch.rad2deg(theta_true).item())
    print("Estimated rotation (deg):", torch.rad2deg(torch.tensor(theta_opt)).item())
    print("Final L2 loss:", final_loss)