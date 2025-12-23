import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from scipy.spatial.transform import Rotation as R_scipy, Slerp

def apply_3d_layout(fig, title=None):
    fig.update_layout(
        title=title,
        height=900,
        margin=dict(l=0, r=0, t=40 if title else 0, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        uirevision="lock_camera"
    )

def plot_clouds_3d(XA, YA, ZA, XB, YB, ZB, labels_A, labels_B, placeholder):
    XA, YA, ZA = pd.Series(XA), pd.Series(YA), pd.Series(ZA)
    XB, YB, ZB = pd.Series(XB), pd.Series(YB), pd.Series(ZB)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=XA, y=YA, z=ZA,
        mode="markers+text",
        text=labels_A, textposition="top center",
        marker=dict(size=5, color="red"),
        name="Cloud A"
    ))
    fig.add_trace(go.Scatter3d(
        x=XB, y=YB, z=ZB,
        mode="markers+text",
        text=labels_B, textposition="bottom center",
        marker=dict(size=5, color="blue"),
        name="Cloud B"
    ))
    apply_3d_layout(fig, title="Point Clouds (3D)")
    placeholder.plotly_chart(fig, use_container_width=True, key=f"plot3d_{time.time_ns()}")

def plot_clouds_with_matches_3d(XA, YA, ZA, XB, YB, ZB, labels_A, labels_B, tmap, placeholder):
    XA, YA, ZA = np.asarray(XA), np.asarray(YA), np.asarray(ZA)
    XB, YB, ZB = np.asarray(XB), np.asarray(YB), np.asarray(ZB)
    tmap = np.asarray(tmap).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=XA, y=YA, z=ZA,
        mode="markers+text",
        text=labels_A, textposition="top center",
        marker=dict(size=5, color="red"),
        name="Cloud A"
    ))
    fig.add_trace(go.Scatter3d(
        x=XB, y=YB, z=ZB,
        mode="markers+text",
        text=labels_B, textposition="bottom center",
        marker=dict(size=5, color="blue"),
        name="Cloud B"
    ))

    # Draw matches
    for i, j in enumerate(tmap):
        if 0 <= j < len(XB):
            fig.add_trace(go.Scatter3d(
                x=[XA[i], XB[j]],
                y=[YA[i], YB[j]],
                z=[ZA[i], ZB[j]],
                mode="lines",
                line=dict(width=3, color="gray"),
                showlegend=False
            ))
    apply_3d_layout(fig, title="Point Clouds with Matches (3D)")
    placeholder.plotly_chart(fig, use_container_width=True, key=f"plot3d_match_{time.time_ns()}")

# ---------------------------------------
# Build 3D animation frames (translation + rotation)
# ---------------------------------------
def build_animation_frames_3d(
    A_np, B_np, A2, B2, R, labels_A, labels_B,
    trans_frames=30, rot_frames=30
):
    frames = []

    def mk_frame(A_pts, B_pts, name):
        return go.Frame(
            data=[
                go.Scatter3d(
                    x=A_pts[:, 0], y=A_pts[:, 1], z=A_pts[:, 2],
                    mode="markers+text",
                    text=labels_A,
                    textposition="top center",
                    marker=dict(size=5, color="red")
                ),
                go.Scatter3d(
                    x=B_pts[:, 0], y=B_pts[:, 1], z=B_pts[:, 2],
                    mode="markers+text",
                    text=labels_B,
                    textposition="bottom center",
                    marker=dict(size=5, color="blue")
                )
            ],
            name=name
        )

    # -----------------------
    # 1) Translation frames
    # -----------------------
    for i in range(trans_frames):
        alpha = (i + 1) / trans_frames
        A_step = (1 - alpha) * A_np + alpha * A2
        B_step = (1 - alpha) * B_np + alpha * B2
        frames.append(mk_frame(A_step, B_step, f"trans_{i}"))

    # -----------------------
    # 2) Rotation frames (SLERP)
    # -----------------------
    key_times = [0.0, 1.0]
    key_rots = R_scipy.from_matrix([np.eye(3), R])
    slerp = Slerp(key_times, key_rots)

    for i in range(rot_frames):
        t = (i + 1) / rot_frames
        R_t = slerp([t]).as_matrix()[0]
        A_rot_step = A2 @ R_t.T   # IMPORTANT: post-multiply
        frames.append(mk_frame(A_rot_step, B2, f"rot_{i}"))

    return frames