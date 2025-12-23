import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time

def plot_clouds(XA, YA, XB, YB, labels_A, labels_B, placeholder):
    XA, YA, XB, YB = pd.Series(XA), pd.Series(YA), pd.Series(XB), pd.Series(YB)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=XA, y=YA, mode="markers+text",
                             text=labels_A, textposition="top center",
                             marker=dict(size=10, color="red"), name="Cloud A"))
    fig.add_trace(go.Scatter(x=XB, y=YB, mode="markers+text",
                             text=labels_B, textposition="bottom center",
                             marker=dict(size=10, color="blue"), name="Cloud B"))

    all_x = pd.concat([XA, XB])
    all_y = pd.concat([YA, YB])
    dx = all_x.max() - all_x.min() if all_x.max() != all_x.min() else 1.0
    dy = all_y.max() - all_y.min() if all_y.max() != all_y.min() else 1.0
    margin = 0.2
    fig.update_xaxes(range=[all_x.min() - margin*dx, all_x.max() + margin*dx], showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(range=[all_y.min() - margin*dy, all_y.max() + margin*dy], showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_layout(title="Point Clouds", width=800, height=700, plot_bgcolor="white")

    placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_{time.time_ns()}")


def plot_clouds_with_matches(XA, YA, XB, YB, labels_A, labels_B, tmap, placeholder):
    XA, YA, XB, YB = np.asarray(XA), np.asarray(YA), np.asarray(XB), np.asarray(YB)
    tmap = np.asarray(tmap).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=XA, y=YA, mode="markers+text",
                             text=labels_A, textposition="top center",
                             marker=dict(size=10, color="red"), name="Cloud A"))
    fig.add_trace(go.Scatter(x=XB, y=YB, mode="markers+text",
                             text=labels_B, textposition="bottom center",
                             marker=dict(size=10, color="blue"), name="Cloud B"))

    for i in range(len(XA)):
        if i < len(tmap):
            j = tmap[i]
            if 0 <= j < len(XB):
                fig.add_trace(go.Scatter(
                    x=[XA[i], XB[j]], y=[YA[i], YB[j]],
                    mode="lines", line=dict(width=3, color="gray"), showlegend=False
                ))

    all_x = np.concatenate([XA, XB])
    all_y = np.concatenate([YA, YB])
    dx = all_x.max() - all_x.min() if all_x.max() != all_x.min() else 1.0
    dy = all_y.max() - all_y.min() if all_y.max() != all_y.min() else 1.0
    margin = 0.2
    fig.update_xaxes(range=[all_x.min() - margin*dx, all_x.max() + margin*dx], showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(range=[all_y.min() - margin*dy, all_y.max() + margin*dy], showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_layout(title="Point Clouds + Matches", width=800, height=700, plot_bgcolor="white")

    placeholder.plotly_chart(fig, use_container_width=True, key=f"plot_match_{time.time_ns()}")


def build_animation_frames(A_np, B_np, A2, B2, A_final, theta_end, labels_A, labels_B):
    frames = []
    trans_frames = 30
    rot_frames = 30

    def mk_frame(A_pts, B_pts, name):
        return go.Frame(
            data=[
                go.Scatter(x=A_pts[:, 0], y=A_pts[:, 1], mode="markers+text",
                           text=labels_A, textposition="top center",
                           marker=dict(size=10, color="red")),
                go.Scatter(x=B_pts[:, 0], y=B_pts[:, 1], mode="markers+text",
                           text=labels_B, textposition="bottom center",
                           marker=dict(size=10, color="blue"))
            ],
            name=name
        )

    for i in range(trans_frames):
        alpha = (i + 1) / trans_frames
        A_step = (1 - alpha) * A_np + alpha * A2
        B_step = (1 - alpha) * B_np + alpha * B2
        frames.append(mk_frame(A_step, B_step, f"trans_{i}"))

    for i in range(rot_frames):
        t = (i + 1) / rot_frames
        theta = t * theta_end
        c, s = np.cos(theta), np.sin(theta)
        R_t = np.array([[c, -s], [s, c]])
        A_rot_step = A2 @ R_t
        frames.append(mk_frame(A_rot_step, B2, f"rot_{i}"))

    return frames
