import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import gc
import io
import json

from TorchRotFinder import ComputeIsometryWithMatchingnD
from Cloudgen_2D import make_batch_cpu

st.set_page_config(layout="wide")
st.title("Point Cloud Editor and Plotter")

# ---------------- Default Data ----------------
default_A = pd.DataFrame({
    "L": ["A1", "A2", "A3"],
    "X": [0.0, 1.0, 2.0],
    "Y": [0.0, 1.0, 0.5]
})

default_B = pd.DataFrame({
    "L": ["B1", "B2", "B3"],
    "X": [1.0, 2.0, 3.0],
    "Y": [2.0, 1.5, 1.0]
})

# ---------------- Initialize session_state ----------------
if "cloud_A" not in st.session_state:
    st.session_state["cloud_A"] = default_A.copy()
if "cloud_B" not in st.session_state:
    st.session_state["cloud_B"] = default_B.copy()

if "cloud_A_editor_key" not in st.session_state:
    st.session_state["cloud_A_editor_key"] = "cloud_A_editor_initial"
if "cloud_B_editor_key" not in st.session_state:
    st.session_state["cloud_B_editor_key"] = "cloud_B_editor_initial"

if "compute_output" not in st.session_state:
    st.session_state["compute_output"] = None

# ---------------- CSV Uploads ----------------
col_up1, col_up2 = st.columns(2)
with col_up1:
    file_A = st.file_uploader("Upload CSV for Cloud A", type=["csv"], key="uploader_A")
    if file_A is not None:
        st.write("Uploaded:", file_A.name)
    if st.button("Load CSV → Table A"):
        if file_A is None:
            st.warning("No file uploaded for A.")
        else:
            try:
                A_temp = pd.read_csv(file_A)
                if all(col in A_temp.columns for col in ["L", "X", "Y"]):
                    st.session_state["cloud_A"] = A_temp.copy()
                    st.session_state["cloud_A_editor_key"] = str(time.time_ns())
                    st.success("Cloud A loaded into table.")
                else:
                    st.error("CSV A must contain columns: L, X, Y")
            except Exception as e:
                st.error(f"Failed to read CSV A: {e}")

with col_up2:
    file_B = st.file_uploader("Upload CSV for Cloud B", type=["csv"], key="uploader_B")
    if file_B is not None:
        st.write("Uploaded:", file_B.name)
    if st.button("Load CSV → Table B"):
        if file_B is None:
            st.warning("No file uploaded for B.")
        else:
            try:
                B_temp = pd.read_csv(file_B)
                if all(col in B_temp.columns for col in ["L", "X", "Y"]):
                    st.session_state["cloud_B"] = B_temp.copy()
                    st.session_state["cloud_B_editor_key"] = str(time.time_ns())
                    st.success("Cloud B loaded into table.")
                else:
                    st.error("CSV B must contain columns: L, X, Y")
            except Exception as e:
                st.error(f"Failed to read CSV B: {e}")

# ---------------- Random Point Cloud Generator ----------------
st.subheader("Random Cloud Generator (From Cloudgen_2D)")
col_rng1, col_rng2, col_rng3, col_rng_btn = st.columns([1, 1, 1, 1])

with col_rng1:
    N = st.number_input("N (number of points)", min_value=1, max_value=20000,
                        value=10, step=1)
with col_rng2:
    eps = st.number_input("eps (float)", min_value=0.0, max_value=10.0,
                          value=0.01, step=0.001)
with col_rng3:
    sep = st.number_input("sep (float)", min_value=0.0, max_value=2.0,
                          value=1.5, step=0.001)
with col_rng_btn:
    generate_clicked = st.button("Generate Random Cloud")

if generate_clicked:
    st.write("Generating random point cloud…")
    Xs, Ys, inv_perms, mats, types = make_batch_cpu(
        B=1, N=N, K=2, eps=eps, delta=0,
        mode="rotation", generationmode="box"
    )
    X = np.asarray(Xs[0])
    Y = np.asarray(Ys[0])

    X = X - np.array([(1/2) * sep, 0])
    Y = Y + np.array([(1/2) * sep, 0])

    dfA = pd.DataFrame({
        "L": [f"A{i}" for i in range(len(X))],
        "X": X[:, 0],
        "Y": X[:, 1]
    })
    dfB = pd.DataFrame({
        "L": [f"B{i}" for i in range(len(Y))],
        "X": Y[:, 0],
        "Y": Y[:, 1]
    })

    # Full session reset
    st.session_state["cloud_A"] = dfA
    st.session_state["cloud_B"] = dfB
    st.session_state["compute_output"] = None

    # Generate fresh keys for data editors to reset widget state
    st.session_state["cloud_A_editor_key"] = str(time.time_ns())
    st.session_state["cloud_B_editor_key"] = str(time.time_ns())

    st.success("Random point cloud generated!")

# ---------------- Tables + Controls ----------------
col_left, col_right = st.columns([1, 2])
with col_left:
    st.subheader("Editable Tables")
    # Data editors with unique keys
    A = st.data_editor(st.session_state["cloud_A"], num_rows="dynamic",
                       height=300, key=st.session_state["cloud_A_editor_key"])
    B = st.data_editor(st.session_state["cloud_B"], num_rows="dynamic",
                       height=300, key=st.session_state["cloud_B_editor_key"])

    # Save edits back to session state
    st.session_state["cloud_A"] = A
    st.session_state["cloud_B"] = B

    # Expose algorithm parameters so user can change them pre-compute
    st.markdown("### Compute parameters")
    maxchunksize = st.number_input("maxchunksize", min_value=1, max_value=200000, value=200, step=1)
    sinkhorn_iters = st.number_input("sinkhorn_iters", min_value=1, max_value=10000, value=50, step=1)
    reg = st.number_input("reg (float)", min_value=1e-9, max_value=10.0, value=1e-3, format="%.6g")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        plot_clicked = st.button("Plot Point Clouds")

    with col2:
        center_clicked = st.button("Center Clouds")

    with col3:
        compute_clicked = st.button("Compute Isometry")

    with col4:
        show_conn_clicked = st.button("Show Connections")

with col_right:
    plot_placeholder = st.empty()
    output_placeholder = st.empty()

# ---------------- Plotting helper functions ----------------
def plot_clouds(XA, YA, XB, YB, labels_A, labels_B):
    XA = pd.Series(XA)
    YA = pd.Series(YA)
    XB = pd.Series(XB)
    YB = pd.Series(YB)

    fig = go.Figure()
    # Cloud A above
    fig.add_trace(go.Scatter(
        x=XA, y=YA, mode="markers+text",
        text=labels_A, textposition="top center",
        marker=dict(size=10, color="red"), name="Cloud A"
    ))
    # Cloud B below
    fig.add_trace(go.Scatter(
        x=XB, y=YB, mode="markers+text",
        text=labels_B, textposition="bottom center",
        marker=dict(size=10, color="blue"), name="Cloud B"
    ))

    all_x = pd.concat([XA, XB])
    all_y = pd.concat([YA, YB])
    dx = all_x.max() - all_x.min() if all_x.max() != all_x.min() else 1.0
    dy = all_y.max() - all_y.min() if all_y.max() != all_y.min() else 1.0
    margin = 0.2

    fig.update_xaxes(range=[all_x.min() - margin*dx, all_x.max() + margin*dx],
                     showgrid=True, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(range=[all_y.min() - margin*dy, all_y.max() + margin*dy],
                     showgrid=True, gridcolor="lightgray", zeroline=False)

    fig.update_layout(title="Point Clouds", width=800, height=700,
                      plot_bgcolor="white")

    plot_placeholder.plotly_chart(fig,
        use_container_width=True,
        key=f"plot_{time.time_ns()}"
    )

def plot_clouds_with_matches(XA, YA, XB, YB, labels_A, labels_B, tmap):
    XA = np.asarray(XA)
    YA = np.asarray(YA)
    XB = np.asarray(XB)
    YB = np.asarray(YB)
    tmap = np.asarray(tmap).astype(int)

    fig = go.Figure()

    # Cloud A
    fig.add_trace(go.Scatter(
        x=XA, y=YA, mode="markers+text",
        text=labels_A, textposition="top center",
        marker=dict(size=10, color="red"),
        name="Cloud A"
    ))

    # Cloud B
    fig.add_trace(go.Scatter(
        x=XB, y=YB, mode="markers+text",
        text=labels_B, textposition="bottom center",
        marker=dict(size=10, color="blue"),
        name="Cloud B"
    ))

    # --- thin match lines ---
    nA = len(XA)
    for i in range(nA):
        if i < len(tmap):
            j = int(tmap[i])
            if 0 <= j < len(XB):
                fig.add_trace(go.Scatter(
                    x=[XA[i], XB[j]],
                    y=[YA[i], YB[j]],
                    mode="lines",
                    line=dict(width=3, color="gray"),
                    showlegend=False
                ))

    # axis ranges
    all_x = np.concatenate([XA, XB])
    all_y = np.concatenate([YA, YB])
    dx = all_x.max() - all_x.min() if all_x.max() != all_x.min() else 1.0
    dy = all_y.max() - all_y.min() if all_y.max() != all_y.min() else 1.0
    margin = 0.2

    fig.update_xaxes(
        range=[all_x.min() - margin*dx, all_x.max() + margin*dx],
        showgrid=True, gridcolor="lightgray", zeroline=False
    )
    fig.update_yaxes(
        range=[all_y.min() - margin*dy, all_y.max() + margin*dy],
        showgrid=True, gridcolor="lightgray", zeroline=False
    )

    fig.update_layout(
        title="Point Clouds + Matches",
        width=800, height=700,
        plot_bgcolor="white"
    )

    plot_placeholder.plotly_chart(fig,
        use_container_width=True,
        key=f"plot_match_{time.time_ns()}"
    )

# ---------------- Button Actions ----------------
if plot_clicked:
    plot_clouds(A["X"], A["Y"], B["X"], B["Y"], A["L"], B["L"])

if center_clicked:
    XA_orig, YA_orig = A["X"].copy(), A["Y"].copy()
    XB_orig, YB_orig = B["X"].copy(), B["Y"].copy()

    cxA, cyA = XA_orig.mean(), YA_orig.mean()
    cxB, cyB = XB_orig.mean(), YB_orig.mean()

    frames_anim = 30
    for i in range(frames_anim):
        t = (i + 1) / frames_anim
        XA = XA_orig - t * cxA
        YA = YA_orig - t * cyA
        XB = XB_orig - t * cxB
        YB = YB_orig - t * cyB
        plot_clouds(XA, YA, XB, YB, A["L"], B["L"])
        time.sleep(0.02)

# Helper: compact JSON string (no spaces)
def compact_json(obj):
    return json.dumps(obj, separators=(',', ':'))

if compute_clicked:
    # raw clouds
    A_np = st.session_state["cloud_A"][["X", "Y"]].values  # (N,2)
    B_np = st.session_state["cloud_B"][["X", "Y"]].values  # (M,2)

    # Validate sizes
    if A_np.shape[0] < 1 or B_np.shape[0] < 1:
        st.error("Both clouds must have at least one point.")
    else:
        # Compute isometry with user-specified parameters
        with st.spinner("Computing isometry (this may take a while)..."):
            try:
                A2, B2, min_val, R, G, tmap = ComputeIsometryWithMatchingnD(
                    A_np, B_np, maxchunksize=int(maxchunksize),
                    sinkhorn_iters=int(sinkhorn_iters), reg=float(reg)
                )
            except Exception as e:
                st.exception(f"ComputeIsometryWithMatchingnD failed: {e}")
                raise

        # sanitize outputs
        A2 = np.asarray(A2)
        B2 = np.asarray(B2)
        R = np.asarray(R)
        tmap = np.asarray(tmap).astype(int).flatten()

        # Compute final rotated A (apply R^T to A2)
        R_end = R.T
        theta_end = np.arctan2(R_end[1, 0], R_end[0, 0])
        A_final = (A2 @ R_end)

        # --- Build animation frames (translation -> rotation -> final matches) ---
        trans_frames = 30
        rot_frames = 30
        frames = []

        def mk_frame(A_pts, B_pts, name):
            return go.Frame(
                data=[
                    go.Scatter(x=A_pts[:, 0], y=A_pts[:, 1], mode="markers+text",
                               text=A["L"], textposition="top center",
                               marker=dict(size=10, color="red")),
                    go.Scatter(x=B_pts[:, 0], y=B_pts[:, 1], mode="markers+text",
                               text=B["L"], textposition="bottom center",
                               marker=dict(size=10, color="blue"))
                ],
                name=name
            )

        # Translation frames: gradually morph original -> A2/B2
        for i in range(trans_frames):
            alpha = (i + 1) / trans_frames
            A_step = (1 - alpha) * A_np + alpha * A2
            B_step = (1 - alpha) * B_np + alpha * B2
            frames.append(mk_frame(A_step, B_step, f"trans_{i}"))

        # Rotation frames: rotate A2 towards A_final
        for i in range(rot_frames):
            t = (i + 1) / rot_frames
            theta = t * theta_end
            c, s = np.cos(theta), np.sin(theta)
            R_t = np.array([[c, -s], [s, c]])
            A_rot_step = A2 @ R_t
            frames.append(mk_frame(A_rot_step, B2, f"rot_{i}"))

        # Final match frame with lines
        match_traces = [
            go.Scatter(x=A_final[:, 0], y=A_final[:, 1], mode="markers+text",
                       text=A["L"], textposition="top center",
                       marker=dict(size=10, color="red")),
            go.Scatter(x=B2[:, 0], y=B2[:, 1], mode="markers+text",
                       text=B["L"], textposition="bottom center",
                       marker=dict(size=10, color="blue"))
        ]
        for i in range(len(A_final)):
            if i < len(tmap):
                j = int(tmap[i])
                if 0 <= j < len(B2):
                    match_traces.append(go.Scatter(
                        x=[A_final[i, 0], B2[j, 0]],
                        y=[A_final[i, 1], B2[j, 1]],
                        mode="lines",
                        line=dict(width=3, color="gray"),
                        showlegend=False
                    ))

        frames.append(go.Frame(data=match_traces, name="matches"))

        # --- Create figure with Play/Pause controls ---
        initial_data = frames[0].data if len(frames) > 0 else match_traces
        fig = go.Figure(
            data=initial_data,
            frames=frames,
            layout=go.Layout(
                title="Point Clouds Animation",
                width=900, height=800,
                plot_bgcolor="white",
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=50, redraw=True),
                                              fromcurrent=True, mode="immediate")]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode="immediate")])
                    ],
                    x=0.1, y=1.1
                )]
            )
        )

        plot_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        # Also show final static matching plot so user always sees matches
       # plot_clouds_with_matches(A_final[:, 0], A_final[:, 1], B2[:, 0], B2[:, 1], A["L"], B["L"], tmap)
        # --- Show final static connection plot only when the user clicks the button ---


        # Output results - compact JSON and downloads
        tmap_list = tmap.tolist()
        R_list = R.tolist()

        output_placeholder.markdown(f"### Computation Results  \n**Distance:** `{float(min_val):.6f}`  \n")

        st.subheader("Bijection (tmap) — mapping index i in A -> tmap[i] in B")
        st.text_area("tmap (copyable)", compact_json(tmap_list), height=140, key=f"tmap_{time.time_ns()}")
        st.download_button("Download tmap (JSON)", data=compact_json(tmap_list), file_name="tmap.json", mime="application/json")

        st.subheader("Rotation matrix R (copyable)")
        st.text_area("R (copyable)", compact_json(R_list), height=140, key=f"R_{time.time_ns()}")
        st.download_button("Download R (JSON)", data=compact_json(R_list), file_name="rotation_matrix_R.json", mime="application/json")

        # Also show G (transport plan) if present
        try:
            if G is not None:
                buf = io.BytesIO()
                np.save(buf, np.asarray(G))
                buf.seek(0)
                st.download_button("Download G (numpy .npy)", data=buf, file_name="G.npy", mime="application/octet-stream")
        except Exception:
            pass

        # Save compute output to session
        st.session_state["compute_output"] = dict(A2=A2, B2=B2, min_val=float(min_val), R=R, G=G, tmap=tmap_list)

        # Cleanup

print("ok")

# -------------------------------------------------------
# SHOW CONNECTIONS BUTTON HANDLER (outside compute block!)
# -------------------------------------------------------
if show_conn_clicked:
    if st.session_state.get("compute_output") is None:
        st.warning("Run Compute Isometry first before showing connections.")
    else:
        out = st.session_state["compute_output"]
        A2 = np.asarray(out["A2"])
        B2 = np.asarray(out["B2"])
        R = np.asarray(out["R"])
        tmap = np.asarray(out["tmap"]).astype(int)

        # Apply rotation to get final A
        A_final = A2 @ R.T

        # Plot with matches
        plot_clouds_with_matches(
            A_final[:, 0], A_final[:, 1],
            B2[:, 0], B2[:, 1],
            st.session_state["cloud_A"]["L"],
            st.session_state["cloud_B"]["L"],
            tmap
        )

