import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import json
import plotly.graph_objects as go
from utils.Cloudgen_2D import make_batch_cpu
from core.compute_2d import compute_isometry
from plotting.plot_2d import plot_clouds, plot_clouds_with_matches, build_animation_frames
from ui.controls import initialize_session, handle_table_editing
from ui.app_nd import display_compute_output

st.set_page_config(layout="wide")

# --- Navigation ---
nav_col1, nav_col2 = st.columns([1, 9])
with nav_col1:
    if st.button("⬅ Main menu"):
        st.switch_page("app.py")

st.title("Point Cloud Editor and Plotter")

# ---------------- Default Clouds ----------------
default_A = pd.DataFrame({"L": ["A1","A2","A3"], "X":[0,1,2], "Y":[0,1,0.5]})
default_B = pd.DataFrame({"L": ["B1","B2","B3"], "X":[1,2,3], "Y":[2,1.5,1]})

initialize_session(default_A, default_B)

# ---------------- CSV Uploads ----------------
col_up1, col_up2 = st.columns(2)
file_A = col_up1.file_uploader("Upload CSV for Cloud A", type=["csv"], key="uploader_A")
file_B = col_up2.file_uploader("Upload CSV for Cloud B", type=["csv"], key="uploader_B")

for col, file, cloud_name, key_name in [(col_up1, file_A, "cloud_A", "cloud_A_editor_key"),
                                        (col_up2, file_B, "cloud_B", "cloud_B_editor_key")]:
    if col.button(f"Load CSV → Table {cloud_name[-1]}"):
        if file is None:
            st.warning(f"No file uploaded for {cloud_name[-1]}.")
        else:
            try:
                df = pd.read_csv(file)
                if all(c in df.columns for c in ["L","X","Y"]):
                    st.session_state[cloud_name] = df.copy()
                    st.session_state[key_name] = str(time.time_ns())
                    st.success(f"Cloud {cloud_name[-1]} loaded into table.")
                else:
                    st.error("CSV must contain columns: L, X, Y")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

# ---------------- Random Cloud Generator ----------------
st.subheader("Random Cloud Generator")
col_rng1, col_rng2, col_rng3,col_rng4, col_rng_btn = st.columns([1,1,1,1,1])
N = col_rng1.number_input("N (number of points)", 1, 20000, 10)
eps = col_rng2.number_input("eps (float)", 0.0, 10.0, 0.01, step=0.001, format="%.6f")
sep = col_rng3.number_input("sep (float)", 0.0, 2.0, 1.5, step=0.001)
mode = col_rng4.selectbox(
    "Transform",
    options=["rotation", "reflection"],
    index=0  # default = rotation
)

generate_clicked = col_rng_btn.button("Generate Random Cloud")

if generate_clicked:
    Xs, Ys, _, _, _ = make_batch_cpu(B=1, N=N, K=2, eps=eps, delta=0, mode=mode, generationmode="box")
    X = np.asarray(Xs[0]) - np.array([sep/2,0])
    Y = np.asarray(Ys[0]) + np.array([sep/2,0])
    st.session_state["cloud_A"] = pd.DataFrame({"L":[f"A{i}" for i in range(len(X))], "X":X[:,0], "Y":X[:,1]})
    st.session_state["cloud_B"] = pd.DataFrame({"L":[f"B{i}" for i in range(len(Y))], "X":Y[:,0], "Y":Y[:,1]})
    st.session_state["cloud_A_editor_key"] = str(time.time_ns())
    st.session_state["cloud_B_editor_key"] = str(time.time_ns())
    st.session_state["compute_output"] = None
    st.success("Random point cloud generated!")

# ---------------- Tables + Controls ----------------
col_left, col_right = st.columns([1,2])
with col_left:
    A, B = handle_table_editing()
    st.markdown("### Compute parameters")
    maxchunksize = st.number_input("maxchunksize",1,200000,200)
    sinkhorn_iters = st.number_input("sinkhorn_iters",1,10000,50)
    rot_iters = st.number_input("rot_iters",1,10000,2)
    reg = st.number_input("reg (float)",1e-9,10.0,1e-3,format="%.6g")

    col1, col2, col3, col4 = st.columns(4)
    plot_clicked = col1.button("Plot Point Clouds")
    center_clicked = col2.button("Center Clouds")
    compute_clicked = col3.button("Compute Isometry")
    show_conn_clicked = col4.button("Show Connections")

with col_right:
    plot_placeholder = st.empty()
    output_placeholder = st.empty()


# ---------------- Button Actions ----------------
if plot_clicked:
    plot_clouds(A["X"], A["Y"], B["X"], B["Y"], A["L"], B["L"], plot_placeholder)

if center_clicked:
    XA_orig, YA_orig = A["X"].copy(), A["Y"].copy()
    XB_orig, YB_orig = B["X"].copy(), B["Y"].copy()
    cxA, cyA = XA_orig.mean(), YA_orig.mean()
    cxB, cyB = XB_orig.mean(), YB_orig.mean()
    frames_anim = 30
    for i in range(frames_anim):
        t = (i+1)/frames_anim
        plot_clouds(XA_orig - t*cxA, YA_orig - t*cyA, XB_orig - t*cxB, YB_orig - t*cyB, A["L"], B["L"], plot_placeholder)
        time.sleep(0.02)

if compute_clicked:
    A_np = st.session_state["cloud_A"][["X","Y"]].values
    B_np = st.session_state["cloud_B"][["X","Y"]].values

    if A_np.shape[0]<1 or B_np.shape[0]<1:
        st.error("Both clouds must have at least one point.")
    else:
        with st.spinner("Computing isometry…"):
            try:
                out = compute_isometry(A_np, B_np, maxchunksize, sinkhorn_iters, rot_iters, reg)
            except Exception as e:
                st.exception(f"Compute failed: {e}")
                raise

        # --- Build animation frames ---
        frames = build_animation_frames(A_np, B_np, out["A2"], out["B2"], out["A_final"], out["theta_end"], A["L"], B["L"])
        initial_data = frames[0].data if len(frames)>0 else []
        fig = go.Figure(data=initial_data, frames=frames,
                        layout=go.Layout(title="Point Clouds Animation",
                                         width=900,height=800,
                                         plot_bgcolor="white",
                                         updatemenus=[dict(type="buttons", showactive=False,
                                                           buttons=[dict(label="Play", method="animate",
                                                                         args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode="immediate")]),
                                                                    dict(label="Pause", method="animate",
                                                                         args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])],
                                                           x=0.1,y=1.1)]))
        plot_placeholder.plotly_chart(fig, use_container_width=True, config={"displayModeBar":True})

        # Show JSON outputs
        display_compute_output(out)

        # Save output to session
        st.session_state["compute_output"] = out

if show_conn_clicked:
    if st.session_state.get("compute_output") is None:
        st.warning("Run Compute Isometry first before showing connections.")
    else:
        out = st.session_state["compute_output"]
        A_final = out["A_final"]
        B2 = out["B2"]
        tmap = np.asarray(out["tmap"]).astype(int)

        # Pass the placeholder explicitly
        plot_clouds_with_matches(
            A_final[:,0], A_final[:,1],
            B2[:,0], B2[:,1],
            st.session_state["cloud_A"]["L"],
            st.session_state["cloud_B"]["L"],
            tmap,
            plot_placeholder  # <- this was missing
        )

        # Redisplay JSON outputs
        display_compute_output(out)

