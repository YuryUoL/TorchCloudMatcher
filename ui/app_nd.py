# app_core.py
import streamlit as st
import numpy as np
import io
import json


def compact_json(obj):
    """Convert arrays to lists so JSON serialization works."""
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    elif isinstance(obj, (list, tuple)):
        # recursively convert nested arrays
        obj = [x.tolist() if isinstance(x, np.ndarray) else x for x in obj]
    return json.dumps(obj, separators=(',', ':'))


def display_compute_output(out, output_placeholder=None):
    """
    Unified function to display computation outputs for 2D, 3D, or nD point clouds.
    """

    if output_placeholder is None:
        output_placeholder = st

    # Ensure JSON-serializable lists
    tmap_list = out.get("tmap", [])
    R_list = out.get("R", [])

    if isinstance(tmap_list, np.ndarray):
        tmap_list = tmap_list.tolist()
    if isinstance(R_list, np.ndarray):
        R_list = R_list.tolist()

    # Show distance metrics
    min_val = out.get("min_val", np.nan)
    bddist = out.get("bddist", np.nan)
    output_placeholder.markdown(
        f"### Computation Results\n"
        f"**Distance:** `{min_val:.6f}`  \n"
        f"**Bottleneck distance (euclidean):** `{bddist:.6f}`"
    )

    # Show tmap
    output_placeholder.subheader("Bijection (tmap)")
    output_placeholder.text_area("tmap (copyable)", compact_json(tmap_list), height=140)
    output_placeholder.download_button("Download tmap (JSON)", compact_json(tmap_list), "tmap.json")

    # Show rotation or transformation
    output_placeholder.subheader("Rotation / Transformation matrix R")
    output_placeholder.text_area("R (copyable)", compact_json(R_list), height=140)
    output_placeholder.download_button("Download R (JSON)", compact_json(R_list), "R.json")

    # Optional G output
    G = out.get("G", None)
    if G is not None:
        if isinstance(G, np.ndarray):
            G = G.tolist()  # ensure JSON-compatible if needed
        buf = io.BytesIO()
        np.save(buf, np.asarray(G))
        buf.seek(0)
        output_placeholder.download_button("Download G (numpy .npy)", buf, "G.npy")

