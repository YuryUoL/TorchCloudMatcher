import streamlit as st
import pandas as pd
import numpy as np
import time

def initialize_session(default_A, default_B):
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


def compact_json(obj):
    import json
    return json.dumps(obj, separators=(',', ':'))


def handle_table_editing():
    st.subheader("Editable Tables")
    A = st.data_editor(st.session_state["cloud_A"], num_rows="dynamic",
                       height=300, key=st.session_state["cloud_A_editor_key"])
    B = st.data_editor(st.session_state["cloud_B"], num_rows="dynamic",
                       height=300, key=st.session_state["cloud_B_editor_key"])

    st.session_state["cloud_A"] = A
    st.session_state["cloud_B"] = B
    return A, B
