import streamlit as st
import pandas as pd
import time

def init_state_2d():
    if "cloud_A" not in st.session_state:
        st.session_state["cloud_A"] = pd.DataFrame({
            "L": ["A1", "A2", "A3"],
            "X": [0.0, 1.0, 2.0],
            "Y": [0.0, 1.0, 0.5],
        })

    if "cloud_B" not in st.session_state:
        st.session_state["cloud_B"] = pd.DataFrame({
            "L": ["B1", "B2", "B3"],
            "X": [1.0, 2.0, 3.0],
            "Y": [2.0, 1.5, 1.0],
        })

    st.session_state.setdefault("cloud_A_editor_key", "cloud_A_editor_initial")
    st.session_state.setdefault("cloud_B_editor_key", "cloud_B_editor_initial")
    st.session_state.setdefault("compute_output", None)


def reset_editors():
    st.session_state["cloud_A_editor_key"] = str(time.time_ns())
    st.session_state["cloud_B_editor_key"] = str(time.time_ns())
