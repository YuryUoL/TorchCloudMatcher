import streamlit as st

st.set_page_config(page_title="Point Cloud App", layout="centered")

st.title("Point Cloud Application")
st.markdown("Choose dimensionality:")

col1, col2 = st.columns(2)

with col1:
    if st.button("2D Point Clouds"):
        st.switch_page("pages/app_2d.py")

with col2:
    if st.button("3D Point Clouds"):
        st.switch_page("pages/app_3d.py")
