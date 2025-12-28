import streamlit as st

st.set_page_config(page_title="Point Cloud App", layout="centered")

st.title("Point Cloud Application")

# Description about the program
st.markdown("""
Welcome to the **Point Cloud Application**!  

This program allows you to work with **2D and 3D point clouds** and find an **optimal rotation matrix** that aligns two unorganized point clouds.  
Good parameters are already set for testing, so you can quickly explore the functionality.  

⚠️ **Note on reflection mode:**  
When generating random point clouds in reflection mode, the app will **not search for an optimal reflection matrix**. Instead, it will still compute the **optimal rotation matrix**. This is an **intentional feature, not a bug**.  

Select a mode below to get started.
""")

# Buttons to select dimensionality
col1, col2 = st.columns(2)

with col1:
    if st.button("2D Point Clouds"):
        st.switch_page("pages/app_2d.py")

with col2:
    if st.button("3D Point Clouds"):
        st.switch_page("pages/app_3d.py")
