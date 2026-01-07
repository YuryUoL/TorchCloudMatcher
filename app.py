import streamlit as st

st.set_page_config(page_title="Point Cloud App", layout="centered")

st.title("Point Cloud Application")

# Short description
st.markdown("""
Welcome to the **Point Cloud Application**!  

This program allows you to work with **2D and 3D point clouds** and find an **optimal rotation matrix** that aligns two unorganized point clouds.  
Good parameters are already set for testing, so you can quickly explore the functionality.  

⚠️ **Note on reflection mode:**  
When generating random point clouds in reflection mode, the app will **not search for an optimal reflection matrix**. Instead, it will still compute the **optimal rotation matrix**. This is an **intentional feature, not a bug**.  

Select a mode below to get started.
""")

# Collapsible detailed description
with st.expander("Show detailed algorithm description"):
    st.markdown("""
### Detailed Algorithm Description: Soft OT-Procrustes Point Cloud Alignment

This algorithm aligns two point clouds, **A** and **B**, in dimension \(N\), finding an optimal **rotation matrix**, **soft point correspondences**, and **hard matching**, while minimizing a soft Earth Mover’s Distance (EMD)-type cost.  

The process is carefully designed to be robust even when point correspondences are initially unknown.

---

#### Step 1: Centering the Point Clouds
- Compute the centroid of each cloud.  
- Subtract the centroid from all points so that both clouds are **centered at the origin**.  
- This ensures that alignment focuses only on **rotation**, not translation.

---

#### Step 2: Selecting Candidate Subspaces

**For cloud A:**
1. Generate **all ordered permutations of \(N-1\) points** from A.  
2. For each subset of \(N-1\) points, compute the **(N−1)-dimensional volume** they span using the Gram determinant:
   \[
   \text{volume} = \sqrt{\det(V^T V)}
   \]  
   where \(V\) is the matrix of the selected vectors.  
3. **Select the subset of A with the maximum volume**.  
   - This subset provides the most informative vectors for determining rotation, avoiding degenerate or nearly collinear configurations.  
   - Denote these points as \(x\) (size: \(N-1 \times N\)).

**For cloud B:**
- Generate **all ordered permutations of \(N-1\) points**.  
- Each permutation will serve as a **candidate subset** for alignment with the N−1 points of A.

---

#### Step 3: Generating Initial Rotation Hypotheses
- For each candidate subset of **N−1 points from B**, compute a **rotation matrix \(R\)** such that:  
  \[
  R \cdot x \approx Y_{\text{candidate}}
  \]  
  - Here, \(x\) are the **N−1 points from A** selected in Step 2.  
  - \(Y_{\text{candidate}}\) are the **N−1 points from the current candidate subset of B**.  
  - This ensures a **point-to-point alignment** of the N−1 vectors between A and B.
- Compute \(R\) using **Orthogonal Procrustes**:
  1. Covariance: \(M = Y_{\text{candidate}}^T x\)  
  2. SVD: \(M = U \Sigma V^T\)  
  3. Rotation: \(R = U V^T\)  
  4. If \(\det(R) < 0\), flip the last column of \(U\) to enforce a **proper rotation** (no reflection).  
- Store all rotation matrices as **initial hypotheses**.

---

#### Step 4: OT-Procrustes Alternating Optimization
- For each rotation hypothesis:
  1. Apply the rotation to **all points in A**.  
  2. Compute a **soft point correspondence** between rotated A and B using the **log-Sinkhorn algorithm**, which approximates the **Earth Mover’s Distance (EMD)**.  
  3. Update the rotation using a **weighted Orthogonal Procrustes solution** based on the soft correspondences.  
- Repeat this **alternating optimization** for several iterations to jointly refine the rotation and the soft correspondences.  
- This ensures both rotation and point matching are optimized in a **probabilistic sense**.

---

#### Step 5: Selecting the Best Candidate
- For each candidate subset of B and its refined rotation, compute the **final soft EMD distance**.  
- Select the candidate subset and rotation that **minimizes this final distance**.  

**Outputs include:**
1. **Soft matching matrix \(G\)**: NxN probabilistic correspondences between points of A and B.  
2. **Hard matching (permutation)**: Obtained from \(G\) via the **Hungarian algorithm**.  
3. **Bottleneck distance**: Maximum point-wise displacement according to the hard matching.  
4. **Rotation matrix \(R\)**: Final rotation aligning A to B.

---

### Summary
- The algorithm uses the **most informative N−1 points from A** (largest volume subset) and considers **all N−1 subsets of B**.  
- It generates rotation hypotheses, refines them with **alternating OT-Procrustes optimization**, and selects the combination that minimizes the **soft EMD**.  
- This approach provides **robust alignment, rotation, and matching** of high-dimensional point clouds even when point correspondences are initially unknown.
""")

# Buttons to select dimensionality
col1, col2 = st.columns(2)

with col1:
    if st.button("2D Point Clouds"):
        st.switch_page("pages/app_2d.py")
        #st.switch_page("pages/app_2d.py")

with col2:
    if st.button("3D Point Clouds"):
        st.switch_page("pages/app_3d.py")
       # st.switch_page("pages/app_3d.py")
