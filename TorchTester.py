import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import math
import gc



from TorchRotFinder import ComputeIsometryWithMatchingnD
from Cloudgen_2D import make_batch_cpu

def runFirstTest():
    N = 1000
    eps = 0.01
    Xs, Ys, inv_perms, mats, types = make_batch_cpu(
        B=1, N=N, K=2, eps=eps, delta=0,
        mode="rotation", generationmode="box"
    )
    ts = time.time()
    Anew,Bnew,min_val, rotFnew, G, t = ComputeIsometryWithMatchingnD(Xs[0],Ys[0],200,50,reg = 1e-3)
    te = time.time()
    print("Calculation time: ", te-ts)
    print(min_val)
    print(rotFnew)
    print(mats)

runFirstTest()
