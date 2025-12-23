import time

from core.TorchRotFinder import ComputeIsometryWithMatchingnD
from core import TorchRotFinderOptimized
from utils.Cloudgen_2D import make_batch_cpu

def runFirstTest():
    N = 200
    eps = 0.01
    Xs, Ys, inv_perms, mats, types = make_batch_cpu(
        B=1, N=N, K=2, eps=eps, delta=0,
        mode="rotation", generationmode="box"
    )
    print("old rot: ", mats)
    ts = time.time()
    Anew,Bnew,min_val, rotFnew, G, t = ComputeIsometryWithMatchingnD(Xs[0],Ys[0],200,50,reg = 1e-3)
    te = time.time()
    print("Calculation time: ", te-ts)
    print(min_val)
    print(rotFnew)
 #   print(mats)

def runSecondTest():
    N = 100
    eps = 0.02
    K = 3
    Xs, Ys, inv_perms, mats, types = make_batch_cpu(
        B=1, N=N, K=K, eps=eps, delta=0,
        mode="rotation", generationmode="box"
    )
    print("old rot: ", mats)
    ts = time.time()
    Anew,Bnew,min_val, rotFnew, G, t = TorchRotFinderOptimized.ComputeIsometryWithMatchingnD(Xs[0], Ys[0], 200, 50, rot_iters= 2, reg = 1e-3)
    te = time.time()
    print("Calculation time: ", te-ts)
    print(min_val)
    print(rotFnew)

#runFirstTest()
runSecondTest()