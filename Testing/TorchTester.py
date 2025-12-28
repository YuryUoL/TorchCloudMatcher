import time
import numpy as np
from core.TorchRotFinder import ComputeIsometryWithMatchingnD
from core import TorchRotFinderOptimized
from utils.Cloudgen_2D import make_batch_cpu
from core import TorchEMD

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
    N = 30
    eps = 0.0
    K = 2
    B = 1
    Xs, Ys, inv_perms, mats, types = make_batch_cpu(
        B=B, N=N, K=K, eps=eps, delta=0,
        mode="rotation", generationmode="box"
    )

    Xcompare = Xs @ mats.transpose(0, 2, 1)
    Ycompare = np.stack([Ys[j][inv_perms[j]] for j in range(B)], axis=0)
    dataBD = TorchEMD.run_sinkhorn_torch(Xcompare, Ys, metric='linf')
    emd_original_s = dataBD["expected"].cpu().numpy()

    bd_original_s = []
    for j in range(B):
        Gtmp = dataBD["G"][j].cpu().numpy()
        perm = TorchRotFinderOptimized.ComputeMatching(Gtmp)
        bd_tmp = TorchRotFinderOptimized.ComputeBD(Xcompare[j], Ys[j], perm)
        print("Bd tmp: ", bd_tmp)
        bd_original_s.append(bd_tmp)


    print("old rot: ", mats)
    ts = time.time()
    Anew,Bnew,min_val, rotFnew, G, t = TorchRotFinderOptimized.ComputeIsometryWithMatchingnD(Xs[0], Ys[0], maxchunksize=200, sinkhorn_iters=50, metric = 'linf', rot_iters= 2, reg = 1e-3)
    te = time.time()
    print("Calculation time: ", te-ts)
    print("Min val was: ", min_val)
    print(rotFnew)

    bddist = TorchRotFinderOptimized.ComputeBD(Anew @ np.transpose(rotFnew), Bnew , t)
    print(bddist)

#runFirstTest()
runSecondTest()