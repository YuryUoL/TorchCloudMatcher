import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from core import TorchRotFinderOptimized
from core import TorchEMDRotOptimizedHypotheses
from utils import Cloudgen_2D as CloudGenAdvanced
import torch
from core import TorchEMD

def performanceTimeTest(outpath):
    dim = 3
    PointNumber = 10 * np.arange(1, 11)  # [100, 200, ..., 1000]
   # PointNumber = 25 * np.arange(1, 11)  # [100, 200, ..., 1000]
    Repeat = 10
    ErrorMeasure = 0.01
    mode = 'rotation'
    constCAlternative = 3 * math.sqrt(2)

    times = []

    # OUTER LOOP WITH PROGRESS BAR
    for i in tqdm(range(len(PointNumber)), desc="Point cloud sizes"):

        N = PointNumber[i]
        X, Y, perm, R, t = CloudGenAdvanced.make_batch_cpu(
            10, N, dim, ErrorMeasure, 0, mode
        )

        timesumm = 0.0
        #FastCalc
        # INNER REPEAT LOOP (optionally add progress too)
        for j in range(Repeat):
            tstart = time.time()
            Anew,Bnew,min_val, rotFnew, G, t = TorchRotFinderOptimized.ComputeIsometryWithMatchingnD(X[j],Y[j],200,50,reg = 1e-3)

            #

           # print(j,  " ", d)
            tend = time.time()
            timesumm += (tend - tstart)

            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.empty_cache()

        avgtime = timesumm / Repeat
        times.append(avgtime)

    # Write CSV *after* the loop, not inside it
    df = pd.DataFrame({
        "points_number": PointNumber,
        "time_s": times
    })

    df.to_csv(
        outpath,
        index=False
    )

def precisionExperiments(gend,fname):
    dim = 3
    sinkhorn_iters = 300
    reg = (1e-3)
    genError = gend * np.arange(1, 11)
    Repeat = 50
    mode = 'rotation'
    constCAlternative = 3 * math.sqrt(2)
    NumberOfPoints = 50

    ### Parameter to save:

    matching_distance = []
    emd_original = []
    bd_original = []
    emd_final = []
    bd_final = []

    emd_final_t = []
    bd_final_t = []

    rotmat_linf_bijection = []
    rotmat_linf_transport = []
    emd_absolute_error = []
    emd_relative_error = []
    bd_absolute_error = []
    bd_relative_error = []
    emd_absolute_error_t = []
    emd_relative_error_t = []
    bd_absolute_error_t = []
    bd_relative_error_t = []

    for i in tqdm(range(len(genError)), desc="Error measures", position=0):

        ErrorMeasure = genError[i]
        X, Y, perm, R, t = CloudGenAdvanced.make_batch_cpu(
            Repeat, NumberOfPoints, dim, ErrorMeasure, 0, mode
        )

        Xcompare = X @ R.transpose(0, 2, 1)
        Ycompare = np.stack([Y[j][perm[j]] for j in range(Repeat)], axis=0)
        dataBD = TorchEMD.run_sinkhorn_torch(Xcompare, Y, metric = 'linf')
        emd_original_s = dataBD["expected"].cpu().numpy()

        bd_original_s = []
        for j in range(Repeat):
            Gtmp = dataBD["G"][j].cpu().numpy()
            perm = TorchRotFinderOptimized.ComputeMatching(Gtmp)
            bd_tmp = TorchRotFinderOptimized.ComputeBD(Xcompare[j], Y[j], perm)
            #print("Bd tmp: ", bd_tmp)
            bd_original_s.append(bd_tmp)

        bd_original_s = np.array(bd_original_s)
       # bd_original_s = dataBD["bottleneck_bij"]


        emd_original.append(np.mean(emd_original_s))
        bd_original.append(np.mean(bd_original_s))

        matchdist = 0
        emd_f = 0
        bd_f = 0
        rotmat_linf_bijection_e = 0
        rotmat_linf_transport_e = 0
        emd_absolute_e = 0
        emd_relative_e = 0
        bd_absolute_e = 0
        bd_relative_e = 0
        emd_f_t = 0
        bd_f_t = 0
        emd_absolute_e_t = 0
        emd_relative_e_t = 0
        bd_absolute_e_t = 0
        bd_relative_e_t = 0
        # Af,Bf, best_val, best_rot, best_G,perm
        for j in tqdm(range(Repeat), desc=f"Repeat {i+1}", position=1, leave=False):
            Af, Bf, dist, rResult,transport_m,match_bij = TorchRotFinderOptimized.ComputeIsometryWithMatchingnD(X[j],Y[j],maxchunksize=250,metric = 'linf')

            matchdist = matchdist + dist
            rotmat_linf_bijection_e = rotmat_linf_bijection_e + np.max(np.abs(rResult - R[j]))
#            XT = XT[np.newaxis, :, :]
#            YT = YT[np.newaxis, :, :]


            emd_star = dist
            bd_star = TorchRotFinderOptimized.ComputeBD(Af @ np.transpose(rResult) , Bf , match_bij)
            emd_f = emd_f + emd_star
            bd_f = bd_f + bd_star

            ## FILL emd_absolute, emd_relative_ erorr, bd_absolute_e, bd_relative_error pelase
            ## by comparing emd_star with emd_original_s[j] and bd_original_s[j] with bd_f

            emd_absolute_e += float(abs(emd_star - emd_original_s[j]))
            emd_relative_e += float(abs(emd_star - emd_original_s[j]) / (abs(emd_original_s[j]) + 1e-12))

            bd_absolute_e += float(abs(bd_star - bd_original_s[j]))
            bd_relative_e += float(abs(bd_star - bd_original_s[j]) / (abs(bd_original_s[j]) + 1e-12))


            emd_star_t = emd_star
            bd_star_t = bd_star
            emd_f_t = emd_f_t + emd_star_t
            bd_f_t = bd_f_t + bd_star_t

            ## FILL emd_absolute, emd_relative_ erorr, bd_absolute_e, bd_relative_error pelase
            ## by comparing emd_star with emd_original_s[j] and bd_original_s[j] with bd_f

            emd_absolute_e_t += float(abs(emd_star_t - emd_original_s[j]))
            emd_relative_e_t += float(abs(emd_star_t - emd_original_s[j]) / (abs(emd_original_s[j]) + 1e-12))

            bd_absolute_e_t += float(abs(bd_star_t - bd_original_s[j]))
            bd_relative_e_t += float(abs(bd_star_t - bd_original_s[j]) / (abs(bd_original_s[j]) + 1e-12))

            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.empty_cache()







            #cp.get_default_memory_pool().free_all_blocks()
            #cp.get_default_pinned_memory_pool().free_all_blocks()

        matchdist = matchdist / Repeat
        emd_f = emd_f / Repeat
        bd_f = bd_f / Repeat
        rotmat_linf_bijection_e = rotmat_linf_bijection_e / Repeat
        rotmat_linf_transport_e = rotmat_linf_transport_e / Repeat
        emd_absolute_e = emd_absolute_e / Repeat
        emd_relative_e = emd_relative_e / Repeat
        bd_absolute_e = bd_absolute_e / Repeat
        bd_relative_e = bd_relative_e / Repeat

        emd_f_t = emd_f_t / Repeat
        bd_f_t = bd_f_t / Repeat
        emd_absolute_e_t = emd_absolute_e_t / Repeat
        emd_relative_e_t = emd_relative_e_t / Repeat
        bd_absolute_e_t = bd_absolute_e_t / Repeat
        bd_relative_e_t = bd_relative_e_t / Repeat



        matching_distance.append(matchdist)
        emd_final.append(emd_f)
        bd_final.append(bd_f)
        rotmat_linf_bijection.append(rotmat_linf_bijection_e )
        rotmat_linf_transport.append(rotmat_linf_transport_e)
        emd_absolute_error.append(emd_absolute_e)
        emd_relative_error.append(emd_relative_e)
        bd_absolute_error.append(bd_absolute_e)
        bd_relative_error.append(bd_relative_e )

        emd_final_t.append(emd_f_t)
        bd_final_t.append(bd_f_t)
        emd_absolute_error_t.append(emd_absolute_e_t)
        emd_relative_error_t.append(emd_relative_e_t)
        bd_absolute_error_t.append(bd_absolute_e_t)
        bd_relative_error_t.append(bd_relative_e_t)


## Correct this DF bro add all this information from above, starting from   ErrorMeasure  and save it to
    ## /home/yury/Projects/CloudMatchingProject/TablesOutput27November2025/Speed but make it precision.csv filename

    df = pd.DataFrame({
        "error_measure": genError,
        "matching_distance": matching_distance,
        "emd_original": emd_original,
        "bd_original": bd_original,
        "emd_final": emd_final,
        "bd_final": bd_final,
        "rotmat_linf_bijection": rotmat_linf_bijection,
        "rotmat_linf_transport": rotmat_linf_transport,
        "emd_absolute_error": emd_absolute_error,
        "emd_relative_error": emd_relative_error,
        "bd_absolute_error": bd_absolute_error,
        "bd_relative_error": bd_relative_error,
        "emd_final_transport": emd_final_t,
        "bd_final_transport": bd_final_t,
        "emd_absolute_error_transport": emd_absolute_error_t,
        "emd_relative_error_transport": emd_relative_error_t,
        "bd_absolute_error_transport": bd_absolute_error_t,
        "bd_relative_error_transport": bd_relative_error_t

    })
    #"/home/yury/Projects/CloudMatchingProject/TablesOutput30November2025/CSV/precision_small_error.csv"
    df.to_csv(fname,index=False )








outpath = "/home/yury/Projects/CloudMatchingProject/TablesDecember28_3D_2025/time_euclidean.csv"
performanceTimeTest(outpath)


    #! ComputeIsometryWithMatching(A,B, InnerCalc , metric = 'euclid', maxchunksize = 1000, constC = 1):

#print("Hallelujah")
#precisionExperiments()
#print("Time experiments")
#performanceTimeTest()
#print("Low-bound experiments")
precisionExperiments(0.002,"/home/yury/Projects/CloudMatchingProject/TablesDecember28_3D_2025/precision_small_error.csv")
#print("High tide experiments")
precisionExperiments(0.005,"/home/yury/Projects/CloudMatchingProject/TablesDecember28_3D_2025/precision_big_error.csv")
#testFive()
#testFour()
#testThree()
#testTwo()
#testOne()
#cupy_health_check()