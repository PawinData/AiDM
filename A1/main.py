import pandas as pd
import numpy as np
import random
from time import process_time
from Naive import RMSE, MAE, Naive_A, Naive_B, Naive_C, Naive_D, compare_vis, Mix_Averages, add_avg
from SVD import UV_approach, Comb_approach
from pickle import load, dump
from collections import namedtuple

DATA = pd.read_csv('ratings.csv')  # read in dataset
N = DATA.shape[0]
K = 5
Subsets = load(open("rowID_split.p","rb"))  # read in subsets of row IDs

Train = namedtuple("Train", ["RMSE","MAE"])
Test  = namedtuple("Test", ["RMSE","MAE"])



# Naive Approaches

NAIVE = dict()
T_naive = 0
for k in range(K):
    # contruct training & test set in this run
    TEST = DATA.loc[Subsets[k],:]
    TRAIN = DATA.loc[[i for i in range(N) if not i in Subsets[k]],:]
    # compute errors by each of the four naive algorithms
    errors = [Naive_A(TRAIN,TEST)]
    errors.append(Naive_B(TRAIN,TEST))
    errors.append(Naive_C(TRAIN,TEST))
    t = process_time()
    errors.append(Naive_D(TRAIN,TEST))
    T_naive += process_time() - t
    # kth run --> a list of tuples of errors
    NAIVE[k] = errors
T_naive /= K
DF = pd.DataFrame(NAIVE, index=["Naive A","Naive B","Naive C","Naive D"]).transpose()   # organize results
DF.index = ["k = "+str(k+1) for k in range(K)]
DF.to_pickle('Performance.pkl')                                                         # save results
CT = [T_naive]
dump(CT, open("ComputingTime.p","wb"))                                                  # save average computing time
print("Naive models done.")




# Mixed Averages

M = load(open("TrainingMatrix.p","rb"))
Mix = list()
for k in range(K):
    Mix.append(Mix_Averages(M[k], DATA.loc[Subsets[k],:]))
DF["Mixed"] = Mix
compare_vis(DF, title="MixedAverages.eps")                     # save comparison results
print("An extension of Naive D model done.")




# UV Decomposition of Utility Matrix

random.seed(100)
UV = list()
t = process_time()
for k in range(K):
    UV.append( UV_approach(M[k], DATA.loc[Subsets[k],:], xi=1, repeat=10) )
T_uv = (process_time() - t)/K  
DF = pd.read_pickle("Performance.pkl")                                                
DF["UV"] = UV
DF.to_pickle("Performance.pkl")                    # save results
CT = load(open("ComputingTime.p","rb"))
CT.append(T_uv)
dump(CT, open("ComputingTime.p","wb"))             # save average computing time
print("UV decomposition done.")




# Matrix Factorization combined with Regularization and Naive Averages

lbd = 0.05
num_iter = 75
num_features = 10
eta = 0.005

random.seed(200)
COMB = list()
t = process_time()
for k in range(K):
    res = Comb_approach(M[k], DATA.loc[Subsets[k],:], num_features, eta, lbd, xi=num_iter, repeat=10)
    COMB.append(res)   
T_comb = (process_time() - t)/K
DF = pd.read_pickle("Performance.pkl")                                                
DF["Combined"] = COMB
DF.to_csv("Performance.csv", index=False)                    # save results
CT = load(open("ComputingTime.p","rb"))
CT.append(T_comb)
dump(CT, open("ComputingTime.p","wb"))             # save average computing time
print("Matrix Factorization combined with Regularization and Naive Averages done.")
compare_vis(DF, title="Performance.eps")



# Hyperparameters

random.seed(300)
k = np.random.randint(0,K,size=1)[0] # randomly pick a training set to experiment
X = M[k].to_numpy(copy=True)
TEST = DATA.loc[Subsets[k],:]
H = dict()
penalty = [0.01, 0.05, 0.1]
num_iter = [50, 75, 100]
num_features = 10
eta = 0.005
for lbd in penalty:
    lst = list()
    for xi in num_iter:
        a,b = Comb_approach(M[k], DATA.loc[Subsets[k],:], num_features, eta, lbd, xi, repeat=10)
        lst.append((a.RMSE,b.RMSE))
    H[lbd] = lst  
H = pd.DataFrame(H,index=[r"$\xi$ = "+str(ele) for ele in num_iter])
H.columns = [r"$\lambda$ = "+str(ele) for ele in penalty]
H.to_csv("Hyperparameters.csv", index=False)  # save results




# Compare Performance

DF = add_avg(pd.read_pickle("Performance.pkl"))  # add a row of average scores of each approach to the dataframe

AVERAGE = dict()
AVERAGE["Computing Time"] = load(open("ComputingTime.p","rb"))
AVERAGE["RMSE Training"] = [a.RMSE for a,b in DF.loc["Average",]][-3:]
AVERAGE["RMSE Test"] = [b.RMSE for a,b in DF.loc["Average",]][-3:]
AVERAGE["MAE Training"] = [a.MAE for a,b in DF.loc["Average",]][-3:]
AVERAGE["MAE Test"] = [b.MAE for a,b in DF.loc["Average",]][-3:]
AVERAGE = pd.DataFrame(AVERAGE, index=DF.columns[-3:]).transpose()
AVERAGE.to_csv("AVERAGE.csv", index=False)