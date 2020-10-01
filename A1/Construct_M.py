import numpy as np
import pandas as pd
from pickle import dump
from Naive import split_ID

K = 5
DATA = pd.read_csv("ratings.csv")
N = DATA.shape[0]
Subsets = split_ID(DATA,K)

# generate utility matrix from raw dataframe
def generate(data, blank=-666):
    M = np.zeros([len(np.unique(data.userId)),len(np.unique(data.movieId))])
    for i,user in enumerate(np.unique(data.userId)):
        for j,movie in enumerate(np.unique(data.movieId)):
            bear = data[(data["userId"]==user) & (data["movieId"]==movie)].rating.to_list()
            M[i,j] = bear[0] if len(bear)>0 else blank
    return M
    

M = dict()
for k in range(K):
    data = DATA.loc[[row for row in range(N) if not row in Subsets[k]],:]
    M[k] = generate(data)
    
dump(M, open("UtilityMatrix.p", "wb"))
dump(Subsets, open("rowID_split", "wb"))