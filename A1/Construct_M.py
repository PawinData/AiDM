import numpy as np
import pandas as pd
from pickle import dump
from random import sample

# split row IDs of a dataframe, DATA, into K subsets of almost equal size
# store in dictionary: subset ID --> a list of row IDs
def split_ID(DATA, K):
    N = DATA.shape[0]
    Subsets = dict()
    pool = [i for i in range(N)]
    for subset_ID in range(K-1):
        Subsets[subset_ID] = sample(pool, int(N/K))
        pool = [i for i in pool if not i in Subsets[subset_ID]]
    Subsets[K-1] = pool
    return Subsets
    

K = 5
DATA = pd.read_csv("ratings.csv")
N = DATA.shape[0]
Subsets = split_ID(DATA,K)
dump(Subsets, open("rowID_split.p","wb"))


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
    M[k] = pd.DataFrame(generate(data), 
                        columns=np.unique(data.movieId), 
                        index=np.unique(data.userId)
                       )
    
dump(M, open("TrainingMatrix.p", "wb"))
