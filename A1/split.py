import pandas as pd
from random import sample, seed
from pickle import load, dump

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

seed(999)
Subsets = split_ID(DATA,K)
dump(Subsets, open("rowID_split.p","wb"))