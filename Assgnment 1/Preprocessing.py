import numpy as np
import pandas as pd
from pickle import dump, load

K = 5
DATA = pd.read_csv("ratings.csv")
N = DATA.shape[0]

# randomly split row indexes into K subsets
Subsets = dict()
pool = [i for i in range(N)]
for k in range(K-1):
    Subsets[k] = list(np.random.choice(pool, int(N/K)))
    pool = [i for i in pool if not i in Subsets[k]]
Subsets[K-1] = pool
dump(data, open("rowID_split.p","wb"))   # save results

# generate utility matrix from raw dataframe
def generate(data, blank=np.nan):
    R, C = np.unique(data.userId), np.unique(data.movieId)
    M = np.zeros([len(R),len(C)])
    for i,user in enumerate(R):
        for j,movie in enumerate(C):
            bear = [score for u,m,score in zip(data.userId,data.movieId,data.rating) if u==user and m==movie]
            M[i,j] = bear[0] if len(bear)>0 else blank
    return M
    

# a dictionary: k --> dataframe = utility matrix
M = dict()
for k in range(K):
    data = DATA.loc[[row for row in range(N) if not row in Subsets[k]],:]
    M[k] = pd.DataFrame(generate(data), 
                        columns=np.unique(data.movieId), 
                        index=np.unique(data.userId)
                       )
    
dump(M, open("TrainingMatrix.p", "wb"))
