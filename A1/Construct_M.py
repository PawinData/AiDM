import numpy as np
import pandas as pd
from pickle import dump, load

K = 5
DATA = pd.read_csv("ratings.csv")
Subsets = load(open("rowID_split.p","rb"))  # read in subsets of row IDs

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
