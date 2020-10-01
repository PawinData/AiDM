import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from collections import namedtuple

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
    

# compute Root Mean Squared Error
def RMSE(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.sqrt(np.mean((v1-v2)**2))
    
   
# compute Mean Absolute Error
def MAE(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.mean(abs(v1-v2))


# organize the results of each algorithm   
Train = namedtuple("Train", ["RMSE","MAE"])
Test  = namedtuple("Test", ["RMSE","MAE"])


# Naive Approach: predict by overall average rating
def Naive_A(TRAIN, TEST):  
    # predict
    Rating_overall = np.mean(TRAIN.rating)
    # compute errors
    RMSE_train = RMSE([Rating_overall] * TRAIN.shape[0], TRAIN.rating)
    RMSE_test  = RMSE([Rating_overall] * TEST.shape[0], TEST.rating)
    MAE_train  = MAE([Rating_overall] * TRAIN.shape[0], TRAIN.rating)
    MAE_test   = MAE([Rating_overall] * TEST.shape[0], TEST.rating)
    return ( Train(RMSE_train,MAE_train), Test(RMSE_test,MAE_test) )
    


# Naive Approach: predict by average rating per User ID    
def Naive_B(TRAIN, TEST):   
    
    alt = np.mean(TRAIN.rating) # alternative if user not in training set
    # build model   
    Rating_user = dict()    
    for ID in np.unique(TRAIN.userId):
        data = TRAIN.loc[TRAIN.userId==ID]
        Rating_user[ID] = np.mean(data.rating)
    # predict
    pred_train = [Rating_user[ID] for ID in TRAIN.userId]
    pred_test  = [Rating_user.get(ID,alt) for ID in TEST.userId]
    # compute errors
    RMSE_train = RMSE(pred_train, TRAIN.rating)
    RMSE_test  = RMSE(pred_test, TEST.rating)
    MAE_train  = MAE(pred_train, TRAIN.rating)
    MAE_test   = MAE(pred_test, TEST.rating)
    
    return ( Train(RMSE_train,MAE_train), Test(RMSE_test,MAE_test) )
    
    

# Naive Approach: predict by average rating per movie ID    
def Naive_C(TRAIN, TEST): 
    
    alt = np.mean(TRAIN.rating)  # alternative if movie not in training set
    # build model
    Rating_movie = dict()
    for index in np.unique(TRAIN.movieId):
        data = TRAIN.loc[TRAIN.movieId==index]
        Rating_movie[index] = np.mean(data.rating)
    # predict
    pred_train = [Rating_movie[index] for index in TRAIN.movieId]
    pred_test  = [Rating_movie.get(index,alt) for index in TEST.movieId]
    # compute errors
    RMSE_train = RMSE(pred_train, TRAIN.rating)
    RMSE_test  = RMSE(pred_test, TEST.rating)
    MAE_train  = MAE(pred_train, TRAIN.rating)
    MAE_test   = MAE(pred_test, TEST.rating)
    
    return ( Train(RMSE_train,MAE_train), Test(RMSE_test,MAE_test) )
    
    

# Naive Approach: predict by combined average rating    
def Naive_D(TRAIN, TEST): 
    
    alt = np.mean(TRAIN.rating) # alternative if user not in training set
    
    # average rating per user ID
    Rating_user = dict()   
    for ID in np.unique(TRAIN.userId):
        data = TRAIN.loc[TRAIN.userId==ID]
        Rating_user[ID] = np.mean(data.rating)  
        
    # average rating per movie ID
    Rating_movie = dict()  
    for index in np.unique(TRAIN.movieId):
        data = TRAIN.loc[TRAIN.movieId==index]
        Rating_movie[index] = np.mean(data.rating)
        
    # linear regression
    A = [(Rating_user[u],Rating_movie[m]) for u,m in zip(TRAIN.userId,TRAIN.movieId)]
    A = np.hstack((np.ones([len(A),1]), np.array(A)))
    alpha,beta,gamma = np.linalg.lstsq(A, np.array(TRAIN.rating), rcond=None)[0]
    coef = np.array([alpha,beta,gamma]).reshape([3,1])
    
    # predict
    pred_train = np.clip(np.dot(A,coef).flatten(), 1, 5)
    B = [(Rating_user.get(u,alt),Rating_movie.get(m,Rating_user.get(u,alt))) for u,m in zip(TEST.userId,TEST.movieId)]
    B = np.hstack((np.ones([len(B),1]), np.array(B)))
    pred_test  = np.clip(np.dot(B,coef).flatten(), 1, 5)
    
    # compute errors
    RMSE_train = RMSE(pred_train , TRAIN.rating)
    RMSE_test  = RMSE(pred_test , TEST.rating)
    MAE_train  = MAE(pred_train , TRAIN.rating)
    MAE_test   = MAE(pred_test , TEST.rating)
    
    return ( Train(RMSE_train,MAE_train), Test(RMSE_test,MAE_test) )
    
    
    
# visualize results of errors organized in dataframe DF
def compare_vis(DF, title=None, FigSize=(15,8), MarkerSize=10):

    fig,ax = plt.subplots(figsize=FigSize)

    for row in DF.index:
    
        ax.plot(DF.columns, [a.RMSE for a,b in DF.loc[row,]], 
                color='darkblue', marker='o', markersize=MarkerSize, linewidth=0,
                label="RMSE for Training" if row=='k = 1' else None)
    
        ax.plot(DF.columns, [a.MAE for a,b in DF.loc[row,]],
                color="darkblue", marker='x', markersize=MarkerSize, linewidth=0,
                label="MAE for Training" if row=='k = 1' else None)
    
        ax.plot(DF.columns, [b.RMSE for a,b in DF.loc[row,]], 
                color="darkolivegreen", marker='o', markersize=MarkerSize, linewidth=0,
                label="RMSE for Testing" if row=='k = 1' else None)
    
        ax.plot(DF.columns, [b.MAE for a,b in DF.loc[row,]],
                color="darkolivegreen", marker='x', markersize=MarkerSize, linewidth=0,
                label="MAE for Testing" if row=='k = 1' else None)

    for elements in ax.lines:    # add random noise to x-coordinates of points
        xs = elements.get_xydata()[:, 0] 
        jittered_xs = xs + np.random.uniform(-0.3, 0.3, xs.shape)
        elements.set_xdata(jittered_xs)
    ax.relim()
    ax.autoscale(enable=True)

    plt.xlabel("Naive Approaches", fontsize=18)
    plt.ylabel("Errors", fontsize=18)
    plt.title("Compare 4 Naive Approaches", fontsize=20)
    plt.legend(loc="upper right", fontsize=16)
    
    if title is not None:
        plt.savefig(title)
   
    plt.show()



# add a row of average scores to dataframe DF
def add_avg(DF):
    avg_scores = dict()
    for approach in DF.columns:
        train_errors = Train(np.mean([a.RMSE for a,b in DF[approach]]), np.mean([a.MAE  for a,b in DF[approach]]))
        test_errors  = Test(np.mean([b.RMSE for a,b in DF[approach]]), np.mean([b.MAE  for a,b in DF[approach]]))
        avg_scores[approach] = [(train_errors, test_errors)]
    DF = DF.append(pd.DataFrame(avg_scores, index=["Average"]))   
    return DF