import numpy as np
import pandas as pd
from collections import namedtuple
from Naive import RMSE, MAE

Train = namedtuple("Train", ["RMSE","MAE"])
Test  = namedtuple("Test", ["RMSE","MAE"])

def rmse(A,B):
    bear = (A*(A> -100) - B*(A> -100))**2
    return np.sqrt(np.mean(bear))
    
    
def mae(A,B):
    return np.mean(abs(A*(A> -100) - B*(A> -100)))


def standardize(X):
    Z = np.copy(X)
    # subtract from each column the column average, excluding large negative value
    col_avg = np.sum(Z*(Z> -100),axis=0)/np.sum(Z> -100,axis=0)
    Z = Z - col_avg
    # subtract from each row the row average, excluding large negative values
    Z = Z.transpose()
    row_avg = np.sum(Z*(Z> -100),axis=0)/np.sum(Z> -100,axis=0)
    Z = Z - row_avg
    Z = Z.transpose()
    return Z,row_avg,col_avg
	
	
def initialize(X, d, randomness=1):
    m,n = X.shape
    U = np.random.normal(0,randomness,(m,d))
    V = np.random.normal(0,randomness,(d,n))
    return (U,V)

	
def update(X,U,V, eps, drop):
    old = np.zeros(X.shape)
    new = np.dot(U,V)
    count = 1
    while rmse(old,new)>=eps and count<=drop:
        old = new
        # update U
        for r in range(U.shape[0]):
            nonblank =  np.where(X[r,:]> -100)[0] # a list of j s.t X[r,j]> -100
            for s in range(U.shape[1]):
                U[r,s] = sum([V[s,j]*(X[r,j] - np.dot(U[r,:],V[:,j]) + U[r,s]*V[s,j]) for j in nonblank])
                U[r,s] /= sum([V[s,j]**2 for j in nonblank])
        # update V
        for r in range(V.shape[0]):
            nonblank = np.where(X[:,s]> -100)  # a list of i s.t X[i,s]> -100
            for s in range(V.shape[1]):
                V[r,s] = np.sum([U[i,r]*(X[i,s] - np.dot(U[i,:],V[:,s]) + U[i,r]*V[r,s]) for i in nonblank])
                V[r,s] /= np.sum([U[i,r]**2 for i in nonblank])
        new = np.clip(np.dot(U,V), -2.5, 2.5)
        count += 1
    return (U,V)



# estimate missing values in X by UV Decomposition
# generate X_estimate
def estimate(X,d, eps=10**(-2), drop=50):
    Z,row_avg,col_avg = standardize(X)
    U,V = initialize(Z,d)
    U,V = update(Z,U,V,eps,drop)
    X_estimate = np.dot(U,V)
    # undo standardization
    X_estimate = X_estimate.transpose()
    X_estimate = X_estimate + row_avg
    X_estimate = X_estimate.transpose()
    X_estimate = X_estimate + col_avg
    return np.clip(np.round(X_estimate), 1, 5)
    
 
def col_avg(X):
    row = np.sum(X*(X> -100), axis=0) / np.sum((X> -100), axis=0)
    return np.clip(np.round(row ), 1,5)   
    
    
# DF is a dataframe of training utility matrix
# X is a matrix estimated by SVD
def df_predict(DF, X_estimate):      
    df = pd.DataFrame(X_estimate, columns=DF.columns, index=DF.index)
    A = DF.to_numpy()
    one_row = col_avg(A).reshape((1,DF.shape[1]))
    df = df.append(pd.DataFrame(one_row, index=["Movie Average"], columns=DF.columns))
    one_col = list(col_avg(A.transpose()))
    one_col.append(-666)
    df["User Average"] = one_col
    return df



def UV_approach(DF_train, TEST, xi, repeat):

    # training
    X = DF_train.to_numpy(copy=True)
    m,n = X.shape
    d = int(np.sum(X > -100)/(m+n))
    # average out the randomness in decomposition
    X_est = np.mean([estimate(X, d, drop=xi) for run in range(repeat)], axis=0)
    grizzly = Train(rmse(X,X_est), mae(X,X_est))  # training errors
    
    # testing
    pred = Predict(df_predict(DF_train,X_est), TEST)   
    panda = Test(RMSE(pred,TEST.rating), MAE(pred,TEST.rating))  # test errors
    
    return (grizzly,panda)
    
    
    
def Predict(DF, TEST):
    res = list()
    for u,m in zip(TEST.userId,TEST.movieId):
        if m in DF.columns and u in DF.index:
            res.append(DF.loc[u,m])
        elif m in DF.columns:
            res.append(DF.loc["Movie Average",m])
        elif u in DF.index:
            res.append(DF.loc[u,"User Average"])
        else:
            bear = np.mean(df.loc["Movie Average", :]) + np.mean(df.loc[:,"User Average"])
            res.append(bear/2)
    return np.clip(np.round(np.array(res)), 1,5)
    
    
    
def approx(X, d, lr, lam, drop, randomness=1, eps=10**(-2)):

    # standardize X
    XX,row_avg,col_avg = standardize(X)

    # randomly initialize W, Z
    m,n = XX.shape
    avg = np.mean(XX*(XX> -100))
    W,Z = np.random.normal(0, randomness, (m,d)), np.random.normal(0, randomness, (d,n))
    old = np.zeros(XX.shape)
    new = np.dot(W,Z) 
    
    # update W, Z by gradient descent with regularization
    count = 1
    while rmse(old,new)>=eps and count<=drop:
        old = new
        for i in range(m):
            for j in range(n):
                if XX[i,j] > -100:
                    E = XX[i,j] - old[i,j]
                    W[i,:], Z[:,j] = W[i,:]+lr*(2*E*Z[:,j]-lam*W[i,:]), Z[:,j]+lr*(2*E*W[i,:]-lam*Z[:,j])
        new = np.clip(np.dot(W,Z), -2.5, 2.5)
        count += 1 

    # undo standardization
    new = new.transpose()
    new = new + row_avg
    new = new.transpose()
    new = new + col_avg
        
    return np.clip(np.round(new), 1, 5)
    
    
   

def linear_model(X, X_approx):
    movie_avg = col_avg(X)
    user_avg = col_avg(X.transpose())
    A,b = [],[]
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > -100:
                b.append(X[i,j])
                A.append([1, movie_avg[j], user_avg[i], X_approx[i,j]])
    A,b = np.array(A),np.array(b)
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    pred_train = np.clip(np.round(np.dot(A,coef)), 1,5)
    grizzly,panda = RMSE(pred_train,b),MAE(pred_train,b)
    return (coef, Train(grizzly,panda))
    
    
    
def Predict_comb(df, TEST, coef):
    B = list()
    for u,m in zip(TEST.userId, TEST.movieId):
        if u in df.index and m in df.columns:
            B.append([1, df.loc["Movie Average",m], df.loc[u,"User Average"], df.loc[u,m]])
        elif m in df.columns:
            bear = df.loc["Movie Average",m]
            B.append([1,bear,bear,bear])
        elif u in df.index:
            bear = df.loc[u,"User Average"]
            B.append([1, bear, bear, bear])
        else:
            bear = np.mean(df.loc["Movie Average", :]) + np.mean(df.loc[:,"User Average"])
            bear /= 2
            B.append([1,bear,bear,bear])
    return np.clip(np.round(np.dot(np.array(B),coef)), 1,5)
    
    
    
    
def Comb_approach(DF_train, TEST, num_features, eta, lbd, xi, repeat):

    # training
    X = DF_train.to_numpy(copy=True)
    X_approx = np.mean([approx(X, d=num_features, lr=eta, lam=lbd, drop=xi) for run in range(repeat)], axis=0)
    beta, train_errors = linear_model(X, X_approx)
    
    # testing
    pred = Predict_comb(df_predict(DF_train,X_approx), TEST, beta)  
    a,b = RMSE(pred,TEST.rating), MAE(pred,TEST.rating)   
    
    return (train_errors, Test(a,b))