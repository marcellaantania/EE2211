import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

def make_polynomial(X, degree=1, bias=True):
    poly = PolynomialFeatures(degree=degree, include_bias=bias)
    X_new = poly.fit_transform(X) # add polynomial
    return X_new

def polynomial_regression_full(X,y,ld=0, bias=True, ridge=False, degree=1): # IN: X,y ; lambda ; OUT: w (no bias)
    print(f"Don't forget to check bias because bias = {bias}")
    X = np.array(X) # data
    y = np.array(y) # target
    ridge_identity = 0
    X_new = make_polynomial(X, degree=degree, bias=bias)
    m = X_new.shape[0] # samples
    d = X_new.shape[1] # parameters
    print(X_new.shape)
    if m > d:
        if ridge == True:
            ridge_identity = ld * np.eye(d) #add regularization
        w = (np.linalg.inv((X_new.T @ X_new) + ridge_identity) @ X_new.T @ y)
        print("primal form (OD)")
    elif m < d:
        if ridge == True:
            ridge_identity = ld * np.eye(m) #add regularization
        w = (X_new.T @ np.linalg.inv((X_new @ X_new.T) + ridge_identity) @ y)
        print("dual form (UD)")
    else:
        w = np.linalg.inv(X_new) @ y
        print(f"X is square-matrix (ED)")
    return w

def predict_polynomial(X,y, Xtest, ld=0, bias=True, ridge=False, degree=1):
    w = polynomial_regression_full(X,y, ld=ld, bias=bias, ridge=ridge, degree=degree)
    Xtest_new = make_polynomial(Xtest, degree=degree, bias=bias)
    return Xtest_new @ w

def classification_onehot(X,y,ld=0, bias=True, ridge=False, degree=1):
    one = OneHotEncoder(sparse=False)
    y_new = one.fit_transform(y.reshape(-1, 1))
    print(f"One Hot encoder class → {one.categories_}")
    w= polynomial_regression_full(X,y_new,ld=ld, bias=bias, ridge=ridge, degree=degree)
    #w = np.linalg.pinv(X) @ y_new
    return w

def prediction_classification_onehot(X,y, Xtest, ld=0, bias=True, ridge=False, degree=1):
    w = classification_onehot(X,y,ld=ld, bias=bias, ridge=ridge, degree=degree)
    Xtest_new = make_polynomial(Xtest, bias=bias, degree=degree)
    index_class = np.argmax(Xtest_new @ w, axis=1)
    return index_class

def binary_classification(X, y):
    w = polynomial_regression_full(X,y)
    return w

def prediction_binary(X, y, Xtest, mapping=None):
    w = polynomial_regression_full(X,y)
    result = np.sign(Xtest.T @ w)
    return result

def gini_impurity(distribution):
    Q = 1
    for i in range(len(distribution)):
        Q-= (distribution[i]/sum(distribution))**2
    return Q

def entropy(distribution):
    if type(distribution) is not np.ndarray:
        distribution = np.array(distribution)
    Q = 0
    for i in range(len(distribution)):
        Q-= (distribution[i]/np.sum(distribution))*np.log2(distribution[i]/np.sum(distribution))
    return Q

def miss_rate(distribution):
    Q = 1 - (max(distribution)/sum(distribution))
    return Q

# k-means++’ : 
# selects initial cluster centers for k-mean clustering in a smart way to speed up convergence

# ‘random’: choose n_clusters observations (rows) at random from data for the initial centroids.

# If an array is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.

# If a callable is passed, it should take arguments X, n_clusters and a random state and return an initialization.
"""
n_clusters = 2
init = np.array([[0, 0], [3, 0]])
data = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [3, 0], [3, 1], [4, 0], [4, 1]])
"""

def kmeans(n_clusters, data, init="random"):
    if init is not "random" or not "k-means++":
        if type(init) is not np.ndarray:
            init = np.array(init)
    if type(data) is not np.ndarray:
        data = np.array(data)
    kmeans = KMeans(n_clusters = n_clusters, init=init)
    kmeans.fit(data)
    print(f"Hasil centroid adalah → \n{kmeans.cluster_centers_}")
    print(f"Hasil cluster adalah → \n{kmeans.labels_}")


""" X = np.array([1,1,1,1]).reshape(-1,2)
print (X)
y = np.array([1,0,0,1]).reshape(-1,2)
print(polynomial_regression_full(X, y, ld=0.01, ridge=True, degree=3))

X = np.array([1,3,-2,-4,0,-1,3,1,8,2,1,6,8,4,6]).reshape(-1,3)
y = np.array([1,1,2,3,3]).reshape(-1,1)
Xtest = np.array([1,-2,4]).reshape(-1,3)

print(prediction_classification_onehot(X,y,Xtest, degree=3))
"""

