import numpy as np
import scipy
from scipy import spatial


def calculateMahalanobis(y=None, data=None, cov=None):
  
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

#a=[[0.1, 0.2, 0.4]]
a=np.random.rand(1, 2048)
a=a.T
print(a.shape)
#b=[[1, 0.75, 0.6]]
b=np.random.rand(1,2048)
b=b.T
#c=np.concatenate((a,b), 0)
#print(c)
#c_mu=np.mean(c, 0)
#print(c_mu)
#a=a-c_mu
#b=b-c_mu
#print(a)
#cov=np.cov(a,b)
#cov=[[1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]]
cov=np.cov(a.T, b.T)
print(cov)
inv=np.linalg.inv(cov)
print(inv)
m=spatial.distance.mahalanobis(a,b, inv)
print(m)

