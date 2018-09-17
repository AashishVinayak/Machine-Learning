# implementing k-means clustering
# 15/11/2017

# dependencies
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sc

# dataset is a matlab file opened using scipy.io.loadmat
# loading data
file = sc.loadmat("kmeans.mat")
# data holds the datset of dimensions 50 x 2
data = np.array(file['X'])


# initialising values
# m is the length of the dataset
m = np.size(data[:, 0])
# initialising 3 centroids in the numpy array centroid of shape 3 x 2
centroid = np.array([1,2,5,4,6,7]).reshape(3,2)
# mindist will be used to store the distance of a data point x(i) from all the 3 centroids
mindist = np.zeros([3, 1], dtype=float)
# minindex will be used to store the index of the minimum value in the array mindist
minindex = np.zeros([m, 1], dtype=float)
# sum holds the sum of all the datapoints with same cluster
sum = np.zeros([3, 2], dtype=float)
# counter counts the number of datapoints assigned to a cluster
counter = np.zeros([3, 1], dtype=float)


# process starts from here
for iter in range(1000):
    
# ==> cluster assignment step
    for i in range(m):
        for k in range(3):
            mindist[k] = (np.sqrt((data[i, 0] - centroid[k, 0])**2 + (data[i, 1] - centroid[k, 1])**2))
        minindex[i] = (np.argmin(mindist, axis=0)[0])

# ==> centroid update step
    for i in range(m):
        for k in range(3):
            if minindex[i] == k:
                sum[k] += data[i]
                counter[k] += 1

    # taking average
    for k in range(3):
        centroid[k, :] = (1/counter[k]) * sum[k, :]

# plotting all the clusters
plt.plot(data[np.where(minindex == 0), 0], data[np.where(minindex == 0), 1], 'xr')
plt.plot(data[np.where(minindex == 1), 0], data[np.where(minindex == 1), 1], 'xg')
plt.plot(data[np.where(minindex == 2), 0], data[np.where(minindex == 2), 1], 'xb')
plt.plot(centroid[:, 0], centroid[:, 1], 'om')
plt.show()