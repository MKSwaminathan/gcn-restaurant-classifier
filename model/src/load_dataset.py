import os, sys
import os.path
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable


from math import radians, cos, sin, asin, sqrt
def distance(lat1, lat2, lon1, lon2):
    # The math module contains a function named 
    # radians which converts from degrees to radians. 
    lon1 = radians(lon1) 
    lon2 = radians(lon2) 
    lat1 = radians(lat1) 
    lat2 = radians(lat2) 

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    c = 2 * asin(sqrt(a))

    # Radius of the earth in km. Use 3956 for miles
    r = 6371

    return (c * r)


def calc_adjacency_mat(datasetDF):
    A = np.zeros((N,N), float)
    for i in range(len(datasetDF)):
        print('Percentage done: ',(i/len(datasetDF))*100,'% ',i,'/',len(datasetDF))
        for j in range(len(datasetDF)):
            lat1 = datasetDF.iloc[i]['latitude']
            lon1 = datasetDF.iloc[i]['longitude']
            lat2 = datasetDF.iloc[j]['latitude']
            lon2 = datasetDF.iloc[j]['longitude']
            dist = distance(lat1, lat2, lon1, lon2)
            if dist <= 5:
                A[i][j] = 1
    return A


# END HELPER FXNS
datasetDF = pd.read_csv('../data/toronto_data_filtered.csv')

# Generate labels and create dataset in torch
datasetDF['label'] = datasetDF['stars'].apply(lambda x: 1 if x >=4 else 0)

# Cut dataset for testing
datasetDF = datasetDF.head(len(datasetDF)//16)

Xdat = pd.concat([datasetDF.latitude,
                  datasetDF.longitude,
                  datasetDF.review_count,
                  datasetDF.is_open,
                  datasetDF.pos_reviews], axis='columns').to_numpy()

Ids = datasetDF['business_id'].to_numpy()
ydat = datasetDF['label'].to_numpy()

# Number of nodes in graph
N = len(Ids)

# Init random Adjacency Matrix
if os.path.isfile(os.getcwd()+'/AdMat-16.npy'):
    print('Loading Adjacency Matrix')
    Adat = np.load('AdMat-16.npy')
    print('Finished loading Adjacency Matrix')
else:
    print('Creating Adjacency Matrix.')
    Adat = calc_adjacency_mat(datasetDF)
    print('Finished creating Adjacency Matrix.')
    

# Add Identity to A and normalize by diag
Adat = np.asmatrix(Adat)
I = np.matrix(np.eye(Adat.shape[0]))
Adat = Adat + I
D = np.array(np.sum(Adat, axis=0))[0]
D = np.matrix(np.diag(D))
Adat = D**-1 * Adat
Adat = np.asarray(Adat)


# Normalize data
#from sklearn import preprocessing
#Xdat = preprocessing.scale(Xdat)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
dtype = torch.FloatTensor

XTensor = torch.from_numpy(Xdat).type(dtype).to(device)
ATensor = torch.from_numpy(Adat).type(dtype).to(device)
yTensor = torch.from_numpy(ydat).type(dtype).to(device)

X = Variable(XTensor)
A = Variable(ATensor)
y = Variable(yTensor, requires_grad=False)

num_features = list(X.size())[1]


# Partitioning into training and testing:
#   create own adjacency matrices for both  

