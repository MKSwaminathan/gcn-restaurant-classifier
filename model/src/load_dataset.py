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
            if dist <= 0.5:
                A[i][j] = 1
    return A


# END HELPER FXNS
DF = pd.read_csv('../data/toronto_data_filtered.csv')
datasetDF2 = pd.read_csv('../data/Alldone_1hot_v1.csv')

# Generate labels and create dataset in torch
DF['label'] = DF['stars'].apply(lambda x: 1 if x >=4 else 0)

# Make mini-batches
bin_size = 48
idxs = len(DF)//bin_size
datasetDF = DF.head(idxs)
testDF = DF[idxs:(2*idxs)]


#Xdat = datasetDF.drop(['stars','latitude','longitude'],axis='columns').to_numpy()
Xdat = pd.concat([datasetDF.review_count,
                  datasetDF.is_open,
                  datasetDF.pos_reviews], axis='columns').to_numpy()
#                  datasetDF2.fbcheckins,
#                  datasetDF2['engagement.count']], axis='columns').to_numpy()

Ids = datasetDF['business_id'].to_numpy()
ydat = datasetDF['label'].to_numpy()

Xdat_test = pd.concat([testDF.review_count,
                  testDF.is_open,
                  testDF.pos_reviews], axis='columns').to_numpy()
ydat_test = testDF['label'].to_numpy()

# Number of nodes in graph
N = len(datasetDF)

# Init random Adjacency Matrix
#f = 'AdMat-16.npy'
#f = 'AdMat-64.npy'
#f = 'AdMat-4.npy'
#f = 'AdMat-64-new.npy'
f = 'AdMat-48.npy'
if os.path.isfile(os.getcwd()+'/'+f):
    print('Loading Adjacency Matrix')
    Adat = np.load(f)
    print('Finished loading Adjacency Matrix')
else:
    print('Creating Adjacency Matrix.')
    Adat = calc_adjacency_mat(datasetDF)
    np.save(f,Adat)
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

#
# Build Test Set
#
# Gen A
f_test = 'AdMat-48-test.npy'
if os.path.isfile(os.getcwd()+'/'+f_test):
    print('Loading Adjacency Matrix')
    Adat_test = np.load(f_test)
    print('Finished loading Adjacency Matrix')
else:
    print('Creating Adjacency Matrix.')
    Adat_test = calc_adjacency_mat(testDF)
    np.save(f_test,Adat_test)
    print('Finished creating Adjacency Matrix.')
 
Adat_test = np.asmatrix(Adat_test)
I = np.matrix(np.eye(Adat_test.shape[0]))
Adat_test = Adat_test + I
D = np.asarray(np.sum(Adat_test, axis=0))[0]
D = np.matrix(np.diag(D))
Adat_test = D**-1 * Adat_test
Adat_test = np.asarray(Adat_test)

XTensor_test = torch.from_numpy(Xdat_test).type(dtype).to(device)
ATensor_test = torch.from_numpy(Adat_test).type(dtype).to(device)
yTensor_test = torch.from_numpy(ydat_test).type(dtype).to(device)

X_test = Variable(XTensor_test)
A_test = Variable(ATensor_test)
y_test = Variable(yTensor_test, requires_grad=False)



