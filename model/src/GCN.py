import numpy as np
import math
import torch

import torch.nn
import torch.nn.functional as F
import torch.nn.init as tinit

from torch.autograd import Variable

dtype = torch.FloatTensor

class GCNLayer(torch.nn.Module):
    def __init__(self, nodes, in_features, out_features):
        
        super(GCNLayer, self).__init__()
        self.kernel = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.out_features = out_features
        self.nodes = nodes
        self.init_params()

    def init_params(self):
        # He initializations are generally better for ReLU activations:
        # Source: He, K. et al. (2015)
        tinit.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        
    def forward(self, A, H):
        # Iterate over all nodes and "convolve"
        H_out = torch.empty(self.nodes,self.out_features).type(dtype)
        H_out.fill_(0)
        for index in range(self.nodes):
            #print('Iteration percentage: ',(index/self.nodes)*100,'%')
            aggregate = self.make_aggregate(index, A, H)
            H_out[index] = F.linear(aggregate, self.kernel)
        return H_out

    def make_aggregate(self, index, A, H):
        aggregate = torch.empty(1,list(H.size())[1]).type(dtype)
        aggregate.fill_(0)
        n_neighbors = 1
        for neighbor_index in range(self.nodes):
            if A[index,neighbor_index] != torch.tensor(0).type(dtype):
                aggregate = aggregate + H[neighbor_index]
                n_neighbors+=1
        return aggregate/n_neighbors



class GCN(torch.nn.Module):
    def __init__(self, nodes, x_dim, h1_dim, classes):

        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(nodes, x_dim, h1_dim)
        self.gcn2 = GCNLayer(nodes, h1_dim, classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, A, X):
        h = self.gcn1(A, X)
        h = torch.relu(h)
        h = self.gcn2(A, h)
        h = self.sigmoid(h)
        return h



from load_dataset import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#
# DEFINE MODEL
#
model = GCN(N,num_features,2,1)

#
# HYPERPARAMS
#
lr = 0.01
betas = (0.9,0.99)
weight_decay = 0
epochs = 10

loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=lr, 
                             betas=betas,
                             weight_decay=0)

import tqdm
from sklearn import metrics
for t in range(epochs):
    print('Epoch: ',t,'/',epochs)
    y_pred = model(A,X)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Find AUC
    y_hat = y_pred.data.numpy()
    y_true = y.data.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat)
    AUC = metrics.auc(fpr, tpr)
    print('AUC SCORE: ',AUC, ' Loss: ',loss.item())
