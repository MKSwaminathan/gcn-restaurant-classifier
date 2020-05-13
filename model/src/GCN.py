import numpy as np
import math
import torch

import torch.nn
import torch.nn.functional as F
import torch.nn.init as tinit

from torch.autograd import Variable

dtype = torch.FloatTensor

class GCNLayer(torch.nn.Module):
    def __init__(self, nodes, in_features, out_features, isrelu=True):
        super(GCNLayer, self).__init__()
        self.kernel = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.out_features = out_features
        self.nodes = nodes
        if isrelu:
            self.init_params_relu()
        else:
            self.init_params()

    def init_params_relu(self):
        # He initializations are generally better for ReLU activations:
        # Source: He, K. et al. (2015)
        tinit.kaiming_uniform_(self.kernel, a=math.sqrt(5))
    
    def init_params(self):
        # Xavier initializations are universally good - He is better for relu though 
        tinit.xavier_uniform_(self.kernel, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, A, H):
        # Iterate over all nodes and "convolve"
        H_out = torch.empty(self.nodes,self.out_features).type(dtype)
        H_out.fill_(0)
        for index in range(self.nodes):
            #print('Iteration percentage: ',(index/self.nodes)*100,'%')
            # NOTE: here you can define which aggregate policy to use
            aggregate = self.make_aggregate_mean(index, A, H)
            H_out[index] = F.linear(aggregate, self.kernel)
        return H_out

    def make_aggregate_mean(self, index, A, H):
        aggregate = torch.empty(1,list(H.size())[1]).type(dtype)
        aggregate.fill_(0)
        n_neighbors = 1
        for neighbor_index in range(self.nodes):
            if A[index,neighbor_index] != torch.tensor(0).type(dtype):
                aggregate = aggregate + H[neighbor_index]
                n_neighbors+=1
        return aggregate/n_neighbors

    def make_aggregate_weighted_mean(self, index, A, H):
        aggregate = torch.empty(1,list(H.size())[1]).type(dtype)
        aggregate.fill_(0)
        n_neighbors = 1
        for neighbor_index in range(self.nodes):
            if A[index,neighbor_index] != torch.tensor(0).type(dtype):
                aggregate = aggregate + H[neighbor_index]
                n_neighbors+=1
        # This is a param that can be changed
        self_weight_factor = n_neighbors
        aggregate = aggregate + self_weight_factor*H[index]

        return aggregate/(n_neighbors + self_weight_factor)

   def make_aggregate_sum(self, index, A, H):
        aggregate = torch.empty(1,list(H.size())[1]).type(dtype)
        aggregate.fill_(0)
        n_neighbors = 1
        for neighbor_index in range(self.nodes):
            if A[index,neighbor_index] != torch.tensor(0).type(dtype):
                aggregate = aggregate + H[neighbor_index]
                n_neighbors+=1
        return aggregate




class GCN(torch.nn.Module):
    def __init__(self, nodes, x_dim, h1_dim, h2_dim, h3_dim, classes):

        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(nodes, x_dim, h1_dim)
        self.gcn2 = GCNLayer(nodes, h1_dim, h2_dim)
        self.gcn3 = GCNLayer(nodes, h2_dim, h3_dim)
        self.gcn4 = GCNLayer(nodes, h3_dim, classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, A, X):
        h = self.gcn1(A, X)
        h = torch.relu(h)
        h = self.gcn2(A, h)
        h = torch.relu(h)
        h = self.gcn3(A, h)
        h = torch.relu(h)
        h = self.gcn4(A, h)
        #h = self.sigmoid(h)
        return h



from load_dataset import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

#
# DEFINE MODEL
#
model = GCN(N,num_features,10,5,2,1)

#
# HYPERPARAMS
#
lr = 0.01
betas = (0.9,0.99)
weight_decay = 0
epochs = 20

loss_fn = torch.nn.MSELoss(size_average=False)
#loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
#loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=lr, 
                             betas=betas,
                             weight_decay=0)


best_auc = 0
best_model = GCN(N,num_features,10,5,2,1)

import tqdm
from sklearn import metrics
for t in range(epochs):
    print('Epoch: ',t,'/',epochs)
    y_pred = model(A,X)
    #loss = loss_fn(y_pred, y)
    #print(y.size())
    #print(y_pred.size())
    #loss = torch.sum((y - y_pred)**2))
    hinge_loss = y - y_pred + 1
    hinge_loss[hinge_loss < 0] = 0
    loss = torch.sum(hinge_loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Find AUC
    #y_predi = torch.abs(1-y_pred)
    y_hat = y_pred.data.numpy()
    y_true = y.data.numpy()
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat)
    AUC = metrics.auc(fpr, tpr)
    if AUC > best_auc:
        best_auc = AUC
        best_model = model
    print('AUC SCORE: ',AUC, ' Loss: ',loss.item())


# Find AUC for Test Set
y_pred = best_model(A_test,X_test)
y_true = y_test.data.numpy()
y_hat = y_pred.data.numpy()
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat)
AUC = metrics.auc(fpr, tpr)
print('Test AUC SCORE: ',AUC)



