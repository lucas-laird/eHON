import torch
import torch.nn as nn
import torch.functional as F
from HON_networks import *
import egnn_clean as eg
from util import compute_centroids

class EquivariantHON(torch.nn.Module):
    def __init__(
        self,
        input_nfs, # Dim of h features for every cell rank. Should be of size K where K is the max rank cell
        hidden_nfs, # Intermediate dimension of the hidden h features for ever cell rank. Should be of size NxK where N = number of message-passing rounds.
        output_dim, # Number of classes in the classification problem
        mpl_intermed = 64,
        depth = 4,
        mlp_hidden = 2, # Number of layers in the final MLP classification model
        mlp_dims = [128,256], # Dimension of the final mlp layers
        mlp_activation = nn.ReLU(), # Final MLP activation function
        eHON_type = "boundary",
        pooling = "max", 
        coords_agg = "mean",
        residual = False
    ):
        super(EquivariantHON, self).__init__()

        
        self.input_nfs = input_nfs
        self.output_dim = output_dim
        self.hidden_nfs = hidden_nfs
        #self.x_nfs = x_nfs
        self.mlp_hidden = mlp_hidden
        self.mlp_activation = mlp_activation
        self.pooling = pooling
        self.residual = residual
        self.K = len(input_nfs)
        self.N = depth
        self.init_layers = nn.ModuleList()
        for j in range(self.K):
            layer = nn.Linear(input_nfs[j], hidden_nfs[j])
            self.init_layers.append(layer)
        cell_layers = nn.ModuleList() #List of lists of layers. MPL round i, cell rank j is cell_layers[i][j]
        #init_layers = nn.ModuleList()
        #for j in range(self.K):
        #    if j == 0:
        #        init_layer = eHON_MPL_boundary(input_nfs[j], hidden_nfs[0][j], 
        #                                       hidden_nf = mpl_intermed, upper_nf = input_nfs[j+1], lower_nf = None, aggregate = coords_agg)
        #    elif j == self.K-1:
        #        init_layer = eHON_MPL_boundary(input_nfs[j], hidden_nfs[0][j], 
        #                                       hidden_nf = mpl_intermed, upper_nf = None, lower_nf = input_nfs[j-1], aggregate = coords_agg)
        #    else:
        #        init_layer = eHON_MPL_boundary(input_nfs[j], hidden_nfs[0][j], 
        #                                       hidden_nf = mpl_intermed, upper_nf = input_nfs[j+1], lower_nf = input_nfs[j-1], aggregate = coords_agg)
        #    init_layers.append(init_layer)
        #cell_layers.append(init_layers)
        
        for i in range(self.N):
            mpl_layers = nn.ModuleList()
            for j in range(self.K):
                in_nf = hidden_nfs[j]
                out_nf = hidden_nfs[j]
                if j == 0:
                    upper_nf = hidden_nfs[j+1]
                    lower_nf = None # There isn't anything below
                elif j == self.K-1:
                    upper_nf = None # There isn't anything above
                    lower_nf = hidden_nfs[j-1]
                else:
                    upper_nf = hidden_nfs[j+1] 
                    lower_nf = hidden_nfs[j-1]
                layer = eHON_MPL_boundary(in_nf, out_nf, 
                                          hidden_nf = mpl_intermed, upper_nf = upper_nf, lower_nf = lower_nf, aggregate = coords_agg, residual = self.residual)
                mpl_layers.append(layer)
            cell_layers.append(mpl_layers)
        
        #Final flattened output size will be (sum(final hidden feature dim for each rank) + num cells*3)
        self.output_size = sum(hidden_nfs[:])
        self.MPL_layers = cell_layers
        
        final_layers = nn.ModuleList()
        final_layers.append(nn.Linear(self.output_size, mlp_dims[0]))
        for i in range(1,mlp_hidden):
            final_layers.append(nn.Linear(mlp_dims[i-1], mlp_dims[i]))
            final_layers.append(mlp_activation)
        final_layers.append(nn.Linear(mlp_dims[-1], output_dim))
        self.class_model = nn.Sequential(*final_layers)
        #print(self.class_model)
    
    def forward(self, h_1, h_2, h_3, x_1, x_2, x_3, b_1, b_2, batch1, batch2, batch3):
        # H should be list of length K where H[0] = vertices, H[1] = edges, etc.
        # X should be list of length K same as above
        # A should be list of length K-1 where A[0] = adjacency for vertices, etc. No adjacency for the highest cell rank since there is no upper adj.
        # C should be list of length K-1 where C[0] = co-adj for EDGES, C[1] = co-adj for faces, etc. No co-adj for vertices since there is no lower adj.
        # B_up should be list of length K-1 where B_up[0] = Boundary from vertices to edges, etc.
        # B_down should be list of length K-1 where B_down[0] = Boundary down from edges to vertices, etc.
        
        # H_batch should be the list (len = K) of batch assignments for each cell rank
        # X_batch should be the list (len = K) of batch assignments for each cell rank
        h_1 = self.init_layers[0](h_1)
        h_2 = self.init_layers[1](h_2)
        h_3 = self.init_layers[2](h_3)
        
        for i in range(self.N):
            h_1_new, x_1_new = self.MPL_layers[i][0](h_1, h_2, None, x_1, x_2, None, b_1, None)
            h_2_new, x_2_new = self.MPL_layers[i][1](h_2, h_3, h_1, x_2, x_3, x_1, b_2, b_1)
            h_3_new, x_3_new = self.MPL_layers[i][2](h_3, None, h_2, x_3, None, x_2, None, b_2)
            
            h_1 = h_1_new
            h_2 = h_2_new
            h_3 = h_3_new
            x_1 = x_1_new
            x_2 = x_2_new
            x_3 = x_3_new
        if self.pooling == "mean":
            h_1_batch = global_mean_pool(h_1_new, batch1)
            h_2_batch = global_mean_pool(h_2_new, batch2)
            h_3_batch = global_mean_pool(h_3_new, batch3)
            
        elif self.pooling == "max":
            h_1_batch = global_max_pool(h_1_new, batch1)
            h_2_batch = global_max_pool(h_2_new, batch2)
            h_3_batch = global_mean_pool(h_3_new, batch3)
        
        H = torch.cat([h_1_batch, h_2_batch, h_3_batch], dim = -1)
        logits = self.class_model(H)
        return F.log_softmax(logits, dim = -1)

class EquivariantHON_centroid(torch.nn.Module):
    def __init__(
        self,
        input_nfs, # Dim of h features for every cell rank. Should be of size K where K is the max rank cell
        hidden_nfs, # Intermediate dimension of the hidden h features for ever cell rank. Should be of size NxK where N = number of message-passing rounds.
        output_dim, # Number of classes in the classification problem
        mpl_intermed = 64,
        depth = 4,
        mlp_hidden = 2, # Number of layers in the final MLP classification model
        mlp_dims = [128,256], # Dimension of the final mlp layers
        mlp_activation = nn.ReLU(), # Final MLP activation function
        eHON_type = "boundary",
        pooling = "max", 
        coords_agg = "mean",
        residual = True
    ):
        super(EquivariantHON_centroid, self).__init__()

        
        self.input_nfs = input_nfs
        self.output_dim = output_dim
        self.hidden_nfs = hidden_nfs
        #self.x_nfs = x_nfs
        self.mlp_hidden = mlp_hidden
        self.mlp_activation = mlp_activation
        self.pooling = pooling
        
        self.K = len(input_nfs)
        self.N = depth
        cell_layers = nn.ModuleList() #List of lists of layers. MPL round i, cell rank j is cell_layers[i][j]
        self.init_layers = nn.ModuleList()
        for j in range(self.K):
            layer = nn.Linear(input_nfs[j], hidden_nfs[j])
            self.init_layers.append(layer)
        
        for i in range(self.N):
            mpl_layers = nn.ModuleList()
            for j in range(self.K):
                in_nf = hidden_nfs[j]
                out_nf = hidden_nfs[j]
                learn_x = False
                if j == 0:
                    upper_nf = hidden_nfs[j+1]
                    lower_nf = None # There isn't anything below
                    learn_x = True
                elif j == self.K-1:
                    upper_nf = None # There isn't anything above
                    lower_nf = hidden_nfs[j-1]
                else:
                    upper_nf = hidden_nfs[j+1] 
                    lower_nf = hidden_nfs[j-1]
                layer = eHON_MPL_centroid(in_nf, out_nf, 
                                          hidden_nf = mpl_intermed, upper_nf = upper_nf, lower_nf = lower_nf, learn_x = learn_x, aggregate = coords_agg, residual = residual)
                mpl_layers.append(layer)
            cell_layers.append(mpl_layers)
        
        #Final flattened output size will be (sum(final hidden feature dim for each rank) + num cells*3)
        self.output_size = sum(hidden_nfs[:])
        self.MPL_layers = cell_layers
        
        final_layers = nn.ModuleList()
        final_layers.append(nn.Linear(self.output_size, mlp_dims[0]))
        for i in range(1,mlp_hidden):
            final_layers.append(nn.Linear(mlp_dims[i-1], mlp_dims[i]))
            final_layers.append(mlp_activation)
        final_layers.append(nn.Linear(mlp_dims[-1], output_dim))
        self.class_model = nn.Sequential(*final_layers)
        #print(self.class_model)
        
    def forward(self, h_1, h_2, h_3, x_1, b_1, b_2, batch1, batch2, batch3):
        # H should be list of length K where H[0] = vertices, H[1] = edges, etc.
        # X should be list of length K same as above
        # A should be list of length K-1 where A[0] = adjacency for vertices, etc. No adjacency for the highest cell rank since there is no upper adj.
        # C should be list of length K-1 where C[0] = co-adj for EDGES, C[1] = co-adj for faces, etc. No co-adj for vertices since there is no lower adj.
        # B_up should be list of length K-1 where B_up[0] = Boundary from vertices to edges, etc.
        # B_down should be list of length K-1 where B_down[0] = Boundary down from edges to vertices, etc.
        
        # H_batch should be the list (len = K) of batch assignments for each cell rank
        # X_batch should be the list (len = K) of batch assignments for each cell rank
        h_1 = self.init_layers[0](h_1)
        h_2 = self.init_layers[1](h_2)
        h_3 = self.init_layers[2](h_3)
        for i in range(self.N):
            x_2 = compute_centroids(x_1, b_1, simplex_dim = 2)
            x_3 = compute_centroids(x_2, b_2, simplex_dim = 3)
            
            h_1_new, x_1_new = self.MPL_layers[i][0](h_1, h_2, None, x_1, x_2, None, b_1, None)
            h_2_new = self.MPL_layers[i][1](h_2, h_3, h_1, x_2, x_3, x_1, b_2, b_1)
            h_3_new = self.MPL_layers[i][2](h_3, None, h_2, x_3, None, x_2, None, b_2)
            
            h_1 = h_1_new
            h_2 = h_2_new
            h_3 = h_3_new
            x_1 = x_1_new
        if self.pooling == "mean":
            h_1_batch = global_mean_pool(h_1_new, batch1)
            h_2_batch = global_mean_pool(h_2_new, batch2)
            h_3_batch = global_mean_pool(h_3_new, batch3)

            X = global_mean_pool(x_1_new, batch1)
        elif self.pooling == "max":
            h_1_batch = global_max_pool(h_1_new, batch1)
            h_2_batch = global_max_pool(h_2_new, batch2)
            h_3_batch = global_max_pool(h_3_new, batch3)

            X = global_max_pool(x_1_new, batch1)
        
        H = torch.cat([h_1_batch, h_2_batch, h_3_batch], dim = -1)
        
        logits = self.class_model(H)
        return F.log_softmax(logits, dim = -1)

class en_gnn(torch.nn.Module):
    def __init__(self,in_nfs, hidden_nf, out_nf, num_classes = 55, mlp_dims = [128,128], residual = False, coords_agg = "mean", pooling = "max"):
        super(en_gnn, self).__init__()
        self.pooling = pooling
        self.egnn = eg.EGNN(in_node_nf = in_nfs, hidden_nf = hidden_nf, out_node_nf = out_nf, in_edge_nf = 0, residual = residual, coords_agg = coords_agg)
        self.fc = nn.Sequential(
            nn.Linear(out_nf, mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], num_classes)
        )
    def forward(self, h, x, edge_index, batch, edge_attr = None):
        h, x = self.egnn(h,x,edge_index, edge_attr)
        if self.pooling == "mean":
            h_pool = global_mean_pool(h, batch)
            #x_pool = global_mean_pool(x, batch)
        elif self.pooling == "max":
            h_pool = global_max_pool(h, batch)
            #x_pool = global_max_pool(x, batch)
        logits = self.fc(h_pool)
        return F.log_softmax(logits, dim = -1)