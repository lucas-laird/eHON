import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F
from util import unsorted_segment_sum, unsorted_segment_mean

class eHON_MPL_boundary(nn.Module):
    def __init__(
        self,
        input_nf, #Size of h features on cell
        output_nf, #Output size of next hf features on cell
        hidden_nf,
        upper_nf = None,
        lower_nf = None,
        mlp_activation=nn.ReLU(inplace = True),
        aggregate = "sum",
        residual = True):
        super(eHON_MPL_boundary, self).__init__()
        
        self.input_nf = input_nf
        self.upper_nf = upper_nf
        self.lower_nf = lower_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.aggregate = aggregate
        x_diff_nf = 1
        self.mlp_activation = mlp_activation
        self.residual = residual
        
        self.coord_weights = torch.nn.Parameter(data = torch.Tensor(2), requires_grad = True)
        self.coord_weights.data.uniform_(-1,1)
        
        mess_nf = input_nf
        
        if upper_nf is not None:
        
            #adj_mlp takes in concatenated h_i,h_j,||x_i - x_j|| and 
            #outputs m_adj_ij
            self.boundary_up_mlp = nn.Sequential(
                nn.Linear(input_nf+upper_nf+x_diff_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, hidden_nf)
            )
            mess_nf += hidden_nf
        
        if lower_nf is not None:
            self.boundary_down_mlp = nn.Sequential(
                nn.Linear(input_nf+lower_nf+x_diff_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, hidden_nf)
            )
            mess_nf += hidden_nf
        #cell_mlp takes in concatenated h_i,agg(messages) and
        #outputs a new h_i with dim = output_nf
        self.cell_mlp = nn.Sequential(
            nn.Linear(mess_nf, hidden_nf),
            nn.Dropout(p = 0.5, inplace = True),
            mlp_activation,
            nn.Linear(hidden_nf, output_nf)
        )
        if upper_nf is not None:
            self.coord_up_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )
        if lower_nf is not None:
            self.coord_down_mlp = nn.Sequential(
                nn.Linear(hidden_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, 1), 
                nn.Sigmoid()
            )
    
    def boundary_model(self, h1, h2, x1, x2, b_i, b_j):
        assert isinstance(h1, Tensor)
        assert isinstance(x1, Tensor)
        assert isinstance(h2, Tensor)
        assert isinstance(x2, Tensor)
        assert h1.shape[0] == x1.shape[0] #Every cell has both an h and an x feature so first dim must be the same size
        assert h2.shape[0] == x2.shape[0] #Every cell has both an h and an x feature so first dim must be the same size
        
        h_i = h1.index_select(-2, b_i) #Num nonzero in B x F
        h_j = h2.index_select(-2, b_j) #Num nonzero in B x F
        x_i = x1.index_select(-2, b_i) #Get all x_i
        x_j = x2.index_select(-2, b_j) #Get all x_j
        x_messages = torch.sum(torch.square(x_i - x_j), -1).unsqueeze(-1) #Get squared euclidean distance, unsqueeze for broadcasting
        
        #values = b.coalesce().values()
        #h_j = torch.mul(values.unsqueeze(-1), h_j)
        #x_messages = torch.mul(values.unsqueeze(-1), x_messages)
        
        #message_input = torch.cat([h_i, h_j, x_messages, vol_up], dim = -1) # Num_nonzero x F+F+1
        message_input = torch.cat([h_i, h_j, x_messages], dim = -1) # Num_nonzero x F+F+1
        return message_input
        #messages = self.boundary_up_mlp(message_input) # Num_nonzero x hidden_nf - m_ij
        
        #return messages
    
    def cell_model(self, h, m):
        h_upd = self.cell_mlp(torch.cat([h,m], dim = -1))
        if self.residual:
            h = h+h_upd
        else:
            h = h_upd
        return h
    
    def coord_model(self, x1, x2, b_i, b_j, m, direction = 1):
        #values = b.coalesce().values().unsqueeze(-1)
        
        x_i = x1.index_select(-2, b_i)
        x_j = x2.index_select(-2, b_j)
        x_ij = x_i - x_j
        #x_ij = torch.mul(values, x_i-x_j)
        
        if direction == 1:
            scale = self.coord_up_mlp(m)
        elif direction == 0:
            scale = self.coord_down_mlp(m)
        x_ij = torch.mul(x_ij, scale)
        if self.aggregate == "sum":
            return unsorted_segment_sum(x_ij, b_i, x1.size(0))
        elif self.aggregate =="mean":
            return unsorted_segment_mean(x_ij, b_i, x1.size(0))
    
    def forward(self, h, h_up, h_down, x, x_up, x_down, b_up, b_down):
        m_list = []
        x_list = []
        if b_up is not None:
            b_up_i, b_up_j = b_up
            #print(b_up.shape)
            #print(b_up_i.shape)
            #print(b_up_j.shape)
            m_up_in = self.boundary_model(h, h_up, x, x_up, b_up_i, b_up_j)
            m_up_in = self.boundary_up_mlp(m_up_in)
            #print(b_up_i)
            #print(m_up_ij.size())
            m_up = unsorted_segment_sum(m_up_in, b_up_i, h.size(0))
            m_list.append(m_up)
            
            x_up = self.coord_model(x, x_up, b_up_i, b_up_j, m_up_in, direction = 1)
            x_up = self.coord_weights[0]*x_up
            x_list.append(x_up)
            
        if b_down is not None:
            b_down_j, b_down_i = b_down #Take transpose of boundary relations by switching i and j.
            m_down_in = self.boundary_model(h, h_down, x, x_down, b_down_i, b_down_j)
            m_down_in = self.boundary_down_mlp(m_down_in)
            m_down = unsorted_segment_sum(m_down_in, b_down_i, h.size(0))
            m_list.append(m_down)
            
            x_down = self.coord_model(x, x_down, b_down_i, b_down_j, m_down_in, direction = 0)
            x_down = self.coord_weights[1]*x_down
            x_list.append(x_down)
        
        m = torch.cat(m_list, dim = -1)
        h = self.cell_model(h, m)
        
        for x_val in x_list:
            x += x_val
        return (h, x)

class eHON_MPL_centroid(nn.Module):
    def __init__(
        self,
        input_nf, #Size of h features on cell
        output_nf, #Output size of next hf features on cell
        hidden_nf,
        upper_nf = None,
        lower_nf = None,
        mlp_activation=nn.ReLU(inplace = True),
        learn_x = True, 
        aggregate = "sum",
        residual = True
    ):
        super(eHON_MPL_centroid, self).__init__()
        
        self.input_nf = input_nf
        self.upper_nf = upper_nf
        self.lower_nf = lower_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.learn_x = learn_x
        x_diff_nf = 1
        self.mlp_activation = mlp_activation
        self.aggregate = aggregate
        
        self.coord_weights = torch.nn.Parameter(data = torch.Tensor(2), requires_grad = True)
        self.coord_weights.data.uniform_(-1,1)
        
        mess_nf = input_nf
        
        if upper_nf is not None:
        
            #adj_mlp takes in concatenated h_i,h_j,||x_i - x_j|| and 
            #outputs m_adj_ij
            self.boundary_up_mlp = nn.Sequential(
                nn.Linear(input_nf+upper_nf+x_diff_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, hidden_nf)
            )
            mess_nf += hidden_nf
        
        if lower_nf is not None:
            self.boundary_down_mlp = nn.Sequential(
                nn.Linear(input_nf+lower_nf+x_diff_nf, hidden_nf),
                nn.Dropout(p = 0.5, inplace = True),
                mlp_activation,
                nn.Linear(hidden_nf, hidden_nf)
            )
            mess_nf += hidden_nf
        #cell_mlp takes in concatenated h_i,agg(messages) and
        #outputs a new h_i with dim = output_nf
        self.cell_mlp = nn.Sequential(
            nn.Linear(mess_nf, hidden_nf),
            nn.Dropout(p = 0.5, inplace = True),
            mlp_activation,
            nn.Linear(hidden_nf, output_nf)
        )
        if self.learn_x:
            if upper_nf is not None:
                self.coord_up_mlp = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.Dropout(p = 0.5, inplace = True),
                    mlp_activation,
                    nn.Linear(hidden_nf, 1),
                    nn.Sigmoid()
                )
            if lower_nf is not None:
                self.coord_down_mlp = nn.Sequential(
                    nn.Linear(hidden_nf, hidden_nf),
                    nn.Dropout(p = 0.5, inplace = True),
                    mlp_activation,
                    nn.Linear(hidden_nf, 1), 
                    nn.Sigmoid()
                )
    
    def boundary_model(self, h1, h2, x1, x2, b_i, b_j):
        assert isinstance(h1, Tensor)
        assert isinstance(x1, Tensor)
        assert isinstance(h2, Tensor)
        assert isinstance(x2, Tensor)
        assert h1.shape[0] == x1.shape[0] #Every cell has both an h and an x feature so first dim must be the same size
        assert h2.shape[0] == x2.shape[0] #Every cell has both an h and an x feature so first dim must be the same size
        
        h_i = h1.index_select(-2, b_i) #Num nonzero in B x F
        h_j = h2.index_select(-2, b_j) #Num nonzero in B x F
        x_i = x1.index_select(-2, b_i) #Get all x_i
        x_j = x2.index_select(-2, b_j) #Get all x_j
        x_messages = torch.sum(torch.square(x_i - x_j), -1).unsqueeze(-1) #Get squared euclidean distance, unsqueeze for broadcasting
        
        #values = b.coalesce().values()
        #h_j = torch.mul(values.unsqueeze(-1), h_j)
        #x_messages = torch.mul(values.unsqueeze(-1), x_messages)
        
        #message_input = torch.cat([h_i, h_j, x_messages, vol_up], dim = -1) # Num_nonzero x F+F+1
        message_input = torch.cat([h_i, h_j, x_messages], dim = -1) # Num_nonzero x F+F+1
        return message_input
        #messages = self.boundary_up_mlp(message_input) # Num_nonzero x hidden_nf - m_ij
        
        #return messages
    
    def cell_model(self, h, m):
        return self.cell_mlp(torch.cat([h,m], dim=-1))
    
    def coord_model(self, x1, x2, b_i, b_j, m, direction = 1):
        #values = b.coalesce().values().unsqueeze(-1)
        
        x_i = x1.index_select(-2, b_i)
        x_j = x2.index_select(-2, b_j)
        x_ij = x_i - x_j
        #x_ij = torch.mul(values, x_i-x_j)
        
        if direction == 1:
            scale = self.coord_up_mlp(m)
        elif direction == 0:
            scale = self.coord_down_mlp(m)
        x_ij = torch.mul(x_ij, scale)
        if self.aggregate == "sum":
            return unsorted_segment_sum(x_ij, b_i, x1.size(0))
        elif self.aggregate =="mean":
            return unsorted_segment_mean(x_ij, b_i, x1.size(0))
    
    def forward(self, h, h_up, h_down, x, x_up, x_down, b_up, b_down):
        m_list = []
        if self.learn_x:
            x_list = []
        if b_up is not None:
            b_up_i, b_up_j = b_up
            #print(b_up.shape)
            #print(b_up_i.shape)
            #print(b_up_j.shape)
            m_up_in = self.boundary_model(h, h_up, x, x_up, b_up_i, b_up_j)
            m_up_in = self.boundary_up_mlp(m_up_in)
            #print(b_up_i)
            #print(m_up_ij.size())
            m_up = unsorted_segment_sum(m_up_in, b_up_i, h.size(0))
            m_list.append(m_up)
            
            if self.learn_x:
                x_up = self.coord_model(x, x_up, b_up_i, b_up_j, m_up_in, direction = 1)
                x_up = self.coord_weights[0]*x_up
                x_list.append(x_up)
            
        if b_down is not None:
            b_down_j, b_down_i = b_down #Take transpose of boundary relations by switching i and j.
            m_down_in = self.boundary_model(h, h_down, x, x_down, b_down_i, b_down_j)
            m_down_in = self.boundary_down_mlp(m_down_in)
            m_down = unsorted_segment_sum(m_down_in, b_down_i, h.size(0))
            m_list.append(m_down)
            
            if self.learn_x:
                x_down = self.coord_model(x, x_down, b_down_i, b_down_j, m_down_in, direction = 0)
                x_down = self.coord_weights[1]*x_down
                x_list.append(x_down)
        
        m = torch.cat(m_list, dim = -1)
        h = self.cell_model(h, m)
        if self.learn_x:
            for x_val in x_list:
                x += x_val
            return (h, x)
        else:
            return h