import torch 
import os
import os.path as osp
import shutil
from pathlib import Path
from util import coo_2_torch_tensor
import networkx as nx
from torch_geometric.transforms import BaseTransform, FaceToEdge
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected, to_networkx
import toponetx as tnx
import numpy as np

class ShapeNet_core(InMemoryDataset):
    def __init__(self, root, transform = None, pre_transform = None, pre_filter = None, 
                 name = None, mini = False, split = "train"):
        self.mini = mini
        self.name = name
        super(ShapeNet_core, self).__init__(root, transform, pre_transform, pre_filter)
        #self.root = root
        #self.name = name
        #self.mini = mini
        #self.pre_transform = pre_transform
        if split == "train":
            path = self.processed_paths[0]
        elif split == "test":
            path = self.processed_paths[1]
        self.data, self.slices = torch.load(path)
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        if "subsampled1000" in self.name:
            return ["subsampled1000_train_test_dict.json"]
        elif "even10k" in self.name:
            return ["even10k_train_test_dict.json"]
        elif "even1k" in self.name:
            return ["even1k_train_test_dict.json"]
        else:
            return ["subsampled_train_test_dict.json"]
            
    @property
    def processed_file_names(self):
        return ["train.pth", "test.pth"]
    
    def process(self):
        with open(self.raw_paths[0], "r") as f:
            train_test_dict = json.load(f)
        train_names = train_test_dict["train"]
        if self.mini:
            train_names = train_names[0:10000]
        train_data = []
        for name in train_names:
            #print(name)
            file_name = name + ".pth"
            data = torch.load(osp.join(self.raw_dir, file_name))
            #print(self.pre_transform)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            train_data.append(data)
            #print(data)
        torch.save(self.collate(train_data), self.processed_paths[0])
        
        
        test_data = []
        test_names = train_test_dict["test"]
        if self.mini:
            test_names = test_names[0:2000]
        for name in test_names:
            #print(name)
            file_name = name + ".pth"
            data = torch.load(osp.join(self.raw_dir, file_name))
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            test_data.append(data)
            #print(data)
        torch.save(self.collate(test_data), self.processed_paths[1])
        
class SimplexData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'bound0_index':
            return torch.tensor([[self.h0.size(0)], [self.h1.size(0)]])
        if key == 'bound1_index':
            return torch.tensor([[self.h1.size(0)], [self.h2.size(0)]])
        if key == 'cobound1_index':
            return torch.tensor([[self.h1.size(0)], [self.h0.size(0)]])
        if key == 'cobound2_index':
            return torch.tensor([[self.h2.size(0)], [self.h1.size(0)]])
        if key == 'edge_index':
            return torch.tensor([[self.h0.size(0)], [self.h0.size(0)]])
        return super().__inc__(key, value, *args, kwargs)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0

class create_simplex_data(BaseTransform):
    
    def __call__(self, d):
        num_nodes = d.x.size(0)
        f2e = FaceToEdge()
        edge_list = f2e(Data(face = d.faces)).edge_index
        temp = Data(edge_index = edge_list, num_nodes = num_nodes)
        G = to_networkx(temp, to_undirected = True, remove_self_loops = True)
        SC = tnx.SimplicialComplex(G)
        faces = d.faces.t().unique(dim = 0).numpy()
        try:
            SC.add_simplices_from(faces)
        except:
            num_faces = faces.shape[0]
            bad_faces = []
            for i in range(num_faces):
                face = faces[i,:]
                if face[0] == face[1] or face[1] == face[2] or face[0] == face[2]:
                    bad_faces.append(i)
            faces = np.delete(faces, bad_faces, axis = 0)
            SC.add_simplices_from(faces)
        N = SC.dim
        num_cells = SC.shape
        x0 = d.x
        data_dict = {}
        data_dict["num_nodes"] = num_nodes
        data_dict["x0"] = x0
        data_dict["y"] = d.y
        for i in range(N+1):
            if i < N:
                bound = coo_2_torch_tensor(SC.incidence_matrix(i+1)).coalesce()
                bound_i, bound_j = bound.indices()
                bound_list = torch.stack([bound_i, bound_j], dim = 1).t()
                data_dict["bound{}_index".format(i)] = bound_list
                #data_dict["bound{}".format(i)] = coo_2_torch_tensor(SC.incidence_matrix(i+1)).to_dense() # Boundary matrix should be of size num_cells[i] x num_cells[i+1], This describes the upper messages (messages received from rank k+1)
                #data_dict["adj{}".format(i)] = coo_2_torch_tensor(SC.adjacency_matrix(i)).to_dense()
            if i > 0:
                cobound = coo_2_torch_tensor(SC.coincidence_matrix(i)).coalesce()
                cobound_i, cobound_j = cobound.indices()
                cobound_list = torch.stack([cobound_i, cobound_j], dim = 1).t()
                data_dict["cobound{}_index".format(i)] = cobound_list
                skeleton = SC.skeleton(i)
                skeleton = [torch.IntTensor(list(s)) for s in skeleton]
                x = torch.cat([x0.index_select(0, skeleton[j]).unsqueeze(0) for j in range(len(skeleton))]).mean(1)
                data_dict["x{}".format(i)] = x
            data_dict["h{}".format(i)] = torch.ones([num_cells[i],1])
        data = SimplexData.from_dict(data_dict)
        return(data)

class reformat_data(BaseTransform):
    def __call__(self, d):
        num_nodes = d.x.size(0)
        f2e = FaceToEdge()
        edge_list = f2e(Data(face = d.faces)).edge_index
        data = Data(x = d.x, edge_index = edge_list, y = d.y, face = d.faces)
        return(data)