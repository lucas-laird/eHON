import os
os.chdir('../')
import sys
sys.path.append(os.getcwd())

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import os.path as osp
import shutil
from pathlib import Path
import json
import time
import argparse
from dataset import ShapeNet_core, SimplexData, create_simplex_data
from models import EquivariantHON_centroid

def train_HON(model, train_loader, optimizer, device = 'cuda:0'):
    model.train()
    accs = []
    losses = []
    for train_step, batch in enumerate(train_loader):
        h1, h2, h3 = batch.h0.to(device), batch.h1.to(device), batch.h2.to(device)
        x1 = batch.x0.to(device)
        batch1, batch2, batch3 = batch.h0_batch.to(device), batch.h1_batch.to(device), batch.h2_batch.to(device)
        b1, b2 = batch.bound0_index.to(device), batch.bound1_index.to(device)

        labels = batch.y.to(device)

        optimizer.zero_grad()
        #pred = model(H, X, B_up, B_down, H_batch, X_batch)
        pred = model(h1, h2, h3, x1, b1, b2, batch1, batch2, batch3)
        #print("==================== Train Step {} =============================".format(train_step))
        #print(torch.cuda.memory_allocated(device = device))

        loss = F.nll_loss(pred, labels)
        loss.backward()
        optimizer.step()
        #print("==================== Train Step {} =============================".format(train_step))
        #print(loss.item())
        pred_labels = torch.argmax(pred, dim = -1)
        
        tmp = (labels == pred_labels).cpu().float().mean()
        accs.append(tmp.item())
        losses.append(loss.cpu().item())
    return accs, losses, np.mean(accs), np.mean(losses)

def eval_HON(model, test_loader, device = 'cuda:0'):
    with torch.no_grad():
        model.eval()
        accs = []
        losses = []
        for test_step, batch in enumerate(test_loader):
            h1, h2, h3 = batch.h0.to(device), batch.h1.to(device), batch.h2.to(device)
            x1 = batch.x0.to(device)
            batch1, batch2, batch3 = batch.h0_batch.to(device), batch.h1_batch.to(device), batch.h2_batch.to(device)
            b1, b2 = batch.bound0_index.to(device), batch.bound1_index.to(device)
            labels = batch.y.to(device)

            #pred = model(H, X, B_up, B_down, H_batch, X_batch)
            pred = model(h1, h2, h3, x1, b1, b2, batch1, batch2, batch3)

            loss = F.nll_loss(pred, labels)
            pred_labels = torch.argmax(pred, dim = -1)
            tmp = (labels == pred_labels).cpu().float().mean()
            accs.append(tmp.item())
            losses.append(loss.cpu().item())
        return np.mean(accs), np.mean(losses)

def train_epochs(model, train_loader, test_loader, optimizer, num_epochs = 10, scheduler = None, log_iter = 10, device = 'cuda:0', log_dir = "eHON_checkpoints/"):
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    full_training_accs = []
    full_training_losses = []
    full_avg_acc = []
    full_avg_loss = []
    epoch_times = []
    previous_checkpoint = -1
    for e in tqdm(range(num_epochs)):
        t1 = time.time()
        training_accs, training_losses, avg_acc, avg_loss = train_HON(model, train_loader, optimizer, device = device)
        full_training_accs = full_training_accs + training_accs
        full_training_losses = full_training_losses + training_losses
        full_avg_acc.append(avg_acc)
        full_avg_loss.append(avg_loss)
        if scheduler is not None:
            scheduler.step()
        t2 = time.time()
        epoch_times.append(t2-t1)
        if (e > 0) and (e%log_iter == 0):
            path = osp.join(log_dir, "model_checkpoint_latest")
            if previous_checkpoint > 0:
                new_path = osp.join(log_dir, "model_checkpoint{}".format(previous_checkpoint))
                if not osp.exists(new_path):
                    Path(new_path).mkdir(parents = True, exist_ok = True)
                for f in os.listdir(path):
                    shutil.move(osp.join(path, f), osp.join(new_path, f))
            
            previous_checkpoint = e
            if not osp.exists(path):
                Path(path).mkdir(parents = True, exist_ok = True)
            
            torch.save(model.state_dict(), osp.join(path, "model_state_dict.pth"))
            eval_acc, eval_loss = eval_HON(model, test_loader, device = device)
            results_dict= {"train acc": full_training_accs,
                           "train loss": full_training_losses, 
                           "avg epoch acc": full_avg_acc,
                           "avg epoch loss": full_avg_loss,
                           "eval acc": eval_acc,
                           "eval loss": eval_loss, 
                           "num epochs trained": e,
                           "trainable params": model_total_params}
            with open(osp.join(path, "results_dict.json"), "w") as f:
                json.dump(results_dict, f)
                
    path = osp.join(log_dir, "model_checkpoint_latest")
    if previous_checkpoint < num_epochs-1:
        new_path = osp.join(log_dir, "model_checkpoint{}".format(previous_checkpoint))
        if not osp.exists(new_path):
            Path(new_path).mkdir(parents = True, exist_ok = True)
        for f in os.listdir(path):
            shutil.move(osp.join(path, f), osp.join(new_path, f))
    
    if not osp.exists(path):
        Path(path).mkdir(parents = True, exist_ok = True)
    Path(path).mkdir(parents = True, exist_ok = True)

    torch.save(model.state_dict(), osp.join(path, "model_state_dict.pth"))
    eval_acc, eval_loss = eval_HON(model, test_loader, device = device)
    results_dict= {"train acc": full_training_accs,
                   "train loss": full_training_losses, 
                   "avg epoch acc": full_avg_acc,
                   "avg epoch loss": full_avg_loss,
                   "eval acc": eval_acc,
                   "eval loss": eval_loss, 
                   "num epochs trained": num_epochs,
                   "trainable params": model_total_params}
    with open(osp.join(path, "results_dict.json"), "w") as f:
        json.dump(results_dict, f)
    #print("average training epoch time = {}".format(np.mean(epoch_times)))
    return full_training_accs, full_training_losses, full_avg_acc, full_avg_loss
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Process some args.")
    parser.add_argument('--lr', default = 1e-4, type = float, help = 'learning rate')
    parser.add_argument('--ne', default = 50, type = int, help = 'number of training epochs')
    parser.add_argument('--log', default = "models/", type = str, help = "log output folder")
    parser.add_argument("--dataset_name", type = str, default = "simplex_data", help = "Two options 10k training ex. or 1k training ex")
    parser.add_argument("--log_iter", default = 10, type = int, help = "how often to log")
    parser.add_argument("--pooling", default = "mean", type = str, help = "What type of global pooling to use")
    parser.add_argument("--x_agg", default = "sum", type = str, help = "what type of aggregation to use for x feats.")
    parser.add_argument("--residual", default = True, type = eval, help = "Use residual h features not?")
    parser.add_argument("--depth", default = 4, type = int, help = "Network Depth")
    
    args = parser.parse_args()
    depth = args.depth
    residual = args.residual
    pooling = args.pooling
    x_agg = args.x_agg
    data_name = args.dataset_name
    lr = args.lr
    num_epochs = args.ne
    if residual:
        log_dir = "/work/gllab/laird.lucas/" + args.log + "centroid_residual_agg-{}_pooling-{}".format(x_agg, pooling)
    else:
        log_dir = "/work/gllab/laird.lucas/" + args.log + "centroid_agg-{}_pooling-{}".format(x_agg, pooling)
    log_iter = args.log_iter

    root_dir = "/work/gllab/laird.lucas/ShapeNet_core/"
    train_dataset = ShapeNet_core(root_dir, name = data_name, mini = False, pre_transform = create_simplex_data(), split = 'train')
    test_dataset = ShapeNet_core(root_dir, name = data_name, mini = False, pre_transform = create_simplex_data(), split = 'test')

    follow_batch = []
    for i in range(3):
        follow_batch.append("h{}".format(i))
        follow_batch.append("x{}".format(i))
    train_loader = DataLoader(train_dataset, batch_size = 8, num_workers = 4, follow_batch = follow_batch)
    test_loader = DataLoader(train_dataset, batch_size = 1, num_workers = 1, follow_batch = follow_batch)

    in_nfs = [1,1,1]
    hidden_nfs = [8,8,8]
    output_dim = 55
    device = 'cuda:0'
    model = EquivariantHON_centroid(in_nfs, hidden_nfs, output_dim, depth = depth, mpl_intermed = 24, mlp_dims = [128, 128], coords_agg = x_agg, pooling = pooling, residual = residual)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr = lr,
                           weight_decay = 1e-12)
    scheduler = None
    #scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                      10,
    #                                      gamma = 0.9)
    #accs, losses, avg_acc, avg_loss = train_HON(model, train_loader, optimizer, device)
    #print(accs)
    #print(losses)
    #print(avg_acc, avg_loss)
    
    full_training_accs, full_training_losses, full_avg_acc, full_avg_loss = train_epochs(model, train_loader, test_loader, num_epochs = num_epochs,
                                                                                         optimizer = optimizer, scheduler = scheduler, 
                                                                                         log_iter = log_iter, log_dir = log_dir)