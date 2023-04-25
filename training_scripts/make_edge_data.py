import os
os.chdir('../')
import sys
sys.path.append(os.getcwd())

from dataset import ShapeNet_core, reformat_data

root_dir = "/work/gllab/laird.lucas/ShapeNet_core/"
dataset = ShapeNet_core(root_dir, name = "even10k_edge_data", mini = False, pre_transform = reformat_data())