import os
os.chdir('../')
import sys
sys.path.append(os.getcwd())

from dataset import ShapeNet_core, create_simplex_data

root_dir = "/work/gllab/laird.lucas/ShapeNet_core/"
dataset = ShapeNet_core(root_dir, name = "additional5k_simplex_data", mini = False, pre_transform = create_simplex_data())