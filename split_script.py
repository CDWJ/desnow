import numpy as np
import importlib.machinery
import sys
import os
import copy
import pandas as pd
from matplotlib import image
from sklearn import model_selection

path = "./all/synthetic/"
dir_list = os.listdir(path)
train_input, val_input = model_selection.train_test_split(dir_list, test_size=int(len(dir_list) * 0.3), train_size=int(len(dir_list) * 0.7))
np.save('./all/train_data.npy', train_input)
np.save('./all/validate_data.npy', val_input)
# dict_csv = []
# count = 0
# for i in dir_list:
#     dict_csv.append([image.imread(f"./all/mask/{i}"), image.imread(f"./all/gt/{i}"), image.imread(f"./all/synthetic/{i}")])         
#     count += 1
#     if count % 1000 == 0:
#         print(count, end='#')

# train_input, val_input = model_selection.train_test_split(dict_csv, 
#                                                            test_size=int(len(dict_csv) * 0.3), 
#                                                            train_size=int(len(dict_csv) * 0.7))
# np.save('./all/train_dataset.npy', train_input)
# np.save('./all/validate_dataset.npy', val_input)

