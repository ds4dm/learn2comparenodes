#!/usr/bin/env python
# coding: utf-8

# In[90]:


import sys
import os
import re
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
from utils import record_stats, display_stats, distribute
from pathlib import Path 


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[91]:


n_cpu = 16
n_instance = -1
nodesels = [ 'estimate_dummy', 'expert_dummy', 'gnn_dummy_nprimal=2', 'svm_dummy_nprimal=2']
problem = 'WPMS'
normalize = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbose = True
on_log = False
default = True
delete = False

if delete:
    try:
        import shutil
        shutil.rmtree(os.path.join(os.path.abspath(''), 
                                       f'stats/{problem}'))
    except:
        ''



instances = list(Path(os.path.join(os.path.abspath(''), 
                                   f"./problem_generation/data/{problem}/test")).glob("*.lp"))
if n_instance == -1 :
    n_instance = len(instances)

import random
random.shuffle(instances)
instances = instances[:n_instance]

print("Evaluation")
print(f"  Problem:                    {problem}")
print(f"  n_instance/problem:         {len(instances)}")
print(f"  Nodeselectors evaluated:    {','.join( ['default' if default else '' ] + nodesels)}")
print(f"  Device for GNN inference:   {device}")
print(f"  Normalize features:         {normalize}")
print("----------------")



# In[92]:


#Run benchmarks

processes = [  Process(name=f"worker {p}", 
                                target=partial(record_stats,
                                                nodesels=nodesels,
                                                instances=instances[p1:p2], 
                                                problem=problem,
                                                device=torch.device(device),
                                                normalize=normalize,
                                                verbose=verbose,
                                                default=default))
                for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]  


try:
    set_start_method('spawn')
except RuntimeError:
    ''

a = list(map(lambda p: p.start(), processes)) #run processes
b = list(map(lambda p: p.join(), processes)) #join processes

