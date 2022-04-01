#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""

def load_src(name, fpath):
     import os, imp
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )
load_src("model", "../learning/model.py" )

import torch
import time
import numpy as np
from pyscipopt import Nodesel
from data_type import BipartiteGraphPairData
from model import GNNPolicy
from line_profiler import LineProfiler
from joblib import dump, load



class CustomNodeSelector(Nodesel):

    def __init__(self, sel_policy='', comp_policy=''):
        self.sel_policy = sel_policy
        self.comp_policy = comp_policy

        
    def nodeselect(self, policy=None):
        
        policy = policy if policy is not None else self.sel_policy
        
        if policy == 'estimate':
            res = self.estimate_nodeselect()
        elif policy == 'dfs':
            res = self.dfs_nodeselect()
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif policy == 'bfs':
            res = self.bfs_nodeselect()
        elif policy == 'random':
            res = self.random_nodeselect()
        else:
            res = {"selnode": self.model.getBestNode()}
            
        return res
    
    def nodecomp(self, node1, node2, policy=None):
        
        policy = policy if policy is not None else self.comp_policy
        
        if policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif policy == 'random':
            res = self.random_nodecomp(node1, node2)
        else:
            res = 0
            
        return res
    
    
    #Estimate 
    def estimate_nodeselect(self):
        return {'selnode':self.model.getEstimateSelNode() }
    
    def estimate_nodecomp(self, node1,node2):
        
        #estimate 
        estimate1 = node1.getEstimate()
        estimate2 = node2.getEstimate()
        if (self.model.isInfinity(estimate1) and self.model.isInfinity(estimate2)) or \
            (self.model.isInfinity(-estimate1) and self.model.isInfinity(-estimate2)) or \
            self.model.isEQ(estimate1, estimate2):
                lb1 = node1.getLowerbound()
                lb2 = node2.getLowerbound()
                
                if self.model.isLT(lb1, lb2):
                    return -1
                elif self.model.isGT(lb1, lb2):
                    return 1
                else:
                    ntype1 = node1.getType()
                    ntype2 = node2.getType()
                    CHILD, SIBLING = 3,2
                    
                    if (ntype1 == CHILD and ntype2 != CHILD) or (ntype1 == SIBLING and ntype2 != SIBLING):
                        return -1
                    elif (ntype1 != CHILD and ntype2 == CHILD) or (ntype1 != SIBLING and ntype2 == SIBLING):
                        return 1
                    else:
                        return -self.dfs_nodecomp(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
        
        
    # Depth first search        
    def dfs_nodeselect(self):
        
        selnode = self.model.getPrioChild()  #aka best child of current node
        if selnode == None:
            
            selnode = self.model.getPrioSibling() #if current node is a leaf, get 
            # a sibling
            if selnode == None: #if no sibling, just get a leaf
                selnode = self.model.getBestLeaf()
                
        return {"selnode": selnode}
    
    def dfs_nodecomp(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
    
    
    # Breath first search
    def breadthfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breadthfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
     
     #random
    def random_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    def random_nodecomp(self, node1,node2):
        return -1 if np.random.rand() < 0.5 else 1

    



class OracleNodeSelectorAbdel(CustomNodeSelector):

    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0):

        super().__init__() 
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
    
    
    
    def nodecomp(self, node1, node2, return_type=False):
        
        if self.oracle_type == "optimal_plunger":            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            
            if d1 and d2:
                res, comp_type = self.dfs_nodecomp(node1, node2), 0
            elif d1:
                res = comp_type = -1
            
            elif d2:
                res = comp_type = 1
            
            else:
                res, comp_type = super().nodecomp(node1, node2, policy=self.prune_policy), 10              
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError

    
    def is_sol_in_domaine(self, sol, node):
        #By partionionning, it is sufficient to only check what variable have
        #been branched and if sol is in [lb, up]_v for v a branched variable
        
        bvars, bbounds, btypes = node.getAncestorBranchings()
        
        for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
            if btype == 0:#LOWER BOUND
                if sol[bvar] < bbound:
                    return False
            else: #btype==1:#UPPER BOUND
                if sol[bvar] > bbound:
                    return False
        
        return True
            
            
    def setOptsol(self, optsol):
        self.optsol = optsol
        

class OracleNodeSelectorEstimator_SVM(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer):
        super().__init__()
        
        self.policy = load(f'./learning/policy_{problem}_svm.pkl')
        self.comp_featurizer = comp_featurizer
        self.counter = 0
        self.primal_changes = 0
        self.best_primal = np.inf
        
    def nodecomp(self, node1, node2):

        n = 4
        
        if self.primal_changes >= n: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
         
        f1, f2 = (self.comp_featurizer.get_features(node1),
                  self.comp_featurizer.get_features(node2))
        
        X = np.hstack((f1,f2))
        X = X[np.newaxis, :]
        
        decision = self.policy.predict(X)[0]
        self.counter += 1
        return -1 if decision < 0.5 else 1

    
        
    
    
class OracleNodeSelectorEstimator(CustomNodeSelector):
    
    def __init__(self, problem, comp_featurizer, device, feature_normalizor, use_trained_gnn=True):
        super().__init__()
        
        policy = GNNPolicy()
        if use_trained_gnn: 
            print(policy.load_state_dict(torch.load(f"./learning/policy_{problem}.pkl", map_location=device))) #run from main
        else:
            print("Using randomly initialized gnn")
        policy.to(device)
        
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.device = device
        self.feature_normalizor = feature_normalizor
        
        self.best_primal = np.inf #minimization
        self.primal_changes = 0
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = 0
        self.counter = 0
        
        self.scores = dict()
        
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        
        self.fe_time = 0
        self.fn_time = 0
        self.inference_time = 0
        self.counter = 0
    
    def nodecomp(self, node1,node2):
        
        n = 4

        if self.primal_changes >= n: #infer until obtained nth best primal solution
            return self.estimate_nodecomp(node1, node2)
        
        curr_primal = self.model.getSolObjVal(self.model.getBestSol())
        
        if self.model.getObjectiveSense() == 'maximize':
            curr_primal *= -1
            
        if curr_primal < self.best_primal:
            self.best_primal = curr_primal
            self.primal_changes += 1
            
        comp_scores = [-1,-1]
        
        for comp_idx, node in enumerate([node1, node2]):
            n_idx = node.getNumber()
        
            if n_idx in self.scores:
                comp_scores[comp_idx] = self.scores[n_idx]
            else:
                start = time.time()
                g =  self.comp_featurizer.get_graph_for_inf(self.model, node)
                self.fe_time += (time.time() - start)
                
                start = time.time()
                g = self.feature_normalizor(*g)[:-1]
                self.fn_time += (time.time() - start)
                
                start = time.time()
                score = self.policy.forward_graph(*g).item()
                self.scores[n_idx] = score 
                comp_scores[comp_idx] = score
                self.inference_time += (time.time() - start)
                
        self.counter += 1
        
        return -1 if comp_scores[0] > comp_scores[1] else 1
