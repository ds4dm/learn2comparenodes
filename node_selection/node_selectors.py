#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""

def load_src(name, fpath):
     import os, imp
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("model", "../learning/model.py" )

import torch
from pyscipopt import Nodesel
from model import GNNPolicy
import time



class OracleNodeSelectorAbdel(Nodesel):
    
    def __init__(self, oracle_type, optsol=0, prune_policy='estimate', inv_proba=0.05):
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
        self.inv_proba = inv_proba
    
    
    def nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    
    
    
    def nodecomp(self, node1, node2, return_type=False):
        
        if self.oracle_type == "optimal_plunger":
            import numpy as np
            
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            inv = np.random.rand() < self.inv_proba
            
            if d1 and d2:
                res = self.dfs_compare(node1, node2) 
                comp_type = 0
            
            elif d1:
                res = -1
                comp_type = -1
            
            elif d2:
                res = 1
                comp_type = 1
            
            else:
                if self.prune_policy == "estimate":
                    res = self.estimate_compare(node1, node2)
                elif self.prune_policy == "dfs":
                    res = self.dfs_compare(node1, node2)
                else:
                    raise NotImplementedError
                comp_type = 10                
            
            inv_res = -1 if res == 1 else 1
            res = inv_res if inv else res
            return res if not return_type  else  (res, comp_type)
        else:
            raise NotImplementedError
    
            
    
    
    def estimate_compare(self, node1, node2):#SCIP 
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
                        return -self.dfs_compare(node1, node2)
     
        
        elif self.model.isLT(estimate1, estimate2):
            return -1
        else:
            return 1
        
                
            
    
    # def prune_sel(self):
    #     leaves, children, siblings = self.model.getOpenNodes()
        
    #     selnode = min(leaves, key=lambda x: x.getNumber(), default=None)
    #     if selnode == None:
    #         selnode = min(children, key=lambda x:x.getNumber(), default=None)
    #         if selnode == None:
    #             selnode = min(siblings,  key=lambda x:x.getNumber(), default=None)
        
        
        
    #     return {"selnode": selnode}
    
    # #BFS compare
    # def prune_compare(self, node1, node2):
        
    #     d1, d2 = node1.getDepth(), node2.getDepth()
        
    #     if d1 == d2:
    #         #choose the first created node
    #         return node1.getNumber() - node2.getNumber()
        
    #     #less deep node => better
    #     return d1 - d2
        
        
        
    
    def nodeinit(self):
        pass
        
        # solver = Model()
        # prob_policy = self.model.getProbpolicy()
        # solver.readProblem(prob_policy)
        
        
        # try:
        #     self.optsol = solver.readSolFile(prob_policy.replace(".lp", ".sol"))
            
        # except OSError:
        #     solver.optimize()
        #     solver.writeBestSol(prob_policy.replace(".lp", ".sol"))
        #     self.optsol = solver.getBestSol()
        
   


    
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
        
    def dfs_compare(self, node1, node2):
        return -node1.getDepth() + node2.getDepth()
    
            
        
   
    
class OracleNodeSelectorEstimator(OracleNodeSelectorAbdel):
    
    def __init__(self, problem, comp_featurizer,  DEVICE, record_fpath=None, use_trained_gnn=True):
        super().__init__("optimal_plunger", inv_proba=0)
        
        policy = GNNPolicy()
        if use_trained_gnn: 
            print("using trained gnn")
            print(policy.load_state_dict(torch.load(f"./learning/policy_{problem}.pkl", map_location=torch.device('cpu')))) #run from main
        else:
            print("using a randomly initialised gnn")
            
        policy.to(DEVICE)
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.DEVICE = DEVICE
        self.record_fpath = record_fpath
        self.inference_time = 0
        self.fe_time = 0
        
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        self.inference_time = 0
        self.fe_time = 0

    def nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    
    def nodecomp(self, node1,node2):
        
        start = time.time() #measure feature extraction time
        batch = self.comp_featurizer.get_inference_features(self.model, 
                                                            node1, 
                                                            node2).to(self.DEVICE)
        end = time.time()
        
        self.fe_time += (end - start)
        
        start = time.time() #measure inference time
        decision = self.policy(batch).item() 
        oracle_decision = super().nodecomp(node1,node2)
        end = time.time()
        
        self.inference_time += (end - start)
        
        import numpy as np
        if 2*np.round(decision) - 1 != oracle_decision:
            # print("---------------UNMATCHING oracle -----")
            # print(node1.getNumber(), node2.getNumber())
            if np.random.rand() < 0:
                decision = 1 - decision
            #print("UNMATCH")
        else:
            pass
            #print("MATCH")
            
        
        if self.record_fpath != None:
            with open(f"{self.record_fpath}", "a+") as f:
                f.write(f"{decision:0.3f},{oracle_decision}\n")
                f.close()
        
        
        return -1 if decision < 0.5 else 1
    



class CustomNodeSelector(Nodesel):
    def __init__(self, policy):
        self.policy = policy
        
    def nodeselect(self):
        if self.policy == 'estimate':
            res = self.estimate_nodeselect()
        elif self.policy == 'dfs':
            res = self.dfs_nodeselect()
        elif self.policy == 'breadthfirst':
            res = self.breadthfirst_nodeselect()
        elif self.policy == 'bfs':
            res = self.bfs_nodeselect()
        elif self.policy == 'random':
            res = self.random_nodeselect()
            
        return res
    
    def nodecomp(self, node1, node2):
        
        if self.policy == 'estimate':
            res = self.estimate_nodecomp(node1, node2)
        elif self.policy == 'dfs':
            res = self.dfs_nodecomp(node1, node2)
        elif self.policy == 'breadthfirst':
            res = self.breadthfirst_nodecomp(node1, node2)
        elif self.policy == 'bfs':
            res = self.bfs_nodecomp(node1, node2)
        elif self.policy == 'random':
            res = self.random_nodecomp(node1, node2)
            
        return res
    
    
    
    def estimate_nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    
    #Estimate 
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
                        return -self.dfs_compare(node1, node2)
     
        
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
    def breathfirst_nodeselect(self):
        
        selnode = self.model.getPrioSibling()
        if selnode == None: #no siblings to be visited (all have been LP-solved), since breath first, 
        #we take the heuristic of taking the best leaves among all leaves
            
            selnode = self.model.getBestLeaf() #DOESTN INCLUDE CURENT NODE CHILD !
            if selnode == None: 
                selnode = self.model.getPrioChild()
        
        return {"selnode": selnode}
    
    def breathfirst_nodecomp(self, node1, node2): 
        
        d1, d2 = node1.getDepth(), node2.getDepth()
        
        if d1 == d2:
            #choose the first created node
            return node1.getNumber() - node2.getNumber()
        
        #less deep node => better
        return d1 - d2
        
        

    

