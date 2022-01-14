#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:43:54 2021

@author: abdel
"""

def load_src(name, fpath):
     import os, imp
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("model", "../../learning/model.py" )

import torch
from pyscipopt import Nodesel
from model import GNNPolicy
import time


class OracleNodeSelectorEstimator(Nodesel):
    
    def __init__(self, problem, comp_featurizer,  DEVICE='cpu'):
        
        policy = GNNPolicy()
        
        print(policy.load_state_dict(torch.load(f"./learning/policy_{problem}.pkl"))) #run from main
        policy.to(DEVICE)
        self.policy = policy
        self.comp_featurizer = comp_featurizer
        self.DEVICE = DEVICE
        self.inference_time = 0
        self.fe_time = 0
        self.decision = []
        
    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_featurizer.set_LP_feature_recorder(LP_feature_recorder)
        self.inference_time = 0
        self.fe_time = 0

    def nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    
    def nodecomp(self, node1,node2):
        start = time.time()
        batch = self.comp_featurizer.get_inference_features(self.model, 
                                                            node1, 
                                                            node2).to(self.DEVICE)
        end = time.time()
        
        self.fe_time += (end - start)
        
        start = time.time()
        results = self.policy(batch).item() 
        end = time.time()
        
        self.inference_time += (end - start)
        
        self.decision += [results]
        
        return 2*(results - 0.5)
    
    
    


class OracleNodeSelectorAbdel(Nodesel):
    
    def __init__(self, oracle_type, optsol=0, prune_policy='estimate'):
        self.oracle_type = oracle_type
        self.optsol = optsol
        self.prune_policy = prune_policy 
    
    
    def nodeselect(self):
        return {"selnode": self.model.getBestNode()}
    
    
    
    def nodecomp(self, node1,node2):
        
        if self.oracle_type == "optimal_plunger":
        
            d1 = self.is_sol_in_domaine(self.optsol, node1)
            d2 = self.is_sol_in_domaine(self.optsol, node2)
            
            if d1 and d2:
                return self.dfs_compare(node1, node2)
            
            elif d1:
                return -1
            
            elif d2:
                return 1
            
            else:
                if self.prune_policy == "estimate":
                    return self.estimate_compare(node1, node2)
                raise NotImplementedError
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
        # prob_name = self.model.getProbName()
        # solver.readProblem(prob_name)
        
        
        # try:
        #     self.optsol = solver.readSolFile(prob_name.replace(".lp", ".sol"))
            
        # except OSError:
        #     solver.optimize()
        #     solver.writeBestSol(prob_name.replace(".lp", ".sol"))
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
    
            
        
            
        

#Maxime
class InternalNodeSelector(Nodesel):
    def __init__(self, nodesel_name):
        self.nodesel_name = nodesel_name

    def nodeselect(self):
        selnode = self.model.executeNodeSel(self.nodesel_name)
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)







class OracleNodeSelectorMaxime(Nodesel):
    def __init__(self, optsol, nodesel_name, instance):
        self.nodesel_name = nodesel_name
        self.optsol = optsol
        self.optnode = None
        self.features = {}
        self.labels = {}
        self.instance = instance 

    def nodeinitsol(self): 
        # TODO: how does nodeinitsol handle restarts ?
        self.optnode = self.model.getRootNode()
        self.optnode_selected = False

    def nodeselect(self):
        # selnode = self.model.executeNodeSel(self.nodesel_name)
        selnode = None
        leaves, children, siblings = self.model.getOpenNodes()
        print(f"    oracle: {len(leaves)} leaves, {len(children)} children, {len(siblings)} siblings")

        # if optimal node was selected last, update it
        if self.optnode_selected:
            print(f"    oracle: updating optimal node")
            # if it has children, one of those is the new optimal node
            if children:
                for child in children:
                    # check whether the child node contains the optimal solution
                    childisopt = True
                    vars, bounds, btypes = child.getParentBranchings()
                    for var, bound, btype in zip(vars, bounds, btypes):
                        optval = self.model.getSolVal(self.optsol, var)
                        # lower bound
                        if btype == 0 and self.model.isLT(optval, bound):
                            childisopt = False
                            break
                        # upper bound
                        if btype == 1 and self.model.isGT(optval, bound): 
                            childisopt = False
                            break
                    # when optimal child is found, stop
                    if childisopt:
                        break
                # assert one child is optimal
                assert childisopt
                self.optnode = child
            # else there is no optimal node any more
            else:
                self.optnode = None
                print(f"    oracle: no more optimal node")

        if self.optnode:
            selnode = self.optnode
            print(f"    oracle: selecting the optimal node")
        else:
            selnode = self.model.executeNodeSel(self.nodesel_name)
            print(f"    oracle: selecting the '{self.nodesel_name}' node")
        # checks whether the selected node is the optimal one
        self.optnode_selected = (self.optnode and self.optnode == selnode)

        if selnode:
            print(f"    selected node {selnode.getNumber()}")
        else:
            print(f"    no node selected")
        
        for child in children: 
            data = NodeFeatureRecorder().record(self.model, child)
            self.features[child.getNumber()] = data
        for sibling in siblings: 
            data = NodeFeatureRecorder().record(self.model, sibling)
            self.features[sibling.getNumber()] = data  
        # Record features here? 
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        decision = self.model.executeNodeComp(self.nodesel_name, node1, node2)
        if decision > 0: 
            self.labels[node1.getNumber()] = 1 
            self.labels[node2.getNumber()] = 0 
        elif decision < 0: 
            self.labels[node1.getNumber()] = 0
            self.labels[node2.getNumber()] = 1     
        return decision 

    def nodeexitsol(self):
        dataset = {}
        features_dict = self.features
        labels_dict = self.labels
        for key, value in features_dict.items():
            if key in labels_dict.keys():
                y = labels_dict[key]
                value.append(y)
                print(value) 
                dataset[key] = value
        keyword = 'features'
        header = []
        for i in range(18): 
            header.append(keyword + '_' + str(i))
        header.append('label')
        csv_file = str(self.instance).strip('.') + '.csv'
        with open(csv_file, 'w', newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for key, value in dataset.items():
                writer.writerow(value)
