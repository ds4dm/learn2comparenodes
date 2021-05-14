from pathlib import Path
from pyscipopt import Model, Nodesel, Eventhdlr
import pyscipopt
import argparse
from feature_recorder_util import NodeFeatureRecorder
import csv


class InternalNodeSelector(Nodesel):
    def __init__(self, nodesel_name):
        self.nodesel_name = nodesel_name

    def nodeselect(self):
        selnode = self.model.executeNodeSel(self.nodesel_name)
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)


class OracleNodeSelector(Nodesel):
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
