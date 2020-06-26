from pyscipopt.scip import Model
import pyscipopt
from pyscipopt import Nodesel, Eventhdlr 
import numpy as np

class NodeFeatureRecorder():
    def __init__(self, model):
        self.node_lb = 0
        self.depth = 0 
        '''
        These features below are the node features. 
        '''
        self.nodeFeatures = []
        self.nodeselGap = 0 
        self.nodeselGapInf = 0 
        self.relativeDepth = 0
        self.lowerBound = 0 
        self.estimate = 0
        self.relativeBound = 0
        self.globalUpperBound = 0
        self.globalUpperBoundInf = 0
        self.plungeDepth = 0  
        self.nodeIsSibling = 0 
        self.nodeIsChild = 0 
        self.nodeIsLeaf = 0 
        '''
        These features below are the features obtained according to the branching 
        status at the time of a particular node being focussed. 
        '''
        self.branchFeatures = []
        self.boundLPDiff = 0 
        self.rootLPDiff = 0
        self.pseudocost = 0 
        self.branchPriorityDown = 0
        self.branchPriorityUp = 0
        self.branchVarInf = 0
        self.model = model

    def record(self, node):
        # Node features
        if self.model:     
            self.maxdepth = self.model.getNBinVars() + self.model.getNintVars()
        else: 
            self.maxdepth = 0
        self.depth = self.model.getDepth()
        if self.depth > self.maxdepth: 
            self.maxdepth = self.depth
        lb_root = abs(self.model.getRootLowerBound())
        lb_node = self.model.getCurrentNode().getLowerbound()
        ub = self.model.getUpperbound()
        lb = self.model.getLowerbound()

        if lb_root == 0: 
            lb_root = 0.0001
        
        if lb == ub: 
            self.nodeselGap = 1
        elif abs(ub) >= 1e+20 or lb == 0: 
            self.nodeselGapInf = 1
        else: 
            self.nodeselGap = (ub - lb) / abs(lb)
        
        if abs(ub) >= 1e20: 
            self.globalUpperBoundInf = 1 
            ub = lb + 0.2 * (ub - lb)
        else: 
            self.globalUpperBound = ub / lb_root
        
        self.relativeDepth = self.depth / (self.maxdepth * 10)
        self.lowerBound = lb / lb_root 
        estimate = self.model.getCurrentNode().getEstimate()
        self.estimate = estimate / lb_root
        if lb != ub: 
            self.relativeBound = (lb_node - lb) / (ub - lb)
        
        self.plungeDepth = self.model.getPlungeDepth()
        
        node_type = self.model.getCurrentNode().getType()
        if node_type == 2: 
            self.nodeIsSibling = 1 
        elif node_type == 3: 
            self.nodeIsChild = 1 
        elif node_type == 4: 
            self.nodeIsLeaf = 1

        self.nodeFeatures.append([self.nodeselGap, 
                        self.nodeselGapInf, 
                        self.relativeDepth, 
                        self.lowerBound, 
                        self.estimate,
                        self.globalUpperBound, 
                        self.globalUpperBoundInf,
                        self.relativeBound, 
                        self.plungeDepth, 
                        self.nodeIsSibling, 
                        self.nodeIsChild, 
                        self.nodeIsLeaf 
                        ])

        # Branching features 
        domainChange = self.model.getCurrentNode().getDomchg()
        if domainChange:
            branchbound = domainChange.getBoundchgs()[0].getNewBound()
            branchvar =  domainChange.getBoundchgs()[0].getVar()
            branchdir =  branchvar.getVarBranchDir()
            # This is broken, it needs some salt and pepper SCIP_TREE stuff to solve this properly 
            haslp = self.model.focusNodeHasLP()
            if haslp == 1: 
                varsol = branchvar.getPseudoSol()
            else: 
                varsol =  branchvar.getLPSol()
            varrootsol =  branchvar.getRootSol()
            boundtype =  domainChange.getBoundchgs()[0].getBoundtype()

            self.boundLPDiff = branchbound - varsol
            self.rootLPDiff = varrootsol - varsol
            self.pseudocost = self.model.getPseudoCost(branchvar, branchbound - varsol)
            if branchdir == 1:
                self.branchPriorityDown = 1 
            elif branchdir == 0: 
                self.branchPriorityUp = 1     
            
            if boundtype == 1: 
                self.branchVarInf = self.model.getAvgInferences(branchvar, 1) / self.maxdepth
            else: 
                self.branchVarInf = self.model.getAvgInferences(branchvar, 0) / self.maxdepth
        
            self.branchFeatures.append([
                self.boundLPDiff, 
                self.rootLPDiff, 
                self.pseudocost, 
                self.branchPriorityDown, 
                self.branchPriorityUp, 
                self.branchVarInf
            ])

            return [self.nodeFeatures, self.branchFeatures]

class NodeIncluder(pyscipopt.Eventhdlr): 
    def __init__(self): 
        self.dataset = []

    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event): 
        if self.model: 
            leaves, children, siblings = self.model.getOpenNodes()
            for child in children: 
                recorder = NodeFeatureRecorder(self.model)
                data = recorder.record(child)
                self.dataset.append(data)

            for leaf in leaves: 
                recorder = NodeFeatureRecorder(self.model)
                data = recorder.record(leaf)
                self.dataset.append(data)
                    
            for sibling in siblings: 
                recorder = NodeFeatureRecorder(self.model)
                data = recorder.record(sibling)
                self.dataset.append(data)

# m = Model()
# m.readProblem("/Users/work/Desktop/LP/instances/er_n=124_m=4524_p=0.60_SET2_setparam=100.00_alpha=0.50_0.lp")
# includer = NodeIncluder()
# m.includeEventhdlr(includer, "Includer", "")
# m.includeNodesel()
# m.hideOutput()
# m.optimize()
# print(len(includer.dataset))
# m.freeProb()
