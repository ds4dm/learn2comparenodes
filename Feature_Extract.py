from pyscipopt.scip import Model
import pyscipopt
from pyscipopt import Nodesel, Eventhdlr 
import numpy as np

class NodeFocusHandler(pyscipopt.Eventhdlr):
    def __init__(self):        
        self.stats = []
        # self.return_ = []
        # self.nb_nodes = []
        # self.nb_lp_iterations = []
        # self.solving_time = []
        self.node_lb = []
        self.depth = []
        self.estimate = []
        self.node_type = []
        self.ub = []
        self.lb = []
        self.lb_root = []
        self.branchbound = []
        self.branchvar = []
        self.plungedepth = []

    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        self.node_lb.append(self.model.getCurrentNode().getLowerbound())
        self.depth.append(self.model.getDepth())
        self.estimate.append(self.model.getCurrentNode().getEstimate())
        self.node_type.append(self.model.getCurrentNode().getType())
        self.ub.append(self.model.getUpperbound())
        self.lb.append(self.model.getLowerbound())
        self.lb_root.append(self.model.getRootLowerbound())
        self.plungedepth.append(self.model.getPlungeDepth())
        # self.data.append(self.model.getLowerbound())
            # self.data.append(self.model.getUpperBound())

class BranchFocusHandler(pyscipopt.Eventhdlr):
    def __init__(self): 
        self.branchvar = []
        self.branchbound = []
        self.branchdir = []
        self.varsol = []
    
    def eventinit(self): 
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.VARCHANGED, self) 
    
    def eventexit(self): 
        self.model.dropEvent(pyscipopt.SCIP_EVENTTYPE.VARCHANGED, self)
    
    def eventexec(self, event): 
        self.branchbound.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getNewBound())
        self.branchvar.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getVar())
        self.branchdir.append(self.branchvar.getVarBranchDir())
        self.varsol.append(self.branchvar.getLPSol())


m = Model()
m.readProblem("assign1-5-8.mps.gz")
node_recorder = NodeFocusHandler()
branch_recorder = BranchFocusHandler()
m.includeEventhdlr(node_recorder, "NodeFocusHandler", "")
m.includeEventhdlr(branch_recorder, "BranchFocusHandler", "")
m.optimize()
print(node_recorder.node_lb[:5])
print(node_recorder.depth[:5])
print(node_recorder.estimate[:5])
print(node_recorder.node_type[:5])
print(node_recorder.lb[:5])
print(node_recorder.ub[:5])
print(node_recorder.lb_root[:5])
print(node_recorder.plungedepth[:5])
print(branch_recorder.branchbound[:5])
print(branch_recorder.branchvar[:5])
print(branch_recorder.branchdir[:5])
print(branch_recorder.varsol[:5])
m.printBestSol()
m.freeProb()
