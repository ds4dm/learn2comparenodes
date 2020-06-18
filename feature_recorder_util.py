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
        self.branchvar = []
        self.branchbound = []
        self.branchdir = []
        self.varsol = []

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
        self.lb_root.append(self.model.getRootLowerBound())
        self.plungedepth.append(self.model.getPlungeDepth())
        if self.model.getCurrentNode().getDomchg():
            self.branchbound.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getNewBound())
            self.branchvar.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getVar())
            self.branchdir.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getVar().getVarBranchDir())
            self.varsol.append(self.model.getCurrentNode().getDomchg().getBoundchgs()[0].getVar().getLPSol())


m = Model()
m.readProblem("/Users/work/Desktop/Learn2SelectNodes/er_n=74_m=1338_p=0.50_SET2_setparam=100.00_alpha=0.75_0.lp")
node_recorder = NodeFocusHandler()
m.includeEventhdlr(node_recorder, "NodeFocusHandler", "")
m.hideOutput()
m.optimize()
print(len(node_recorder.node_lb[1:]))
print(node_recorder.depth[:5])
print(node_recorder.estimate[:5])
print(node_recorder.node_type[:5])
print(node_recorder.lb[:5])
print(node_recorder.ub[:5])
print(node_recorder.lb_root[:5])
print(node_recorder.plungedepth[:5])
print(len(node_recorder.branchbound))
print(node_recorder.branchvar[:5])
print(node_recorder.branchdir[:5])
print(node_recorder.varsol[:5])
m.freeProb()
