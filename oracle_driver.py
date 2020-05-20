# Shadows the file nodesel_oracle.c from SCIP_DAgger 
from pyscipopt import Model
from pyscipopt.scip import Nodesel
import SVMfeatures 
import PySCIPOPT

# TODO: Implement SCIPNodeCheckOptimal (big method - may give problems??)
def nodeCheckOptimal(Model, node, opt_sol):  
    branchvarssize = 1 
    isoptimal = True 
    parent = node.getParent()
    if(parent.getDepth() > 0 and not self.model.isOptimal(parent)): 
    #TODO: Find some way to make this return value work out
        return None

    nbranchvars = Model.getNParentBranchings()
    branchvars, branchbounds, boundtypes = Model.getParentBranchings()
    assert(nbranchvars >= 1) 
    for i in range(nbranchvars):
        optval = self.model.getSolVal(self, opt_sol, branchvars[i])
        if boundtypes[i] == 0 and optval < branchbounds[i] or boundtypes[i] == 1 and optval > branchbounds[i]: 
            isoptimal = False 
            break 
    if isoptimal: 
        # TODO: setOptimal needs to be implemented
        Model.setOptimal(node)
    return None  

#TODO: Give this class some more powers? 
class OracleData(Nodesel): 
    def __init__(self):
        self.solutionfilename = ''
        self.trajectoryfilename = ''
        self.negate = True 
        self.optsol = []
        self.trjfile = open(self.trajectoryfilename)
        self.solfile = open(self.solutionfilename)
        self.feat = []
        self.optfeat = []

class Oracle(Nodesel): 
    def __init__(self):
        self.nodeseldata = OracleData()

    def nodeinit(self):
        self.nodeseldata = 1 # TODO: How do I extract features from the node selector so they can be saved to this variable?
        # read the optimal solution here
        # Assume that the problem has been loaded elsewhere???
        # How to add the problem?
        self.model.readProblem()
        solution_path = root + 'bounded/temp_sol/' + str(problem.name)[:len(problem.name) - 6]
        optimal_solution = self.model.readSolFile(solution_path)
        pass


    def readOptSol(self, filename, solution): 
        error = False 
        unknown_var = False 
        usevartable = False
        # TODO: Write the get BoolParam function 
        self.model.getBoolParam(args..)
        if not usevartable:
            print("Cannot read solution file if vartable is disabled. Make sure parameter 'misc/usevartable' is set to TRUE.")
            return 0  
        if not file_stream: 
            print("File not openable for reading. ")
            return 0 
        
        # TODO: Write the createSol function 
        self.model.createSol(solution)
        # start reading file here 
        with open(filename, "r") as f:

            # while True:
            #     chunk = f.read(1024)
            #     buffer = ''
            #     varname = ''
            #     value_string = ''
            #     var = []

    def nodeselect(self):
        leaves, children, siblings = self.model.getOpenNodes()
        optchild = -1 
        nodeseldata = self.model.getNodeselData()
       # TODO: Translate the SCIP function SCIPNodeIsOptimal, to python 
       # TODO: How to obtain the node selector data? 
       # TODO: third argument to nodecheckoptimal 
        for i, child in enumerate(children): 
            if not self.model.isOptChecked(child): 
                nodeCheckOptimal(self.model, child, ??)
                self.model.setOptChecked(child)
            if self.model.isOptimal(child): 
                optchild = i 
                
        # TODO: Where do we write all these values into? 
        # Children are optimal 
        if nodeseldata.trajectoryfilename != '':
            if optchild != -1: 
                self.optfeat.calcNodeselFeat(children[optchild])
                for _, child in enumerate(children): 
                    if i != optchild: 
                        features = self.feat.calcNodeselFeat(child)
                        nodeseldata.negate ^= 1
                        print(features)
                        # TODO: LIBSVMPrint in python 
                        # self.LIBSVMPrint(args...)
                for _, sibling in enumerate(siblings):  
                    features = self.feat.calcNodeselFeat(sibling)
                    nodeseldata.negate ^= 1
                    print(features)
                        
                    # self.LIBSVMPrint(args...)                
                for _, leaf in enumerate(leaves): 
                    features = self.feat.calcNodeselFeat(leaf)
                    nodeseldata.negate ^= 1
                    print(features) 
                    # self.LIBSVMPrint(args...) 
            else: 
                # Children are not optimal
                assert(len(children) == 0 or (len(children > 0 and nodeseldata[optnodenumber] != -1)))
                for _, child in enumerate(childen): 
                    features = self.calcNodeselFeat(child)
                    nodeseldata.negate ^= 1
                    print(features)
                    # self.LIBSVMPrint(args...)
            
        best = self.model.getBestNode()
        return {"best_node": best}


    def nodecomp(self, node1, node2):
        # TODO: implement isOptimal, SCIPnodeIsOptChecked
        isoptimal1 = self.model.isOptimal(node1)
        isoptimal2 = self.model.isOptimal(node2)
        if(isoptimal1):
            assert(not isoptimal2)
            return -1
        elif isoptimal2:
            return 1
        else:
            depth1 = self.model.getDepth(node1)
            depth2 = self.model.getDepth(node2)
            if (depth1 > depth2):
                return -1
            elif (depth1 < depth2):
                return 1
            else:
                lowerbound1 = self.model.getnodeLowerbound(node1)
                lowerbound2 = self.model.getnodeLowerbound(node2)
                if(self.model.isLT(lowerbound1, lowerbound2)):
                    return -1
                elif(self.model.isGT(lowerbound1, lowerbound2)):
                    return 1
                # Why check the same thing twice? What is the purpose/benefits?
                if lowerbound1 < lowerbound2:
                    return -1
                elif lowerbound1 > lowerbound2:
                    return 1
                else:
                    return 0

def test_nodesel():
    m = Model()
    # include node selector
    m.includeNodesel(Oracle(), "oracle", "Testing the expert node selector.", 1073741823, 536870911)
    # add Variables
    m.readproblem("assign1-5-8.mps.gz")
    m.optimize()
    m.printStatistics()

if __name__ == "__main__":
    test_nodesel()