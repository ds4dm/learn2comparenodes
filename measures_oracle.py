from pyscipopt import Nodesel, Model, Eventhdlr
import pyscipopt
import csv
# from pathlib import Path
import time


class OracleNodeSelector(Nodesel):
    def __init__(self, optsol, nodesel_name):
        self.nodesel_name = nodesel_name
        self.optsol = optsol
        self.optnode = None

    def nodeinitsol(self): 
        # TODO: how does nodeinitsol handle restarts ?
        self.optnode = self.model.getRootNode()
        self.optnode_selected = False

    def nodeselect(self):
        selnode = None

        leaves, children, siblings = self.model.getOpenNodes()
        print(f"    oracle: {len(leaves)} leaves, {len(children)} children, {len(siblings)} siblings")

        # if optimal node was selected last, update it
        if self.optnode_selected:
            print(f"    oracle: updating optimal node")
            # if it has children, one of those is the new optimal node
            if children:
                for child in children:
                    # check whether the child node
                    # contains the optimal solution
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

        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)


class Measures(Eventhdlr): 
    def __init__(self):
        self.primal_bound = []
        self.dual_bound = []
        self.nnodes = []
        self.solving_time = []
        self.pdi = []
        self.nlpiterations = []
        self.start = time.process_time()

    def eventinit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
    
    def eventexit(self):
        self.model.catchEvent(pyscipopt.SCIP_EVENTTYPE.NODEFOCUSED, self)
    
    def eventexec(self, event):
        self.nnodes.append(self.model.getNNodes())
        self.solving_time.append(time.process_time() - self.start)
        self.primal_bound.append(self.model.getPrimalbound())
        self.dual_bound.append(self.model.getDualbound())
        self.nlpiterations.append(self.model.getNLPIterations())
        self.pdi.append(self.model.getStatPrimalDualIntegral())


nodeselector = 'estimate'
# 'bfs',
# 'hybridestim',
# 'restartdfs',
# 'dfs',
# 'breadthfirst',
names = ['instance', 'mode', 'nodeselector', 'solving_time', 'primal_bound', 
         'dual_bound', 'number_nodes', 'lp_iterations', 'pdi_time']
instance = '/Users/work/Desktop/LP/instances/er_n=132_m=6865_p=0.80_SET2_setparam=100.00_alpha=0.50_0.lp'
# for instance in instances_setcover:
print(f"instance {instance}")
sol_file = instance.strip(".") + "_solution.txt"
m = Model()
# m.hideOutput()
m.readProblem(instance)
m.optimize()
m.writeBestSol(sol_file)
m.freeProb()
print(f"optimal solution written to {sol_file}")

with open(instance + "_measures.csv", "w") as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames=names)
    writer.writeheader()    
    # for nodeselector in nodeselectors:
    #     # Oracle
    mode = 'oracle'
    m = Model()
    m.hideOutput()
    m.readProblem(instance)
    solution = m.readSolFile(sol_file)
    m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
    measure_oracle = Measures()
    m.includeEventhdlr(measure_oracle, "abcd", "efgh")
    m.optimize()
    writer.writerow({
        'instance': instance, 
        'mode': mode, 
        'nodeselector': nodeselector,
        'solving_time': measure_oracle.solving_time, 
        'number_nodes': measure_oracle.nnodes, 
        'lp_iterations': measure_oracle.nlpiterations, 
        'pdi_time': measure_oracle.pdi, 
        'primal_bound': measure_oracle.primal_bound, 
        'dual_bound': measure_oracle.dual_bound
    })
    csvfile.flush()
    m.freeProb()
    del solution

    # vanilla SCIP rules
    mode = 'vanilla_scip'
    m = Model()
    m.hideOutput()
    m.readProblem(instance)
    measure_scip = Measures()
    m.includeEventhdlr(measure_scip, "abcde", "fghij")
    m.setParam(f"nodeselection/{nodeselector}/stdpriority", 666666)
    m.setParam(f"nodeselection/{nodeselector}/memsavepriority", 666666)
    m.optimize()
    
    print('writing to csv file!!!')
    writer.writerow({
        'instance': instance,
        'mode': mode, 
        'nodeselector': nodeselector,
        'solving_time': measure_scip.solving_time, 
        'number_nodes': measure_scip.nnodes, 
        'lp_iterations': measure_scip.nlpiterations, 
        'pdi_time': measure_scip.pdi, 
        'primal_bound': measure_scip.primal_bound, 
        'dual_bound': measure_scip.dual_bound
    })
    csvfile.flush()
    m.freeProb()
