from pathlib import Path
from pyscipopt import Model, Nodesel, Eventhdlr
import pyscipopt
import argparse

class InternalNodeSelector(Nodesel):
    def __init__(self, nodesel_name):
        self.nodesel_name = nodesel_name

    def nodeselect(self):
        selnode = self.model.executeNodeSel(self.nodesel_name)
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)


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

        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)


instances = [f"data/instances/indset/transfer_500_4/instance_{i+1}.lp" for i in range(5)]
# instances = Path('data/instances/indset/transfer_500_4/').glob("*.lp")
# instances = Path('data/instances/setcover/transfer_500r_1000c_0.05d/').glob("*.lp")
# instances = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files

instances = [str(instance) for instance in instances]

instances = instances[1:2]

nodeselectors = [
    'estimate',
    # 'bfs',
    # 'hybridestim',
    # 'restartdfs',
    # 'uct',
    # 'dfs',
    # 'breadthfirst',
]


for instance in instances: 
    print(f"instance {instance}")
    sol_file = instance.strip(".") + "_solution.txt"
    m = Model()
    m.hideOutput()
    m.readProblem(instance)
    m.optimize()
    m.writeBestSol(sol_file)
    m.freeProb()
    print(f"optimal solution written to {sol_file}")

    for nodeselector in nodeselectors:

        # Oracle
        m = Model()
        m.hideOutput()
        m.readProblem(instance)
        solution = m.readSolFile(sol_file)
        m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
        m.optimize()
        nnodes = m.getNNodes()
        nlpiterations = m.getNLPIterations()
        stime = m.getSolvingTime()
        pdi = m.getStatPrimalDualIntegral()
        print(f"  oracle_{nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
        m.freeProb()
        del solution

        # vanilla SCIP rules
        m = Model()
        m.hideOutput()
        m.readProblem(instance)
        m.setParam(f"nodeselection/{nodeselector}/stdpriority", 666666)
        m.setParam(f"nodeselection/{nodeselector}/memsavepriority", 666666)
        m.optimize()
        nnodes = m.getNNodes()
        nlpiterations = m.getNLPIterations()
        stime = m.getSolvingTime()
        pdi = m.getStatPrimalDualIntegral()
        print(f"  {nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
        m.freeProb()

        # # python SCIP rules
        # m = Model()
        # m.hideOutput()
        # m.readProblem(instance)
        # m.includeNodesel(InternalNodeSelector(nodeselector), f"py_{nodeselector}", "", 666666, 666666)
        # m.optimize()
        # nnodes = m.getNNodes()
        # nlpiterations = m.getNLPIterations()
        # stime = m.getSolvingTime()
        # pdi = m.getStatPrimalDualIntegral()
        # print(f"  py_{nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PID")
        # m.freeProb()

