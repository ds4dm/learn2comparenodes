from pathlib import Path
from pyscipopt import Model, Nodesel, Eventhdlr
import pyscipopt
import argparse
from feature_recorder_util import NodeFeatureRecorder


class InternalNodeSelector(Nodesel):
    def __init__(self, nodesel_name):
        self.nodesel_name = nodesel_name

    def nodeselect(self):
        selnode = self.model.executeNodeSel(self.nodesel_name)
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        return self.model.executeNodeComp(self.nodesel_name, node1, node2)


class OracleNodeSelector(Nodesel):
    def __init__(self, optsol, nodesel_name, feature_file = 'features.txt'):
        self.nodesel_name = nodesel_name
        self.optsol = optsol
        self.optnode = None
        self.feature_file = feature_file

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
    
        # Record features here? 
        return {"selnode": selnode}

    def nodecomp(self, node1, node2): 
        decision = self.model.executeNodeComp(self.nodesel_name, node1, node2)
        if self.model: 
            recorder = NodeFeatureRecorder(self.model)
            if node1 and node2: 
                data1 = recorder.record(node1)
                data2 = recorder.record(node2)
                print(data1, data2, decision, ~decision)
        return decision 


# instances_indset = [f"/Users/work/Desktop/learn2branch/data/instances/indset/transfer_500_4/instance_{i+1}.lp" for i in range(10)]
# instances_setcover = [f"/Users/work/Desktop/learn2branch/data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp" for i in range(10)]
# # instances = Path('data/instances/indset/transfer_500_4/').glob("*.lp")
# # instances = Path('data/instances/setcover/transfer_500r_1000c_0.05d/').glob("*.lp")
# nodeselectors = [
#     'estimate',
#     'bfs',
#     'hybridestim',
#     'restartdfs',
#     'uct',
#     'dfs',
#     'breadthfirst',
# ]


# instances_indset = [str(instance) for instance in instances_indset]
# instances_setcover = [str(instance) for instance in instances_setcover]
# names = ['mode','method', 'instance', 'time', 'nnodes', 'pdi']

# instances = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')
# import csv
# with open("results_mik.csv", 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=names)
#     writer.writeheader()
#     for instance in instances: 
#         instance = str(instance)
#         print(f"instance {instance}")
#         sol_file = instance.strip(".") + "_solution.txt"
#         m = Model()
#         m.hideOutput()
#         m.readProblem(instance)
#         m.optimize()
#         m.writeBestSol(sol_file)
#         m.freeProb()
#         print(f"optimal solution written to {sol_file}")

#         for nodeselector in nodeselectors:
#             # Oracle
#             mode = 'oracle'
#             m = Model()
#             m.hideOutput()
#             m.readProblem(instance)
#             solution = m.readSolFile(sol_file)
#             m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
#             m.optimize()
#             nnodes = m.getNNodes()
#             nlpiterations = m.getNLPIterations()
#             stime = m.getSolvingTime()
#             pdi = m.getStatPrimalDualIntegral()
#             writer.writerow({
#                 'mode': mode, 
#                 'method': nodeselector, 
#                 'instance': instance, 
#                 'time': stime, 
#                 'nnodes': nnodes, 
#                 'pdi': pdi,
#             })
#             csvfile.flush()
#             m.freeProb()
#             del solution

#             # vanilla SCIP rules
#             mode = 'vanilla_scip'
#             m = Model()
#             m.hideOutput()
#             m.readProblem(instance)
#             m.setParam(f"nodeselection/{nodeselector}/stdpriority", 666666)
#             m.setParam(f"nodeselection/{nodeselector}/memsavepriority", 666666)
#             m.optimize()
#             nnodes = m.getNNodes()
#             nlpiterations = m.getNLPIterations()
#             stime = m.getSolvingTime()
#             pdi = m.getStatPrimalDualIntegral()
#             writer.writerow({
#                 'mode': mode, 
#                 'method': nodeselector, 
#                 'instance': instance, 
#                 'time': stime, 
#                 'nnodes': nnodes, 
#                 'pdi': pdi,
#             })
#             csvfile.flush()
#             m.freeProb()


# for instance in instances_setcover: 
#     print(f"instance {instance}")
#     sol_file = instance.strip(".") + "_solution.txt"
#     m = Model()
#     m.hideOutput()
#     m.readProblem(instance)
#     m.optimize()
#     m.writeBestSol(sol_file)
#     m.freeProb()
#     print(f"optimal solution written to {sol_file}")

#     for nodeselector in nodeselectors:

#         # Oracle
#         m = Model()
#         m.hideOutput()
#         m.readProblem(instance)
#         solution = m.readSolFile(sol_file)
#         m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
#         m.optimize()
#         nnodes = m.getNNodes()
#         nlpiterations = m.getNLPIterations()
#         stime = m.getSolvingTime()
#         pdi = m.getStatPrimalDualIntegral()
#         print(f"  oracle_{nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
#         m.freeProb()
#         del solution

#         # vanilla SCIP rules
#         m = Model()
#         m.hideOutput()
#         m.readProblem(instance)
#         m.setParam(f"nodeselection/{nodeselector}/stdpriority", 666666)
#         m.setParam(f"nodeselection/{nodeselector}/memsavepriority", 666666)
#         m.optimize()
#         nnodes = m.getNNodes()
#         nlpiterations = m.getNLPIterations()
#         stime = m.getSolvingTime()
#         pdi = m.getStatPrimalDualIntegral()
#         print(f"  {nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
#         m.freeProb()

# instances = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files
# for instance in instances: 
#     instance = str(instance)
#     print(f"instance {instance}")
#     sol_file = instance.strip(".") + "_solution.txt"
#     m = Model()
#     m.hideOutput()
#     m.readProblem(instance)
#     m.optimize()
#     m.writeBestSol(sol_file)
#     m.freeProb()
#     print(f"optimal solution written to {sol_file}")

#     for nodeselector in nodeselectors:

#         # Oracle
#         m = Model()
#         m.hideOutput()
#         m.readProblem(instance)
#         solution = m.readSolFile(sol_file)
#         m.includeNodesel(OracleNodeSelector(solution, nodeselector), f"py_{nodeselector}", "", 666666, 666666)
#         m.optimize()
#         nnodes = m.getNNodes()
#         nlpiterations = m.getNLPIterations()
#         stime = m.getSolvingTime()
#         pdi = m.getStatPrimalDualIntegral()
#         print(f"  oracle_{nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
#         m.freeProb()
#         del solution

#         # vanilla SCIP rules
#         m = Model()
#         m.hideOutput()
#         m.readProblem(instance)
#         m.setParam(f"nodeselection/{nodeselector}/stdpriority", 666666)
#         m.setParam(f"nodeselection/{nodeselector}/memsavepriority", 666666)
#         m.optimize()
#         nnodes = m.getNNodes()
#         nlpiterations = m.getNLPIterations()
#         stime = m.getSolvingTime()
#         pdi = m.getStatPrimalDualIntegral()
#         print(f"  {nodeselector}: {nnodes} nodes, {stime} solving time, {pdi} PDI")
#         m.freeProb()
