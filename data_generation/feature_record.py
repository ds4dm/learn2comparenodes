import sys

pkg_path = "/".join(__file__.split("/")[:-2])
sys.path.append(pkg_path)

from node_selectors.oracle_selectors import OracleNodeSelector
from data_generation.feature_recorder_util import NodeFeatureRecorder 
from pyscipopt import Model
from pathlib import Path

instances = Path('data_generation/LP/instances').glob("*.lp")
instance = next(instances)
sol_file = str(instance).strip(".") + "_solution.txt"
nodeselector = "estimate"
instance = "/Users/work/Desktop/LP/instances/er_n=124_m=4524_p=0.60_SET2_setparam=100.00_alpha=0.50_0.lp"
for instance in instances: 
    nodeselector = 'estimate'
    print(f"instance {instance}")
    sol_file = str(instance).strip(".") + "_solution.txt"
    try:
        f = open(sol_file)
        # file alreaty existing
    except FileNotFoundError:
        # that should  be base case,normal flow
        m = Model()
        m.hideOutput()
        m.readProblem(str(instance))
        m.optimize()
        m.writeBestSol(str(instance) + "_solution.txt")
        sol_file = str(instance).strip(".") + "_solution.txt"

m = Model()
m.hideOutput()
m.readProblem(instance)
m.includeNodesel(OracleNodeSelector(m.readSolFile(sol_file), nodeselector, instance), f"py_{nodeselector}", "", 666666, 666666)
m.optimize()
m.writeBestSol()
m.freeProb()

