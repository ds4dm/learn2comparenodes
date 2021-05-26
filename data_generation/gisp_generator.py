import os
import sys
import networkx as nx
import random


def dimacsToNx(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if line[0] == 'e':
                g.add_edge(int(arr[1]), int(arr[2]))
    return g


def generateRevsCosts(g, whichSet, setParam):
    if whichSet == 'SET1':
        for node in g.nodes():
            g.nodes[node]['revenue'] = random.randint(1, 100)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = (g.node[u]['revenue'] /
                            + g.node[v]['revenue'])/float(setParam)
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.nodes[node]['revenue'] = float(setParam)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = 1.0


def generateE2(g, alphaE2):
    E2 = set()
    for edge in g.edges():
        if random.random() <= alphaE2:
            E2.add(edge)
    return E2


def createIP(g, E2, ipfilename):
    with open(ipfilename, 'w') as lp_file:
        val = 100
        lp_file.write("maximize\nOBJ:")
        lp_file.write("100x0")
        count = 0
        for node in g.nodes():
            if count:
                lp_file.write(" + " + str(val) + "x" + str(node))
            count += 1
        for edge in E2:
            lp_file.write(" - y" + str(edge[0]) + '_' + str(edge[1]))
        lp_file.write("\n Subject to\n")
        constraint_count = 1
        for node1, node2, edge in g.edges(data=True):
            if (node1, node2) in E2:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+x" + str(node2) + "-y" + str(node1) + "_"
                              + str(node2) + " <=1 \n")
            else:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+" + "x" + str(node2) + " <=1 \n")
            constraint_count += 1

        lp_file.write("\nbinary\n")
        for node in g.nodes():
            lp_file.write(f"x{node}\n")


if __name__ == "__main__":
    instance = None
    exp_dir = "./instances"
    min_n = 5
    max_n = 20
    er_prob = 0.6
    whichSet = 'SET2'
    setParam = 100.0
    alphaE2 = 0.5
    timelimit = 7200.0
    solveInstance = False
    # seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-instance':
            instance = sys.argv[i + 1]
        if sys.argv[i] == '-exp_dir':
            exp_dir = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-er_prob':
            er_prob = float(sys.argv[i + 1])
        if sys.argv[i] == '-whichSet':
            whichSet = sys.argv[i + 1]
        if sys.argv[i] == '-setParam':
            setParam = float(sys.argv[i + 1])
        if sys.argv[i] == '-alphaE2':
            alphaE2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-timelimit':
            timelimit = float(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(sys.argv[i + 1])
        if sys.argv[i] == '-seed':
            seed = int(sys.argv[i + 1])
    assert exp_dir is not None
    if instance is None:
        assert min_n is not None
        assert max_n is not None

    lp_dir = "data_generation/LP/" + exp_dir
    try:
        os.makedirs(lp_dir)
    except OSError:
        if not os.path.exists(lp_dir):
            raise
    # Seed generator
    for seed in range(100):
        random.seed(seed)
        print(whichSet)
        print(setParam)
        print(alphaE2)
        if instance is None:
            # Generate random graph
            numnodes = random.randint(min_n, max_n)
            g = nx.erdos_renyi_graph(n=numnodes, p=er_prob, seed=seed)
            lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f_%d"
                    % (numnodes, nx.number_of_edges(g), er_prob, whichSet,
                        setParam, alphaE2, seed))
        else:
            g = dimacsToNx(instance)
            # instanceName = os.path.splitext(instance)[1]
            instanceName = instance.split('/')[-1]
            lpname = ("%s_%s_%g_%g_%d" % (instanceName, whichSet, alphaE2,
                    setParam, seed))

        # Generate node revenues and edge costs
        generateRevsCosts(g, whichSet, setParam)
        # Generate the set of removable edges
        E2 = generateE2(g, alphaE2)
        # Create IP, write it to file, and solve it with CPLEX
        print(lpname)
        # ip = createIP(g, E2, lp_dir + "/" + lpname)
        createIP(g, E2, lp_dir + "/" + lpname + ".lp")
