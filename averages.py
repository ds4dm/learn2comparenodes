from pathlib import Path 
# list_files_bounded = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files
list_files_setcover = Path('/Users/work/Desktop/learn2branch/data/instances/setcover/transfer_500r_1000c_0.05d/').glob("*.lp")


for method in ('oracle', 'oracle_nodesel', 'default'):
    total_time = []
    primal_dual_integral = []
    nodes = [] 
    list_files_bounded = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files
    for path in list_files_bounded:
        with open((str(path) + method).strip(".") + "stats.txt", 'r') as f: 
            for line in f: 
                if 'Total Time' in line:
                    for t in line.split():
                        try:
                            total_time.append(float(t))
                        except ValueError:
                            pass
                if 'Avg. Gap' in line:
                    for t in line.split():
                        try:
                            primal_dual_integral.append(float(t))
                        except ValueError:
                            pass 
                if 'nodes' in line: 
                    if '(total)' in line:
                        for t in line.split():  
                            try:
                                nodes.append(float(t))
                            except ValueError:
                                pass 
        
    print('The method is:', method)
    nodes = [nodes[i] for i in range(len(nodes)) if i % 2 != 0]
    print('Average nodes:', sum(nodes)/len(nodes))
    print('Average PDI:', sum(primal_dual_integral)/len(primal_dual_integral))
    print('Average solving time:', sum(total_time)/len(total_time))

