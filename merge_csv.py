# from pathlib import Path
# import csv
# filenames = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files)
# nodeselectors = [
#     'estimate',
#     'bfs',
#     'hybridestim',
#     'restartdfs',
#     'uct',
#     'dfs',
#     'breadthfirst',
# ]

# with open('some.csv', 'w', newline='\n') as f:
#     for instance in filenames:
#         for nodeselector in nodeselectors:
#             with open(str(instance) + '_' + nodeselector + '.txt') as infile:
#                 for line in infile: 
#                     writer = csv.writer(f)
#                     writer.writerows(line)


from pathlib import Path 
import csv

with open("results_mik_oracle.csv", 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=names)
    writer.writeheader()
    for method in ('dfs','estimate', 'hybridestim', 'bfs','breadthfirst','uct','restartdfs'):
        instances = Path('/Users/work/Desktop/Learn2SelectNodes/mik.data/bounded/').rglob('*.mps.gz')    # 90 bounded files
        for instance in instances: 
            with open(str(instance) + '_oracle_' + method + '.txt', 'r') as f: 
                for line in f:
                    if 'Total Time' in line:
                        print(line)
                        for t in line.split():
                            try:
                                print(float(t))
                                time = float(t)
                            except ValueError:
                                pass
                    if 'Avg. Gap' in line:
                        print(line)
                        for t in line.split():
                            try:
                                print(float(t))
                                pdi = float(t)
                            except ValueError:
                                pass 
                    if 'nodes' in line: 
                        if '(total)' in line:
                            print(line)
                            for t in line.split():  
                                try:
                                    print(float(t))
                                    nnodes = float(t)
                                except ValueError:
                                    pass 
                    
                    # if time and nnodes and pdi: 
                    #     writer = csv.DictWriter(f, fieldnames=names)
                    #     writer.writerow({
                    #         'method':method, 
                    #         'instance':str(instance), 
                    #         'time': time, 
                    #         'nnodes':nnodes, 
                    #         'pdi': pdi
                    #     })
                    #     csvfile.flush()