source activate l2sn
python behaviour_generation.py -oracle optimal_plunger -problem GISP -data_partition train -n_cpu 8
python behaviour_generation.py -oracle optimal_plunger -problem GISP -data_partition valid -n_cpu 8
