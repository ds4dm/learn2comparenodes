source activate l2sn
python gisp_generator.py -min_n 50 -max_n 60 -exp_dir data/GISP/train -solve 1 -n_instance 700 -n_cpu 8
python gisp_generator.py -min_n 50 -max_n 60 -exp_dir data/GISP/valid -solve 1 -n_instance 100 -n_cpu 8
python gisp_generator.py -min_n 50 -max_n 70 -exp_dir data/GISP/test -solve 1  -n_instance 200  -n_cpu 8
