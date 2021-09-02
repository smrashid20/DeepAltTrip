import args_kdiverse
import os
import glob

if not os.path.exists('model_files'):
    os.mkdir('model_files')
else:
    files = glob.glob('model_files/*')
    for f in files:
        os.remove(f)

if not os.path.exists('recset_ddtwos'):
    os.mkdir('recset_ddtwos')
else:
    files = glob.glob('recset_ddtwos/*')
    for f in files:
        os.remove(f)

args_kdiverse.dat_ix = 2
args_kdiverse.FOLD = 5
args_kdiverse.test_index = 1
args_kdiverse.copy_no = 0

from kfold_dataset_generator import generate_ds

generate_ds(args_kdiverse.dat_ix, args_kdiverse.FOLD, args_kdiverse.test_index, args_kdiverse.copy_no)

from kdiverse_generator import generate_result

Ns = [(3, 3), (5, 5), (7, 7), (9, 9)]

for Nmn, Nmx in Ns:
    if Nmn == 3:
        generate_result(False, K=3, N_min=Nmn, N_max=Nmx)
    else:
        generate_result(True, K=3, N_min=Nmn, N_max=Nmx)
