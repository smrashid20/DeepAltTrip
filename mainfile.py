import args_kdiverse
import os
import glob
import zipfile

if os.path.exists('model_files'):
    files = glob.glob('model_files/*')
    for f in files:
        os.remove(f)

if not os.path.exists('recset_myalgo'):
    os.mkdir('recset_myalgo')
else:
    files = glob.glob('recset_myalgo/*')
    for f in files:
        os.remove(f)

args_kdiverse.dat_ix = 6
args_kdiverse.FOLD = 5
args_kdiverse.test_index = 1
args_kdiverse.copy_no = 0

K = 3

model_zip_name = 'model_files_ds_' + str(args_kdiverse.dat_ix) + '_index_' \
                 + str(args_kdiverse.test_index) + '.zip'

with zipfile.ZipFile(os.path.join('model_repository', model_zip_name), 'r') as zip_ref:
    zip_ref.extractall('.')

from kfold_dataset_generator import generate_ds

generate_ds(args_kdiverse.dat_ix, args_kdiverse.FOLD, args_kdiverse.test_index, args_kdiverse.copy_no)

from kdiverse_generator import generate_result

Ns = [(3, 3), (5, 5), (7, 7), (9, 9)]

for Nmn, Nmx in Ns:
    generate_result(True, K=K, N_min=Nmn, N_max=Nmx)
