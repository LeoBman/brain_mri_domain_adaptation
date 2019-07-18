from __future__ import print_function

import sys
import os
from multiprocessing import Pool, cpu_count
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

# get array number
iter_num = int(sys.argv[1])

# set wdata and sdata paths w/ HPC in mind
if os.path.isdir("/Dedicated"):
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    sdata = '/sdata'
    wdata = '/wdata'


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def unwarp_bias_field_correction(arg, **kwarg):
    return bias_field_correction(*arg, **kwarg)


def bias_field_correction(src_path, dst_path):
    print("N4ITK on: ", src_path)
    try:
        n4 = N4BiasFieldCorrection()
        n4.inputs.input_image = src_path
        n4.inputs.output_image = dst_path
        n4.inputs.dimension = 3
        n4.inputs.n_iterations = [100, 100, 60, 40]
        n4.inputs.shrink_factor = 3
        n4.inputs.convergence_threshold = 1e-4
        n4.inputs.bspline_fitting_distance = 300
        n4.run()
    except RuntimeError:
        print("\tFailed on: ", src_path)
    return


parent_dir = os.path.join(sdata, 'neuroimaging', 'processed')

src_dir_list = ['abide1','abide2','adhd200','openneuro'][iter_num-1]
dst_dir_list = ['abide1','abide2','adhd200','openneuro'][iter_num-1]

data_src_paths, data_dst_paths = [], []

in_dir = src_dir_list
out_dir = dst_dir_list

src_label_dir = os.path.join(parent_dir, in_dir, "03_skullstrip")
dst_label_dir = os.path.join(parent_dir, in_dir, "04_biascorrect")
create_dir(dst_label_dir)
for subject in os.listdir(src_label_dir):
    data_src_paths.append(os.path.join(src_label_dir, subject))
    data_dst_paths.append(os.path.join(dst_label_dir, subject))

# Test
# bias_field_correction(data_src_paths[0], data_dst_paths[0])

# Multi-processing
paras = zip(data_src_paths, data_dst_paths)
pool = Pool(processes=cpu_count()-1)
pool.map(unwarp_bias_field_correction, paras)