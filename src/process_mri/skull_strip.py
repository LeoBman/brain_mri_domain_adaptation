from __future__ import print_function

import os
import subprocess
from multiprocessing import Pool, cpu_count

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


def bet(src_path, dst_path, frac="0.5"):
    command = ["bet", src_path, dst_path, "-R", "-f", frac, "-g", "0"]
    subprocess.call(command)
    return


def unwarp_strip_skull(arg, **kwarg):
    return strip_skull(*arg, **kwarg)


def strip_skull(src_path, dst_path, frac="0.4"):
    print("Working on :", src_path)
    try:
        bet(src_path, dst_path, frac)
    except RuntimeError:
        print("\tFailed on: ", src_path)
    return


parent_dir = os.path.join(sdata, 'neuroimaging', 'processed')

src_dir_list = ['abide1','abide2','adhd200','openneuro']
dst_dir_list = ['abide1','abide2','adhd200','openneuro']

data_src_paths, data_dst_paths = [], []
for in_dir, out_dir in zip(src_dir_list, dst_dir_list):
    src_label_dir = os.path.join(parent_dir, in_dir, "02_registration")
    dst_label_dir = os.path.join(parent_dir, in_dir, "03_skullstrip")
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))



# Test
#strip_skull(data_src_paths[0], data_dst_paths[0])

# Multi-processing
paras = zip(data_src_paths, data_dst_paths)
pool = Pool(processes=cpu_count()-1)
pool.map(unwarp_strip_skull, paras)