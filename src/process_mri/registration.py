# script adapted from https://github.com/quqixun/BrainPrep
#
# explanation of registration types:
# https://www.reddit.com/r/neuro/comments/3u8gqb/what_is_the_difference_between_an_affine_vs_rigid/
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT/UserGuide , see "Restrict the transformation type. For 3D to 3D..."

#import sys
#batch_num = sys.argv[1]

import os
import subprocess
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# set wdata and sdata paths w/ HPC in mind
if os.path.isdir("/Dedicated"):
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    sdata = '/sdata'
    wdata = '/wdata'

def plot_middle(data, slice_no=None):
    if not slice_no:
        slice_no = data.shape[-1] // 2
    plt.figure()
    plt.imshow(data[..., slice_no], cmap="gray")
    plt.show()
    return

# "-searchrx", "0", "0",
               #"-searchry", "0", "0", "-searchrz", "0", "0"

def registration(src_path, dst_path, ref_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command, stdout=open(os.devnull, "r"),
                    stderr=subprocess.STDOUT)
    return


def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)
    return


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def unwarp_main(arg, **kwarg):
    return main(*arg, **kwarg)


def main(src_path, dst_path, ref_path):
    print("Registration on: ", src_path)
    try:
        orient2std(src_path, dst_path)
        registration(dst_path, dst_path, ref_path)
    except RuntimeError:
        print("\tFalied on: ", src_path)
    return

parent_dir = os.path.join(sdata, 'neuroimaging', 'processed')

src_dir_list = ['abide1','abide2','adhd200','openneuro']
dst_dir_list = ['abide1','abide2','adhd200','openneuro']

data_src_paths, data_dst_paths = [], []
for in_dir, out_dir in zip(src_dir_list, dst_dir_list):
    src_label_dir = os.path.join(parent_dir, in_dir, "01_reorganize")
    dst_label_dir = os.path.join(parent_dir, in_dir, "02_registration")
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))

ref_path = os.path.join(wdata, "lbrueggeman/fsl/data/standard/MNI152lin_T1_1mm.nii.gz")


# Test
#main(data_src_paths[0], data_dst_paths[0], ref_path)

# Multi-processing
paras = zip(data_src_paths, data_dst_paths,
            [ref_path] * len(data_src_paths))
pool = Pool(processes=cpu_count())
pool.map(unwarp_main, paras)






# OLD
#data_dir = os.path.join(parent_dir, "01_abide_reorganize")

#data_src_dir = data_dir
#data_dst_dir = os.path.join(parent_dir, '02_abide_registration')

#data_labels = ["abide1", "abide2"]
#create_dir(data_dst_dir)

#data_src_paths, data_dst_paths = [], []
#for label in data_labels:
#    src_label_dir = os.path.join(data_src_dir, label)
#    dst_label_dir = os.path.join(data_dst_dir, label)
#    create_dir(dst_label_dir)
#    for subject in os.listdir(src_label_dir):
#        data_src_paths.append(os.path.join(src_label_dir, subject))
#        data_dst_paths.append(os.path.join(dst_label_dir, subject))