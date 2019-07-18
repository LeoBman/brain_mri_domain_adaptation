# script adapted from https://github.com/quqixun/BrainPrep
# written to process abide1, abide2, adhd200, and openneuro data
# all data downloaded via datalad

import os
import glob
import shutil
from tqdm import *
from subprocess import check_call

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
    return

src_dir_list = ["abide1", "abide2", "adhd200", "openneuro"]
dst_dir_list = ["abide1", "abide2", "adhd200", "openneuro"]

parent_dir = os.path.join(sdata, "neuroimaging")

############
# Block of code to define allowed and not allowed files, testing only
############
#data_src_dir = '/Dedicated/jmichaelson-sdata/neuroimaging/'
#subjects1 = glob.glob(os.path.join(data_src_dir, "**", "**", "**", "**", "*.nii.gz"))
#subjects2 = glob.glob(os.path.join(data_src_dir,"**", "**", "**", "**", "**", "*.nii.gz"))
#subjects3 = glob.glob(os.path.join(data_src_dir,"**", "**", "**", "**", "**" , "**", "*.nii.gz"))
#subjects = subjects1 + subjects2 + subjects3 
#ban = ['func','moldON','T2', 'processed', 'topup','fmap','dwi','defacemask','ehalfhalf', 'flipangle','rest', 'FLASH', 'displace','/pet/', 'sourcedata','derivatives', 'hires', 'dcm2bids', 'RawDataBIDS']
#banned = [s for s in subjects if any(xs in s for xs in ban)]
#kept = [s for s in subjects if not any(xs in s for xs in banned)]
#with open('/Dedicated/jmichaelson-sdata/neuroimaging/subjects.txt', 'w') as f:
#    for item in kept:
#        f.write("%s\n" % item)
#with open('/Dedicated/jmichaelson-sdata/neuroimaging/subjects_ban.txt', 'w') as f:
#    for item in banned:
#        f.write("%s\n" % item)


for in_dir, out_dir in zip(src_dir_list, dst_dir_list):
    data_src_dir = os.path.join(parent_dir, in_dir)
    data_dst_dir = os.path.join(parent_dir, "processed", out_dir, "01_reorganize")
    create_dir(data_dst_dir)
    print("Move files\nfrom: {0}\nto {1}".format(data_src_dir, data_dst_dir))
    # get list of subjects, banned terms correspond to unwanted files
    subjects1 = glob.glob(os.path.join(data_src_dir, "**", "**", "**", "*.nii.gz"))
    subjects2 = glob.glob(os.path.join(data_src_dir, "**", "**", "**", "**", "*.nii.gz"))
    subjects3 = glob.glob(os.path.join(data_src_dir, "**", "**", "**", "**" , "**", "*.nii.gz"))
    subjects = subjects1 + subjects2 + subjects3 
    ban = ['func','moldON','T2', 'processed', 'topup','fmap','dwi','defacemask','ehalfhalf', 'flipangle','rest', 'FLASH', 'displace','/pet/', 'sourcedata','derivatives', 'hires', 'dcm2bids', 'RawDataBIDS']
    banned = [s for s in subjects if any(xs in s for xs in ban)]
    kept = [s for s in subjects if not any(xs in s for xs in banned)]
    # copy file to new location
    for subject in tqdm(kept):
        if 'openneuro' in subject:
            if 'ses' in subject:
                new_subject_name = os.path.normpath(subject).split(os.sep)[5] + ":" + os.path.normpath(subject).split(os.sep)[6] + ":" + os.path.normpath(subject).split(os.sep)[7]
            else:
                new_subject_name = os.path.normpath(subject).split(os.sep)[5] + ":" + os.path.normpath(subject).split(os.sep)[6]
        else:
            if 'ses' in subject:
                new_subject_name = os.path.normpath(subject).split(os.sep)[6] + ":" + os.path.normpath(subject).split(os.sep)[7] + ":" + os.path.normpath(subject).split(os.sep)[8]
            else:
                new_subject_name = os.path.normpath(subject).split(os.sep)[6] + ":" + os.path.normpath(subject).split(os.sep)[7]
        dst_path = os.path.join(data_dst_dir, new_subject_name + ".nii.gz")
        try:
            shutil.copyfile(subject, dst_path)
        except:
            print('missing file')

