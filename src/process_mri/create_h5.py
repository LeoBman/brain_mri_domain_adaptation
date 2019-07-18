import numpy as np
from glob import glob
from scipy import ndimage
import keras
import nibabel as nib
import re
import os

nifti_files = glob("/Dedicated/jmichaelson-sdata/abide/abide_1and2/05_abide_enhancement/**/**.nii.gz")

t1_data = np.zeros(shape=(len(nifti_files), 182, 218, 182))
labels = []

for idx, val in enumerate(nifti_files):
    # label process
    subject = os.path.split(val)[1]
    subject = re.split(r'sub-', subject)[-1]
    subject = re.split(r'.nii.gz', subject)[0]
    labels.append(subject)
    # image process
    img = nib.load(val)
    img = img.get_data()
    t1_data[idx] = img
    if idx%100 == 0:
        print("image processed: ", idx, " of ", len(nifti_files))

np.save("/Dedicated/jmichaelson-sdata/abide/abide_1and2/06_abide_h5file/t1_data.npy", t1_data_128)
np.save("/Dedicated/jmichaelson-sdata/abide/abide_1and2/06_abide_h5file/labels.npy", labels)