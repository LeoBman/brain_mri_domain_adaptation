import numpy as np
from glob import glob
from scipy import ndimage
import nibabel as nib
import re
import os

# set wdata and sdata paths w/ HPC in mind
if os.path.isdir("/Dedicated"):
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    sdata = '/sdata'
    wdata = '/wdata'

# PROCESSED DATA
nifti_files = glob(os.path.join(sdata, "neuroimaging/processed/**/05_enhancement/**.nii.gz")
t1_data = np.zeros(shape=(len(nifti_files), 91, 109, 91))
labels = []

def rescaler_05(np_image):
    image_size = np_image.shape
    # in case of channels...
    if len(image_size) == 4:
        if image_size[3] == 1:
            f1 = 91/image_size[0] ## X direction
            f2 = 109/image_size[1] ## Y direction
            f3 = 91/image_size[2] ## Z direction
            np_image = ndimage.zoom(np_image[:,:,:,0],(f1,f2,f3))
        elif image_size[0] == 1:
            f1 = 91/image_size[1] ## X direction
            f2 = 109/image_size[2] ## Y direction
            f3 = 91/image_size[3] ## Z direction
            np_image = ndimage.zoom(np_image[0,:,:,:],(f1,f2,f3))
    else:
        f1 = 91/image_size[0] ## X direction
        f2 = 109/image_size[1] ## Y direction
        f3 = 91/image_size[2] ## Z direction
        np_image = ndimage.zoom(np_image,(f1,f2,f3))
    return np_image

for idx, val in enumerate(nifti_files):
    # label process
    subject = os.path.split(val)[1]
    subject = re.split(r'.nii.gz', subject)[0]
    labels.append(subject)
    # image process
    img = nib.load(val)
    img = img.get_data()
    img = rescaler_05(img)
    t1_data[idx] = img
    if idx%100 == 0:
        print("image processed: ", idx, " of ", len(nifti_files))

np.save(os.path.join(sdata, "neuroimaging/processed/numpy/processed_t1_data.npy", t1_data))
np.save(os.path.join("neuroimaging/processed/numpy/processed_labels.npy", labels))

# UNPROCESSED DATA
nifti_files = glob(os.path.join(sdata, "neuroimaging/processed/**/01_reorganize/**.nii.gz"))
t1_data = np.zeros(shape=(len(nifti_files), 91, 109, 91))
labels = []

for idx, val in enumerate(nifti_files):
    # label process
    subject = os.path.split(val)[1]
    subject = re.split(r'.nii.gz', subject)[0]
    # image process
    img = nib.load(val)
    img = img.get_data()
    try:
        img = rescaler_05(img)
    except:
        print('error')
        continue
    labels.append(subject)
    t1_data[idx] = img
    if idx%100 == 0:
        print("image processed: ", idx, " of ", len(nifti_files))

np.save(os.path.join(sdata , "neuroimaging/processed/numpy/unprocessed_t1_data.npy"), t1_data)
np.save(os.path.join(sdata , "neuroimaging/processed/numpy/unprocessed_labels.npy"), labels)


