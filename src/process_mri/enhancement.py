from __future__ import print_function

import os
import numpy as np
import nibabel as nib
from scipy.signal import medfilt
from multiprocessing import Pool, cpu_count

if 'Dedicated' in os.getcwd():
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    sdata = '/sdata'
    wdata = '/wdata'

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def load_nii(path):
    nii = nib.load(path)
    return nii.get_data(), nii.get_affine()

def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return

def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])
    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1
    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    volume[np.where(volume < 0)] = 0
    volume[np.where(volume > max_value)] = max_value
    return volume

def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]
    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    volume = volume/255.0
    return volume

def unwarp_enhance(arg, **kwarg):
    return enhance(*arg, **kwarg)

def enhance(src_path, dst_path, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    print("Preprocess on: ", src_path)
    try:
        volume, affine = load_nii(src_path)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, 256)
        save_nii(volume, dst_path, affine)
    except RuntimeError:
        print("\tFailed on: ", src_path)

parent_dir = os.path.join(sdata, 'abide', 'abide_1and2')
data_dir = os.path.join(parent_dir, "04_abide_biascorrect")
data_src_dir = data_dir
data_dst_dir = os.path.join(parent_dir, '05_abide_enhancement')
data_labels = ["abide1", "abide2"]
create_dir(data_dst_dir)

data_src_paths, data_dst_paths = [], []
for label in data_labels:
    src_label_dir = os.path.join(data_src_dir, label)
    dst_label_dir = os.path.join(data_dst_dir, label)
    create_dir(dst_label_dir)
    for subject in os.listdir(src_label_dir):
        data_src_paths.append(os.path.join(src_label_dir, subject))
        data_dst_paths.append(os.path.join(dst_label_dir, subject))

kernel_size = 3
percentils = [0.5, 99.5]
bins_num = 0
eh = True

# Test
# enhance(data_src_paths[0], data_dst_paths[0],
#         kernel_size, percentils, bins_num, eh)

# Multi-processing
subj_num = len(data_src_paths)
paras = zip(data_src_paths, data_dst_paths,
            [kernel_size] * subj_num,
            [percentils] * subj_num,
            [bins_num] * subj_num,
            [eh] * subj_num)
pool = Pool(processes=cpu_count())
pool.map(unwarp_enhance, paras)




# produce example plots of this process
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from skimage import exposure


img, affine = load_nii(data_src_paths[0])
img_copy = img.flatten()
img_copy.sort()
img_copy_y = np.arange(len(img_copy)) / len(img_copy)

img_rescale = rescale_intensity(img, bins_num=bins_num)
img_rescale_sort = img_rescale.flatten()
img_rescale_sort.sort()
img_rescale_sort_y = np.arange(len(img_rescale_sort)) / len(img_rescale_sort)

img_rescale_eh = equalize_hist(img_rescale)
img_rescale_eh_sort = img_rescale_eh.flatten()
img_rescale_eh_sort.sort()
img_rescale_eh_sort_y = np.arange(len(img_rescale_eh_sort)) / len(img_rescale_eh_sort)

fig, axes = plt.subplots(nrows=2,ncols=3)

axes[0,0].imshow(img[75,:,:])
axes4a = axes[1,0].twinx()
axes[1,0].hist(img_copy, normed=True, histtype='stepfilled', alpha=0.2)
axes4a.plot(img_copy, img_copy_y)

axes[0,1].imshow(img_rescale[75,:,:])
axes5a = axes[1,1].twinx()
axes[1,1].hist(img_rescale_sort, normed=True, histtype='stepfilled', alpha=0.2)
axes5a.plot(img_rescale_sort, img_rescale_sort_y)

axes[0,2].imshow(img_rescale_eh[75,:,:])
axes6a = axes[1,2].twinx()
axes[1,2].hist(img_rescale_eh_sort, normed=True, histtype='stepfilled', alpha=0.2)
axes6a.plot(img_rescale_eh_sort, img_rescale_eh_sort_y)

plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/process_mris/bias_cor.png')
