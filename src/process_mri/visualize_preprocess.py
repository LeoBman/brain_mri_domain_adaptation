matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import nibabel as nib

### load phenotypes
# a1
a1_df = pd.read_csv("/Dedicated/jmichaelson-sdata/abide/abide1/Phenotypic_V1_0b.csv")
a1_df['participant_id'] = np.core.defchararray.add(np.array('00'), np.array(a1_df['SUB_ID']).astype(np.str))
# a2
abide2_files = glob("/Dedicated/jmichaelson-sdata/abide/abide2/RawData/**/participants.tsv")
a2_df = pd.DataFrame()
for file in abide2_files:
    a2_df = a2_df.append(pd.read_csv(file, sep='\t'))

a2_df['participant_id'] = np.core.defchararray.add(np.array('sub-'), np.array(a2_df['participant_id']).astype(np.str))

### grab a sample participant name
# a1
a1_ids = np.array(())
for site in np.unique(a1_df['SITE_ID']):
    a1_ids = np.append(a1_ids,a1_df[a1_df['SITE_ID'] == site].sample(1)['participant_id'])

# a2
a2_ids = np.array(())
for site in np.unique(a2_df['site_id']):
    a2_ids = np.append(a2_ids,a2_df[a2_df['site_id'] == site].sample(1)['participant_id'])

### get image file paths
img_paths_01 = np.array(())
for ids in np.append(a1_ids, a2_ids):
    img_paths_01 = np.append(img_paths_01, glob("/Dedicated/jmichaelson-sdata/abide/abide_1and2/01_abide_reorganize/**/" + ids + "*"))

img_paths_02 = np.array(())
for ids in np.append(a1_ids, a2_ids):
    img_paths_02 = np.append(img_paths_02, glob("/Dedicated/jmichaelson-sdata/abide/abide_1and2/02_abide_registration/**/" + ids + "*"))

img_paths_03 = np.array(())
for ids in np.append(a1_ids, a2_ids):
    img_paths_03 = np.append(img_paths_03, glob("/Dedicated/jmichaelson-sdata/abide/abide_1and2/03_abide_skullstrip/**/" + ids + "*"))


img_paths = np.concatenate((img_paths_01, img_paths_02, img_paths_03))

### plot pre and post registration images
plt.figure(figsize=(30,20))
for i in range(108):
    plt.subplot(12,9,i+1)
    plt.imshow(nib.load(img_paths[i]).get_data()[75,:,:])

plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/process_mris/test2.png')

plt.show()

import nilearn.image

nib.load(img_paths_01[10]).get_data().shape
nib.load(img_paths_02[10]).get_data().shape


nib.load("/wdata/lbrueggeman/fsl/data/atlases/MNI/MNI-prob-1mm.nii.gz").get_data().shape


foo = nib.load(img_paths_01[0]).get_data()
plt.subplot(1,2,1)
plt.imshow(foo[75,:,:])
plt.subplot(1,2,2)
plt.imshow(foo[75,:,:])
plt.show()
