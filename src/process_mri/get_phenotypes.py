import pandas as pd
import numpy as np
from glob import glob
import re
import os

if os.path.isdir("/Dedicated"):
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    sdata = '/sdata'
    wdata = '/wdata'

# LOAD PHENOTYPE DATA
# get file paths to phenotypes
a2_files = glob(os.path.join(sdata, "neuroimaging/abide2/RawData/**/participants.tsv"))
on_files = glob(os.path.join(sdata, "neuroimaging/openneuro/**/participants.tsv"))
adhd_files = glob(os.path.join(sdata, "neuroimaging/adhd200/RawData/**_phenotypic.csv"))

# create dataframe of phenotypes for each study
#abide1
# hi
a1_df = pd.read_csv(os.path.join(sdata, "neuroimaging/abide1/Phenotypic_V1_0b_preprocessed.csv"))
a1_df = a1_df[['SITE_ID','subject','SEX']]
a1_df.rename(columns={'SITE_ID':'site_id', 'subject':'participant_id', 'SEX':'sex'}, inplace=True)
a1_df['participant_id'] = [str(x).strip('0') for x in a1_df['participant_id']]
a1_df['site_id'] = a1_df['site_id'].str.lower()
a1_df['site_part_id'] = np.core.defchararray.add(np.array(a1_df['site_id']).astype(np.str), ":")
a1_df['site_part_id'] = np.core.defchararray.add(np.array(a1_df['site_part_id']).astype(np.str), np.array(a1_df['participant_id']).astype(np.str))
a1_df['study'] = 'abide1' 
#abide2
a2_df = pd.DataFrame()
for fpath in a2_files:
    a2_df = a2_df.append(pd.read_csv(fpath, sep='\t'))
a2_df = a2_df[['site_id','participant_id','sex']]
a2_df['participant_id'] = np.core.defchararray.add(np.array('sub-'), np.array(a2_df['participant_id']).astype(np.str))
a2_df['participant_id'] = [str(x).strip('0') for x in a2_df['participant_id']]
a2_df['site_id'] = a2_df['site_id'].str.lower()
a2_df['site_id'] = [re.split('abideii-', x)[1] for x in np.array(a2_df['site_id'])]
a2_df['site_part_id'] = np.core.defchararray.add(np.array(a2_df['site_id']).astype(np.str), ":")
a2_df['site_part_id'] = np.core.defchararray.add(np.array(a2_df['site_part_id']).astype(np.str), np.array(a2_df['participant_id']).astype(np.str)) 
a2_df['study'] = 'abide2'
#openneuro
on_df = pd.DataFrame()
for fpath in on_files:
    foo = pd.read_csv(fpath, sep='\t')
    foo['site_id'] = fpath.split(os.sep)[-2]
    foo.columns = map(str.lower, foo.columns)
    if 'gender' in foo.columns:
        foo.rename(columns={'gender':'sex'}, inplace=True)
    if 'sex' not in foo.columns:
        print(fpath)
        continue
    on_df = on_df.append(foo[['site_id', 'participant_id','sex']])
on_df['participant_id'] = [str(x).strip('0') for x in on_df['participant_id']]
on_df['site_id'] = on_df['site_id'].str.lower()
on_df['site_part_id'] = np.core.defchararray.add(np.array(on_df['site_id']).astype(np.str), ":")
on_df['site_part_id'] = np.core.defchararray.add(np.array(on_df['site_part_id']).astype(np.str), np.array(on_df['participant_id']).astype(np.str)) 
on_df['study'] = 'on'
#adhd200
adhd_df = pd.DataFrame()
for fpath in adhd_files:
    foo = pd.read_csv(fpath, sep=',')
    foo['site_id'] = fpath.split(os.sep)[-1].split('_')[0]
    if 'ID' in foo.columns:
        foo.rename(columns={'ID':'ScanDir ID'}, inplace=True)
    foo = foo[['site_id', 'ScanDir ID', 'Gender']]
    foo.rename(columns={'ScanDir ID':'participant_id','Gender':'sex'}, inplace=True)
    adhd_df = adhd_df.append(foo)
adhd_df['participant_id'] = [str(x).strip('0') for x in adhd_df['participant_id']]
adhd_df['site_id'] = adhd_df['site_id'].str.lower()
adhd_df['site_part_id'] = np.core.defchararray.add(np.array(adhd_df['site_id']).astype(np.str), ":")
adhd_df['site_part_id'] = np.core.defchararray.add(np.array(adhd_df['site_part_id']).astype(np.str), np.array(adhd_df['participant_id']).astype(np.str))
adhd_df['study'] = 'adhd'
# combine phenotype dfs across studies
df = a1_df.append(a2_df)
df = df.append(on_df)
df = df.append(adhd_df)

# PROCESSED IMAGES
# get labels derived from image file names
img = np.load(os.path.join(sdata, 'neuroimaging/processed/numpy/processed_t1_data.npy'))
labels = np.load(os.path.join(sdata, 'neuroimaging/processed/numpy/processed_labels.npy'))
sites = [re.split(":", x)[0] for x in labels]
sites = list(map(str.lower, sites))
ids = [re.split(":", x)[1].strip('0') for x in labels]
sess = [re.split(":", x)[2] if (len(re.split(":", x)) == 3) else 'NA' for x in labels]
site_part_id = np.core.defchararray.add(sites, ":")
site_part_id = np.core.defchararray.add(site_part_id, ids)

# get subset of objects for which we have their phenotypes
bool_vec = np.isin(site_part_id, df['site_part_id'])
site_part_id = site_part_id[bool_vec]
df = df.set_index('site_part_id')
df = df.loc[site_part_id]

img = img[bool_vec]
sex = df['sex']
site = df['site_id']
part_id = df['participant_id']
site_part_id = np.array(df.index)
study = df['study']

# drop sites w/ < 5 samples
comb = np.core.defchararray.add(np.array(study).astype(np.str), "-")
comb = np.core.defchararray.add(np.array(comb).astype(np.str), np.array(site).astype(np.str))
comb_tbl = pd.DataFrame(comb)[0].value_counts()
foo = np.array(comb_tbl[comb_tbl < 5].index)
bar = np.array([False if x in foo else True for x in comb])
bool_vec = bar
# save
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_img.npy"), img[bool_vec])
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_sex.npy"), sex[bool_vec])
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_site.npy"), site[bool_vec])
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_ID.npy"), part_id[bool_vec])
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_site_ID.npy"), site_part_id[bool_vec])
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_study.npy"), study[bool_vec])

# UNPROCESSED IMAGES
# get labels derived from image file names
img = np.load(os.path.join(sdata, 'neuroimaging/processed/numpy/unprocessed_t1_data.npy'))
labels = np.load(os.path.join(sdata, 'neuroimaging/processed/numpy/unprocessed_labels.npy'))
sites = [re.split(":", x)[0] for x in labels]
sites = list(map(str.lower, sites))
ids = [re.split(":", x)[1].strip('0') for x in labels]
sess = [re.split(":", x)[2] if (len(re.split(":", x)) == 3) else 'NA' for x in labels]
site_part_id = np.core.defchararray.add(sites, ":")
site_part_id = np.core.defchararray.add(site_part_id, ids)

# get in same order as the processed data
foo = np.load(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-processed_filtered_site_ID.npy"), allow_pickle=True)
bar = [np.where(x == site_part_id)[0][0] for x in foo]
bar = np.array(bar)
img = img[bar]
np.save(os.path.join(sdata, "neuroimaging/processed/numpy/2019-07-22-unprocessed_filtered_img.npy"), img)


