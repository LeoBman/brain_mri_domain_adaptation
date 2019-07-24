import numpy as np
import pandas as pd

sites = np.load("/sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_site.npy", allow_pickle=True)
study = np.load("/sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_study.npy", allow_pickle=True)

# array of sites-study
comb = np.core.defchararray.add(np.array(study).astype(np.str), "-")
comb = np.core.defchararray.add(np.array(comb).astype(np.str), np.array(sites).astype(np.str))

# table of occurences
comb_tbl = pd.DataFrame(comb)[0].value_counts()

# every fifth site goes into the same fold
f1 = np.arange(100) % 5 == 0
f2 = ((np.arange(100)-1) % 5) == 0
f3 = ((np.arange(100)-2) % 5) == 0
f4 = ((np.arange(100)-3) % 5) == 0
f5 = ((np.arange(100)-4) % 5) == 0


def split_site(site, sites, train_ratio, val_ratio):
    # given a site, split sites array into three appropriately sized arrays (based on ratios)
    # get number of samples in the splits
    ind = np.where(site == sites)[0]
    train = round(len(ind) * train_ratio)
    val = round(len(ind) * val_ratio)
    test = len(ind) - (train+val) 
    #   make split
    train = np.random.choice(ind, train, replace=False)
    val = np.random.choice(ind[np.isin(ind, train, invert=True)], val, replace=False)
    test = ind[np.isin(ind, np.append(train,val), invert=True)]
    return(train, val, test)


def split_fold(train_ratio, val_ratio, fold_bool, all_sites, sites_array):
    # sites split into folds, then split into train/val/test
    # given a fold, calls split_site function on each site to fully split the fold into train,val,test
    fold_sites = all_sites[fold_bool].index
    foo = [split_site(x, sites_array, 0.6, 0.2) for x in fold_sites]
    train = [item[0] for item in foo]
    train = np.hstack(train).squeeze()
    val = [item[1] for item in foo]
    val = np.hstack(val).squeeze()
    test = [item[2] for item in foo]
    test = np.hstack(test).squeeze()
    return(train, val, test)

fold1 = split_fold(train_ratio = 0.6,
    val_ratio = 0.2,
    fold_bool = f1,
    all_sites = comb_tbl,
    sites_array = comb)

fold2 = split_fold(train_ratio = 0.6,
    val_ratio = 0.2,
    fold_bool = f2,
    all_sites = comb_tbl,
    sites_array = comb)

fold3 = split_fold(train_ratio = 0.6,
    val_ratio = 0.2,
    fold_bool = f3,
    all_sites = comb_tbl,
    sites_array = comb)

fold4 = split_fold(train_ratio = 0.6,
    val_ratio = 0.2,
    fold_bool = f4,
    all_sites = comb_tbl,
    sites_array = comb)

fold5 = split_fold(train_ratio = 0.6,
    val_ratio = 0.2,
    fold_bool = f5,
    all_sites = comb_tbl,
    sites_array = comb)

folds = np.array([fold1, fold2, fold3, fold4, fold5])
np.save("/sdata/neuroimaging/processed/numpy/2019-07-22-folds.npy", folds)





