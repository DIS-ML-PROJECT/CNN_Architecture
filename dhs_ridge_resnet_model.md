

# dhs_ridge_resnet_model.ipynb

## Imports and Constants

Inline imports

```python
%load_ext autoreload

%autoreload 2
%matplotlib inline
```

Import libraries

import local modules

- batchers

  - batcher
  - dataset_constants

- models

  - linear_model

    - ridge_cv

      > For every fold F (the test fold):
      >       1. uses leave-one-fold-out CV on all other folds
      >          to tune ridge model alpha parameter
      >             2. using best alpha, trains ridge model on all folds except F
      >                   3. runs trained ridge model on F
      >             Saves predictions for each fold on test.
      >                         savedir/test_preds_{subset_name}.npz if subset_name is given
      >                         savedir/test_preds.npz otherwise
      >             Saves ridge regression weights to savedir/ridge_weights.npz
      >                         if save_weight=True

    - train_linear_logo

- utils

  - general
    - load_npz

```python
from collections import defaultdict
from glob import glob
import pickle
import os
import re
import sys

import numpy as np

sys.path.append('../')
from batchers import batcher, dataset_constants
from models.linear_model import ridge_cv, train_linear_logo
from utils.general import load_npz
```

### set main parameters

```python
FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
DATASET_NAME = '2009-17'
LOGS_ROOT_DIR = '../logs/'
COUNTRIES = dataset_constants.DHS_COUNTRIES

KEEPS = [0.05, 0.1, 0.25, 0.5]
SEEDS = [123, 456, 789]
```

### create a model dirs dictionary

{ model_name : model_directory}

```python
MODEL_DIRS = {
    'resnet_ms_A': '2009-17A_18preact_ms_samescaled_b64_fc01_conv01_lr0001',
    'resnet_ms_B': '2009-17B_18preact_ms_samescaled_b64_fc001_conv001_lr0001',
    'resnet_ms_C': '2009-17C_18preact_ms_samescaled_b64_fc001_conv001_lr001',
    'resnet_ms_D': '2009-17D_18preact_ms_samescaled_b64_fc001_conv001_lr01',
    'resnet_ms_E': '2009-17E_18preact_ms_samescaled_b64_fc01_conv01_lr001',
    'resnet_nl_A': '2009-17A_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'resnet_nl_B': '2009-17B_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'resnet_nl_C': '2009-17C_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'resnet_nl_D': '2009-17D_18preact_nl_random_b64_fc1.0_conv1.0_lr01',
    'resnet_nl_E': '2009-17E_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'resnet_rgb_A': '2009-17A_18preact_rgb_same_b64_fc001_conv001_lr01',
    'resnet_rgb_B': '2009-17B_18preact_rgb_same_b64_fc001_conv001_lr0001',
    'resnet_rgb_C': '2009-17C_18preact_rgb_same_b64_fc001_conv001_lr0001',
    'resnet_rgb_D': '2009-17D_18preact_rgb_same_b64_fc1.0_conv1.0_lr01',
    'resnet_rgb_E': '2009-17E_18preact_rgb_same_b64_fc001_conv001_lr0001',
    'resnet_rgb_transfer': 'transfer_2009-17nl_nlcenter_18preact_rgb_b64_fc001_conv001_lr0001',
    'resnet_ms_transfer': 'transfer_2009-17nl_nlcenter_18preact_ms_b64_fc001_conv001_lr0001',

    'incountry_resnet_ms_A': 'DHSIncountry/incountryA_18preact_ms_samescaled_b64_fc01_conv01_lr001',
    'incountry_resnet_ms_B': 'DHSIncountry/incountryB_18preact_ms_samescaled_b64_fc1_conv1_lr001',
    'incountry_resnet_ms_C': 'DHSIncountry/incountryC_18preact_ms_samescaled_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_ms_D': 'DHSIncountry/incountryD_18preact_ms_samescaled_b64_fc001_conv001_lr0001',
    'incountry_resnet_ms_E': 'DHSIncountry/incountryE_18preact_ms_samescaled_b64_fc001_conv001_lr0001',
    'incountry_resnet_nl_A': 'DHSIncountry/incountryA_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_B': 'DHSIncountry/incountryB_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_C': 'DHSIncountry/incountryC_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_D': 'DHSIncountry/incountryD_18preact_nl_random_b64_fc1.0_conv1.0_lr0001',
    'incountry_resnet_nl_E': 'DHSIncountry/incountryE_18preact_nl_random_b64_fc01_conv01_lr001',
    }
    

# add in keep_frac/seed dirs

KEEP_MODEL_DIRS = sorted(glob(os.path.join(LOGS_ROOT_DIR, '*keep*seed*')))

for model_dir in KEEP_MODEL_DIRS:
    model_dir = os.path.basename(model_dir)
    regex = r'2009-17(\w)_18preact_(\w+)_keep(.+)_seed(\w+)_b64.+'
    m = re.match(regex, model_dir)
    fold, bands_name, keep, seed = m.groups()
    model_name = 'resnet_{b}_{f}_keep{k}_seed{s}'.format(
        b=bands_name,
        f=fold,
        k=keep,
        s=seed)
    MODEL_DIRS[model_name] = model_dir
```
## 2 Load Saved Data

### 2.1 Load labels

Load labels, locs and years from npz (later from tfrecords / wealth_index.csv)

```
file_path = '../data/dhs_image_hists.npz' #adapted to current directory! original:  '../data/dhs_image_hists.npz'

npz = load_npz(file_path)



labels = npz['labels']

locs = npz['locs']

years = npz['years']



num_examples = len(labels)

assert np.all(np.asarray([len(labels), len(locs), len(years)]) == num_examples)
```



### 2.2 Load `loc_dict`

`loc_dict` has the format:

```python
{
    (lat, lon): {
        'cluster': 1,
        'country': 'malawi',
        'country_year': 'malawi_2012',  # surveyID
        'households': 25,
        'urban': False,
        'wealth': -0.513607621192932,
        'wealthpooled': -0.732255101203918,
        'year': 2012
    }, ...
}
```

NOTE: `year` and `country_year` might differ in the year. `country_year` is the survey ID, which says which year the survey started. However, sometimes the DHS surveys cross the year-boundary, in which case `country_year` will remain the same but `year` will be the next year.

```python
loc_dict_path = '../data/dhs_loc_dict.pkl' #adapted to current directory! original:  '../data/dhs_loc_dict.pkl'

with open(loc_dict_path, 'rb') as f:

  loc_dict = pickle.load(f)
```

### 2.3 `country_indices` and `country_label`

`country_indices` is a dictionary that maps a country name to a sorted `np.array` of its indices

```
{ 'malawi': np.array([ 8530,  8531,  8532, ..., 10484, 10485, 10486]), ... }
```

`country_labels` is a `np.array` that shows which country each example belongs to

```
np.array([0, 0, 0, 0, ..., 22, 22, 22])
```

where countries are indexed by their position in `dataset_constants.DHS_COUNTRIES`

```python
country_indices = defaultdict(list) # country => np.array of indices

country_labels = np.zeros(num_examples, dtype=np.int32) # np.array of country labels



for i, loc in enumerate(locs):

  country = loc_dict[tuple(loc)]['country']

  country_indices[country].append(i)



for i, country in enumerate(COUNTRIES):

  country_indices[country] = np.asarray(country_indices[country])

  indices = country_indices[country]

  country_labels[indices] = i
```



### 2.4 Get folds

```
*_folds = {
    'A': { # (ooc) list of country names, (incountry) np.array of indices
           'train': [...],
           'val': [...],
           'test': [...],
    }, ...
}
```

#### ooc folds

```python
ooc_folds = {

  f: {split: dataset_constants.SURVEY_NAMES[f'2009-17{f}'][split] for split in SPLITS}

  for f in FOLDS

}
```

#### incountry folds

```python
with open('../data/dhs_incountry_folds.pkl', 'rb') as f:

  incountry_folds = pickle.load(f)



incountry_group_labels = np.zeros(num_examples, dtype=np.int32)

for i, fold in enumerate(FOLDS):

  test_indices = incountry_folds[fold]['test']

  incountry_group_labels[test_indices] = i
```

### 3 General Training Code

```python
def get_indices_for_countries(countries):
    indices =  np.sort(np.concatenate([
        country_indices[country] for country in countries
    ]))
    return indices

def countries_to_nums(countries):
    '''
    Args

   - countries: list or set of str, names of countries
     Returns: nums, list of int
     '''
     nums = []
     for c in countries:
         num = COUNTRIES.index(c)
         nums.append(num)
     return nums
```



### 4 OOC

Define ridgecv_ooc_wrapper function

get model features for each ooc fold

perform ridge cv

save ms and nl models separately 

```python
def ridgecv_ooc_wrapper(model_name, savedir):
    features_dict = {}
    for f in FOLDS:
        model_fold_name = f'{model_name}_{f}'
        model_dir = MODEL_DIRS[model_fold_name]
        npz = load_npz(os.path.join(LOGS_ROOT_DIR, 'DHS_OOC', model_dir, 'features.npz'),
                       check={'labels': labels})
        features = npz['features']
        for country in dataset_constants.SURVEY_NAMES[f'2009-17{f}']['test']:
            features_dict[country] = features

    ridge_cv(
        features=features_dict,
        labels=labels,
        group_labels=country_labels,
        group_names=COUNTRIES,
        do_plot=True,
        savedir=savedir,
        save_weights=True,
        save_dict=dict(locs=locs))

model_name = 'resnet_ms'
savedir = '../logs/dhs_resnet/ms/'
ridgecv_ooc_wrapper(model_name, savedir) 

model_name = 'resnet_nl'
savedir = '../logs/dhs_resnet/nl/'
ridgecv_ooc_wrapper(model_name, savedir)
```

#### 4.1 Concatenated MS, NL Features

define ridgecv ooc wrapper function for concatenating nl and ms features

get model features for nl and ms and concatenate them

perform ridge_cv

```python
def ridgecv_ooc_concat_wrapper(model_names, savedir):
    features_dict = {}
    for f in FOLDS:
        concat_features = []  # list of np.array, each shape [N, D_i]
        for model_name in model_names:
            model_dir = MODEL_DIRS[f'{model_name}_{f}']
            npz = load_npz(os.path.join(LOGS_ROOT_DIR, 'DHS_OOC', model_dir, 'features.npz'),
                           check={'labels': labels})
            concat_features.append(npz['features'])
        concat_features = np.concatenate(concat_features, axis=1) # shape [N, D_1 + ... + D_m]
        for country in dataset_constants.SURVEY_NAMES[f'2009-17{f}']['test']:
            features_dict[country] = concat_features

ridge_cv(
    features=features_dict,
    labels=labels,
    group_labels=country_labels,
    group_names=COUNTRIES,
    do_plot=True,
    savedir=savedir,
    save_weights=True,
    save_dict=dict(locs=locs),
    verbose=True)
```

save concatenated model

```python
model_names = ['resnet_ms', 'resnet_nl']
savedir = os.path.join(LOGS_ROOT_DIR, 'dhs_resnet', 'msnl_concat')
ridgecv_ooc_concat_wrapper(model_names, savedir)
```



#### 4.2 Transfer features

define ridgecv ooc wrapper function for transfer models

s. above

```python
def ridgecv_ooc_transfer_wrapper(model_name, savedir):
    model_dir = MODEL_DIRS[model_name]
    features = load_npz(os.path.join(LOGS_ROOT_DIR, model_dir, 'features.npz'),
                        check={'labels': labels})['features']
    ridge_cv(
        features=features,
        labels=labels,
        group_labels=country_labels,
        group_names=COUNTRIES,
        savedir=savedir,
        save_weights=False,
        save_dict=dict(locs=locs),
        verbose=True)
```

```python
model_name = 'resnet_rgb_transfer'
savedir = '../logs/dhs_resnet/rgb_transfer/'
ridgecv_ooc_transfer_wrapper(model_name, savedir)
```

```python
model_name = 'resnet_ms_transfer'
savedir = '../logs/dhs_resnet/ms_transfer/'
ridgecv_ooc_transfer_wrapper(model_name, savedir)
```



### 5 OOC keep_frac

define function to get train, validation and test indices

define function to get randomized samples (size according to keep_frac)

```
def get_split_idxs(dataset: str):
    train_tfrecord_paths = batcher.get_tfrecord_paths(dataset, 'train')
    val_tfrecord_paths = batcher.get_tfrecord_paths(dataset, 'val')
    test_tfrecord_paths = batcher.get_tfrecord_paths(dataset, 'test')
    all_tfrecord_paths = batcher.get_tfrecord_paths(dataset, 'all')

    path_to_idx = {path: idx for idx, path in enumerate(all_tfrecord_paths)}
    train_idxs = [path_to_idx[path] for path in train_tfrecord_paths]
    val_idxs = [path_to_idx[path] for path in val_tfrecord_paths]
    test_idxs = [path_to_idx[path] for path in test_tfrecord_paths]
    return train_idxs, val_idxs, test_idxs

def get_keep_indices(keep_frac: float, seed: int, dataset: str, test_country: str):
    '''
    Args
    - keep_frac: float, fraction of non-test-country data to use for training
    - seed: int
    - dataset: str, one of the keys of dataset_constants.SIZES[dataset]
    - test_country: str
    '''
    train_idxs, val_idxs, test_idxs = get_split_idxs(dataset)
    print(train_idxs)
    test_country_idxs = get_indices_for_countries([test_country]).tolist()
    test_other_idxs = sorted(set(test_idxs) - set(test_country_idxs))  # sort for determinism
    
    num_train = int(dataset_constants.SIZES[dataset]['train'] * keep_frac)
    num_val = int(dataset_constants.SIZES[dataset]['val'] * keep_frac)
    num_test = int(len(test_other_idxs) * keep_frac)

    np.random.seed(seed)
    train_idxs = np.random.choice(train_idxs, size=num_train, replace=False)
    val_idxs = np.random.choice(val_idxs, size=num_val, replace=False)
    test_other_idxs = np.random.choice(test_other_idxs, size=num_test, replace=False)

    return np.sort(np.concatenate([train_idxs, val_idxs, test_other_idxs, test_country_idxs]))
```

### define run_ridgecv_keep

For every country C (the test country):
      1. uses leave-one-country-out CV on all other countries
         to tune ridge model alpha parameter
            2. using best alpha, trains ridge model on all countries except C
                  3. runs trained ridge model on C
            Saves predictions for each country on test.



FOR EACH FOLD:

​	set model name and directory

​	get features for this fold

​	get test countries and create subset

​	get indices for test subset

​	perform linear logarithm on subset and get predictions

​	

```
def run_ridgecv_keep(model_name, labels, locs, country_labels, folds, keep_frac, seed, savedir):
    '''
    For every country C (the test country):
      1. uses leave-one-country-out CV on all other countries
         to tune ridge model alpha parameter
      2. using best alpha, trains ridge model on all countries except C
      3. runs trained ridge model on C
    Saves predictions for each country on test.

    Args
    - model_name: str, format 'resnet_{bands}', e.g. 'resnet_ms'
    - labels: np.array, shape [num_examples]
    - locs: np.array, shape [num_examples, 2]
    - country_labels: np.array, shape [num_examples]
    - folds: dict, fold (str) => dict
    - keep_frac: float, fraction of non-test-country data to use for training
    - seed: int
    - savedir: str
    '''
    test_preds = np.zeros(num_examples, dtype=np.float32)
    all_countries_set = set(COUNTRIES)

    for f in FOLDS:
        print('Fold:', f)
        model_fold_name = f'{model_name}_{f}'
        model_dir = MODEL_DIRS[model_fold_name]
        model_dir = model_dir.replace('b64', f'keep{keep_frac}_seed{seed}_b64')

        npz = load_npz(
            os.path.join(LOGS_ROOT_DIR, 'DHS_OOC', model_dir, 'features.npz'),
            check=dict(labels=labels, locs=locs))
        features = npz['features']
        dataset = '2009-17' + f

        for test_country in folds[f]['test']:
            print('test country:', test_country)
            keep_subset_indices = get_keep_indices(
                keep_frac=keep_frac,
                seed=seed,
                dataset=dataset,
                test_country=test_country)

            test_country_set = {test_country}
            cv_countries_set = all_countries_set - test_country_set
            test_indices = get_indices_for_countries(test_country_set)
            test_preds[test_indices] = train_linear_logo(
                features=features[keep_subset_indices],
                labels=labels[keep_subset_indices],
                group_labels=country_labels[keep_subset_indices],
                cv_groups=countries_to_nums(cv_countries_set),
                test_groups=countries_to_nums(test_country_set),
                plot=False,
                group_names=COUNTRIES)

    # save preds on the test set
    os.makedirs(savedir, exist_ok=True)
    npz_path = os.path.join(savedir, 'test_preds_keep{k}_seed{s}.npz'.format(k=keep_frac, s=seed))
    print('Saving preds to:', savedir)
    np.savez_compressed(npz_path, test_preds=test_preds, labels=labels, locs=locs)
```

run ridge cv for ms data

```
model_name = 'resnet_ms'
savedir = os.path.join(LOGS_ROOT_DIR, 'dhs_resnet', 'ms')

for keep_frac in KEEPS:
    for seed in SEEDS:
        print('-------------------------------------------')
        print(f'---------- keep: {keep_frac:0.02g}, seed: {seed} ----------')
        print('-------------------------------------------')
        run_ridgecv_keep(
            model_name=model_name,
            labels=labels,
            locs=locs,
            country_labels=country_labels,
            folds=ooc_folds,
            keep_frac=keep_frac,
            seed=seed,
            savedir=savedir)
```

run ridge cv for nl data

```
model_name = 'resnet_nl'
savedir = os.path.join(LOGS_ROOT_DIR, 'dhs_resnet', 'nl')

for keep_frac in KEEPS:
    for seed in SEEDS:
        print('-------------------------------------------')
        print(f'---------- keep: {keep_frac:0.02g}, seed: {seed} ----------')
        print('-------------------------------------------')
        run_ridgecv_keep(
            model_name=model_name,
            labels=labels,
            locs=locs,
            country_labels=country_labels,
            folds=ooc_folds,
            keep_frac=keep_frac,
            seed=seed,
            savedir=savedir)
```

#### 5.1 Concatenated MS, NL Features

define keep ridge cv for concatenating ms and nl features



```
def run_ridgecv_keep_concat(model_names, concat_model_name, labels, locs,
                            country_labels, folds, keep_frac, seed):
    '''
    For every country C (the test country):
      1. uses leave-one-country-out CV on all other countries
         to tune ridge model alpha parameter
      2. using best alpha, trains ridge model on all countries except C
      3. runs trained ridge model on C
    Saves predictions for each country on test.

    Args
    - model_names: list of str
    - concat_model_name: str
    - labels: np.array, shape [num_examples]
    - locs: np.array, shape [num_examples, 2]
    - country_labels: np.array, shape [num_examples]
    - folds: dict, fold (str) => dict
    - keep_frac: float, fraction of non-test-country data to use for training
    - seed: int
    '''
    test_preds = np.zeros(num_examples, dtype=np.float32)
    all_countries_set = set(COUNTRIES)

    for f in FOLDS:
        print('Fold:', f)
        concat_features = []  # list of np.array, each shape [N, D_i]
        for model_name in model_names:
            model_fold_name = f'{model_name}_{f}'
            model_dir = MODEL_DIRS[model_fold_name].replace('b64', f'keep{keep_frac}_seed{seed}_b64')
            npz = load_npz(
                os.path.join(LOGS_ROOT_DIR, 'DHS_OOC', model_dir, 'features.npz'),
                check=dict(labels=labels, locs=locs))
            concat_features.append(npz['features'])
        concat_features = np.concatenate(concat_features, axis=1) # shape [N, D_1 + ... + D_m]
        dataset = '2009-17' + f

        do_plot = True
        for test_country in folds[f]['test']:
            print('test country:', test_country)
            keep_subset_indices = get_keep_indices(
                keep_frac=keep_frac,
                seed=seed,
                dataset=dataset,
                test_country=test_country)

            test_country_set = {test_country}
            cv_countries_set = all_countries_set - test_country_set
            test_indices = get_indices_for_countries(test_country_set)
            test_preds[test_indices] = train_linear_logo(
                features=concat_features[keep_subset_indices],
                labels=labels[keep_subset_indices],
                group_labels=country_labels[keep_subset_indices],
                cv_groups=countries_to_nums(cv_countries_set),
                test_groups=countries_to_nums(test_country_set),
                plot=do_plot,
                group_names=COUNTRIES)
            do_plot = False

    # save preds on the test set
    log_dir = os.path.join(LOGS_ROOT_DIR, 'dhs_resnet', concat_model_name)
    os.makedirs(log_dir, exist_ok=True)
    npz_path = os.path.join(log_dir, 'test_preds_keep{k}_seed{s}.npz'.format(k=keep_frac, s=seed))
    np.savez_compressed(npz_path, test_preds=test_preds, locs=locs)
```

```
for keep_frac in KEEPS:
    for seed in SEEDS:
        print('-------------------------------------------')
        print(f'---------- keep: {keep_frac:0.02g}, seed: {seed} ----------')
        print('-------------------------------------------')
        run_ridgecv_keep_concat(
            model_names=['resnet_ms', 'resnet_nl'],
            concat_model_name='msnl_concat',
            labels=labels,
            locs=locs,
            country_labels=country_labels,
            folds=ooc_folds,
            keep_frac=keep_frac,
            seed=seed)
```



### 6 Incountry

Same as above only using incountry data

```
def ridgecv_incountry_wrapper(model_name, savedir):
    features_dict = {}
    for f in FOLDS:
        model_fold_name = f'{model_name}_{f}'
        model_dir = MODEL_DIRS[model_fold_name]
        npz = load_npz(os.path.join(LOGS_ROOT_DIR, model_dir, 'features.npz'),
                       check={'labels': labels})
        features_dict[f] = npz['features']

    ridge_cv(
        features=features_dict,
        labels=labels,
        group_labels=incountry_group_labels,
        group_names=FOLDS,
        savedir=savedir,
        save_weights=True,
        verbose=True)
```

```
model_name = 'incountry_resnet_ms'
savedir = '../logs/dhs_resnet/incountry_ms/'
ridgecv_incountry_wrapper(model_name, savedir)
```

```
model_name = 'incountry_resnet_nl'
savedir = '../logs/dhs_resnet/incountry_nl/'
ridgecv_incountry_wrapper(model_name, savedir)
```



#### 6.1 Concatenated MS, NL Features

```
def ridgecv_incountry_concat_wrapper(model_names, savedir):
    features_dict = {}
    for i, f in enumerate(FOLDS):
        concat_features = []  # list of np.array, each shape [N, D_i]
        for model_name in model_names:
            model_dir = MODEL_DIRS[f'{model_name}_{f}']
            npz = load_npz(os.path.join(LOGS_ROOT_DIR, model_dir, 'features.npz'),
                           check={'labels': labels})
            concat_features.append(npz['features'])
        concat_features = np.concatenate(concat_features, axis=1) # shape [N, D_1 + ... + D_m]
        features_dict[f] = concat_features

    ridge_cv(
        features=features_dict,
        labels=labels,
        group_labels=incountry_group_labels,
        group_names=FOLDS,
        savedir=savedir,
        save_weights=True,
        verbose=True)
```

```
model_names = ['incountry_resnet_ms', 'incountry_resnet_nl']
savedir = os.path.join(LOGS_ROOT_DIR, 'dhs_resnet', 'incountry_msnl_concat')
ridgecv_incountry_concat_wrapper(model_names, savedir)
```



#### 6.2 Transfer Features

```
def ridgecv_incountry_transfer_wrapper(model_name, savedir):
    model_dir = MODEL_DIRS[model_name]
    features = load_npz(os.path.join(LOGS_ROOT_DIR, model_dir, 'features.npz'),
                        check={'labels': labels})['features']
    ridge_cv(
        features=features,
        labels=labels,
        folds=incountry_folds,
        savedir=savedir,
        save_weights=False)
```

```
model_name = 'resnet_rgb_transfer'
savedir = '../logs/dhs_resnet/incountry_rgb_transfer/'
ridgecv_incountry_transfer_wrapper(model_name, savedir)
```

```
model_name = 'resnet_ms_transfer'
savedir = '../logs/dhs_resnet/incountry_ms_transfer/'
ridgecv_incountry_transfer_wrapper(model_name, savedir)
```



