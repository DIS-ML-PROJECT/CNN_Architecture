from utils.analysis import evaluate

import os
import time

import numpy as np
import scipy.spatial

def knn_cv_opt(features, labels, group_labels, group_names, savedir=None, weights=None,
               do_plot=False, subset_indices=None, subset_name=None, save_dict=None,
               distance_metric='cityblock'):
    '''Similar to knn_cv(), but pre-computes a distance matrix to use for all folds.

    For every fold F (the test fold):
      1. uses leave-one-fold-out CV on all other folds
         to tune KNN k parameter
      2. using best k, trains KNN model on all folds except F
      3. runs trained ridge model on F

    Saves predictions for each fold on test.
        savedir/test_preds_{subset_name}.npz if subset_name is given
        savedir/test_preds.npz otherwise

    Args
    - features: np.array, shape [N, D]
    - labels: np.array, shape [N]
    - group_labels: np.array, shape [N], type int
    - group_names: list of str, a group_label of X corresponds to group_names[X]
    - savedir: str, path to directory to save predictions
    - weights: np.array, shape [N], optional
    - do_plot: bool, whether to plot alpha vs. mse curve for 1st fold
    - subset_indices: np.array, indices of examples to include for both
        training and testing
    - subset_name: str, name of the subset
    - save_dict: dict, str => np.array, saved with test preds npz file
    - distance_metric: str, see documentation for scipy.spatial.distance.pdist

    Returns
    - test_preds: np.array, shape [N]
    '''
    N = len(labels)
    assert len(features) == N
    assert len(group_labels) == N

    if save_dict is None:
        save_dict = {}
    else:
        save_dict = dict(save_dict)  # make a copy

    if subset_indices is None:
        assert subset_name is None
        filename = 'test_preds.npz'
    else:
        assert subset_name is not None
        features = features[subset_indices]
        labels = labels[subset_indices]
        group_labels = group_labels[subset_indices]

        filename = f'test_preds_{subset_name}.npz'
        for key in save_dict:
            save_dict[key] = save_dict[key][subset_indices]

    if savedir is not None:
        npz_path = os.path.join(savedir, filename)
        assert not os.path.exists(npz_path)

    print('Pre-computing distance matrix...', end='')
    start = time.time()
    dists = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(features, metric=distance_metric)
    )
    elapsed = time.time() - start
    print(f' took {elapsed:.2f} seconds.')

    test_preds = np.zeros_like(labels, dtype=np.float32)
    for i, f in enumerate(group_names):
        print('Group:', f)
        test_mask = (group_labels == i)
        if np.sum(test_mask) == 0:
            print(f'no examples corresponding to group {f} were found')
            continue
        test_preds[test_mask] = train_knn_logo_opt(
            dists=dists,
            features=features,
            labels=labels,
            group_labels=group_labels,
            cv_groups=[x for x in range(len(group_names)) if x != i],
            test_groups=[i],
            weights=weights,
            plot=do_plot,
            group_names=group_names)

        # only plot the curve for the first group
        do_plot = False

    evaluate(labels=labels, preds=test_preds, weights=weights, do_print=True, title='Pooled test preds')

    # save preds on the test set
    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)

        # build up save_dict
        if 'labels' in save_dict:
            assert np.array_equal(labels, save_dict['labels'])
        save_dict['labels'] = labels
        if weights is not None:
            save_dict['weights'] = weights
        save_dict['test_preds'] = test_preds

        print('saving test preds to:', npz_path)
        np.savez_compressed(npz_path, **save_dict)

    return test_preds