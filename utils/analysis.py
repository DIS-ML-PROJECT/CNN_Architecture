import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics


def calc_score(labels, preds, metric, weights=None):
    '''
    See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    for the weighted correlation coefficient formula.
    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - score: str, one of ['r2', 'R2', 'mse', 'rank']
        - 'r2': (weighted) squared Pearson correlation coefficient
        - 'R2': (weighted) coefficient of determination
        - 'mse': (weighted) mean squared-error
        - 'rank': (unweighted) Spearman rank correlation coefficient
    - weights: np.array, shape [N]
    Returns: float
    '''
    if metric == 'r2':
        if weights is None:
            return scipy.stats.pearsonr(labels, preds)[0] ** 2
        else:
            mx = np.average(preds, weights=weights)
            my = np.average(labels, weights=weights)
            cov_xy = np.average((preds - mx) * (labels - my), weights=weights)
            cov_xx = np.average((preds - mx) ** 2, weights=weights)
            cov_yy = np.average((labels - my) ** 2, weights=weights)
            return cov_xy ** 2 / (cov_xx * cov_yy)
    elif metric == 'R2':
        return sklearn.metrics.r2_score(y_true=labels, y_pred=preds,
                                        sample_weight=weights)
    elif metric == 'mse':
        return np.average((labels - preds) ** 2, weights=weights)
    elif metric == 'rank':
        return scipy.stats.spearmanr(labels, preds)[0]
    else:
        raise ValueError(f'Unknown metric: "{metric}"')



def evaluate_df(df, cols, labels_col='label', weights_col=None, index_name=None):
    '''
    Args
    - df: pd.DataFrame, columns include cols and labels_col
    - cols: list of str, names of cols in df to evaluate
    - labels_col: str, name of labels column
    - weights_col: str, name of weights column, optional
    - index_name: str, name of index for returned df
    Returns
    - results_df: pd.DataFrame, columns are ['r2', 'R2', 'mse', 'rank']
        row index are `cols`
    '''
    labels = df[labels_col]
    weights = None if weights_col is None else df[weights_col]
    records = []
    for col in cols:
        row = evaluate(labels=labels, preds=df[col], weights=weights)
        records.append(row)
    index = pd.Index(data=cols, name=index_name)
    results_df = pd.DataFrame.from_records(
        records, columns=['r2', 'R2', 'mse', 'rank'], index=index)
    return results_df