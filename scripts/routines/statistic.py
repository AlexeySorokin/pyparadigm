import os
import sys

import numpy as np
import scipy.stats as scs

LOG_MIN_THRESHOLD = 0.000005

def bound(value, min_value, max_value):
    '''
    restricts value to segment [min_value, max_value]
    '''
    return max([min_value, min([value, max_value])])

def log_odds(value):
    value = zeroone_bound(value, LOG_MIN_THRESHOLD)
    return np.log(value / (1.0 - value))

def zeroone_bound(value, threshold = LOG_MIN_THRESHOLD):
    return bound(value, threshold, 1.0 - threshold)

def probs_to_score(probs):
    '''
    Transforms array (1 - p, p) to log(p / (1-p))
    '''
    assert np.array(probs).shape == (2,),\
           "Probs must have shape (2, ), actual shape is {0}".format(np.array(probs).shape)
    return np.log(zeroone_bound(probs[1])) - np.log(zeroone_bound(probs[0]))

def entropy(x):
    '''
    calculates entropy of a given probability distribution
    '''
    x = np.array(x, dtype = 'float64')
    x = np.copy(x[np.nonzero(x)])
    return -np.dot(x, np.log(x))

def entropy_normalized(x):
    '''
    calculates entropy of a given empirical count distribution
    (the distribution may be unnormalized)
    '''
    x = np.array(x, dtype = 'float64')
    if np.sum(x) == 0.0:
        return 0.0
    return entropy(x / np.sum(x))

def conditional_entropy(a):
    '''
    calculates the conditional entropy H(y | x) provided the counts
    n(y_1 | x_1) ... n(y_1 | x_m)
    ...          ... ...
    n(y_n | x_1) ... n(y_n | x_m)
    '''
    a = np.array(a, dtype='float64')
    assert len(a.shape) <= 2, "Input must be a contingency table"
    if len(a.shape) == 1:
        a = a[:, np.newaxis]
    b = np.sum(a, axis = 1)
    N = np.sum(b)
    return np.dot(b / N, np.apply_along_axis(entropy_normalized, 1, a))

def information_gain(a):
    '''
    calculates information gain H(y) - H(y | x) provided the counts
    n(y_1 | x_1) ... n(y_n | x_1)
    ...          ... ...
    n(y_1 | x_m) ... n(y_n | x_m)
    '''
    return entropy_normalized(np.sum(a, axis = 0)) - conditional_entropy(a)

def gain_ratio(a):
    '''
    calculates information gain ratio 1.0 - H(y | x) / H(y) provided the counts
    n(y_1 | x_1) ... n(y_n | x_1)
    ...          ... ...
    n(y_1 | x_m) ... n(y_n | x_m)
    '''
    return  1.0 - conditional_entropy(a) / entropy_normalized(np.sum(a, axis = 0))

def precision(a):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      FP TP ) and returns precision
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0)
    TN, FN, FP, TP = tuple(a)
    if TP + FP > 0.0:
        return TP / (TP + FP)
    else:
        return 1.0

def recall(a):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      FP TP ) and returns recall
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0)
    TN, FN, FP, TP = tuple(a)
    if TP + FN > 0.0:
        return TP / (TP + FN)
    else:
        return 1.0

def f_measure(a):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      TP FP ) and returns f-measure
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0)
    TN, FN, FP, TP = tuple(a)
    if TP + FP + FN > 0.0:
        return TP / (TP + 0.5 * (FN + FP))
    else:
        return 1.0

def symmetric_f_measure(a):
    '''
    returns symmetrized version of F-measure
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0) and np.sometrue(a > 0.0)
    TN, FN, FP, TP = tuple(a)
    return 1.0 - 0.5 * (FN + FP) / (max([TN, TP]) + 0.5 * (FN + FP))

def bns(a):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      FP TP ) and returns BNS-coefficient (Forman, 2003)
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0)
    TN, FN, FP, TP = tuple(a)
    pos, neg = TP + FN, TN + FP
    tpr, fpr = TP / pos, FP / neg
    tpr, fpr = map(lambda x: zeroone_bound(x, LOG_MIN_THRESHOLD), (tpr, fpr))
    return max([np.fabs(scs.norm.ppf(tpr) - scs.norm.ppf(fpr)),
                np.fabs(scs.norm.ppf(1.0 - fpr) - scs.norm.ppf(1.0 - tpr))])

def chi_squared(a):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      FP TP ) and returns chi-squared statistics
    '''
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0) and np.sometrue(a > 0.0)
    TN, FN, FP, TP = tuple(a)
    odds_diff = TN * TP - FN * FP
    if max([FN * FP, TN * TP]) > 0.0:
        return odds_diff * odds_diff / ((TN + FN) * (TN + FP) * (TP + FP) * (TP + FN))
    if min([TN + TP, FN + FP]) == 0.0:
        return 1.0
    else:
        return 0.0

def chi_squared_generalized(a):
    '''
    takes contingency tabel
    n(y_1 | x_1) ... n(y_n | x_1)
    ...          ... ...
    n(y_1 | x_m) ... n(y_n | x_m)
    and calculates the normalized chi-squared statistics
    '''
    a = np.array(a, dtype='float64')
    assert len(a.shape) <= 2, "Input must be a contingency table"
    if len(a.shape) == 1:
        a = a[:, np.newaxis]
    #calculating counts for chi-squared statistics
    #if a is a 2*2 contingency table, using explicit formula
    if a.shape == (2, 2):
        return chi_squared(a)
    rs, cs = np.sum(a, axis = 1), np.sum(a, axis = 0)
    nonzero_rows, nonzero_cols = rs.nonzero(), cs.nonzero()
    a, rs, cs = a[nonzero_rows, nonzero_cols], rs[nonzero_rows], cs[nonzero_cols]
    N = np.sum(rs)
    expected = np.outer(rs, cs) / N
    return np.sum((a - expected) * (a - expected)  / expected) / N

def odds_ratio(a, alpha = 1.0):
    '''
    takes tuple (TN, FN, FP, TP) or array
    ( TN FN
      TP FP ) and returns odds_ratio
    '''
    #adding alpha to prevent zero_division
    a = np.asarray(a, dtype = 'float64').reshape(4)
    assert np.all(a >= 0) and np.sometrue(a > 0.0)
    TN, FN, FP, TP = tuple(a)
    return np.log((TP + alpha) * (TN + alpha) / ((FP + alpha) * (FN + alpha)))

if __name__ == '__main__':
    a = np.array([[3, 0, 1], [0, 0, 0], [2, 0, 2]])
    print (chi_squared_generalized(a))
    b = np.array([[3, 1],[2, 2]])
    print (chi_squared(b))

