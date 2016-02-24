#-------------------------------------------------------------------------------
# Name:        local_paradigm_transformer.py
#-------------------------------------------------------------------------------

import sys
from itertools import chain
from collections import OrderedDict
import bisect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm as sksvm
from sklearn import linear_model as sklm
from scipy.sparse import csr_matrix

from paradigm_detector import ParadigmFragment, make_flection_paradigm, get_flection_length
from feature_selector import MulticlassFeatureSelector

class LocalTransformation:
    """
    Класс для представления транссформации внутри парадигмы

    Атрибуты:
    ---------
    trans: tuple of strs,
    is_suffix: bool(optional, default=False), происходит ли трансформация в конце слова
    """
    def __init__(self, trans, is_suffix, is_prefix=False):
        self.trans = tuple(trans)
        self.is_suffix = is_suffix
        self.is_prefix = is_prefix

    def __str__(self):
        return "#".join(self.trans)

    def to_tuple(self):
        return (self.trans, self.is_suffix, self.is_prefix)

    def __lt__(self, other):
        return self.to_tuple() < other.to_tuple()

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def __repr__(self):
        return "{0} {1} {2}".format(*self.to_tuple())


def descr_to_transforms(descr, return_lemma=True, has_prefix_changes=False):
    """
    Преобразует описание парадигмы формата вида
    1+е+2+ь#1+е+2+ь#1+2+и#1+2+я#1+2+ей#1+2+ю#1+2+ям#1+е+2+ь#1+2+и#1+2+ем#1+2+ями#1+2+е#1+2+ях
    в трансформации
    (e, '', '', '', '', '', e, '', '', '', '', ''), False и
    (ь, и, я, ей, ю, ям, ь, и, ем, ями, е, ях), True
    """
    splitted_descr = descr.split('#')
    # в нулевой ячейке лемма
    if not return_lemma:
        splitted_descr = splitted_descr[1:]
    paradigm_fragments, paradigm_indexes = [], []
    for i, pattern in enumerate(splitted_descr):
        if pattern in ['-']:
            continue
        paradigm_indexes.append(i)
        paradigm_fragments.append(ParadigmFragment(pattern))
    constant_fragments = [fragment.const_fragments for fragment in paradigm_fragments]
    constant_fragments = list(zip(*constant_fragments))
    for i, elem in enumerate(constant_fragments):
        new_elem = ['-'] * len(splitted_descr)
        for j, substr in zip(paradigm_indexes, elem):
            new_elem[j] = substr
        constant_fragments[i] = new_elem
    answer = [LocalTransformation(elem, False) for elem in constant_fragments[1:-1]]
    answer.append(LocalTransformation(constant_fragments[-1], True))
    if has_prefix_changes:
        answer.append(LocalTransformation(constant_fragments[0], False, True))
    return answer

def test():
    tests = ['1#1#1+ы#1+а#1+ов#1+у#1+ам#1#1+ы#1+ом#1+ами#1+е#1+ах',
             '1+о+2#1+о+2#1+2+и#1+2+а#1+2+ов#1+2+у#1+2+ам#1+о+2#1+2+и#1+2+ом#1+2+ами#1+2+е#1+2+ах']
    for descr in tests:
        print(descr)
        for elem in descr_to_transforms(descr):
            print(elem.trans)
        print("")


if __name__ == "__main__":
    test()