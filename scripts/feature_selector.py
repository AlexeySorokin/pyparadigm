#-------------------------------------------------------------------------------
# Name:        feature_selector.py
# Purpose:     модуль, реализующий различные стратегии выбора признаков
#
# Created:     05.09.2015
#-------------------------------------------------------------------------------

import sys
from functools import reduce

import numpy as np

from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from routines.statistic import odds_ratio, information_gain

class MulticlassFeatureSelector(SelectorMixin):
    '''
    Класс, реализующий различные стратегии выбора признаков
    для мультиклассовой классификации

    Атрибуты:
    ---------
    local: bool, optional(default=True)
        если True, то признаки отбираются для каждого класса в отдельности,
        если False, то производится общая оценка признаков

    method: str('log_odds' or 'ambiguity'), optional(default='ambiguity')
        метод отбора признаков

    nfeatures: int, optional(default=-1)
        число признаков, если -1, то оставляются все признаки

    threshold: float, array or None, optional(default=None)
        пороговое значение веса признаков,
        отбираются только признаки с весом >= threshold,
        если threshold=None, то отбираются nfeatures лучших признаков

    order: str('fixed' or 'random'), optional(default='fixed')
        порядок рассмотрения признаков, фиксированный или случайный

    random_state: int, optional(default=0)
        значение для инициализации генератора случайных чисел
    '''

    _METHODS = ['ambiguity', 'log_odds']
    _CONTINGENCY_METHODS = ['log_odds']

    def __init__(self, local=True, method='ambiguity', min_count=2,
                 nfeatures=-1, threshold=None, order='fixed', random_state=0):
        self.local = local
        self.method = method
        self.min_count = min_count
        self.nfeatures = nfeatures
        self.threshold = threshold
        self.order = order
        self.random_state = random_state

    def fit(self, X, y):
        if self.method not in self._METHODS:
            raise ValueError("Method should be one of {0}".format(
                    ", ".join(self._METHODS)))
        X = check_array(X, accept_sparse='csr')
        self._N, self._ndata_features = X.shape

        self.classes_, y = np.unique(y, return_inverse=True)
        nclasses = self.classes_.shape[0]
        Y_new = np.zeros(shape=(self._N, nclasses), dtype=int)
        Y_new[np.arange(self._N), y] = 1

        if (self.threshold is None and
            (self.nfeatures >= self._ndata_features or self.nfeatures == -1)):
            self.scores_ = np.ones(shape=(nclasses, self._ndata_features))
        else:
            self.scores_ = self._calculate_scores(X, Y_new)
        # print(np.count_nonzero(self.scores_))
        self.mask_ = self._extract_mask()
        return self

    def _calculate_scores(self, X, Y):
        '''
        Вычисляет веса по алгоритму, соответствующему self.method
        '''
        classes_counts = Y.sum(axis=0) # встречаемость классов
        counts = X.sum(axis=0).A1
        counts_by_classes  = safe_sparse_dot(Y.T, X)
        # print(np.count_nonzero(counts_by_classes))

        # обнуляем редко встречающиеся признаки
        rare_indices = np.where(counts < self.min_count)
        counts[rare_indices] = 0
        counts_by_classes[:,rare_indices] = 0
        # print(np.count_nonzero(counts_by_classes))

        # вычисляем таблицы сопряжённости
        if self.method in self._CONTINGENCY_METHODS:
            nclasses = Y.shape[1]
            contingency_table = np.zeros(shape=(nclasses, ndata_features, 2, 2),
                                         dtype=np.float64)
            for i in range(nclasses):
                for j in range(self._ndata_features):
                    count = counts_by_classes[i,j]
                    feature_count, class_count = counts[j], classes_counts[i]
                    rest = self._N - count - feature_count + count
                    contingency_table[i, j] = [[rest, feature_count - count],
                                               [class_count - count, count]]
            if self.selection == 'log_odds':
                func = (lambda x: odds_ratio(x, alpha=0.1))
            else:
                func = information_gain
            # КАК СДЕЛАТЬ БЕЗ ПРЕОБРАЗОВАНИЙ
            scores = np.array([[func(contingency_table[i][j])
                                for j in range(ndata_features)]
                               for i in range(nclasses)])
        elif self.method == 'ambiguity':
            scores = counts_by_classes / counts
        scores[np.isnan(scores)] = 0.0
        return scores

    def _extract_mask(self):
        '''
        Извлекает булеву маску для признаков на основе весов
        '''
        mask = np.zeros(dtype=bool, shape=(self._ndata_features,))
        if not self.local:
            raise NotImplementedError()
        else:
            # устанавливаем порядок на классах
            classes_order = np.arange(self.classes_.shape[0])
            if self.order == 'random':
                np.random.seed(self.random_state)
                classes_order = np.random.permutation(classes_order)
            # сортируем индексы для каждого класса
            indexes_to_select = list(self._extract_feature_indexes(classes_order))
            mask[indexes_to_select] = True
        return mask

    def _extract_feature_indexes(self, classes_order):
        '''
        Возвращает множество индексов остающихся переменных
        '''
        if self.threshold is not None:
            return set(np.where(self.scores_ >= self.threshold)[1])
        sorted_feature_indexes = [_sort_feature_indexes_by_scores(elem) for elem in self.scores_]
        selected = set()
        for i in range(self.scores_.shape[1]):
            for label in classes_order:
                selected.add(sorted_feature_indexes[label][i])
                if len(selected) == self.nfeatures:
                    return selected
        return selected

    def _get_support_mask(self):
        check_is_fitted(self, "mask_")
        return self.mask_

def _sort_feature_indexes_by_scores(scores):
    '''
    Сортирует индексы по убыванию весов,
    в случае равных весов сортировка ведётся по возрастанию индексов
    '''
    negated_scores = -scores
    indexes = np.arange(scores.shape[0])
    return np.lexsort((indexes, negated_scores))


if __name__ == "__main__":
    a = [3, 2, 3, 1, 2]
    a = np.array(a)
    print(_sort_feature_indexes_by_scores(a))




