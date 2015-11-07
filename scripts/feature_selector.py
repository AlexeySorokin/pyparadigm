# -------------------------------------------------------------------------------
# Name:        feature_selector.py
# Purpose:     модуль, реализующий различные стратегии выбора признаков
#
# Created:     05.09.2015
# -------------------------------------------------------------------------------

import sys
from functools import reduce

import numpy as np
from scipy.sparse import issparse, csc_matrix, csr_matrix

from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer as SK_Binarizer
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted

from routines.statistic import odds_ratio, information_gain, bns


class MulticlassFeatureSelector(SelectorMixin):
    """
    Класс, реализующий различные стратегии выбора признаков
    для мультиклассовой классификации

    Атрибуты:
    ---------
    local: bool, optional(default=True)
        если True, то признаки отбираются для каждого класса в отдельности,
        если False, то производится общая оценка признаков

    method: str('log_odds' or 'ambiguity'), optional(default='ambiguity')
        метод отбора признаков

    min_count: int, optional(default=2)
        минимальное число, которое признак должен встретиться
        в обучающей выборке, чтобы он мог быть отобран

    nfeatures: int, optional(default=-1)
        число признаков, если -1, то оставляются все признаки

    threshold: float, array or None, optional(default=None)
        пороговое значение веса признаков,
        отбираются только признаки с весом >= threshold,
        если threshold=None, то отбираются nfeatures лучших признаков

    binarization_method: str('log_odds' or 'BNS'), optional(default='log_odds')
        метод подбора порога при бинаризации признаков

    divide_to_bins: bool, optional(default=True),
        следует ли приводить количественные признаки
        к целочисленным значениям

    bins_number: int, optional(default=10),
         число возможных целочисленных значений признаков
         при приведении к целочисленным значениям

    order: str('fixed' or 'random'), optional(default='fixed')
        порядок рассмотрения признаков, фиксированный или случайный

    random_state: int, optional(default=0)
        значение для инициализации генератора случайных чисел
    """

    _METHODS = ['ambiguity', 'log_odds', 'information gain']
    _CONTINGENCY_METHODS = ['log_odds', 'information gain']

    def __init__(self, local=True, method='ambiguity', min_count=2,
                 nfeatures=-1, threshold=None, binarization_method='log_odds',
                 divide_to_bins=True, bins_number=10, order='fixed', random_state=0):
        self.local = local
        self.method = method
        self.min_count = min_count
        self.nfeatures = nfeatures
        self.threshold = threshold
        self.binarization_method = binarization_method
        self.divide_to_bins = divide_to_bins
        self.bins_number = bins_number
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

        if (self.threshold is None and (self.nfeatures is None or
                                        self.nfeatures >= self._ndata_features or
                                        self.nfeatures == -1)):
            # не нужно отбирать признаки
            self.scores_ = np.ones(shape=(nclasses, self._ndata_features))
            self.nfeatures = self._ndata_features
        else:
            print("Fitting selector...")
            self.scores_ = self._calculate_scores(X, Y_new)
        self._make_mask()
        return self

    def _calculate_scores(self, X, Y):
        """
        Вычисляет веса по алгоритму, соответствующему self.method
        """
        # проверяем, содержит ли матрица только 0 или 1, для чего
        # проверяем её на равенство с результатом приведения к двоичному виду
        X_binary = check_array(X, accept_sparse=['csr', 'csc'], dtype='bool')
        if issparse(X):
            is_binary = not (X != X_binary).data.any()
        else:
            is_binary = (X == X_binary).all()
        if is_binary:
            scores = self._calculate_scores_for_binary(X, Y)
        else:
            # сначала бинаризуем данные
            binarizer = Binarizer(method=self.binarization_method,
                                  divide_to_bins=self.divide_to_bins,
                                  bins_number=self.bins_number)
            binarizer.fit(X, Y)
            scores = binarizer.scores_
        return scores

    def _calculate_scores_for_binary(self, X, Y):
        """
        Вычисляет полезность для бинарных признаков

        Аргументы:
        ----------
        X: array-like, shape=(nobj, self.nfeatures)
            массив с обучающими данными
        Y: array-like, shape=(nobj, nclasses)
            массив значений классов, закодированных с помощью двоичных векторов

        Возвращает:
        -----------
        scores: array-like, shape=(self.nfeatures, nclasses)
            scores[i][j] показывает полезность i-го признака для j-го класса
        """
        classes_counts = Y.sum(axis=0)  # встречаемость классов
        counts = X.sum(axis=0).A1  # встречаемость признаков
        # совместная встречаемость признаков и классов
        counts_by_classes = safe_sparse_dot(Y.T, X)

        # обнуляем редко встречающиеся признаки
        rare_indices = np.where(counts < self.min_count)
        counts[rare_indices] = 0
        counts_by_classes[:, rare_indices] = 0

        if self.method in self._CONTINGENCY_METHODS:
            # вычисляем таблицы сопряжённости
            nclasses = Y.shape[1]
            contingency_table = np.zeros(shape=(nclasses, self._ndata_features, 2, 2),
                                         dtype=np.float64)
            for i in range(nclasses):
                for j in range(self._ndata_features):
                    count = counts_by_classes[i, j]
                    feature_count, class_count = counts[j], classes_counts[i]
                    rest = self._N - count - feature_count + count
                    contingency_table[i, j] = [[rest, feature_count - count],
                                               [class_count - count, count]]
            if self.method == 'log_odds':
                func = (lambda x: odds_ratio(x, alpha=0.1))
            else:
                func = information_gain
            # КАК СДЕЛАТЬ БЕЗ ПРЕОБРАЗОВАНИЙ
            scores = np.array([[func(contingency_table[i][j])
                                for j in range(self._ndata_features)]
                               for i in range(nclasses)])
        elif self.method == 'ambiguity':
            scores = counts_by_classes / counts
        else:
            raise ValueError("Wrong feature selection method: {0}, "
                             "only the following methods are available: {1}".format(
                                self.method, self._CONTINGENCY_METHODS))
        scores[np.isnan(scores)] = 0.0
        return scores

    def _make_mask(self):
        """
        Извлекает булеву маску для признаков на основе весов
        """
        if self.nfeatures >= self._ndata_features:
            self.mask_ = np.ones(dtype=bool, shape=(self._ndata_features,))
            return
        self.mask_ = np.zeros(dtype=bool, shape=(self._ndata_features,))
        if not self.local:
            raise NotImplementedError()
        else:
            # устанавливаем порядок на классах
            classes_order = np.arange(self.classes_.shape[0])
            if self.order == 'random':
                np.random.seed(self.random_state)
                classes_order = np.random.permutation(classes_order)
            # сортируем индексы для каждого класса
            # indexes_to_select = list(self._extract_feature_indexes(classes_order))
            # mask[indexes_to_select] = True
            self.mask_ = self._extract_feature_indexes(classes_order)

    def _extract_feature_indexes(self, classes_order):
        """
        Возвращает множество индексов остающихся переменных

        Аргументы:
        ----------
        classes_order: array-like, shape=self.classes_.shape,
            порядок предъявления классов (сейчас не используется)
        """
        if self.threshold is not None:
            return set(np.where(self.scores_ >= self.threshold)[1])
        # сортируем признаки по полезности отдельно для каждого класса
        # argsort нужен для стабильности сортировки
        sorted_feature_indexes = np.argsort(-self.scores_, axis=1)
        selected = np.zeros(shape=self.mask_.shape, dtype=bool)
        for indexes in sorted_feature_indexes.T:
            np.put(selected, indexes, True)
            if np.count_nonzero(selected) >= self.nfeatures:
                break
        return selected

    def _get_support_mask(self):
        """
        Возвращает булеву маску для признаков
        """
        check_is_fitted(self, "mask_")
        return self.mask_


class Binarizer(TransformerMixin):
    """
    Реализует различные стратегии бинаризации признаков,
    вычисляя оптимальные пороги и производя бинаризацию с данными порогами

    Аргументы:
    ----------
    method: str('random', 'log_odds' or 'bns'), метод бинаризации признаков
    divide_to_bins: bool(optional, default=True),
        индикатор приведения количественных признаков к целочисленным
    bins_number: int(optional, default=10),
        число возможных значений целочисленных признаков при бинаризации
    """
    _UNSUPERVISED_METHODS = ['random']
    _SUPERVISED_METHODS = ['log_odds', 'bns']
    _CONTINGENCY_METHODS = ['log_odds', 'bns']

    def __init__(self, method, divide_to_bins=True, bins_number=10):
        self.method = method
        self.divide_to_bins = divide_to_bins
        self.bins_number = bins_number

    def fit(self, X, y=None):
        """
        Обучает бинаризатор на данных
        """
        print("Fitting binarizer...")
        methods = Binarizer._UNSUPERVISED_METHODS + Binarizer._SUPERVISED_METHODS
        if self.method not in methods:
            raise ValueError("Method should be one of {0}".format(", ".join(methods)))
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if issparse(X):
            X = X.tocsc()
        if self.method in Binarizer._UNSUPERVISED_METHODS:
            self._fit_unsupervised(X)
            self.joint_thresholds_ = self.thresholds_
            self.joint_scores_ = self.scores_
        else:
            if y is None:
                raise ValueError("y must not be None for supervised binarizers.")
            # вынести в отдельную функцию
            y = np.array(y)
            if len(y.shape) == 1:
                self.classes_, y = np.unique(y, return_inverse=True)
                nclasses = self.classes_.shape[0]
                Y_new = np.zeros(shape=(y.shape[0], nclasses), dtype=int)
                Y_new[np.arange(y.shape[0]), y] = 1
            else:
                self.classes_ = np.arange(y.shape[1])
                Y_new = y
            if X.shape[0] != Y_new.shape[0]:
                raise ValueError("X and y have incompatible shapes.\n"
                                 "X has %s samples, but y has %s." %
                                 (X.shape[0], Y_new.shape[0]))
            self._fit_supervised(X, Y_new)
            if len(self.classes_) <= 2:
                self.joint_thresholds_ = self.thresholds_[:, 0]
                self.joint_scores_ = self.scores_[:, 0]
            else:
                min_class_scores = np.min(self.scores_, axis=0)
                max_class_scores = np.max(self.scores_, axis=0)
                diffs = max_class_scores - min_class_scores
                diffs[np.where(diffs == 0)] = 1
                normalized_scores = (self.scores_ - min_class_scores) / diffs
                # находим для каждого признака тот класс, для которого он наиболее полезен
                # НАВЕРНО, МОЖНО СДЕЛАТЬ ПО_ДРУГОМУ
                optimal_indexes = np.argmax(normalized_scores, axis=1)
                nfeat = self.thresholds_.shape[0]
                # в качестве порога бинаризации каждого признака
                # берём значение для класса, где он наиболее полезен
                self.joint_thresholds_ = self.thresholds_[np.arange(nfeat), optimal_indexes]
                self.joint_scores_ = self.scores_[np.arange(nfeat), optimal_indexes]
        # передаём пороги в sklearn.SK_Binarizer
        self.binarize_transformer_ = SK_Binarizer(self.joint_thresholds_)
        return self

    def transform(self, X):
        """
        Применяем бинаризатор к данным
        """
        print("Transforming binarizer...")
        if hasattr(self, 'binarize_transformer_'):
            return self.binarize_transformer_.transform(X)
        else:
            raise ValueError("Transformer is not fitted")

    def _fit_unsupervised(self, X):
        """
        Управляющая функция для методов подбора порога без учителя
        """
        if self.method == 'random':
            # случайные пороги и полезности
            if issparse(X):
                minimums = X.min(axis=0).toarray()
                maximums = X.max(axis=0).toarray()
            else:
                minimums = np.min(X, axis=0)
                maximums = np.max(X, axis=0)
            random_numbers = np.random.rand(X.shape[1], 1).reshape((X.shape[1],))
            self.thresholds_ = minimums + (maximums - minimums) * random_numbers
            self.scores_ = np.random.rand(X.shape[1], 1).reshape((X.shape[1],))
        return self

    def _fit_supervised(self, X, y):
        """
        Выполняет подбор порогов с учителем
        """
        # приводим X к целочисленным значениям, если нужно
        if self.divide_to_bins:
            bin_divider = BinDivider(bins_number=self.bins_number)
            X = bin_divider.fit_transform(X)
        thresholds, scores = [], []
        for i in range(X.shape[1]):
            threshold, score = self._find_optimal_thresholds(X[:, i], y)
            thresholds.append(threshold)
            scores.append(score)
        self.thresholds_ = np.asarray(thresholds, dtype=np.float64)
        self.scores_ = np.asarray(scores, dtype=np.float64)
        return self

    def _find_optimal_thresholds(self, column, y):
        """
        Вычисляет пороги для бинаризации

        Аргументы:
        ----------
        column: array-like, shape=(nobj,), колонка значений признаков
        y: array-like, shape=(nobj, nclasses), 0/1-матрица классов
        """
        classes_number = y.shape[1]
        # вычисляем частоты встречаемости признаков для разных классов
        values, counts = \
            _collect_column_statistics(column, y, classes_number=classes_number, precision=6)
        if self.method in Binarizer._CONTINGENCY_METHODS:
            if classes_number == 2:
                counts = [counts]
            else:
                summary_counts = np.sum(counts, axis=1)
                counts = [np.array((summary_counts - counts[:, i], counts[:, i])).T
                          for i in np.arange(classes_number)]
            best_thresholds = [None] * len(counts)
            best_scores = [None] * len(counts)
            for i in np.arange(len(counts)):
                current_thresholds, current_tables = \
                    _collect_contingency_tables(values, counts[i])
                if self.method == "log_odds":
                    func = (lambda x: odds_ratio(x, alpha=0.1))
                elif self.method == 'information_gain':
                    func = information_gain
                elif self.method == 'bns':
                    func = bns
                else:
                    raise ValueError("Wrong binarization method: {0}".format(self.method))
                scores = [func(table) for table in current_tables]
                best_score_index = np.argmax(scores)
                best_thresholds[i] = current_thresholds[best_score_index]
                best_scores[i] = scores[best_score_index]
        return best_thresholds, best_scores


class BinDivider(TransformerMixin):
    """
    Приводит данные к целочисленному виду, разделяя диапазон значений каждого
    признака на k одинаковых участков и кодируя i-ый диапазон значением i

    Аргументы:
    -----------
    bins_number: int, число участков, на которые делится диапазон
    allow_negative: bool, optional(default=False), разрешены ли негативные значения признаков
    fit_minima: bool, optional(default=False), вычисляется ли минимальное значение диапазона
        если fit_minima=False, то в качестве минимального значения берётся 0
    keep_integers: bool, optional(default=True), сохраняются ли целочисленные значения,
        меньшие self.bins_number
    """

    def __init__(self, bins_number, allow_negative=False, fit_minima=False, keep_integers=True):
        self.bins_number = bins_number
        self.allow_negative = allow_negative
        self.fit_minima = fit_minima
        self.keep_integers = keep_integers

    def fit(self, X, y=None):
        """
        Обучаем преобразователь, вычисляя значения минимума и максимума по данным
        """
        if self.allow_negative:
            if not self.minima:
                raise ValueError("fit_minima must be 'True' with allow_negative=True")
            if self.keep_integers:
                raise ValueError("keep_integers must be 'False' with allow_negative=True")
        X = check_array(X, accept_sparse=['csc', 'csr'])
        if issparse(X):
            if self.allow_negative:
                raise ValueError("allow_negative must be 'False' with sparse matrices")
            if self.fit_minima:
                raise ValueError("fit_minima must be 'False' with sparse matrices")
            X = X.tocsc()
        nfeat = X.shape[1]
        if isinstance(self.bins_number, int):
            self.bins_number = np.array([self.bins_number] * nfeat)
        if self.fit_minima:
            self.minima_ = X.min(axis=0)
        else:
            self.minima_ = np.zeros(shape=(nfeat,))
        if not self.allow_negative and np.min(self.minima_) < 0:
            raise ValueError("allow_negative must be False for matrices with negative values")
        if issparse(X):
            self.maxima_ = X.max(axis=0).toarray().flatten()
        else:
            self.maxima_ = X.max(axis=0)
        if self.keep_integers:
            # не изменяем целочисленные значения от 0 до self.bins_number
            for i in np.arange(nfeat):
                if self.maxima_[i] > self.bins_number[i]:
                    continue
                column = X[:, i].toarray().flatten()
                if (column == column.astype(int)).all():
                    self.maxima_[i] = self.bins_number[i]
        self.diff_ = self.maxima_ - self.minima_
        self.diff_[np.where(self.diff_ == 0)] = 1
        return self

    def transform(self, X):
        """
        Применяем преобразование к данным
        """
        X = check_array(X, accept_sparse=['csc', 'csr'])
        if not self.allow_negative and X.min() < 0:
            raise ValueError("allow_negative must be False "
                             "for matrices with negative values")
        if issparse(X):
            X = X.tocsc()
            for i in range(X.shape[1]):
                row_data = X.data[X.indptr[i]:X.indptr[i + 1]]
                X.data[X.indptr[i]:X.indptr[i + 1]] = np.minimum(row_data, self.maxima_[i])
        else:
            X = np.minimum(X, self.maxima_)
            if self.fit_minima:
                X = np.maximum(X, self.minima_)
                X -= self.minima_
        if issparse(X):
            for i in range(X.shape[1]):
                row_data = (X.data[X.indptr[i]:X.indptr[i + 1]])
                X.data[X.indptr[i]:X.indptr[i + 1]] = (row_data * self.bins_number[i]) / self.maxima_[i]
            X.data = X.data.astype(int)
        else:
            X = (X * self.bins_number / self.diff_).astype(int)
        return X


def _collect_column_statistics(column, y, classes_number=None, precision=6):
    """
    Вычисляет частоты значений в column в зависимости от значений y
    """
    y = np.asarray(y, dtype=int)
    if len(y.shape) == 0:
        # преобразуем y к бинаризованному виду
        classes_, y = np.unique(y, return_inverse=True)
        nclasses = classes_.shape[0]
        nobj = y.shape[0]
        y_new = np.zeros(shape=(nobj, nclasses), dtype=int)
        y_new[np.arange(nobj), y] = 1
    else:
        nobj, nclasses = y.shape
    if issparse(column):
        column = column.toarray().flatten()
    else:
        column = np.asarray(column)
    # поскольку ключи словаря не могут быть float,
    # преобразуем действительные числа в строки
    if np.issubdtype(column.dtype, float):
        value_coder = lambda x: '{0:.{precision}f}'.format(x, precision=precision)
        reverse_value_coder = float
    else:
        value_coder = lambda x: x
        reverse_value_coder = lambda x: x
    value_counts = dict()
    for val, label in zip(column, y):
        val = value_coder(val)
        if val not in value_counts:
            value_counts[val] = np.zeros(shape=nclasses)
        value_counts[val] += label
    value_counts = sorted(value_counts.items(),
                          key=(lambda x: reverse_value_coder(x[0])))
    values = np.fromiter((reverse_value_coder(x[0]) for x in value_counts),
                         dtype=column.dtype)
    counts = np.array([x[1] for x in value_counts])
    return values, counts


def _collect_contingency_tables(values, counts):
    """
    Вычмсляет таблицы совместимости для каждого из возможных значений порога

    Аргументы:
    ----------
    values: list-like, length=m, значения признака, упорядоченные по возрастанию
    counts: list of ints, частоты значений признаков в том же порядке
    """
    if len(values) != len(counts):
        print(len(values), len(counts))
        raise ValueError()
    if len(values) == 1:
        return [values[0]], [np.array([np.array(counts[0]) / 2, np.array(counts[0]) / 2])]
    thresholds = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    cumulative_sums = np.cumsum(counts, axis=0)
    tables = [np.array([elem, cumulative_sums[-1] - elem])
              for elem in cumulative_sums[:-1]]
    return thresholds, tables


def _sort_feature_indexes_by_scores(scores):
    """
    Сортирует индексы по убыванию весов,
    в случае равных весов сортировка ведётся по возрастанию индексов
    """
    negated_scores = -scores
    indexes = np.arange(scores.shape[0])
    return np.lexsort((indexes, negated_scores))


def test_binarizer():
    data = [[1, 2, 0], [1, 2, 2], [2, 0, 0], [1, 1, 1], [3, 3, 2], [2, 2, 1]]
    labels = [0, 1, 0, 0, 1, 1]
    binarizer = Binarizer(method='log_odds')
    binarizer.fit(data, labels)
    print(binarizer.joint_thresholds_)
    print(binarizer.joint_scores_)
    data = binarizer.transform(data)
    print(data)
    return


def test_bin_divider():
    data = csr_matrix([[0, 1, 3, 1.1], [0.25, 0, 4, 0],
                       [2.0, 2, 3, 2], [0.7, 1, 2, 1]])
    bin_divider = BinDivider(3)
    data = bin_divider.fit_transform(data)
    print(data.toarray())


if __name__ == "__main__":
    test_bin_divider()
    # a = [3, 2, 3, 1, 2]
    # a = np.array(a)
    # print(_sort_feature_indexes_by_scores(a))
