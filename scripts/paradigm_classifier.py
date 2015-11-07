#-------------------------------------------------------------------------------
# Name:        paradigm_classifier.py
#-------------------------------------------------------------------------------

import sys
from itertools import chain
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm as sksvm
from sklearn import linear_model as sklm
from scipy.sparse import csr_matrix

from paradigm_detector import ParadigmFragment, make_flection_paradigm, get_flection_length
from feature_selector import MulticlassFeatureSelector


def make_flection_paradigms_table(paradigms_table):
    """
    Получает на вход таблицу стандартных парадигм
    и возвращает таблицу парадигм только для флексии
    """
    codes_mapping = dict()
    new_paradigm_codes = dict()
    flection_lengths = []
    codes_number = 0
    for paradigm, code in paradigms_table.items():
        flection_paradigm = make_flection_paradigm(paradigm)
        flection_length = get_flection_length(flection_paradigm)
        new_code = new_paradigm_codes.get(flection_paradigm, None)
        if new_code is None:
            new_paradigm_codes[flection_paradigm] = new_code = codes_number
            flection_lengths.append(flection_length)
            codes_number += 1
        codes_mapping[code] = new_code
    return codes_mapping, new_paradigm_codes, flection_lengths


# ДОБАВИТЬ ПОДДЕРЖКУ АКТИВНЫХ ПАРАДИГМ

class ParadigmClassifier(BaseEstimator, ClassifierMixin):
    """
    Класс для определения парадигмы по лемме с помощью заданного классификатора

    Аргументы:
    -----------
    paradigm_table: dict, словарь вида {<парадигма>: <код>}
    multiclass: bool, optional(default=False),
        возможен ли возврат нескольких меток для одной леммы
    find_flection: bool, optional(default=False),
        отделяяется ли окончание слова перед классификацией,
        в текущей версии ухудшает качество классификации
    max_length: int, optional(default=False),
        максимальная длина признаков в классификации
    classifier: classifier-like, optional(default=sklearn.linear_model.logistic_regression),
        классификатор, непосредственно выполняющий классификацию,
        обязан поддерживать метод set_params
    selection_method: str, optional(default='ambiguity'), метод отбора признаков
    nfeatures: int, float or None(default=None),
        количество оставляемых признаков, если nfeatures \in [0, 1),
        то интерпретируется как доля оставляемых признаков,
        nfeatures=None означает отсутствие отбора признаков
    smallest_prob: float, default=0.01,
        минимальная разрешённая ненулевая вероятность класса
    min_probs_ratio: float, default=0.9,
        минимально возможное отношение вероятности класса к максимальной,
        при которой данный класс возвращается при multiclass=True
    """
    def __init__(self, paradigm_table, multiclass=False, find_flection=False,
                 max_length=5, use_prefixes=False,
                 classifier=sklm.LogisticRegression, classifier_params = None,
                 selection_method='ambiguity', nfeatures=None,
                 smallest_prob = 0.01, min_probs_ratio=0.9):
        self.paradigm_table = paradigm_table
        self.multiclass = multiclass
        self.find_flection=find_flection  # индикатор отделения окончания перед классификацией
        self.max_length = max_length
        self.use_prefixes = use_prefixes
        self.classifier = classifier
        self.classifier_params = classifier_params
        self.selection_method = selection_method
        self.nfeatures = nfeatures
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio

    def fit(self, X, y):
        """
        Обучает классификатор на данных
        """
        if self.classifier_params is None:
            self.classifier_params = dict()
        self.classifier.set_params(self.classifier_params)
        self._prepare_fragmentors()
        # если отделяем флексии
        # if self.find_flection:
        #     (self._flection_codes_mapping,
        #         self._flection_paradigm_table,
        #         self._flection_lengths) = make_flection_paradigms_table(self.paradigm_table)
        #     self._flection_paradigm_classifier = ParadigmClassifier(self._flection_paradigm_table,
        #                                                             find_flection=False,
        #                                                             max_length=5,
        #                                                             feature_fraction=0.1)
        X_train, y = self._preprocess_input(X, y, create_features=True,
                                            retain_multiple=self.multiclass, return_y=True)
        N, _ = X_train.shape

        if self.nfeatures is None:
            self.nfeatures = self.features_number
        if 0.0 < self.nfeatures and self.nfeatures < 1.0:
            # преобразуем долю признаков в их число
            self.nfeatures = int(self.features_number * self.feature_fraction)
        X_train = self._select_features(X_train, y)
        # print("Training...")
        self.classifier.fit(X_train, y)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X):
        """
        Применяет классификатор к данным

        Аргументы:
        -----------
        X, list of strs, список лемм, парадигму которых надо определить

        Возвращает:
        ------------
        answer, list of lists, для каждого объекта возвращается список классов
        """
        # print("Predicting...")
        probs = self.predict_proba(X)
        # реальная нумерация классов может не совпадать с нумерацией внутри классификатора
        if not self.multiclass:
            answer = [[x] for x in np.take(self.classes_, np.argmax(probs, axis=1))]
        else:
            answer = [np.take(self.classes_, elem)
                      for elem in extract_classes_from_probs(probs, self.min_probs_ratio)]
        return answer

    def fit_predict(self, X, y):
        # fit_predict is not implemented in sklearn.base
        # ПЕРЕПИСАТЬ!
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """
        Предсказывает вероятности классов. Даже если используется обычный predict,
        данная функция всё равно вызывается.

        Аргументы:
        -----------
        X, list of strs, список лемм, парадигму которых надо определить

        Возвращает:
        -----------
        probs, array-like, shape=(len(X), self.nclasses_)
            вероятности классов для объектов тестовой выборки
        """
        # преобразование входа в матрицу объекты-признаки
        X_train = self._preprocess_input(X, retain_multiple=False)
        X_train = self.selector.transform(X_train)
        # print("Predicting probabilities...")
        probs = self.classifier.predict_proba(X_train)
        # print("Calculating answer...")
        answer = np.zeros(dtype=np.float64, shape=probs.shape)
        # чтобы не делать индексацию при каждом поиске парадигмы,
        # сохраняем обработчики для каждого из возможных классов
        temp_fragmentors = [self._paradigm_fragmentors[i]  for i in self.classes_]
        temp_fragmentors_indexes = [self._fragmentors_indexes[i] for i in self.classes_]
        for i, (word, word_probs, final_word_probs) in enumerate(zip(X, probs, answer)):
            probs_sum = 0.0
            indexes_order = np.flipud(np.argsort(word_probs))
            fits_to_fragmentor = [None] * len(self._fragmentors_list)
            for class_index in indexes_order:
                prob = word_probs[class_index]
                # проверяем, что вероятность достаточно велика
                if prob < probs_sum * self.smallest_prob / (1.0 - self.smallest_prob):
                    break
                fragmentor_index = temp_fragmentors_indexes[class_index]
                # ищем в массиве fits_to_fragmentor индикатор того,
                # подходит ли слово под парадигму
                if fits_to_fragmentor[fragmentor_index] is None:
                    # заполняем массив, если значение не определено
                    fits_to_fragmentor[fragmentor_index] =\
                        temp_fragmentors[class_index].fits_to_pattern(word)
                if not fits_to_fragmentor[fragmentor_index]:
                    continue
                final_word_probs[class_index] = prob
                probs_sum += prob
            final_word_probs /= probs_sum
        return answer

    def predict_minus_log_proba(self, X):
        '''
        Преобразует значения вероятностей на логарифмическую шкалу
        по формуле q = (log(p/(1-p)) + L) / 2L, где L = log(p_0 / (1- p_0))

        Аргументы:
        -----------
        X, list of strs, список лемм, парадигму которых надо определить

        Возвращает:
        -----------
        minus_log_probs, array-like, shape=(len(X), self.nclasses_)
            логарифмические ``вероятности'' классов для объектов тестовой выборки
        """
        '''
        probs = self.predict_proba(X)
        probs = np.maximum(probs, self.smallest_prob)
        probs = np.minimum(probs, 1.0 - self.smallest_prob)
        minus_log_probs = np.log(probs) - np.log(1.0 - probs)
        max_value = np.log(1.0 - self.smallest_prob) - np.log(self.smallest_prob)
        # элементы minus_log_probs находились в диапазоне [-max_value; max_value]
        minus_log_probs = 0.5 * (minus_log_probs / max_value + 1)
        return minus_log_probs

    def _find_flections(self, X, y=None, create_features=False):
        """
        Выделяет окончания и основы у слов из X
        """
        if create_features:
            if y is None:
                raise ValueError("y must be given to create features"
                                 " in case of find_flection=True")
            y_flections = [[self._flection_codes_mapping[label[0]]] for label in y]
            self._flection_paradigm_classifier.fit_predict(X, y_flections)
            flection_features =['!{0}'.format(i)
                                for i in range(len(self._flection_codes_mapping))]
            self._flection_features_number = len(flection_features)
        else:
            y_flections = self._flection_paradigm_classifier.predict(X)
            flection_features = None
        # ParadigmClassifier.predict всегда возвращает списки меток
        y_flections = [elem[0] for elem in y_flections]
        X_stemmed = [word[ :(len(word) - self._flection_lengths[label])]
                             for word, label in zip(X, y_flections)]
        return X_stemmed, y_flections, flection_features

    def _preprocess_input(self, X, y=None, create_features=False,
                          retain_multiple=False, return_y=False):
        """
        Преобразует список слов в матрицу X_train
        X_train[i][j] = 1 <=> self.features[j] --- подслово #X[i]#

        Аргументы:
        ----------
        X: list of strs, список лемм, парадигму которых надо определить
        y: list of lists or None, список, i-ый элемент которого содержит классы
            для X[i] из обучающей выборки
        create_features: bool, optional(default=False), создаются ли новые признаки
            или им сопоставляется значение ``неизвестный суффикс/префикс''
        retain_multiple: bool, optional(default=False), может ли возвращаться
            несколько меток классов для одного объекта
        return_y: возвращается ли y (True в случае обработки обучающей выборки,
            т.к. в этом случае y преобразуется из списка списков в один список)
        """
        if self.find_flection:
            # print("FLECTION CLASSIFIER: ")
            X, flection_classes, flection_features = self._find_flections(X, y, create_features)
            # print("MAIN CLASSIFIER: ")
        else:
            flection_classes = [None] * len(X)
            self._flection_features_number = 0
        if create_features:
            # создаём новые признаки, вызывается при обработке обучающей выборки
            self.features_number = 0
            features, self.feature_codes = [], dict()
            #  признаки для суффиксов, не встречавшихся в обучающей выборки
            features.extend(("-" * length + "#")
                            for length in range(1, self.max_length+1))
            if self.use_prefixes:
                features.extend((("#" + "-" * length) for length in range(1, self.max_length+1)))
            for code, feature in enumerate(features):
                self.feature_codes[feature] = code
            features_number = len(features)
            # создаём функцию для обработки неизвестных признаков
            def _process_unknown_feature(feature, is_prefix=False):
                nonlocal features_number
                features.append(feature)
                self.feature_codes[feature] = code = features_number
                features_number += 1
                return code
        else:
            # новые признаки не создаются, вызывается при обработке контрольной выборки
            features_number = self.features_number
            # создаём функцию для обработки неизвестных признаков
            def _process_unknown_feature(feature, is_prefix=False):
                if is_prefix:
                    feature = '#' + '-' * (len(feature)-1)
                else:
                    feature = '-' * (len(feature)-1) + '#'
                return self.feature_codes[feature]
        if y is None:
            y = [[None]] * len(X)
            retain_multiple = False
        y_new = list(chain(*y))
        X_with_temporary_codes = []
        for word in X:
            active_features_codes = []
            for length in range(min(self.max_length, len(word))):
                feature = word[-length-1:] + "#"
                feature_code = self.feature_codes.get(feature, None)
                if feature_code is None:
                    feature_code = _process_unknown_feature(feature)
                active_features_codes.append(feature_code)
            if self.use_prefixes:
                for length in range(min(self.max_length, len(word))):
                    feature = "#" + word[:(length+1)]
                    feature_code = self.feature_codes.get(feature, None)
                    if feature_code is None:
                        feature_code = _process_unknown_feature(feature, is_prefix=True)
                active_features_codes.append(feature_code)
            X_with_temporary_codes.append(active_features_codes)
        if create_features:
            # при сортировке сначала учитывается длина, потом суффикс/префикс
            feature_order = sorted(enumerate(features),
                                   key=(lambda x:(len(x[1]), x[1][0] == '#')))
            self.features = [x[1] for x in feature_order]
            if self.find_flection:
                self.features = flection_features + self.features
            # перекодируем временные признаки
            temporary_features_recoding = [None] * features_number
            for new_code, (code, _) in enumerate(feature_order):
                temporary_features_recoding[code] = new_code + self._flection_features_number
            for i, elem in enumerate(X_with_temporary_codes):
                X_with_temporary_codes[i] = [temporary_features_recoding[code] for code in elem]
            self.feature_codes = {feature: code for code, feature in enumerate(self.features)}
            self.features_number = len(self.features)
        # сохраняем данные для классификации
        rows, cols, curr_row = [], [], 0
        for codes, labels, flection_label in zip(X_with_temporary_codes, y, flection_classes):
            if self.find_flection:
                codes.append(flection_label)
            # каждый объект обучающей выборки размножается k раз,
            # где k --- число классов, к которым он принадлежит, например, пара
            # X[i]=x, y[i]=[1, 2, 6] будет преобразована в <x, 1>, <x, 2>, <x, 6>
            number_of_rows_to_add = 1 if not(retain_multiple) else len(labels)
            for i in range(number_of_rows_to_add):
                rows.extend([curr_row for _ in codes])
                cols.extend(codes)
                curr_row += 1
        # сохраняем преобразованные данные в разреженную матрицу
        data = np.ones(shape=(len(rows,)), dtype=float)
        X_train = csr_matrix((data, (rows, cols)),
                             shape=(curr_row, self.features_number))
        if return_y:
            return X_train, y_new
        else:
            return X_train

    def _add_feature(self, feature):
        """
        Добавление нового признака
        """
        self.features.append(feature)
        self.feature_codes[feature] = self.features_number
        self.features_number += 1

    def _select_features(self, X, y):
        """
        Осуществляет отбор признаков
        """
        self.selector = MulticlassFeatureSelector(
                local=True, method=self.selection_method, min_count=3, nfeatures=self.nfeatures)
        X = self.selector.fit_transform(X, y)
        self._active_feature_codes = self.selector.get_support(indices=True)
        return X

    def _prepare_fragmentors(self):
        """
        Предвычисление обработчиков шаблонов, соответствующих леммам.
        Поскольку таких шаблонов существенно меньше, чем парадигм,
        то вычислять их по ходу нецелесообразно.
        """
        self._paradigm_fragmentors = dict()
        self._fragmentors_indexes = dict()
        self._fragmentors_list = []
        self._patterns = dict()
        fragmentors_indexes_by_patterns = dict()
        fragmentors_number = 0
        for paradigm, code in self.paradigm_table.items():
            # вычисляем шаблон для леммы
            pattern = paradigm.split("#")[0]
            if pattern == "-":
                pattern = paradigm.split("#")[1]
            fragmentor_index = fragmentors_indexes_by_patterns.get(pattern, -1)
            if fragmentor_index < 0:
                # если такой шаблон леммы ещё не возникал, то сохраняем новый обработчик
                self._fragmentors_list.append(ParadigmFragment(pattern))
                fragmentors_indexes_by_patterns[pattern] = fragmentors_number
                fragmentor_index = fragmentors_number
                fragmentors_number += 1
            self._paradigm_fragmentors[code] = self._fragmentors_list[fragmentor_index]
            self._fragmentors_indexes[code] = fragmentor_index
            self._patterns[code] = paradigm
        return


def extract_classes_from_probs(probs, min_probs_ratio):
    """
    Возвращает список классов по их вероятностям
    в случае мультиклассовой классификации
    """
    # возможно, имеет смысл возвращать разреженную матрицу
    max_allowed_probs = np.max(probs, axis=1).reshape((probs.shape[0], 1)) * min_probs_ratio
    remaining_positions = np.where(probs >= max_allowed_probs)
    answer = [[] for _ in range(probs.shape[0])]
    for row, col in zip(*remaining_positions):
        answer[row].append(col)
    return answer

















