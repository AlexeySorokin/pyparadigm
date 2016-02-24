#-------------------------------------------------------------------------------
# Name:        paradigm_classifier.py
#-------------------------------------------------------------------------------

import sys
from itertools import chain
from collections import OrderedDict, Counter, defaultdict
import bisect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn import svm as sksvm
from sklearn import linear_model as sklm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix, csc_matrix, issparse

from paradigm_detector import ParadigmFragment, make_flection_paradigm, get_flection_length
from feature_selector import MulticlassFeatureSelector, SelectingFeatureWeighter, ZeroFeatureRemover
from transformation_classifier import LocalTransformationClassifier, OptimalPositionSearcher,\
    extract_uncovered_substrings
from graph_utilities import Trie, prune_dead_branches
from tools import counts_to_probs

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
    nfeatures: int, float or None, optional (default=None),
        количество оставляемых признаков, если nfeatures \in [0, 1),
        то интерпретируется как доля оставляемых признаков,
        nfeatures=None означает отсутствие отбора признаков
    minfeatures: int, optional (default=100),
        минимальное количество признаков, которое будет оставлено
    smallest_prob: float, optional (default=0.01)
        минимальная разрешённая ненулевая вероятность класса
    min_probs_ratio: float, default=0.9,
        минимально возможное отношение вероятности класса к максимальной,
        при которой данный класс возвращается при multiclass=True
    """
    def __init__(self, paradigm_table=None, multiclass=False, find_flection=False,
                 max_length=5, use_prefixes=False, has_letter_classifiers=True,
                 classifier=sklm.LogisticRegression(), classifier_params = None,
                 selection_method='ambiguity', nfeatures=None, minfeatures=100,
                 weight_features=False, smallest_prob = 0.01, min_probs_ratio=0.9,
                 unique_class_prob = 0.9, to_memorize_suffixes = 3):
        self.paradigm_table = paradigm_table
        self.multiclass = multiclass
        self.find_flection=find_flection  # индикатор отделения окончания перед классификацией
        self.max_length = max_length
        self.use_prefixes = use_prefixes
        self.classifier = classifier
        self.classifier_params = classifier_params
        self.selection_method = selection_method
        self.nfeatures = nfeatures
        self.minfeatures = minfeatures
        self.weight_features = weight_features
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio
        self.has_letter_classifiers = has_letter_classifiers
        self.unique_class_prob = unique_class_prob
        self.to_memorize_suffixes = to_memorize_suffixes

    def _prepare_classifier(self):
        if self.classifier_params is None:
            self.classifier_params = dict()
        self.classifier.set_params(**self.classifier_params)
        self.first_selector = MulticlassFeatureSelector(local=True, method=self.selection_method,
                                                        min_count=3, nfeatures=-1)
        selector = MulticlassFeatureSelector(local=True, method=self.selection_method,
                                             min_count=3, nfeatures=self.nfeatures,
                                             minfeatures=self.minfeatures)
        if not self.weight_features:
            first_stage = ('feature_selection', selector)
        else:
            weighter_params = {'transformation': 'log', 'min_prob': 0.01}
            weighter = SelectingFeatureWeighter(selector, **weighter_params)
            first_stage = ('feature_weighting', weighter)
        single_classifier = Pipeline([first_stage, ('classifier', self.classifier)])
        self.classifier = OneVsRestClassifier(single_classifier)
        return self

    def fit(self, X, y):
        """
        Обучает классификатор на данных
        """
        if self.paradigm_table is None:
            self.paradigm_table = dict()
        self._paradigms_by_codes = {code: descr for descr, code in self.paradigm_table.items()}
        self._prepare_classifier()
        self._prepare_fragmentors()
        X_train_, Y = self._preprocess_input(X, y, create_features=True,
                                            retain_multiple=self.multiclass, return_y=True)
        # сразу определим множество классов
        self.classes_, Y_new = np.unique(Y, return_inverse=True)
        self.reverse_classes = {label: i for i, label in enumerate(self.classes_)}
        # перекодируем классы в новую нумерацию перед подсчётом статистики
        y_new = [[self.reverse_classes[label] for label in labels] for labels in y]
        self._make_label_statistics(X, y_new)
        X_train = self._remove_rare_features(X_train_, Y_new, sparse_type='csr')
        if self.has_letter_classifiers:
            # сначала найдём возможные классы,
            # поскольку их множество используется в предобработке
            data_indexes_by_letters =\
                arrange_indexes_by_last_letters(X, [len(labels) for labels in y])
            # заводим классификаторы для каждой буквы
            self.letter_classifiers_ = dict()
            for letter, indexes in data_indexes_by_letters.items():
                X_curr, y_curr = X_train[indexes,:], Y_new[indexes]
                # print(letter, set(y_curr))
                if letter not in self._single_class_letters:
                    self.letter_classifiers_[letter] = clone(self.classifier)
                    self.letter_classifiers_[letter].fit(X_curr, y_curr)
        else:
            self.classifier.fit(X_train, Y_new)
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
        probs = self._predict_proba(X)
        # реальная нумерация классов может не совпадать с нумерацией внутри классификатора
        if not self.multiclass:
            answer = [[self.classes_[indices[0]]] if len(indices) > 0 else [None]
                      for indices, _ in probs]
        else:
            answer = [np.take(self.classes_, indices) for indices, _ in probs
                      in extract_classes_from_sparse_probs(probs, self.min_probs_ratio)]
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
        probs = self._predict_proba(X)
        answer = np.zeros(dtype=np.float64, shape=(len(X), len(self.classes_)))
        for i, (indices, word_probs) in enumerate(probs):
            if len(indices) > 0:
                answer[i][indices] = word_probs
        return answer

    def _predict_proba(self, X):
        """
        Основная функция для предсказания вероятностей.
        Открытая функция  predict_proba --- интерфейс для данной.
        Аргументы:
        -----------
        X, list of strs, список лемм, парадигму которых надо определить

        Возвращает:
        -----------
        probs, list of tuples of the form (indices, word_probs),
            где word_probs --- список вероятностей классов для каждого слова
            в порядке их убывания, а indices --- список соответствующих классов
        """
        answer = [([], []) for _ in X]
        # предвычисляем ответы в зависимости от суффикса
        indexes_with_known_answers, other_indexes, known_answers =\
            self._precompute_known_answers(X)
        temp_fragmentors = [self._paradigm_fragmentors[i]  for i in self.classes_]
        temp_fragmentors_indexes = [self._fragmentors_indexes[i] for i in self.classes_]
        # извлекаем вероятности из ответов
        for i, labels in zip(indexes_with_known_answers, known_answers):
            word_probs = self._make_probs_from_known_labels(X[i], labels)
            answer[i] = self._extract_word_probs(X[i], word_probs, temp_fragmentors,
                                                 temp_fragmentors_indexes)
        # оставляем только объекты с неизвестными ответами
        X_unanswered = [X[i] for i in other_indexes]
        # после этой строчки X_train будет csr_matrix
        X_train = self._preprocess_input(X_unanswered, retain_multiple=False, sparse_type='csr')
        # чтобы не делать индексацию при каждом поиске парадигмы,
        # сохраняем обработчики для каждого из возможных классов
        if not self.has_letter_classifiers:
            probs = self.classifier.predict_proba(X_train)
            # перекодируем классы в self.classifier.classes_ в элементы self.classes_
            temp_fragmentors =\
                [self._paradigm_fragmentors[self.classes_[i]] for i in self.classifier.classes_]
            temp_fragmentors_indexes =\
                [self._fragmentors_indexes[self.classes_[i]] for i in self.classifier.classes_]
            for i, word, word_probs in zip(other_indexes, X_unanswered, probs):
                answer[i] = self._extract_word_probs(word, word_probs, temp_fragmentors,
                                                     temp_fragmentors_indexes)
        else:
            data_indexes_by_letters = arrange_indexes_by_last_letters(X_unanswered)
            for letter, indexes in sorted(data_indexes_by_letters.items()):
                if letter in self._single_class_letters:
                    sys.exit("Answer for this class has already been calculated. Check your code.")
                if letter not in self.letter_classifiers_:
                    probs = np.tile(self.new_letter_probs, (len(indexes), 1))
                    temp_fragmentors = [self._paradigm_fragmentors[i]  for i in self.classes_]
                    temp_fragmentors_indexes = [self._fragmentors_indexes[i] for i in self.classes_]
                    curr_classes = range(len(self.classes_))
                else:
                    curr_X_train = X_train[indexes,:]
                    cls = self.letter_classifiers_.get(letter)
                    probs = cls.predict_proba(curr_X_train)
                    temp_fragmentors = [self._paradigm_fragmentors[self.classes_[i]]
                                        for i in cls.classes_]
                    temp_fragmentors_indexes = [self._fragmentors_indexes[self.classes_[i]]
                                                for i in cls.classes_]
                    curr_classes = cls.classes_
                for i, word_probs in zip(indexes, probs):
                    cls_class_indices_list, probs_list = self._extract_word_probs(
                        X[other_indexes[i]], word_probs, temp_fragmentors, temp_fragmentors_indexes)
                    answer[other_indexes[i]] =\
                        ([curr_classes[j] for j in cls_class_indices_list], probs_list)
        # for word, (first, _) in zip(words, answer):
        #     print(word, len(first))
        return answer

    def _precompute_known_answers(self, X):
        """
        Возвращает:
        ------------
        known_indexes: list of ints, индексы элементов в X, ответ на которых известен,
        other_indexes: list of ints, индексы остальных элементов
        known_answers: list of pair, список пар вида (labels, flag), где
            labels: list of ints, список предсказанных классов,
            flag: int, тип набора вероятностей, который будет использоваться
            для пополнения в функции _make_probs_from_known_labels
        """
        known_indexes, other_indexes, answers_for_known_indexes = [], [], []
        if self.to_memorize_suffixes:
            suffix_start = -2 if self.has_letter_classifiers else -1
            suffix_end = -1 - self.to_memorize_suffixes
            # print(self.memorized_suffixes)
        for i, word in enumerate(X):
            letter = word[-1]
            if self.to_memorize_suffixes > 0:
                suffix = word[suffix_start:suffix_end:-1]
                if self.has_letter_classifiers:
                    trie, node_data = self.memorized_suffixes.get(letter, None), None
                else:
                    trie, node_data = self.memorized_suffixes, None
                if trie is not None:
                    node_data = trie.partial_path(suffix)[-1].data
                if node_data is not None and len(node_data) > 0:
                    known_indexes.append(i)
                    answers_for_known_indexes.append((node_data, int(self.has_letter_classifiers)))
                    continue
            if self.has_letter_classifiers:
                labels = self._single_class_letters.get(letter)
                if labels is not None:
                    known_indexes.append(i)
                    answers_for_known_indexes.append((labels, 0))
                    continue
            other_indexes.append(i)
        return known_indexes, other_indexes, answers_for_known_indexes

    def _make_probs_from_known_labels(self, word, labels_with_flag):
        """
        Преобразует предвычисленные классы для слова word в набор вероятностей

        Аргументы:
        ----------
        word: слово, чью вероятность мы предсказываем,
        labels: пара (labels, flag), где
            labels: list of ints, набор предсказанных классов
            flag: int, 0, 1, тип вероятности по умолчанию для слова word,
                0 --- берётся общая вероятность по умолчанию,
                1 --- вероятность по умолчанию для последней буквы
        """
        labels, flag = labels_with_flag
        if flag == 0:
            default_probs = self.new_letter_probs
        elif flag == 1:
            default_probs = self._default_letter_probs[word[-1]]
        default_probs *= 1.0 - self.unique_class_prob
        default_probs[labels] += self.unique_class_prob
        return default_probs

    def _extract_word_probs(self, word, word_probs,
                            temp_fragmentors, temp_fragmentors_indexes):
        '''
        Вспомогательная функция для нахождения вероятности одного слова
        '''
        probs_sum = 0.0
        indexes_order = np.flipud(np.argsort(word_probs))
        fits_to_fragmentor = [None] * len(self._fragmentors_list)
        current_indices, current_probs = [], []
        for class_index in indexes_order:
            prob = word_probs[class_index]
            # проверяем, что вероятность достаточно велика
            if prob < probs_sum * self.smallest_prob / (1.0 - self.smallest_prob):
                break
            fragmentor_index = temp_fragmentors_indexes[class_index]
            if fits_to_fragmentor[fragmentor_index] is None:
                # заполняем массив, если значение не определено
                fits_to_fragmentor[fragmentor_index] =\
                    temp_fragmentors[class_index].fits_to_pattern(word)
            if not fits_to_fragmentor[fragmentor_index]:
                continue
            current_indices.append(class_index)
            current_probs.append(prob)
            probs_sum += prob
        current_probs = np.array(current_probs) / probs_sum
        return (current_indices, current_probs)

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
        probs = self._predict_proba(X)
        answer = [None] * len(probs)
        max_value = -np.log(self.smallest_prob) + np.log(1.0 - self.smallest_prob)
        for i, (indices, word_probs) in enumerate(probs):
            # print(X[i])
            # for index, prob in zip(indices, word_probs):
            #     print(self.classes_)
            #     print("{0} {1} {2:.4f}".format(index, self._paradigms_by_codes[self.classes_[index]], prob))
            if len(indices) > 0:
                word_probs = np.minimum(word_probs, 1.0 - self.smallest_prob)
                word_probs = np.maximum(word_probs, self.smallest_prob)
                minus_log_probs = np.log(word_probs) - np.log(1.0 - word_probs)
                # элементы minus_log_probs находились в диапазоне [-max_value; max_value]
                minus_log_probs = 0.5 * (minus_log_probs / max_value + 1.0)
                # print(word_probs)
                # print(minus_log_probs)
                answer[i] = (indices, minus_log_probs)
            else:
                answer[i] = ([], [])
        return answer

    # def _find_flections(self, X, y=None, create_features=False):
    #     """
    #     Выделяет окончания и основы у слов из X
    #     """
    #     if create_features:
    #         if y is None:
    #             raise ValueError("y must be given to create features"
    #                              " in case of find_flection=True")
    #         y_flections = [[self._flection_codes_mapping[label[0]]] for label in y]
    #         self._flection_paradigm_classifier.fit_predict(X, y_flections)
    #         flection_features =['!{0}'.format(i)
    #                             for i in range(len(self._flection_codes_mapping))]
    #         self._flection_features_number = len(flection_features)
    #     else:
    #         y_flections = self._flection_paradigm_classifier.predict(X)
    #         flection_features = None
    #     # ParadigmClassifier.predict всегда возвращает списки меток
    #     y_flections = [elem[0] for elem in y_flections]
    #     X_stemmed = [word[ :(len(word) - self._flection_lengths[label])]
    #                          for word, label in zip(X, y_flections)]
    #     return X_stemmed, y_flections, flection_features

    def _preprocess_input(self, X, y=None, create_features=False,
                          retain_multiple=False, return_y=False, sparse_type='csc'):
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
        if create_features:
            # создаём новые признаки, вызывается при обработке обучающей выборки
            self.features, self.feature_codes = [], dict()
            #  признаки для суффиксов, не встречавшихся в обучающей выборке
            self.features.extend(("-" * length + "#")
                                 for length in range(1, self.max_length+1))
            if self.use_prefixes:
                self.features.extend((("#" + "-" * length)
                                      for length in range(1, self.max_length+1)))
            for code, feature in enumerate(self.features):
                self.feature_codes[feature] = code
            self.features_number = len(self.features)
            # создаём функцию для обработки неизвестных признаков
        if y is None:
            y = [[None]] * len(X)
            retain_multiple = False
        y_new = list(chain(*y))
        X_with_temporary_codes = []
        for word in X:
            active_features_codes = []
            for length in range(min(self.max_length, len(word))):
                feature = word[-length-1:] + "#"
                feature_code = self._get_feature_code(feature, create_new=create_features)
                active_features_codes.append(feature_code)
            if self.use_prefixes:
                for length in range(min(self.max_length, len(word))):
                    feature = "#" + word[:(length+1)]
                    feature_code = self._get_feature_code(feature, create_new=create_features)
                active_features_codes.append(feature_code)
            X_with_temporary_codes.append(active_features_codes)
        if create_features:
            # при сортировке сначала учитывается длина, потом суффикс/префикс
            feature_order = sorted(enumerate(self.features),
                                   key=(lambda x:(len(x[1]), x[1][0] == '#')))
            self.features = [x[1] for x in feature_order]
            # перекодируем временные признаки
            temporary_features_recoding = [None] * self.features_number
            for new_code, (code, _) in enumerate(feature_order):
                temporary_features_recoding[code] = new_code  # + self._flection_features_number
            for i, elem in enumerate(X_with_temporary_codes):
                X_with_temporary_codes[i] = [temporary_features_recoding[code] for code in elem]
            self.feature_codes = {feature: code for code, feature in enumerate(self.features)}
            self.features_number = len(self.features)
        # сохраняем данные для классификации
        rows, cols, curr_row = [], [], 0
        # for codes, labels, flection_label in zip(X_with_temporary_codes, y, flection_classes):
        for codes, labels in zip(X_with_temporary_codes, y):
            # if self.find_flection:
            #     codes.append(flection_label)
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
        if sparse_type == 'csr':
            sparse_matrix = csr_matrix
        elif sparse_type == 'csc':
            sparse_matrix = csc_matrix
        X_train = sparse_matrix((data, (rows, cols)),
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

    def _get_feature_code(self, feature, create_new=False):
        """
        Возвращает код признака feature
        """
        code = self.feature_codes.get(feature, -1)
        if code < 0:
            if create_new:
                self.features.append(feature)
                code = self.features_number
                self.features_number += 1
            else:
                if feature.endswith('#'):
                    partial_features = ['-' * start + feature[start:]
                                        for start in range(1, len(feature))]
                else:
                    partial_features = [feature[:(len(feature)-start)] + '-' * start
                                        for start in range(1, len(feature))][::-1]
                for partial_feature in partial_features:
                    code = self.feature_codes.get(partial_feature, -1)
                    if code > 0:
                        break
            self.feature_codes[feature] = code
        return code

    def _remove_rare_features(self, X, y, sparse_type='csr'):
        """
        Удаляет признаки, встречающиеся меньше self.min_count раз,
        преобразуя соответствующим образом поля класса.
        """
        self.first_selector.fit(X, y)
        mask = self.first_selector.mask_
        indexes = self.first_selector.get_support(indices=True)
        new_features = []
        new_feature_codes = dict()
        new_features_number = 0
        # первый проход: оставляем те признаки, которые точно будут
        features_in_mask = set([self.features[i] for i in indexes])
        # второй проход: сохраняем признаки с учётом их суффиксов, префиксов
        for i, (flag, feature) in enumerate(zip(mask, self.features)):
            to_add = False
            if flag or '-' in feature:
                to_add = True
            else:
                # теперь признак точно не содержит пропусков
                if feature.endswith('#'):
                    partial_features = [feature[start:] for start in range(1, len(feature)-1)]
                else:
                    partial_features = [feature[:end] for end in range(len(feature)-1, 1, -1)]
                # ищем максимальный признак, являющийся совместимым с текущим
                for start, partial_feature in enumerate(partial_features, 1):
                    if partial_feature in features_in_mask:
                        # дополняем прочерками
                        if feature.endswith('#'):
                            feature = '-' * start + partial_feature
                        else:
                            feature = partial_feature + '-' * start
                        if feature not in new_features:
                            to_add = True
                            break
            if to_add:
                new_features.append(feature)
                new_feature_codes[feature] = new_features_number
                new_features_number += 1
            # print(i, self.features[i], flag, feature, to_add)
        # сохранение новых полей класса
        old_features = self.features
        self.features = new_features
        self.feature_codes = new_feature_codes
        self.features_number = new_features_number
        # третий проход: распределение признаков
        codes_for_old_features = [self._get_feature_code(feature)
                                  for feature in old_features]
        if issparse(X):
            X.sort_indices
            rows, cols = X.nonzero()
            cols = [codes_for_old_features[j] for j in cols]
            if sparse_type == 'csr':
                sparse_matrix = csr_matrix
            elif sparse_type == 'csc':
                sparse_matrix = csc_matrix
            X_new = sparse_matrix((X.data, (rows, cols)),
                                    shape=(X.shape[0], self.features_number))
        else:
            raise NotImplementedError
        return X_new

    # def _select_features(self, X, y):
    #     """
    #     Осуществляет отбор признаков
    #     """
    #     self.selector = MulticlassFeatureSelector(
    #             local=True, method=self.selection_method, min_count=3,
    #             nfeatures=self.nfeatures, minfeatures=self.minfeatures)
    #     X = self.selector.fit_transform(X, y)
    #     self._active_feature_codes = self.selector.get_support(indices=True)
    #     return X

    def _make_label_statistics(self, X, y):
        """
        Подсчёт статистики встречаемости классов
        """
        # сначала считаем частоты классов
        label_counts = defaultdict(int)
        if self.has_letter_classifiers:
            label_counts_by_letters = defaultdict(lambda: defaultdict(int))
            constant_labels_for_letters = dict()
        for word, labels in zip(X, y):
            # labels = [self.reverse_classes[label] for label in labels]
            for label in labels:
                label_counts[label] += 1
            if self.has_letter_classifiers:
                letter = word[-1]
                label_counts_for_letter = label_counts_by_letters[letter]
                for label in labels:
                    label_counts_for_letter[label] += 1
                constant_labels_for_letter = constant_labels_for_letters.get(letter)
                if constant_labels_for_letter is None:
                    constant_labels_for_letters[letter] = labels
                else:
                    constant_labels_for_letters[letter] =\
                        [label for label in labels if label in constant_labels_for_letter]
        # теперь вычисляем вероятности по умолчанию
        self.new_letter_probs = counts_to_probs(label_counts, len(self.classes_))
        if self.has_letter_classifiers:
            self._default_letter_probs = dict()
            self._single_class_letters = dict()
            for letter, counts in label_counts_by_letters.items():
                self._default_letter_probs[letter] =\
                    counts_to_probs(counts, len(self.classes_)) * (self.unique_class_prob) +\
                    self.new_letter_probs * (1.0 - self.unique_class_prob)
                # проверяем, не встречаемся ли некоторый класс для всех слов,
                # заканчивающихся данной буквой
                constant_labels_for_letter = constant_labels_for_letters[letter]
                if len(constant_labels_for_letter) > 0:
                    self._single_class_letters[letter] = constant_labels_for_letter
        if self.to_memorize_suffixes > 0:
            self.memorized_suffixes = _memorize_suffixes(
                self.to_memorize_suffixes,X, y, remove_last_letter=self.has_letter_classifiers)
        return

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
        # заводим множество подходящих фрагменторов в зависимости от завершающей буквы
        # self.pattern_suffixes_list = []
        for paradigm, code in self.paradigm_table.items():
            # вычисляем шаблон для леммы
            pattern = paradigm.split("#")[0]
            if pattern == "-":
                pattern = paradigm.split("#")[1]
            last_fragment = pattern.split('+')[-1]
            if last_fragment.isdigit():
                last_fragment = ""
            fragmentor_index = fragmentors_indexes_by_patterns.get(pattern, -1)
            if fragmentor_index < 0:
                # если такой шаблон леммы ещё не возникал, то сохраняем новый обработчик
                self._fragmentors_list.append(ParadigmFragment(pattern))
                fragmentors_indexes_by_patterns[pattern] = fragmentors_number
                fragmentor_index = fragmentors_number
                # self.pattern_suffixes_list.append(last_fragment)
                fragmentors_number += 1
            self._paradigm_fragmentors[code] = self._fragmentors_list[fragmentor_index]
            self._fragmentors_indexes[code] = fragmentor_index
            self._patterns[code] = paradigm
        return

    def fragmentor_by_code(self, code):
        """
        Возвращает обработчик парадигмы по её коду
        """
        return self._paradigm_fragmentors[code]

class JointParadigmClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор, определяющий абстрактную парадигму,
    а также вычисляющий значения переменных
    """
    def __init__(self, paradigm_classifier, transformation_handler,
                 paradigm_classifier_params, transformation_classifier_params,
                 smallest_prob=0.01, min_probs_ratio=0.75):
        self.paradigm_classifier = paradigm_classifier
        self.transformation_handler = transformation_handler
        self.paradigm_classifier_params = paradigm_classifier_params
        self.transformation_classifier_params = transformation_classifier_params
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio

    def fit(self, X, y):
        """
        Обучает классификатор по данным
        """
        self.paradigm_classifier_params['smallest_prob'] = self.smallest_prob
        self.paradigm_classifier_params['min_probs_ratio'] = self.min_probs_ratio
        self.paradigm_classifier.set_params(**self.paradigm_classifier_params)
        training_data = list(chain.from_iterable(
            [(lemma, code, values) for code, values in label]
            for lemma, label in zip(X, y)))
        X_flat, labels, var_values = list(list(elem) for elem in zip(*training_data))
        self.paradigm_classifier.fit(X_flat, [[x] for x in labels])
        self.classes_ = self.paradigm_classifier.classes_
        transformation_classifier =\
            LocalTransformationClassifier(**self.transformation_classifier_params)
        self.position_searcher = OptimalPositionSearcher(self.transformation_handler.transformations_by_paradigms,
                                                         self.transformation_handler.paradigms_by_codes,
                                                         transformation_classifier)
        data_for_trans_training, trans_labels =\
                self.transformation_handler.create_training_data(training_data)
        self.position_searcher.fit(data_for_trans_training, [[x] for x in trans_labels])
        return self

    def predict(self, X):
        labels = self.paradigm_classifier.predict(X)
        # var_values = [[self.position_searcher.predict_variables(code, word) for code in codes]
        #                for word, codes in zip(X, labels)]
        labels_for_variable_prediction = list(chain.from_iterable(x for x in labels if x[0] is not None))
        words_for_variable_prediction =\
            list(chain.from_iterable((word for _ in word_labels)
                                     for word, word_labels in zip(X, labels)
                                     if word_labels[0] is not None))
        # print(labels_for_variable_prediction)
        # print(words_for_variable_prediction)
        if len(labels_for_variable_prediction) > 0:
            var_values = self.position_searcher.predict_variables(labels_for_variable_prediction,
                                                                  words_for_variable_prediction)
        end = 0
        answer = []
        for word_labels, word in zip(labels, X):
            if word_labels[0] is not None:
                start, end = end, end + len(word_labels)
                answer.append(list(zip(word_labels, var_values[start:end])))
            else:
                answer.append([(None, [])])
        return answer

    def predict_probs(self, X):
        """
        Возвращает возможные наборы ((код, переменные), вероятность) в порядке убывания вероятности
        """
        codes_with_probs = self.paradigm_classifier._predict_proba(X)
        labels_for_variable_prediction =\
            list(chain.from_iterable(np.take(self.classes_, indices)
                                     for indices, probs in codes_with_probs))
        words_for_variable_prediction =\
            list(chain.from_iterable((word for _ in indices)
                                     for (indices, probs), word in zip(codes_with_probs, X)))
        var_values = self.position_searcher.predict_variables(labels_for_variable_prediction,
                                                              words_for_variable_prediction)
        end = 0
        answer = []
        for indices, probs in codes_with_probs:
            start = end
            end = start + len(indices)
            answer.append((list(zip([self.classes_[index] for index in indices],
                                    var_values[start:end])),
                           probs))
        return answer


class CombinedParadigmClassifier(BaseEstimator, ClassifierMixin):
    """
    Классификатор, определяющий абстрактную парадигму
    с помощью комбинации базового классификатора
    и классификаторов для локальных трансформаций,
    а также вычисляющий значения переменных
    """
    def __init__(self, paradigm_classifier, transformation_handler,
                 paradigm_classifier_params=None, transformation_classifier_params=None,
                 smallest_prob=0.01, min_probs_ratio=0.75):
        """
        Аргументы:
        ----------
        paradigm_classifier --- классификатор для парадигм,
        joint_classifier_params --- параметры классификатора,
            объединяющего общую и локальную модель
        """
        self.paradigm_classifier = paradigm_classifier
        self.transformation_handler = transformation_handler
        self.paradigm_classifier_params = paradigm_classifier_params
        self.transformation_classifier_params = transformation_classifier_params
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio

    def fit(self, X, y):
        """
        Обучает классификатор по данным
        """
        if self.paradigm_classifier_params is None:
            self.paradigm_classifier_params = {'smallest_prob': 0.01}
        if self.transformation_classifier_params is None:
            self.transformation_classifier_params = dict()
        self.paradigm_classifier.set_params(**self.paradigm_classifier_params)
        training_data = list(chain.from_iterable(
            [(lemma, code, values) for code, values in label]
            for lemma, label in zip(X, y)))
        X_flat, labels, var_values = list(list(elem) for elem in zip(*training_data))
        self.paradigm_classifier.fit(X_flat, [[x] for x in labels])
        self.classes_ = self.paradigm_classifier.classes_
        transformation_classifier =\
            LocalTransformationClassifier(**self.transformation_classifier_params)
        self.position_searcher = OptimalPositionSearcher(self.transformation_handler.transformations_by_paradigms,
                                                         self.transformation_handler.paradigms_by_codes,
                                                         transformation_classifier)
        data_for_trans_training, trans_labels =\
                self.transformation_handler.create_training_data(training_data)
        self.position_searcher.fit(data_for_trans_training, [[x] for x in trans_labels])
        return self


    def predict(self, X):
        probs = self.predict_probs(X)
        if not self.multiclass:
            class_indexes = [indices[0] for indices, _ in probs]
            answer = [[x] for x in np.take(self.classes_, class_indexes)]
        else:
            answer = [np.take(self.classes_, indices) for indices
                      in extract_classes_from_sparse_probs(probs, self.min_probs_ratio)]
        return answer

    def predict_probs(self, X):
        # заводим сокращённые имена для классификаторов
        par_cls = self.paradigm_classifier
        pos_searcher = self.position_searcher
        trans_cls = self.position_searcher.classifier
        # считаем вероятности парадигм
        class_indices_with_probs = self.paradigm_classifier._predict_proba(X)
        # ищем оптимальные значения переменных для каждой из возможных парадигм
        codes_for_span_prediction = [par_cls.classes_[index]
                                     for indices, probs in class_indices_with_probs
                                     for index in indices]
        words_for_span_prediction = [word for word, (indices, _) in zip(X, class_indices_with_probs)
                                     for index in indices]
        predicted_spans, predicted_spans_probs =\
            pos_searcher._predict_constant_spans(codes_for_span_prediction,
                                                 words_for_span_prediction,
                                                 return_scores=True)
        end = 0
        symbol_spans, epsilon_spans = [], []
        symbol_spans_positions, epsilon_spans_positions = [], []
        for i, (word, (indices, probs)) in enumerate(zip(X, class_indices_with_probs)):
            # находим участки, соответствующие различным парадигмам для данного слова
            # в массиве predicted_spans
            start, end = end, end + len(indices)
            are_symbols_covered = np.zeros(shape=(len(word), ), dtype=bool)
            are_epsilons_covered = np.zeros(shape=(len(word) + 1, ), dtype=bool)
            for curr_spans_list in predicted_spans[start:end]:
                for span_start, span_end in curr_spans_list:
                    are_symbols_covered[span_start: span_end] = True
                    are_epsilons_covered[span_start: (span_end + 1)] = True
            covered_symbol_positions = are_symbols_covered.nonzero()[0]
            covered_epsilon_positions = are_epsilons_covered.nonzero()[0]
            # копируем участки для будущего запроса
            symbol_spans.extend(((word[j], word[:j], word[(j+1): ]) for j in covered_symbol_positions))
            symbol_spans_positions.extend(((i, pos) for pos in covered_symbol_positions))
            epsilon_spans.extend(("", word[:j], word[j: ]) for j in covered_symbol_positions)
            epsilon_spans_positions.extend(((i, pos) for pos in covered_epsilon_positions))
        zero_class_index = pos_searcher.transformation_classes[0]
        symbol_spans_probs = trans_cls.predict_proba(symbol_spans)[:,zero_class_index]
        epsilon_spans_probs = trans_cls.predict_proba(epsilon_spans)[:,zero_class_index]
        # сохраняем вероятности тождественных трансформаций
        # для фрагментов каждого из слов
        probs_for_symbols = [np.ones(shape=(len(word), ), dtype=float) for word in X]
        probs_for_epsilons = [np.ones(shape=(len(word) + 1, ), dtype=float) for word in X]
        for (i, pos), prob in zip(symbol_spans_positions, symbol_spans_probs):
            probs_for_symbols[i][pos] = prob
        for (i, pos), prob in zip(epsilon_spans_positions, epsilon_spans_probs):
            probs_for_epsilons[i][pos] = prob
        cumulative_log_probs_for_symbols = []
        for probs in probs_for_symbols:
            log_probs = np.log(np.where(probs > self.smallest_prob, probs, self.smallest_prob))
            cumulative_log_probs_for_symbols.append([0.0] + list(np.cumsum(log_probs)))
        cumulative_log_probs_for_epsilons = []
        for probs in probs_for_epsilons:
            log_probs = np.log(np.where(probs > self.smallest_prob, probs, self.smallest_prob))
            cumulative_log_probs_for_epsilons.append([0.0] + list(np.cumsum(log_probs)))
        answer = []
        end = 0
        for word, (indices, probs), curr_cum_log_probs_for_symbols,\
            curr_cum_log_probs_for_epsilons in zip(X, class_indices_with_probs,
                                                   cumulative_log_probs_for_symbols,
                                                   cumulative_log_probs_for_epsilons):
            start, end = end, end + len(indices)
            curr_joint_log_probs = []
            curr_predicted_spans, curr_predicted_probs = predicted_spans[start:end], predicted_spans_probs[start:end]
            for curr_spans_list, score in zip(curr_predicted_spans, curr_predicted_probs):
                for span_start, span_end in curr_spans_list:
                    # вычитаем стоимости нетождественных трансформаций, заменившихся на тождественные
                    score -= (curr_cum_log_probs_for_symbols[span_end] -
                              curr_cum_log_probs_for_symbols[span_start])
                    # последний эпсилон берётся справа за концом отрезка
                    score -= (curr_cum_log_probs_for_epsilons[span_end + 1] -
                              curr_cum_log_probs_for_epsilons[span_start])
                curr_joint_log_probs.append(score)
            # преобразуем логарифмы вероятностей в вероятности
            curr_probs = np.exp(curr_joint_log_probs)
            curr_probs /= np.sum(curr_probs)
            indices_order = np.flipud(np.argsort(curr_probs))
            var_values = [extract_uncovered_substrings(word, spans)
                          for spans in curr_predicted_spans]
            reordered_indices_with_variables = [(self.classes_[indices[i]], var_values[i])
                                                for i in indices_order]
            answer.append((reordered_indices_with_variables, curr_probs[indices_order]))
        return answer

def _memorize_suffixes(length, X, y, remove_last_letter=False):
    if remove_last_letter:
        answer = dict()
    else:
        answer = Trie()
    offset = int(remove_last_letter)
    for word, labels in zip(X, y):
        suffix = word[(-1-offset):(-length-1):-1]
        if remove_last_letter:
            trie = answer.get(word[-1])
            if trie is None:
                trie = answer[word[-1]] = Trie()
        else:
            trie = answer
        if suffix not in trie:
            trie[suffix] = None
        path = trie.path(suffix)
        for node in path:
            if node.data is None:
                node.data = labels
            else:
                node.data = [label for label in labels if label in node.data]
    if remove_last_letter:
        for letter, trie in answer.items():
            for node in trie.nodes:
                node.is_terminal = (len(node.data) > 0)
            answer[letter] = prune_dead_branches(trie)
    else:
        for node in answer.nodes:
            node.is_terminal = (len(node.data) > 0)
        answer = prune_dead_branches(answer)
    return answer

def extract_classes_from_probs(probs, min_probs_ratio):
    """
    Возвращает список классов по их вероятностям
    в случае мультиклассовой классификации
    """
    # возможно, имеет смысл возвращать разреженную матрицу
    answer = [[] for _ in range(probs.shape[0])]
    max_allowed_probs = np.max(probs, axis=1).reshape((probs.shape[0], 1)) * min_probs_ratio
    remaining_positions = np.where(probs >= max_allowed_probs)
    for i, col in zip(*remaining_positions):
        answer[i].append(col)
    return answer

def extract_classes_from_sparse_probs(probs, min_probs_ratio):
    """
    Возвращает список классов по их вероятностям
    в случае мультиклассовой классификации
    """
    answer = [[] for i in range(len(probs))]
    for i, (indices, word_probs) in enumerate(probs):
        if len(word_probs) == 0:
            continue
        max_allowed_prob = word_probs[0] * min_probs_ratio
        for end, prob in enumerate(word_probs):
            if prob < max_allowed_prob:
                break
        answer[i] = indices[:end]
    return answer


def arrange_indexes_by_last_letters(words, reps=None):
    if reps is None:
        reps = [1] * len(words)
    answer = defaultdict(list)
    i = 0
    for word, rep in zip(words, reps):
        answer[word[-1]].extend(range(i, i+rep))
        i += rep
    return answer















