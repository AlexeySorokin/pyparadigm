import sys
import os
from itertools import chain, product
from collections import defaultdict, Counter
import bisect
import getopt
import copy
import math

import numpy as np
import scipy.sparse as scsp
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn import svm as sksvm
from sklearn import linear_model as sklm
import sklearn.metrics as skm
import sklearn.cross_validation as skcv
import sklearn.naive_bayes as sknb
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

import statprof

from input_reader import process_codes_file, process_lemmas_file, read_paradigms, read_lemmas
from learn_paradigms import write_data, write_probs_data, make_lemmas, output_accuracies
from paradigm_classifier import ParadigmClassifier,JointParadigmClassifier, CombinedParadigmClassifier, \
    arrange_indexes_by_last_letters
from paradigm_detector import Paradigm, ParadigmSubstitutor
from transformation_classifier import TransformationsHandler
from feature_selector import MulticlassFeatureSelector
from common import get_categories_marks, LEMMA_KEY
from tools import extract_classes_from_sparse_probs

from contextlib import contextmanager

DEBUG = False

@contextmanager  # для перенаправления stdout в файл в процессе работы
def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target  # replace sys.stdout
    try:
        yield new_target  # run some code with the replaced stdout
    finally:
        sys.stdout = old_target  # restore to the previous value


def read_counts(infile, convert_to_lower=True):
    counts = defaultdict(int)
    with open(infile, "r") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split()
            if len(splitted_line) != 2:
                continue
            word, count = splitted_line
            count = int(count)
            counts[word] += count
            if convert_to_lower:
                lower_word = word.lower()
                counts[lower_word] += count
    return counts


class ParadigmCorporaClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, paradigms_list, word_counts=None,
                 max_length=6, multiclass=False, has_letter_classifiers=False,
                 selection_method='ambiguity', binarization_method='bns',
                 nfeatures=None, inner_feature_fraction=0.1,
                 smallest_prob=0.01, min_probs_ratio=0.75, min_corpus_frac = 0.5,
                 active_paradigm_codes=None, paradigm_counts=None,
                 alpha=1.0, beta=0.1, gamma = 1.0, multiplier=100.0, sparse=True,
                 class_counts_alpha=0.2):
        # self.marks = marks
        # self.classes_number = len(paradigms_list)
        self.paradigms_list = paradigms_list
        self.word_counts = word_counts
        self.tmp_word_counts = dict()
        self.max_length = max_length
        self.multiclass = multiclass
        self.has_letter_classifiers = has_letter_classifiers
        self.selection_method = selection_method
        self.binarization_method = binarization_method
        self.nfeatures = nfeatures
        self.inner_feature_fraction = inner_feature_fraction
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio
        self.min_corpus_frac = min_corpus_frac
        self.active_paradigm_codes = active_paradigm_codes
        # задаём коды активных парадигм
        if active_paradigm_codes is None:
            active_paradigm_codes = self.paradigms_list.values()
        self.active_paradigm_codes = set(active_paradigm_codes)
        # на всякий случай сохраним счётчики для парадигм
        self.paradigm_counts = paradigm_counts
        self.alpha = alpha  # сглаживающий параметр в вычислении first_score
        self.beta = beta  # сглаживающий параметр в вычислении form_probabilities_for_paradigms
        self.gamma = gamma # сглаживающий параметр в вычислении вероятностей парадигм
        self.multiplier = multiplier  # масштабный коэффициент для первого классификатора
        self.unique_class_prob = 0.9
        self.method = 'form'
        self.sparse = sparse
        self.class_counts_alpha = class_counts_alpha

    def fit(self, X, y, X_dev=None, y_dev=None):
        if len(X) != len(y):
            raise ValueError("Data and labels should have equal length")
        self._prepare_parameters()
        self._prepare_classifiers()
        if isinstance(self.nfeatures, float):
            if self.nfeatures < 0.0 or self.nfeatures > 1.0:
                raise ValueError("If nfeatures is float, it should be from 0.0 to 1.0")
        # выбираем, производится ли классификация отдельно для каждой буквы
        if self.has_letter_classifiers:
            self.classes_, Y_new = np.unique(y, return_inverse=True)
            # ДОБАВЛЯЕМ КЛАССЫ ИЗ paradigm_table
            classes_set, self.classes_  = set(self.classes_), list(self.classes_)
            for code in self.paradigms_list.values():
                if code not in classes_set:
                    self.classes_.append(code)
            self.classes_ = np.array(self.classes_)
            # ПЕРЕКОДИРУЕМ КЛАССЫ
            recoded_paradigm_table = {self.descrs_by_classes[label]: i
                                      for i, label in enumerate(self.classes_)}
            self.paradigm_classifier.set_params(paradigm_table=recoded_paradigm_table)
            data_indexes_by_letters =\
                arrange_indexes_by_last_letters(X, [len(labels) for labels in y])
            X_by_letters, y_by_letters = dict(), dict()
            single_class_letters = dict()
            for letter, indexes in data_indexes_by_letters.items():
                X_curr, y_curr = [X[i] for i in indexes], [Y_new[i] for i in indexes]
                if min(y_curr) < max(y_curr):
                    X_by_letters[letter] = X_curr
                    y_by_letters[letter] = [[label] for label in y_curr]
                else:
                    single_class_letters[letter] = y_curr[0]
            self.letter_classifiers_ = {letter: clone(self.paradigm_classifier)
                                        for letter in X_by_letters}
            self.joint_classifiers_ =  {letter: clone(self.joint_classifier)
                                        for letter in X_by_letters}
            # определяем вероятности для букв, для которых нет классификаторов
            self._make_new_letter_probs(y)
            self._make_default_letter_probs(single_class_letters)
            for letter, X_curr in X_by_letters.items():
                self.letter_classifiers_[letter].fit(X_curr, y_by_letters[letter])
        else:
            self.paradigm_classifier.set_params(paradigm_table = self.paradigms_list)
            self.paradigm_classifier.fit(X, y)
            # self.paradigm_classifier уже содержал все классы,
            # поэтому ничего добавлять не нужно
            self.classes_ = self.paradigm_classifier.classes_
            self.active_classes_number = self.paradigm_classifier.active_classes_number
        # обработчики парадигм
        self.paradigmers = [ParadigmSubstitutor(self.descrs_by_classes[label])
                            for label in self.classes_]
        # обработчики лемм
        self._prepare_lemma_fragmentors()
        # вероятности граммем
        self.form_probabilities_for_paradigms =\
            [np.zeros(shape=(paradigmer.unique_forms_number(return_principal=False),),
                      dtype=np.float64) for paradigmer in self.paradigmers]
        self.reverse_classes = {label: i for (i, label) in enumerate(self.classes_)}
        self._fit_probabilities(X, [[self.reverse_classes[code] for code in labels]
                                    for labels in y])
        if self.has_letter_classifiers:
            for letter, X_curr in X_by_letters.items():
                cls = self.letter_classifiers_[letter]
                cls_classes = cls.classes_[:cls.active_classes_number]
                (_, X_joint, y_joint), _ = self._prepare_to_joint_classifier(
                    cls, cls_classes, X_curr, y_by_letters[letter])
                self.joint_classifiers_[letter].fit(X_joint, y_joint)
        else:
            cls = self.paradigm_classifier
            cls_classes = list(range(cls.active_classes_number))
            (_, X_joint, y_joint), _ =\
                self._prepare_to_joint_classifier(cls, cls_classes, X, y)
            self.joint_classifier.fit(X_joint, y_joint)
        return self

    def _prepare_parameters(self):
        """
        Предобработка параметров, переданных методу self
        """
        # не задан словарь счётчиков словоформ
        if self.word_counts is None:
            self.word_counts = defaultdict(int)
        if len(self.paradigms_list) == 0:
            raise ValueError("Paradigms list cannot be empty")
        # находим число форм, хранящихся в парадигме как число разделителей
        # (исходная форма не учитывается)
        self.forms_number = list(self.paradigms_list.keys())[0].count('#')
        self.descrs_by_classes = {label: descr for descr, label in self.paradigms_list.items()}
        return self

    def _prepare_classifiers(self):
        """
        Подготавливает классификаторы
        """
        # перенумеруем парадигмы в соответствии с классами
        self.paradigm_classifier = ParadigmClassifier(
            paradigm_table=None, multiclass=self.multiclass,
            has_letter_classifiers=not(self.has_letter_classifiers),
            max_length=self.max_length, nfeatures=self.inner_feature_fraction,
            smallest_prob=self.smallest_prob, min_probs_ratio=0.1)
        self.joint_classifier = sklm.LogisticRegression()
        return self

    def predict(self, X):
        if DEBUG:
            print("Predicting...")
        probs = self._predict_proba(X)

        if not self.multiclass:
            class_indexes = [indices[0] for indices, _ in probs]
            answer = [[x] for x in np.take(self.classes_, class_indexes)]
        else:
            answer = [np.take(self.classes_, indices) for indices
                      in extract_classes_from_sparse_probs(probs, self.min_probs_ratio)]
        return answer

    def predict_proba(self, X, sparse=False):
        """
        Возвращает вероятности классов для каждого объекта.

        Данная функция преобразует результат _predict_proba в нужный формат
        """
        probs = self._predict_proba(X)
        if sparse:
            rows = list(chain((i for _ in elem[0]) for i, elem in enumerate(probs)))
            cols = list(chain(elem[0] for elem in enumerate(probs)))
            data = list(chain(elem[1] for elem in enumerate(probs)))
            answer = scsp.csr_matrix((data, (rows, cols)),
                                     shape=(len(X), self.classes_number))
        else:
            answer = np.zeros(dtype=np.float64, shape=(len(X), self.classes_number))
            for i, (indices, data) in enumerate(probs):
                answer[i, indices] = data
        return answer

    def _predict_proba(self, X):
        """
        Функция, возвращающая для каждого объекта список (классы, вероятности)
        в порядке убывания вероятностей. Потом данная функция вызывается в predict и
        predict_proba.
        """
        # оставляем только парадигмы, под которые может подходить слово
        answer = [None] * len(X)
        row_denser = ((lambda x: np.ravel(x.todense())) if self.sparse else (lambda x: x))
        if self.has_letter_classifiers:
            data_indexes_by_letters = arrange_indexes_by_last_letters(X)
            for letter, indexes in data_indexes_by_letters.items():
                curr_X = [X[i] for i in indexes]
                cls = self.letter_classifiers_.get(letter)
                if cls is None:
                    if letter in self._default_letter_probs:
                        probs_row = self._default_letter_probs[letter]
                    else:
                        probs_row = self.new_letter_probs
                    curr_X_probs = np.tile(probs_row, (len(indexes), 1))
                    curr_classes = range(len(self.classes_))
                    active_classes_number = len(curr_classes)
                else:
                    active_classes_number = cls.active_classes_number
                    (train_indexes, X_train), (other_indexes, other_probs) =\
                        self._prepare_to_joint_classifier(cls, cls.classes_, curr_X,
                                                          active_classes_number=active_classes_number)
                    curr_X_probs = self.joint_classifiers_[letter].predict_proba(X_train)
                    curr_classes = cls.classes_
                # объекты, чьи классы встречались в обучающей выборке
                for i, (train_index, word_probs) in enumerate(zip(train_indexes, curr_X_probs)):
                    index = indexes[train_index]
                    if cls is None:
                        row_ = self._fits_to_which_lemma_fragmentors(curr_X[i], negate=True)
                    else:
                        row = row_denser(X_train[i])
                        row_ = [j for j in range(active_classes_number) if row[3*j] == 0.0]
                    indices, probs = self._extract_word_probs(word_probs, row_)
                    indices = [curr_classes[j] for j in indices]
                    answer[index] = (indices, probs)
                # объекты, чьи классы не встречались в обучающей выборке
                # их классы унаследованы от базового классификатора
                for other_index, (indices, probs) in zip(other_indexes, other_probs):
                    index = indexes[other_index]
                    indices = [curr_classes[j] for j in indices]
                    answer[index] = (indices, probs)
        else:
            cls_classes = [i for i, _ in enumerate(self.classes_)]
            active_classes_number = self.paradigm_classifier.active_classes_number
            (train_indexes, X_train), (other_indexes, other_probs) =\
                self._prepare_to_joint_classifier(self.paradigm_classifier, cls_classes, X,
                                                  active_classes_number=active_classes_number)
            probs = self.joint_classifier.predict_proba(X_train)
            # объекты, чьи классы встречались в обучающей выборке
            for i, row, word_probs in zip(train_indexes, X_train, probs):
                # здесь надо разобраться
                row = row_denser(row)
                row_ = [j for j in range(active_classes_number) if row[3*j] == 0.0]
                answer[i] = self._extract_word_probs(word_probs, row_)
            # объекты, чьи классы не встречались в обучающей выборке
            # их классы унаследованы от базового классификатора
            for other_index, (indices, probs) in zip(other_indexes, other_probs):
                indices = [cls_classes[j] for j in indices]
                # print(X[other_index], self.paradigmers[indices[0]].descr)
                answer[other_index] = (indices, probs)
        # sys.exit()
        return answer

    def _extract_word_probs(self, probs, impossible_classes):
        """
        Извлекает вычисленнные вероятности
        """
        probs[impossible_classes] = 0.0
        if np.sum(probs) > 0.0:
            probs /= np.sum(probs)
        probs[np.where(probs < self.smallest_prob)] = 0.0
        if np.sum(probs) > 0.0:
            probs /= np.sum(probs)
        nonzero_indexes = probs.nonzero()[0]
        probs = probs[nonzero_indexes]
        indexes_order = np.flipud(np.argsort(probs))
        current_indices = nonzero_indexes[indexes_order]
        current_probs = probs[indexes_order]
        return (current_indices, current_probs)

    def _prepare_lemma_fragmentors(self):
        """
        Предвычисляет фрагменторы для лемм
        """
        self._lemma_fragmentors_ = [None] * len(self.paradigmers)
        self._lemma_fragmentors_indexes = [None] * len(self.paradigmers)
        self._lemma_fragmentors_list = []
        fragmentors_indexes_by_patterns = dict()
        fragmentors_number = 0
        for code, paradigmer in enumerate(self.paradigmers):
            lemma_fragmentor = paradigmer._principal_paradigm_fragment
            lemma_descr = paradigmer._principal_descr
            fragmentor_index = fragmentors_indexes_by_patterns.get(lemma_descr, -1)
            if fragmentor_index < 0:
                self._lemma_fragmentors_list.append(lemma_fragmentor)
                fragmentors_indexes_by_patterns[lemma_descr] = fragmentor_index = fragmentors_number
                fragmentors_number += 1
            self._lemma_fragmentors_[code] = self._lemma_fragmentors_list[fragmentor_index]
            self._lemma_fragmentors_indexes[code] = fragmentor_index
        return self

    def _fit_probabilities(self, X, y):
        for word, labels in zip(X, y):
            for label in labels:
                paradigmer = self.paradigmers[label]
                current_counts = np.zeros(shape=len(self.form_probabilities_for_paradigms[label]))
                # ОПРЕДЕЛЯТЬ ФОРМЫ ИЗ ОБУЧАЮЩЕЙ ВЫБОРКИ, А НЕ УГАДЫВАНИЕМ
                unique_word_forms = paradigmer.make_unique_forms(word, return_principal=False)
                weight = 1.0 / len(unique_word_forms) / len(labels)
                for forms_list in unique_word_forms:
                    for j, form in enumerate(forms_list):
                        current_counts[j] += self.word_counts[form]
                self.form_probabilities_for_paradigms[label] += weight * current_counts
        for elem in self.form_probabilities_for_paradigms:
            elem += self.beta
            elem /= np.sum(elem)
        return self

    def _prepare_to_joint_classifier(self, cls, cls_classes, X, y=None,
                                     active_classes_number=None):
        """
        cls: object, классификатор, применяемый для получения вероятностей классов
        cls_classes:  позиции возможных классов в массиве self.classes_
        classes_to_use: позиции классов, которые разрешено использовать при обучении
        """
        if y is None:
            y = [[None]] * len(X)
            return_y = False
        else:
            return_y = True
        # y_new = list(chain(*y))
        if active_classes_number is None:
            active_classes_number = len(cls_classes)
        # fout = open("tmp.out", "w", encoding="utf8")
        minus_log_probs = cls.predict_minus_log_proba(X)
        curr_row_number = 0
        tmp_paradigmers = [self.paradigmers[label] for label in cls_classes]
        if self.sparse:
            rows, cols, data = [], [], []
        else:
            X_new = np.zeros(dtype=np.float64, shape=(len(y_new), 3 * active_classes_number))
        y_new = []
        indexes_to_train, other_indexes, X_other = [], [], []
        for j, ((indices, word_probs), word, labels) in enumerate(zip(minus_log_probs, X, y)):
            if min(indices) < active_classes_number:
                has_active_classes = True
            else:
                has_active_classes = False
                scores_by_classes = []
            has_active_paradigm_codes = any(
                (self.classes_[cls_classes[i]] in self.active_paradigm_codes)
                 # and int(val * self.multiplier) > 0)
                for i, val in zip(indices, word_probs))
            # if word == 'щи':
            #     print(indices, word_probs, has_active_classes, has_active_paradigm_codes)
            #     sys.exit()
            if self.sparse:
                current_cols, current_data = [], []
            else:
                curr_row = np.zeros(shape=(3 * len(cls_classes), ), dtype=float)
            # if word in ['музей', 'лето', 'жила']:
            #     print(word)
            #     for i, prob in zip(indices, word_probs):
            #         print(tmp_paradigmers[i].descr, prob)
            for i, prob in zip(indices, word_probs):
                if has_active_classes and i >= active_classes_number:
                    continue
                class_index = cls_classes[i]
                # вычисляем значения для объединённого классификатора
                # value = int(prob * self.multiplier)
                value = prob
                if value == 0 or (has_active_paradigm_codes and
                        self.classes_[class_index] not in self.active_paradigm_codes):
                    continue
                first_score, counts, _, weights = self._make_first_score(tmp_paradigmers[i], word)
                third_score = self._make_third_score(class_index, counts)
                # помещаем значения в массив в зависимости от наличия активного класса и self.sparse
                if has_active_classes:
                    if self.sparse:
                        current_cols.extend((3 * i, 3 * i + 1, 3 * i + 2))
                        current_data.extend((value, first_score, third_score))
                    else:
                        curr_row[3*i: 3*(i + 1)] = (value, first_score, third_score)
                else:
                    # запоминаем тройку (i, first_score, third_score)
                    scores_by_classes.append((i, first_score, third_score))
            # if word in ['музей', 'лето', 'жила']:
            #     print(word)
            #     for i in range(0, len(current_cols), 3):
            #         print("{} {:.4f} {:.4f}".format(*current_data[i: i+3]))
            if has_active_classes:
                indexes_to_train.append(j)
                if self.sparse:
                    for _ in labels:
                        rows.extend(curr_row_number for _ in current_data)
                        cols.extend(current_cols)
                        data.extend(current_data)
                        curr_row_number += 1
                else:
                    X_new[curr_row_number: curr_row_number+len(labels)] = curr_row
                    curr_row_number += len(labels)
                y_new.extend(labels)
            else:
                other_indexes.append(j)
                _, first_score, _ = max(scores_by_classes, key=(lambda x:x[1]))
                best_classes_indexes =\
                    [i for i, score, _ in scores_by_classes if score == first_score]
                best_classes_probs =\
                    [(1.0 / len(best_classes_indexes)) for _ in best_classes_indexes]
                X_other.append((best_classes_indexes, best_classes_probs))
        if self.sparse:
            X_new = scsp.csr_matrix((data, (rows, cols)), dtype=np.float64,
                                    shape=(curr_row_number, 3 * active_classes_number))
        if return_y:
            return (indexes_to_train, X_new, y_new), (other_indexes, X_other)
        else:
            return (indexes_to_train, X_new), (other_indexes, X_other)

    def _make_first_score(self, paradigmer, word):
        word_forms = paradigmer.make_unique_forms(word, return_principal=False)
        # счётчики могут не содержать ё''
        # word_forms = [[form.replace('ё', 'е') for form in forms_list]
        #               for forms_list in word_forms]
        if len(word_forms) == 0:
            print("No forms!", word)
            return None
        pattern_counts = paradigmer.get_pattern_counts(return_principal=False)
        if self.method == 'unique':
            weights = np.ones(shape=(len(pattern_counts)))
        elif self.method == 'form':
            weights = np.array(pattern_counts)
        # current_word_counts = dict()
        log_counts = []
        # for word in chain.from_iterable(word_forms):
        #     if word not in self.tmp_word_counts:
        #         # current_word_counts[word] = self.word_counts.get(word, 0.0) + self.alpha
        #         # current_word_counts[word] = np.log(self.word_counts.get(word, 0.0) + self.alpha)
        #         self.tmp_word_counts[word] = np.log(self.word_counts.get(word, 0.0) + self.alpha)
        for forms_list in word_forms:
            curr_log_counts = []
            for word in forms_list:
                count = self.tmp_word_counts.get(word, -1)
                if count < 0:
                    count = self.tmp_word_counts[word] =\
                        math.log(self.word_counts.get(word, 0.0) + self.alpha)
                    # current_word_counts[word] = self.word_counts.get(word, 0.0) + self.alpha
                    # current_word_counts[word] = np.log(self.word_counts.get(word, 0.0) + self.alpha)
                curr_log_counts.append(count)
            log_counts.append(curr_log_counts)
        best_score, best_corpus_forms_count = -np.inf, 0
        best_counts, best_forms_list = None, None
        had_enough_forms = False
        min_corpus_forms_count = np.ceil(self.min_corpus_frac * self.forms_number)
        log_alpha = np.log(self.alpha)
        # for forms_list in word_forms:
        best_index = -1
        for i, form_counts in enumerate(log_counts):
            # counts = np.array([current_word_counts[word] for word in forms_list])
            # form_counts = np.array([self.tmp_word_counts[word] for word in forms_list])
            corpus_forms_count = 0.0
            for word_count, weight in zip(form_counts, weights):
                # if word_count > self.alpha:
                if word_count > log_alpha:
                    corpus_forms_count += weight
            if corpus_forms_count > min_corpus_forms_count:
                had_enough_forms = True
            # print(forms_list)
            # print(corpus_forms_count, had_enough_forms, best_corpus_forms_count, min_corpus_forms_count)
            if had_enough_forms:
                if corpus_forms_count <= min_corpus_forms_count:
                    continue
                score = np.dot(form_counts, weights)
                # score = np.dot(np.log(counts), weights)
                if score > best_score:
                    # best_counts = form_counts
                    # best_forms_list = forms_list
                    best_index = i
                    best_score = score
            elif corpus_forms_count > best_corpus_forms_count:
                # best_counts = form_counts
                # best_forms_list = forms_list
                best_index = i
                best_corpus_forms_count = corpus_forms_count
        # print()
        if not had_enough_forms:
            best_score = 0.0
            best_forms, best_counts = None, None
        else:
            best_score -= np.log(self.alpha) * np.sum(weights)
            best_forms, best_counts = word_forms[best_index], np.exp(log_counts[best_index])
        # best_counts = np.exp(log_counts[best_index]) # 29.02
        # best_counts = log_counts[i]
        return best_score, best_counts, best_forms, weights
        # return best_score, best_counts, word_forms[best_index], weights

    def _make_second_score(self, class_index, counts, weights):
        second_score = np.dot(-np.log(self.form_probabilities_for_paradigms[class_index]),
                                      counts)
        if self.method == 'form':
            second_score -= np.dot(counts, -np.log(weights))
        counts_sum = np.sum(counts)
        second_score *= (np.log(counts_sum) / counts_sum)
        return second_score

    def _make_third_score(self, class_index, counts):
        if counts is None:
            return 0.0
        p = self.form_probabilities_for_paradigms[class_index]
        counts_sum = np.sum(counts)
        counts /= counts_sum
        answer = np.dot(counts, np.log(counts / p)) * np.log(counts_sum)
        return answer

    # def _get_average_log_counts(self, X, average='log'):
    #     probs = self.predict_proba(X)
    #     answer = [None] * len(X)
    #     for i, (word, prob) in enumerate(zip(X, probs)):
    #         nonzero_indexes = prob.nonzero()[0]
    #         nonzero_probs = prob[nonzero_indexes]
    #         class_indexes = [self.paradigm_classifier.classes_[i]
    #                          for i in nonzero_indexes]
    #         paradigmers = np.take(self.paradigmers, class_indexes)
    #         scores_with_words = []
    #         for paradigmer in paradigmers:
    #             # возвращать лучший из тех вариантов, где нашлось достаточно форм
    #             curr_score = self._make_first_score(paradigmer, word)
    #             if curr_score is None:
    #                 sys.exit("No score!")
    #             else:
    #                 scores_with_words.append(curr_score)
    #         if average == 'log':
    #             score = np.dot(nonzero_probs, [elem[0] for elem in scores_with_words])
    #         elif average == 'count':
    #             total_counts = [sum(elem[1]) for elem in scores_with_words]
    #             score = np.log(np.dot(probs, total_counts))
    #         answer[i] = score
    #     return answer

    def _get_average_paradigm_count(self, probs):
        '''
        Вычисляет среднюю число лемм в соответствии с заданным распределением
        вероятностей на множестве парадигм
        '''
        paradigm_counts_by_classes = [self.paradigm_counts.get(i, 0.0) + self.gamma
                                      for i in self.classes_]
        answer = np.dot(paradigm_counts_by_classes, probs)
        return answer

    def _make_new_letter_probs(self, y):
        """
        Вычисляет вероятности для последних букв, не встречавшихся в обучающей выборке
        """
        counts = Counter(chain.from_iterable(y))
        self.new_letter_probs = np.zeros(shape=(len(self.classes_), ), dtype=float)
        for i, label in enumerate(self.classes_):
            self.new_letter_probs[i] = counts[label] + self.class_counts_alpha
        self.new_letter_probs /= sum(self.new_letter_probs)
        return self

    def _make_default_letter_probs(self, single_class_letters):
        self._default_letter_probs = dict()
        for letter, label in single_class_letters.items():
            curr_probs = (1.0 - self.unique_class_prob) * self.new_letter_probs
            curr_probs[label] += self.unique_class_prob
            self._default_letter_probs[letter] = curr_probs
        return self

    def _fits_to_which_lemma_fragmentors(self, word, negate=False):
        """
        вычисляет индикаторы того, подходит ли слово word под фрагменторы
        """
        answer = []
        fits_to_fragmentor = [False for _ in self._lemma_fragmentors_list]
        for index, fragmentor in enumerate(self._lemma_fragmentors_list):
            fits_to_fragmentor[index] = fragmentor.fits_to_pattern(word)
        if negate:
            answer = [i for i, index in enumerate(self._lemma_fragmentors_indexes)
                      if not fits_to_fragmentor[index]]
        else:
            answer = [i for i, index in enumerate(self._lemma_fragmentors_indexes)
                      if fits_to_fragmentor[index]]
        return answer

def cv_mode(testing_mode, language_code, multiclass, predict_lemmas,
            paradigm_file, counts_file, infile,
            train_fraction, feature_fraction, paradigm_counts_threshold,
            nfolds, selection_method, binarization_method, max_feature_length,
            output_train_dir=None, output_pred_dir=None):
    lemma_descriptions_list = process_lemmas_file(infile)
    data, labels_with_vars = read_lemmas(lemma_descriptions_list, multiclass=multiclass, return_joint=True)
    paradigm_descriptions_list = process_codes_file(paradigm_file)
    paradigm_table, paradigm_counts = read_paradigms(paradigm_descriptions_list)
    word_counts_table = read_counts(counts_file)
    if predict_lemmas:
        paradigm_handlers = {code: ParadigmSubstitutor(descr)
                             for descr, code in paradigm_table.items()}
    else:
        test_lemmas, pred_lemmas = None, None
    # подготовка для кросс-валидации
    classes = sorted(set(chain(*((x[0] for x in elem) for elem in labels_with_vars))))
    active_paradigm_codes = [i for i, count in paradigm_counts.items()
                             if count >= paradigm_counts_threshold]
    # зачем нужны метки???
    marks = [LEMMA_KEY] + [",".join(x) for x in get_categories_marks(language_code)]
    paradigms_by_codes = {code: descr for descr, code in paradigm_table.items()}
    # подготовка данных для локальных трансформаций
    if selection_method is None:
        selection_method = 'ambiguity'
    if nfolds == 0:
        train_data_length = int(train_fraction * len(data))
        train_data, test_data = [data[:train_data_length]], [data[train_data_length:]]
        train_labels_with_vars, test_labels_with_vars =\
            [labels_with_vars[:train_data_length]], [labels_with_vars[train_data_length:]]
        nfolds = 1
    else:
        test_data, train_data = [None] * nfolds, [None] * nfolds
        test_labels_with_vars, train_labels_with_vars = [None] * nfolds, [None] * nfolds
        for fold in range(nfolds):
            train_data[fold], test_data[fold], train_labels_with_vars[fold], test_labels_with_vars[fold] =\
                skcv.train_test_split(data, labels_with_vars, test_size = 1.0 - train_fraction,
                                      random_state = 100 * fold + 13)
        if predict_lemmas:
            test_lemmas = [None] * nfolds
            for fold in range(nfolds):
                test_lemmas[fold] = make_lemmas(paradigm_handlers, test_labels_with_vars[fold])
    predictions = [None] * nfolds
    prediction_probs = [None] * nfolds
    classes_by_cls = [None] * nfolds
    if predict_lemmas:
        pred_lemmas = [None] * nfolds
    # задаём классификатор
    # cls = ParadigmCorporaClassifier(marks, paradigm_table, word_counts_table,
    #                                 multiclass=multiclass, selection_method=selection_method,
    #                                 binarization_method=binarization_method,
    #                                 inner_feature_fraction=feature_fraction,
    #                                 active_paradigm_codes=active_paradigm_codes,
    #                                 paradigm_counts=paradigm_counts , smallest_prob=0.01)
    cls = ParadigmCorporaClassifier(paradigm_table, word_counts_table,
                                    multiclass=multiclass, selection_method=selection_method,
                                    binarization_method=binarization_method,
                                    inner_feature_fraction=feature_fraction,
                                    active_paradigm_codes=active_paradigm_codes,
                                    paradigm_counts=paradigm_counts , smallest_prob=0.001)
    cls_params = {'max_length': max_feature_length}
    transformation_handler = TransformationsHandler(paradigm_table, paradigm_counts)
    transformation_classifier_params = {'select_features': 'ambiguity',
                                        'selection_params': {'nfeatures': 0.1, 'min_count': 2}}
    # statprof.start()
    cls = JointParadigmClassifier(cls, transformation_handler, cls_params,
                                  transformation_classifier_params)
    # cls = CombinedParadigmClassifier(cls, transformation_handler, cls_params,
    #                                  transformation_classifier_params)
    # сохраняем тестовые данные
    # if output_train_dir is not None:
    #     if not os.path.exists(output_train_dir):
    #         os.makedirs(output_train_dir)
    #     for i, (train_sample, train_labels_sample) in\
    #             enumerate(zip(train_data, train_labels_with_vars), 1):
    #         write_joint_data(os.path.join(output_train_dir, "{0}.data".format(i)),
    #                          train_sample, train_labels_sample)
    # применяем классификатор к данным
    for i, (train_sample, train_labels_sample, test_sample, test_labels_sample) in\
            enumerate(zip(train_data, train_labels_with_vars, test_data, test_labels_with_vars)):
        cls.fit(train_sample, train_labels_sample)
        classes_by_cls[i] = cls.classes_
        if testing_mode == 'predict':
            predictions[i] = cls.predict(test_sample)
        elif testing_mode == 'predict_proba':
            prediction_probs[i] = cls.predict_probs(test_sample)
            # в случае, если мы вернули вероятности,
            # то надо ещё извлечь классы
            if not multiclass:
                predictions[i] = [[elem[0][0]] for elem in prediction_probs[i]]
            else:
                raise NotImplementedError()
        if predict_lemmas:
            pred_lemmas[i] = make_lemmas(paradigm_handlers, predictions[i])
    # statprof.stop()
    # with open("statprof_{0:.1f}_{1:.1f}.stat".format(train_fraction,
    #                                                  feature_fraction), "w") as fout:
    #     with redirect_stdout(fout):
    #         statprof.display()
    if output_pred_dir:
        descrs_by_codes = {code: descr for descr, code in paradigm_table.items()}
        test_words = test_data
        if testing_mode == 'predict_proba':
            prediction_probs_for_output = prediction_probs
        else:
            prediction_probs_for_output = None
    else:
        descrs_by_codes, test_words, prediction_probs_for_output = None, None, None
    if not predict_lemmas:
        label_precisions, variable_precisions, form_precisions =\
            output_accuracies(classes, test_labels_with_vars, predictions, multiclass,
                              outfile=output_pred_dir, paradigm_descrs=descrs_by_codes,
                              test_words=test_words, predicted_probs=prediction_probs_for_output,
                              save_confusion_matrices=True)
        print("{0:<.2f}\t{1}\t{2:<.2f}\t{3:<.2f}\t{4:<.2f}".format(
            train_fraction, cls.paradigm_classifier.nfeatures,
            100 * np.mean(label_precisions), 100 * np.mean(variable_precisions),
            100 * np.mean(form_precisions)))
    else:
        label_precisions, variable_precisions, lemma_precisions, form_precisions =\
            output_accuracies(classes, test_labels_with_vars, predictions,
                              multiclass, test_lemmas, pred_lemmas,
                              outfile=output_pred_dir, paradigm_descrs=descrs_by_codes,
                              test_words=test_words, predicted_probs=prediction_probs_for_output,
                              save_confusion_matrices=True)
        print("{0:<.2f}\t{1}\t{2:<.2f}\t{3:<.2f}\t{4:<.2f}\t{5:<.2f}".format(
            train_fraction, cls.paradigm_classifier.nfeatures,
            100 * np.mean(label_precisions), 100 * np.mean(variable_precisions),
            100 * np.mean(lemma_precisions), 100 * np.mean(form_precisions)))

    # statprof.stop()
    # with open("statprof_{0:.1f}_{1:.1f}.stat".format(fraction, feature_fraction), "w") as fout:
    #     with redirect_stdout(fout):
    #         statprof.display()
    # вычисляем точность и обрабатываем результаты
    # for curr_test_values, curr_pred_values in zip(test_values_with_codes, pred_values_with_codes):
    #     for first, second in zip(curr_test_values, curr_pred_values):
    #         first_code, first_vars = first[0].split('_')[0], tuple(first[0].split('_')[1:])
    #         second_code, second_vars = second[0].split('_')[0], tuple(second[0].split('_')[1:])
    #         if first_code == second_code and first_vars != second_vars:
    #             print('{0}\t{1}'.format(first, second))
    # if not multiclass:
    #     confusion_matrices = [skm.confusion_matrix(first, second, labels=classes)
    #                           for first, second in zip(firsts, seconds)]
    # сохраняем результаты классификации
    # if output_pred_dir is not None:
    #     if not os.path.exists(output_pred_dir):
    #         os.makedirs(output_pred_dir)
    #     if testing_mode == 'predict':
    #         for i, (test_sample, pred_labels_sample, true_labels_sample) in\
    #                 enumerate(zip(test_data, predictions, test_labels), 1):
    #             write_data(os.path.join(output_pred_dir, "{0}.data".format(i)),
    #                        test_sample, pred_labels_sample, true_labels_sample)
    #     elif testing_mode == 'predict_proba':
    #         for i, (test_sample, pred_probs_sample, labels_sample, cls_classes) in\
    #                 enumerate(zip(test_data, prediction_probs, test_labels, classes_by_cls), 1):
    #             write_probs_data(os.path.join(output_pred_dir, "{0}.prob".format(i)),
    #                              test_sample, pred_probs_sample, cls_classes, labels_sample)
    # сохраняем матрицы ошибок классификации
    # if not multiclass and nfolds <= 1:
    #     confusion_matrices_folder = "confusion_matrices"
    #     if not os.path.exists(confusion_matrices_folder):
    #         os.makedirs(confusion_matrices_folder)
    #     dest = os.path.join(confusion_matrices_folder,
    #                         "confusion_matrix_{0}_{1:<.2f}_{2:<.2f}.out".format(
    #                             max_length, fraction, feature_fraction))
    #     with open(dest, "w", encoding="utf8") as fout:
    #         fout.write("{0:<4}".format("") +
    #                    "".join("{0:>4}".format(label) for label in classes) + "\n")
    #         for label, elem in zip(cls.classes_, confusion_matrices[0]):
    #             nonzero_positions = np.nonzero(elem)
    #             nonzero_counts = np.take(elem, nonzero_positions)[0]
    #             nonzero_labels = np.take(classes, nonzero_positions)[0]
    #             fout.write("{0:<4}\t".format(label))
    #             fout.write("\t".join("{0}:{1}".format(*pair)
    #                                  for pair in sorted(zip(nonzero_labels, nonzero_counts),
    #                                                     key=(lambda x: x[1]), reverse=True))
    #                        + "\n")
    return


SHORT_OPTIONS = 'mplf:t:T:O:'
LONG_OPTIONS = ['multiclass', 'probs', 'lemmas', 'max_feature_length',
                'paradigm_threshold=', 'train_output=', 'test_output=']



if __name__ == '__main__':
    args = sys.argv[1:]
    # значения по умолчанию для режима тестирования
    # и поддержки множественнных парадигм для одного слова
    testing_mode, multiclass, predict_lemmas = 'predict', False, False
    max_feature_lengths = '5'
    paradigm_count_threshold = 0
    output_train_dir, output_pred_dir = None, None
    # опции командной строки
    opts, args = getopt.getopt(args, SHORT_OPTIONS, LONG_OPTIONS)
    for opt, val in opts:
        if opt in ['-m', '--multiclass']:
            multiclass = True
        elif opt in ['-p', '--probs']:
            testing_mode = 'predict_proba'
        # измеряется качество лемматизации
        elif opt in ['-l', '--lemmas']:
            predict_lemmas = True
        elif opt in ['-f', '--max_feature_length']:
            max_feature_lengths = val
        elif opt in ['-t', '--paradigm_threshold']:
            paradigm_count_threshold = int(val)
        elif opt in ['-T', '--train_output']:
            output_train_dir = val
        elif opt in ['-O', '--test_output']:
            output_pred_dir = val
    max_feature_lengths = [int(elem) for elem in max_feature_lengths.split(',')]
    # аргументы командной строки
    if len(args) < 1:
        sys.exit("You should pass mode as first argument")
    mode, args = args[0], args[1:]
    if mode not in ["train", "cross-validation"]:
        sys.exit("Mode should be train or cross-validation")
    if mode == "train":
        raise NotImplementedError()
    elif mode == "cross-validation":
        # if len(args) not in [7, 10]:
        #     sys.exit("Pass 'cross-validation', language code, coding file, "
        #              "counts file, train file, feature fraction to select, "
        #              "train data fraction, number of folds, "
        #              "[selection_method, binarization_method]")
        if len(args) not in [6, 9]:
            sys.exit("Pass 'cross-validation', coding file, counts file, train file, "
                     "feature fraction to select,  train data fraction, number of folds, "
                     "[selection_method, binarization_method]")
        # language_code, paradigm_file, counts_file, infile = args[:4]
        paradigm_file, counts_file, infile = args[:3]
        # feature_fractions, train_fractions = args[4:6]
        feature_fractions, train_fractions = args[3:5]
        # можно передавать несколько вариантов параметров через запятую
        feature_fractions = list(float(x) if float(x) >= 0.0 else None
                                 for x in feature_fractions.split(','))
        train_fractions = list(float(x) for x in train_fractions.split(','))
        # folds_number = int(args[6])
        folds_number = int(args[5])
        # selection_method = args[7] if len(args) > 7 else 'ambiguity'
        # binarization_method = args[8] if len(args) > 8 else 'bns'
        selection_method = args[6] if len(args) > 7 else 'ambiguity'
        binarization_method = args[7] if len(args) > 8 else 'bns'

        # for train_fraction in train_fractions:
        #     for feature_fraction in feature_fractions:
        for train_fraction, feature_fraction, max_feature_length in\
                product(train_fractions, feature_fractions, max_feature_lengths):
            if output_train_dir is not None:
                if not os.path.exists(output_train_dir):
                    os.makedirs(output_train_dir)
                output_train_dir_ =\
                    os.path.join(output_train_dir,
                                 "{0:.1f}_{1:.1f}".format(train_fraction, feature_fraction))
            if output_pred_dir is not None:
                if not os.path.exists(output_pred_dir):
                    os.makedirs(output_pred_dir)
                output_pred_dir_ =\
                    os.path.join(output_pred_dir,
                                 "{0:.1f}_{1:.1f}".format(train_fraction, feature_fraction))
            # cv_mode(testing_mode, language_code, multiclass=multiclass, predict_lemmas=predict_lemmas,
            #         paradigm_file=paradigm_file, counts_file=counts_file, infile=infile,
            #         train_fraction=train_fraction, feature_fraction=feature_fraction,
            #         paradigm_counts_threshold=paradigm_count_threshold, nfolds=folds_number,
            #         selection_method=selection_method, binarization_method=binarization_method,
            #         output_train_dir=output_train_dir, output_pred_dir=output_pred_dir)
            cv_mode(testing_mode, language_code="RU", multiclass=multiclass, predict_lemmas=predict_lemmas,
                    paradigm_file=paradigm_file, counts_file=counts_file, infile=infile,
                    train_fraction=train_fraction, feature_fraction=feature_fraction,
                    paradigm_counts_threshold=paradigm_count_threshold, nfolds=folds_number,
                    selection_method=selection_method, binarization_method=binarization_method,
                    max_feature_length=max_feature_length,
                    output_train_dir=output_train_dir, output_pred_dir=output_pred_dir)

