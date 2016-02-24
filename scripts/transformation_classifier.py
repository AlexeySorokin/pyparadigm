#-------------------------------------------------------------------------------
# Name:        transformation_classifier.py
#-------------------------------------------------------------------------------

import sys
from itertools import chain
from collections import OrderedDict, defaultdict
import bisect
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import sklearn.cross_validation as skcv
import sklearn.metrics as skm
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model as sklm
from scipy.sparse import csr_matrix, csc_matrix, issparse

from input_reader import process_lemmas_file, process_codes_file, read_paradigms, read_transformations_
import paradigm_classifier
from paradigm_detector import ParadigmFragment, get_first_form_pattern
from feature_selector import MulticlassFeatureSelector, ZeroFeatureRemover, SelectingFeatureWeighter
from utility import find_first_larger_indexes

# Здесь планируется классификатор, определяющий по соседним символам для трансформации, возможна
# ли она в данном окружении

class TransformationsHandler:

    def __init__(self, paradigm_codes, paradigm_counts=None):
        self.paradigm_codes = paradigm_codes
        self.paradigm_counts = paradigm_counts
        self.paradigms_by_codes = {code: descr for descr, code in paradigm_codes.items()}
        self._preprocess_input()

    def _preprocess_input(self):
        # приводим paradigm_counts к словарю
        if self.paradigm_counts is None:
            self.paradigm_counts = defaultdict(int)
        # извлекаем информацию о трансформациях
        transformations_data, transformations_by_strings =\
            read_transformations_(self.paradigms_by_codes, self.paradigm_counts)
        (self.transformations, self.transformation_codes,
         self.transformation_counts, self.transformations_by_paradigms) = transformations_data
        with open("transformation_codes.out", "w", encoding="utf8") as fout:
            for trans, code in sorted(self.transformation_codes.items(), key=(lambda x:x[1])):
                fout.write("{0}\t{1}\n".format(trans.trans, code))
        self.transformations_by_strings = transformations_by_strings
        # строим фрагменторы
        self.fragmentors = {code: ParadigmFragment(descr.split('#')[0])
                            for code, descr in self.paradigms_by_codes.items()}

    def _extract_transformations(self, lemma_descriptions_list):
        # извлекает трансформации из списка парадигм для лемм
        answer = []
        for lemma, code, var_values in lemma_descriptions_list:
            fragmentor = self.fragmentors[code]
            current_transformations_codes = self.transformations_by_paradigms[code]
            spans = fragmentor.find_const_fragments_spans(var_values)
            if len(current_transformations_codes) < len(spans):
                spans = spans[1:]
            current_spans_data =\
                list(((i, j), trans_code, lemma[:i], lemma[j:])
                     for (i, j), trans_code in zip(spans, current_transformations_codes))
            answer.append((lemma, current_spans_data))
        return answer

    def _extract_transformations_for_lm_learning(self, lemma_descriptions_list):
        '''
        Преобразует записи вида (лемма, код, переменные)
        в последовательности элементарных преобразований,
        применяемых к лемме слева направо
        '''
        answer = []
        for lemma, code, var_values in lemma_descriptions_list:
            current_transformations_codes = self.transformations_by_paradigms[code]
            curr_answer = []
            if len(var_values) != len(current_transformations_codes):
                print(var_values)
                print(current_transformations_codes)
                sys.exit()
            for var_value, trans_code in zip(var_values, current_transformations_codes):
                # (0, letter) обозначает тождественное
                # преобразование, применяемое к букве letter
                curr_answer.extend(var_value)
                curr_answer.append(trans_code)
            answer.append(curr_answer)
        return answer

    def create_training_data(self, lemma_descriptions_list):
        """
        Преобразует записи вида (лемма, код, переменные)
        в формат, удобный для обучения классификатора
        """
        lemmas_with_spans = self._extract_transformations(lemma_descriptions_list)
        max_length = max(len(substr) for substr in self.transformations_by_strings)
        spans_with_contexts, codes = [], []
        for lemma, spans_data in lemmas_with_spans:
            spans_data = {elem[0]: elem[1:] for elem in spans_data}
            # находим незанятые участки
            free_spans = find_free_spans(spans_data.keys(), len(lemma))
            for first, second in free_spans:
                for i in range(first, second):
                    j_min = i + bool(i == first)
                    j_max = min(i + max_length, second)
                    for j in range(j_min, j_max+1):
                        substr = lemma[i:j]
                        if substr in self.transformations_by_strings:
                            # нет трансформации
                            spans_data[(i, j)] = (0, lemma[:i], lemma[j:])
            substr_spans_data = [(lemma[start: end], prefix, suffix)
                                 for (start, end), (code, prefix, suffix) in spans_data.items()]
            spans_with_contexts.extend(substr_spans_data)
            codes.extend(elem[0] for elem in spans_data.values())
        return spans_with_contexts, codes

    def output_transformations(self, outfile):
        with open(outfile, "w", encoding="utf8") as fout:
            for substr, codes in self.transformations_by_strings.items():
                for code in sorted(codes, key=(lambda x: self.transformation_counts[x]), reverse=True):
                    fout.write("{0}\t{1}\t{2}\n".format(substr,
                                                        "#".join(self.transformations[code].trans),
                                                        self.transformation_counts[code]))
        return


def find_free_spans(spans, end):
    spans = sorted(spans)
    last = 0
    answer = []
    for first, second in spans:
        if first > last:
            answer.append((last, first))
        last = second
    if end > last:
        answer.append((last, end))
    return answer


class LocalTransformationClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, left_context_length=2, right_context_length=3, has_central_context=True,
                 classifier_name=sklm.LogisticRegression(), classifier_params=None,
                 multiclass=False, select_features=False, selection_params = None,
                 min_count=3, smallest_prob=0.01, min_probs_ratio=0.75):
        self.left_context_length = left_context_length
        self.right_context_length = right_context_length
        self.has_central_context = has_central_context
        self.select_features = True
        self.substrs = []
        self.classifier_name = classifier_name
        self.classifier_params = classifier_params
        self.multiclass = multiclass
        self.select_features = select_features
        self.selection_params = selection_params
        self.smallest_prob = smallest_prob
        self.min_probs_ratio = min_probs_ratio
        self.min_count = min_count
        # инициализация параметров
        self.classifiers = dict()
        self.classifier_classes = dict()
        self.predefined_answers = dict()
        self.selector_name = None
        self.selectors = dict()
        self.feature_codes = dict()
        self.features = []
        self.features_number = 0

    def _prepare_classifiers(self):
        if self.classifier_params is None:
            self.classifier_params = dict()
        self.classifier_name.set_params(**self.classifier_params)
        if self.select_features is not None:
            if self.selection_params is None:
                self.selection_params = {'nfeatures': 0.1, 'min_count': 2, 'minfeatures': 100}
            self.selection_params['method'] = self.select_features
            self.first_selector = MulticlassFeatureSelector(
                local=True, method=self.select_features, min_count=self.min_count, nfeatures=-1)
            feature_selector = MulticlassFeatureSelector(**self.selection_params)
            self.classifier_ = Pipeline([('selector', feature_selector),
                                         ('classifier', self.classifier_name)])
        else:
            self.first_selector = ZeroFeatureRemover()
            self.classifier_ = self.classifier_name
        return self

    def fit(self, X, y):
        """
        Применяет классификатор к данным.

        X --- набор троек вида (фрагмент, парадигма, левый контекст, правый контекст)
        """
        self._prepare_classifiers()
        substrs, X_train, y = self._preprocess_input(X, y, create_features=True,
                                                     retain_multiple=self.multiclass, return_y=True)
        X_train = self.first_selector.fit_transform(X_train, y)
        self._fit_classifiers(substrs, X_train, y)
        return self

    def _fit_classifiers(self, substrs, X_train, y):
        """
        Обучает классификаторы для каждой из подстрок, к которой планируется его применять
        """
        if issparse(X_train):
            X_train = X_train.tocsr()
        substr_indexes_in_train = _find_positions(substrs)
        for substr, indexes in substr_indexes_in_train.items():
            X_curr, y_curr = X_train[indexes,:], [y[i] for i in indexes]
            X_curr.sort_indices()
            max_index = max(y_curr)
            # в случае, если y_curr содержит только один класс, не обучаем классификатор
            if min(y_curr) == max_index:
                self.predefined_answers[substr] = max_index
                continue
            self.classifiers[substr] = clone(self.classifier_)
            self.classifiers[substr].fit(X_curr, y_curr)
            self.classifier_classes[substr] = self.classifiers[substr].classes_
        return

    def predict(self, X):
        probs = self._predict_proba(X)
        if not self.multiclass:
            class_indexes = [indices[0] for indices, _ in probs]
            answer = [[x] for x in np.take(self.classes_, class_indexes)]
        else:
            answer = [np.take(self.classes_, indices) for indices
                      in paradigm_classifier.extract_classes_from_sparse_probs(
                    probs, self.min_probs_ratio)]
        return answer

    def predict_proba(self, X):
        # probs = self._predict_proba(X)
        # answer = np.zeros(dtype=np.float64, shape=(len(X), len(self.classes_)))
        # for i, (indices, elem_probs) in enumerate(probs):
        #     answer[i][indices] = elem_probs
        # return answer
        return self._predict_proba(X, return_arrays=True)


    def _predict_proba(self, X, return_arrays=False):
        """
        Аргументы:
        ------------
        X: list of tuples, список троек вида (преобразуемая подстрока, левый контекст, правый контекст)

        Возвращает:
        ------------
        answer: list of tuples, список пар вида (indices, probs),
            indices: array, shape=(len(self.classes), dtype=int,
                список кодов трансформаций в порядке убывания их вероятности,
            probs: array, shape=(len(self.classes), dtype=float,
                вероятности трансформаций в порядке убывания.
        """
        substrs, X_test = self._preprocess_input(X, create_features=False, retain_multiple=False)
        X_test = self.first_selector.transform(X_test)
        substr_indexes = _find_positions(substrs)
        if return_arrays:
            answer = np.zeros(shape=(len(substrs), len(self.classes_)))
        else:
            answer = [None for _ in substrs]
        for substr, indexes in substr_indexes.items():
            # print(substr, end=" ")
            cls = self.classifiers.get(substr, None)
            if cls is not None:
                # X_curr = self.selectors[substr].transform(X_test[indexes])
                curr_smallest_prob = self.smallest_prob / len(cls.classes_)
                X_curr = X_test[indexes,:]
                probs = cls.predict_proba(X_curr)
                # probs[np.where(probs < curr_smallest_prob)] = 0.0
                probs[np.where(probs < self.smallest_prob)] = 0.0
                if return_arrays:
                    answer[np.ix_(indexes, cls.classes_)] = probs
                else:
                    for index, row in zip(indexes, probs):
                        nonzero_indices = row.nonzero()[0]
                        row = row[nonzero_indices]
                        indices_order = np.flipud(np.argsort(row))
                        current_indices = nonzero_indices[indices_order]
                        answer[index] = (np.take(cls.classes_, current_indices),
                                         row[indices_order] / row.sum())
                # for index, row in zip(indexes, probs):
                #     nonzero_indices = row.nonzero()[0]
                #     row = row[nonzero_indices]
                #     indices_order = np.flipud(np.argsort(row))
                #     current_indices = nonzero_indices[indices_order]
                #     elem_probs = row[indices_order]
                #     probs_sum, last_index = elem_probs[0], 1
                #     while probs_sum < 1.0 - self.smallest_prob:
                #         probs_sum += elem_probs[last_index]
                #         last_index += 1
                #     class_indexes = np.take(cls.classes_, current_indices[:(last_index)])
                #     answer[index] = (class_indexes, elem_probs[:(last_index)] / probs_sum)
            else:
                # такой подстроки не было в обучающей выборке,
                # поэтому для неё возращаем отсутствие замены
                code = self.predefined_answers.get(substr, 0)
                if return_arrays:
                    answer[indexes, code] = 1.0
                else:
                    for index in indexes:
                        answer[index] = ([code], 1.0)
        return answer


    def _preprocess_input(self, X, y=None, create_features=False,
                          retain_multiple=False, return_y=False, sparse_type='csr'):
        """
        Аргументы:
        ----------
        y: list of lists or None, список, i-ый элемент которого содержит классы
            для X[i] из обучающей выборки
        create_features: bool, optional(default=False), создаются ли новые признаки
            или им сопоставляется значение ``неизвестный суффикс/префикс''
        retain_multiple: bool, optional(default=False), может ли возвращаться
            несколько меток классов для одного объекта
        return_y: возвращается ли y (True в случае обработки обучающей выборки,
            т.к. в этом случае y преобразуется из списка списков в один список)
        """
        ## СОКРАТИТЬ
        if create_features:
            # создаём новые признаки, вызывается при обработке обучающей выборки
            self._create_basic_features()
            self.features_number = len(self.features)
        # создаём функцию для обработки неизвестных признаков
        ## ПЕРЕПИСАТЬ КАК В ОБЫЧНОМ КЛАССИФИКАТОРЕ
        # _process_unknown_feature = self._create_unknown_feature_processor(create_features)
        if y is None:
            y = [[None]] * len(X)
            retain_multiple = False
            y_new = list(chain(*y))
        else:
            self.classes_, y_new = np.unique(list(chain(*y)), return_inverse=True)
        X_with_temporary_codes = []
        for _, left_context, right_context in X:
            active_features_codes = []
            # обрабатываем левые контексты
            for length in range(1, self.left_context_length + 1):
                feature = left_context[-length: ]
                feature_code, to_break =\
                    self._get_feature_code(feature, (len(feature) != length),
                                           'left', create_new=create_features)
                active_features_codes.append(feature_code)
                if to_break:
                    break
            # правые контексты
            for length in range(1, self.right_context_length + 1):
                feature = right_context[:length]
                feature_code, to_break =\
                    self._get_feature_code(feature, (len(feature) != length),
                                           'right', create_new=create_features)
                active_features_codes.append(feature_code)
                if to_break:
                    break
            # центральный контекст
            if self.has_central_context:
                feature = left_context[-1:] + '#' + right_context[:1]
                if feature.startswith('#'):
                    feature = '^' + feature
                if feature.endswith('#'):
                    feature += '$'
                feature_code, to_break =\
                    self._get_feature_code(feature, False, 'center', create_new=create_features)
                if feature_code > 0:
                    active_features_codes.append(feature_code)
            X_with_temporary_codes.append(active_features_codes)
        if create_features:
            # при равной длине порядок x$, x#, $x, #x
            feature_order = sorted(enumerate(self.features),
                                   key=(lambda x:(len(x[1]), x[1].startswith('#'),
                                                  x[1].startswith('$'), x[1].endswith('#'))))
            self.features = [x[1] for x in feature_order]
            # перекодируем временные признаки
            temporary_features_recoding = [None] * self.features_number
            for new_code, (code, _) in enumerate(feature_order):
                temporary_features_recoding[code] = new_code
            for i, elem in enumerate(X_with_temporary_codes):
                X_with_temporary_codes[i] = [temporary_features_recoding[code] for code in elem]
            self.feature_codes = {feature: code for code, feature in enumerate(self.features)}
        # сохраняем данные для классификации
        rows, cols, curr_row = [], [], 0
        for codes, labels in zip(X_with_temporary_codes, y):
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
        X_train = sparse_matrix((data, (rows, cols)), shape=(curr_row, self.features_number))
        # сохраняем подстроки, которые и нужно классифицировать
        substrs = [x[0] for x in X]
        if return_y:
            return substrs, X_train, y_new
        else:
            return substrs, X_train

    def _create_basic_features(self):
        """
        Создаёт базовые признаки ещё до чтения входа

        Возвращает:
        -----------
        features: list, список созданных признаков
        """
        self.features, self.feature_codes = [], dict()
        #  признаки для контекстов, не встречавшихся в обучающей выборке
        # правые контексты
        self.features.extend(("-" * length + '#')
                             for length in range(1, self.right_context_length + 1))
        # короткие правые контексты
        self.features.extend(("-" * length + '$')
                             for length in range(self.right_context_length))
        # левые контексты
        self.features.extend(('#' + "-" * length)
                             for length in range(1, self.right_context_length + 1))
        # короткие левые контексты
        self.features.extend(('^' + "-" * length)
                             for length in range(self.right_context_length))
        if self.has_central_context:
            self.features.append('-#-')
        for code, feature in enumerate(self.features):
            self.feature_codes[feature] = code
        return

    # def _create_new_feature(self, feature, is_full, side):
    #     if is_full:
    #         delim = '^' if side == 'left' else '$'
    #         to_break = True
    #     else:
    #         delim, to_break = '#', False
    #     if side == 'left':
    #         feature = delim + feature
    #     elif side == 'right':
    #         feature += delim
    #     feature_code = self.feature_codes.get(feature, None)
    #     if feature_code is None:
    #         feature_code = self._process_unknown_feature(feature, side)
    #     return feature_code, to_break

    def _get_feature_code(self, feature, is_full, side, create_new=False):
        """
        Возвращает код признака feature
        """
        if is_full:
            delim = '^' if side == 'left' else '$'
            to_break = True
        else:
            delim, to_break = '#', False
        if side == 'left':
            feature = delim + feature
        elif side == 'right':
            feature += delim
        code = self.feature_codes.get(feature, -1)
        if code < 0:
            if create_new:
                self.features.append(feature)
                self.feature_codes[feature] = code = self.features_number
                self.features_number += 1
            else:
                if side == 'right':
                    partial_features = ['-' * start + feature[start:]
                                        for start in range(1, len(feature))]
                elif side == 'left':
                    partial_features = [feature[:(len(feature)-start)] + '-' * start
                                        for start in range(1, len(feature))][::-1]
                else:  # side == 'center'
                    partial_features = []
                    code, to_break = -1, None
                for partial_feature in partial_features:
                    code = self.feature_codes.get(partial_feature, -1)
                    if code > 0:
                        break
                if code > 0:
                    self.feature_codes[feature] = code
        return code, to_break

    # def _create_unknown_feature_processor(self, create_features):
    #     if create_features:
    #         def _process_unknown_feature(feature, context_type):
    #             self.features.append(feature)
    #             self.feature_codes[feature] = code = self.features_number
    #             self.features_number += 1
    #             return code
    #     else:
    #         def _process_unknown_feature(feature, context_type):
    #             if context_type == 'left':
    #                 delim = '$' if feature.startswith('$') else '#'
    #                 feature = delim + '-' * (len(feature) - 1)
    #             elif context_type == 'right':
    #                 delim = '^' if feature.endswith('^') else '#'
    #                 feature = '-' * (len(feature) - 1) + delim
    #             elif context_type == 'center':
    #                 center_pos = feature.find('#')
    #                 if center_pos == -1:
    #                     raise ValueError("Wrong feature: {0}".format(feature))
    #                 feature = '-' * center_pos + '#' + '-' * (len(feature) - center_pos - 1)
    #             else:
    #                 raise ValueError("context_type must be 'right' of 'left'")
    #             return self.feature_codes[feature]
    #     self._process_unknown_feature = _process_unknown_feature


class OptimalPositionSearcher:
    """
    Класс для поиска оптимальных позиций элементов шаблона
    """
    def __init__(self, transformations_by_paradigms, paradigms_by_codes,
                 classifier, classifier_params=None):
        """
        transformation_by_paradigms: dict,
            словарь вида (парадигма: список локальных трансформаций)
        paradigm_codes: dict, словарь вида (код: парадигма)
        """
        self.transformations_by_paradigms = transformations_by_paradigms
        self.paradigms_by_codes = paradigms_by_codes
        self.classifier = classifier
        self.classifier_params = classifier_params
        # сохранённые фрагменторы
        self.fragmentors_by_descrs = dict()
        self.fragmentors_by_codes = dict()
        self.first_form_fragmentors_by_codes = dict()
        # для сохранения кодов трансформаций
        self.transformation_classes = dict()
        self.transformation_classes_number = 0

    def fit(self, X, y=None):
        """
        Производит предвычисление фрагменторов и обучение классификатора
        """
        # предвычисляем фрагменторы
        for code, paradigm in self.paradigms_by_codes.items():
            # создаём фрагментор для первого шаблона в парадигме
            basic_descr = paradigm.split('#')[0]
            current_fragmentor = self.fragmentors_by_descrs.get(basic_descr)
            if current_fragmentor is None:
                current_fragmentor =\
                    self.fragmentors_by_descrs[basic_descr] = ParadigmFragment(basic_descr)
            self.fragmentors_by_codes[code] = current_fragmentor
            # создаём фрагментор для начальной формы
            _, first_form_descr = get_first_form_pattern(paradigm)
            self.first_form_fragmentors_by_codes[code] = ParadigmFragment(first_form_descr)
        # обучаем классификатор
        if self.classifier_params is None:
            self.classifier_params = dict()
        self.classifier.fit(X, y)
        self.transformation_classes = {label: i for i, label
                                       in enumerate(self.classifier.classes_)}
        return self

    def _predict_constant_spans(self, paradigm_codes, words, return_scores=False):
        # вначале предсказываем позиции постоянных фрагментов
        constant_fragment_positions, possible_spans = [], []
        transformation_classes = []
        # трёхмерный массив для логарифмических вероятностей
        # scores_by_words[i][j][k] = log(p_{ijk})
        # i --- номер слова, j --- номер трансформируемого фрагмента
        # k --- номер варианта расположения этого фрагмента
        indexes_for_scores, scores_by_words = [], []
        possible_spans = []
        for i, (code, word) in enumerate(zip(paradigm_codes, words)):
            # print("{0} {1}".format(self.paradigms_by_codes[code].split('#')[0], word))
            # print(self.paradigms_by_codes[code])
            fragmentor = self.fragmentors_by_codes[code]
            curr_transformation_codes = self.transformations_by_paradigms[code]
            transformation_classes.append([self.transformation_classes[trans_code]
                                           for trans_code in curr_transformation_codes])
            curr_constant_fragment_positions =\
                fragmentor.find_constant_fragments_positions(word)
            if len(curr_constant_fragment_positions) > len(curr_transformation_codes):
                curr_constant_fragment_positions = curr_constant_fragment_positions[1:]
            constant_fragment_positions.append(curr_constant_fragment_positions)
            curr_scores_by_words = [None] * len(curr_constant_fragment_positions)
            for j, elem in enumerate(curr_constant_fragment_positions):
                curr_scores_by_words[j] = [0] * len(elem)
                possible_spans.extend(((word[start:end], word[:start], word[end:])
                                       for start, end in elem))
                indexes_for_scores.extend(((i, j, k) for k in range(len(elem))))
            scores_by_words.append(curr_scores_by_words)
        probs = self.classifier.predict_proba(possible_spans)
        scores = [curr_probs[transformation_classes[i][j]]
                  for curr_probs, (i, j, _) in zip(probs, indexes_for_scores)]
        scores = np.array(scores)
        # scores = np.zeros(shape=(len(probs),), dtype=float)
        # for m, ((i, j, _, _), curr_probs) in enumerate(zip(data_for_probs, probs)):
        #     scores[m] = curr_probs[transformation_classes[i][j]]
        # sys.exit()
        # for (i, j, k, triple), score in zip(data_for_probs, scores):
        #     if score > 0.01:
        #         print("{4}\t{1}_{0}_{2}\t{3:.4f}".format(triple[0], triple[1], triple[2], score,
        #                                                  transformation_codes[i][j]))
        scores = np.where(scores > self.classifier.smallest_prob, scores, self.classifier.smallest_prob)
        scores = np.log(scores)
        # здесь надо снова разобрать по словам
        for (i, j, k), score in zip(indexes_for_scores, scores):
            scores_by_words[i][j][k] = score
        best_positions = [self._find_best_positions(scores, positions, return_scores=return_scores)
                          for scores, positions in zip(scores_by_words, constant_fragment_positions)]
        if return_scores:
            best_positions, scores = zip(*best_positions)
            return list(best_positions), list(scores)
        else:
            return best_positions

    @classmethod
    def _find_best_positions(cls, scores, fragment_positions, return_scores=False):
        """
        Вычисляет наилучший набор индексов по стоимостям отдельных трансформаций
        """
        # предвычисляем индексы
        minimal_next_indexes =\
            [find_first_larger_indexes([first[1] for first in elem],
                                       [second[0] for second in fragment_positions[i+1]])
             for i, elem in enumerate(fragment_positions[:-1])]
        curr_cumulative_score = scores[-1]
        next_indexes = [[None] * len(elem) for elem in scores[:-1]]
        for i, (curr_scores, curr_minimal_next_indexes) in\
                list(enumerate(zip(scores[:-1], minimal_next_indexes)))[::-1]:
            tail_maximal_indexes = list(range(len(curr_cumulative_score)))
            last_index = len(curr_cumulative_score) - 1
            for j, elem in list(enumerate(curr_cumulative_score[:-1]))[::-1]:
                if curr_cumulative_score[j] > curr_cumulative_score[last_index]:
                    # tail_maximal_indexes[j] = j
                    last_index = j
                else:
                    tail_maximal_indexes[j] = last_index
            prev_cumulative_score = curr_cumulative_score
            curr_cumulative_score = curr_scores
            for j, index in enumerate(curr_minimal_next_indexes):
                best_next_index = tail_maximal_indexes[index]
                curr_cumulative_score[j] += prev_cumulative_score[best_next_index]
                next_indexes[i][j] = best_next_index
        best_start_index = np.argmax(curr_cumulative_score)
        best_indexes = [best_start_index]
        for elem in next_indexes:
            best_indexes.append(elem[best_indexes[-1]])
        answer = [elem[index] for elem, index in zip(fragment_positions, best_indexes)]
        if return_scores:
            return answer, curr_cumulative_score[best_start_index]
        else:
            return answer

    def predict_variables(self, paradigm_codes, words):
        # в случае, если подали один код и одно слово, преобразуем в списки
        if isinstance(paradigm_codes, (int, np.integer)):
            paradigm_codes = [paradigm_codes]
        if isinstance(words, str):
            words = [words]
        constant_spans = self._predict_constant_spans(paradigm_codes, words)
        answer = [extract_uncovered_substrings(word, spans)
                  for word, spans in zip(words, constant_spans)]
        return answer

    def predict_first_form(self, paradigm_codes, words):
        var_values = self.predict_variables(paradigm_codes, words)
        return [self.first_form_fragmentors_by_codes[code].substitute(values)
                for code, values in zip(paradigm_codes, var_values)]


def _find_positions(substrs):
    answer = defaultdict(list)
    for i, substr in enumerate(substrs):
        answer[substr].append(i)
    return answer


def extract_uncovered_substrings(word, spans):
    """
    Извлекает подстроки, не покрытые сегментами из spans
    """
    last, var_values = 0, []
    for start, end in spans:
        var_values.append(word[last:start])
        last = end
    return var_values



def compare_quality(substrs, Y_pred, Y_test, measure=skm.f1_score):
    Y_pred = np.fromiter(chain.from_iterable(Y_pred), dtype=int)
    Y_test = np.fromiter(chain.from_iterable(Y_test), dtype=int)
    substr_indexes = _find_positions(substrs)
    accuracies = dict()
    for substr, indexes in substr_indexes.items():
        curr_pred = Y_pred[indexes]
        curr_test = Y_test[indexes]
        indexes_to_compare = np.where(curr_test > 0)[0]
        test_to_compare = curr_test[indexes_to_compare]
        pred_to_compare = curr_pred[indexes_to_compare]
        curr_pred_binary = (curr_pred > 0)
        curr_test_binary = (curr_test > 0)
        if np.any(curr_test_binary) or np.any(curr_pred_binary):
            score = measure(curr_test, curr_pred, pos_label=None, average='micro')
            simple_score = measure(curr_test_binary, curr_pred_binary)
        else:
            score, simple_score = 1.0, 1.0
        accuracies[substr] = (score, simple_score, np.count_nonzero(curr_pred_binary),
                              np.count_nonzero(curr_test_binary), len(curr_test_binary))
    return accuracies

def output_results(outfile, accuracies):
    with open(outfile, "w", encoding="utf8") as fout:
        for substr, score in sorted(accuracies.items()):
            fout.write("{0}\t{1:.2f}\t{2:.2f}\t{3}\t{4}\t{5}\n".format(substr, *score))

if __name__ == "__main__":
    args = sys.argv[1:]
    lemmas_file, codes_file, trans_codes_outfile, results_outfile = args
    lemmas_data = process_lemmas_file(lemmas_file)
    codes_data = process_codes_file(codes_file)
    paradigm_codes, paradigm_counts = read_paradigms(codes_data)
    transformations_handler = TransformationsHandler(paradigm_codes, paradigm_counts)
    transformations_handler.output_transformations(trans_codes_outfile)
    ltc = LocalTransformationClassifier(
        left_context_length=2, right_context_length=3, select_features='ambiguity',
        selection_params={'nfeatures': 0.1, 'min_count': 3, 'min_features': 100})
    X, y = transformations_handler.create_training_data(lemmas_data)
    X_train, X_test, Y_train, Y_test = skcv.train_test_split(X, y, test_size=0.25, random_state=156)
    Y_train, Y_test = [[x] for x in Y_train], [[x] for x in Y_test]
    # print("Fitting...")
    ltc.fit(X_train, Y_train)
    # print("Predicting...")
    Y_pred = ltc.predict(X_test)
    substrs = [elem[0] for elem in X_test]
    accuracies = compare_quality(substrs, Y_pred, Y_test)
    output_results(results_outfile, accuracies)
    print(ltc._predict_proba([('я', 'ассоциаци', '')]))
