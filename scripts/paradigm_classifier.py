#-------------------------------------------------------------------------------
# Name:        paradigm_classifier.py
#-------------------------------------------------------------------------------

from itertools import chain

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm as sksvm
from sklearn import linear_model as sklm
from scipy.sparse import csr_matrix

from paradigm_detector import ParadigmFragment, fit_to_patterns
from feature_selector import MulticlassFeatureSelector

class ParadigmClassifier(BaseEstimator, ClassifierMixin):
    '''
    Класс, применяющий метод опорных векторов для классификации парадигм
    '''
    def __init__(self, paradigm_table, multiclass=False, max_length=3,
                 use_prefixes=False, SVM_params = None, selection_method='ambiguity',
                 nfeatures=None, feature_fraction=1.0,
                 smallest_prob = 0.001, max_probs_ratio=0.9):
        self.paradigm_table = paradigm_table
        self.multiclass = multiclass
        self.max_length = max_length
        self.use_prefixes = use_prefixes
        self.SVM_params = SVM_params
        self.is_fitted = False
        self.selection_method = selection_method
        self.nfeatures = nfeatures
        self.feature_fraction = feature_fraction
        self.smallest_prob = smallest_prob
        self.max_probs_ratio = max_probs_ratio

    def fit(self, X, y):
        if self.SVM_params is None:
            self.SVM_params = dict()
        self.classifier = sklm.LogisticRegression(**self.SVM_params)
        self._prepare_fragmentors()

        self._collect_features(X)
        X_train, y = self._preprocess_input(X, y, retain_multiple=self.multiclass, return_y=True)
        N, _ = X_train.shape

        if self.nfeatures is None:
            try:
                self.feature_fraction = float(self.feature_fraction)
            except:
                self.feature_fraction = 1.0
            self.nfeatures = int(self.features_number * self.feature_fraction)
        X_train = self._select_features(X_train, y)
        print("Training...")
        self.classifier.fit(X_train, y)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        print("Predicting...")
        probs = self.predict_proba(X)
        if not self.multiclass:
            answer = [[x] for x in np.take(self.classes_, np.argmax(probs, axis=1))]
        else:
            answer = self._extract_classes_from_probs(probs)
        return answer

    def predict_proba(self, X):
        print("Predicting probabilities...")
        X_train = self._preprocess_input(X, retain_multiple=False)
        X_train = self.selector.transform(X_train)
        probs = self.classifier.predict_proba(X_train)
        answer = np.zeros(dtype=np.float64, shape=probs.shape)
        for i, (word, word_probs, final_word_probs) in enumerate(zip(X, probs, answer)):
            probs_sum = 0.0
            while True:
                best_class_index = word_probs.argmax()
                best_class = self.classes_[best_class_index]
                fragmentor = self._paradigm_fragmentors[best_class]
                variable_values = fragmentor.extract_variables(word)
                prob = word_probs[best_class_index]
                if prob < probs_sum * self.smallest_prob / (1.0 - self.smallest_prob):
                    final_word_probs /= probs_sum
                    break
                if len(variable_values) > 0:
                    final_word_probs[best_class_index] = prob
                    probs_sum += prob
                word_probs[best_class_index] = 0.0
        return answer

    def _collect_features(self, X):
        '''
        Извлекаем имена признаков из обучающих данных
        '''
        features = ["-" * length + "#"
                    for length in range(1, self.max_length+1)]
        if self.use_prefixes:
            features.extend((("#" + "-" * length) for length in range(1, self.max_length+1)))
        for word in X:
            features.extend(((word[-length: ] + '#')
                             for length in range(1, min(self.max_length, len(word))+1)))
            if self.use_prefixes:
                features.extend((('#' + word[:length])
                                 for length in range(1, min(self.max_length, len(word))+1)))
        # сортируем признаки по возрастанию длины, чтобы суффиксы шли раньше префиксов
        self.features = sorted(features, key=(lambda x:(len(x), x[0] == '#')))
        self.feature_codes = {feature: code for code, feature in enumerate(self.features)}
        self.features_number = len(self.features)
        return

    def _preprocess_input(self, X, y=None, retain_multiple=False, return_y=False):
        """
        Преобразует список слов в матрицу X_train
        X_train[i][j] = 1 <=> self.features[i] --- подслово #X[i]#
        """
        if y is None:
            y = [[None]] * len(X)
            retain_multiple = False
        y_new = list(chain(*y))
        rows, cols, curr_row = [], [], 0
        for (word, labels) in zip(X, y):
            active_features_codes = []
            for length in range(min(self.max_length, len(word))):
                feature = word[-length-1:] + "#"
                feature_code = self.feature_codes.get(feature, None)
                if feature_code is None:
                    feature = "-" * (length + 1) + "#"
                    feature_code = self.feature_codes[feature]
                active_features_codes.append(feature_code)
            if self.use_prefixes:
                for length in range(min(self.max_length, len(word))):
                    feature = "#" + word[:(length+1)]
                    feature_code = self.feature_codes.get(feature, None)
                    if feature_code is None:
                        feature = "#" + "-" * (length + 1)
                        feature_code = self.feature_codes[feature]
                    active_features_codes.append(feature_code)
            number_of_rows_to_add = 1 if not(retain_multiple) else len(labels)
            for i in range(number_of_rows_to_add):
                rows.extend([curr_row for _ in active_features_codes])
                cols.extend(active_features_codes)
                curr_row += 1
        data = np.ones(shape=(len(rows,)), dtype=float)
        X_train = csr_matrix((data, (rows, cols)),
                             shape=(curr_row, self.features_number))
        if return_y:
            return X_train, y_new
        else:
            return X_train

    def _add_feature(self, feature):
        '''
        Добавление нового признака
        '''
        self.features.append(feature)
        self.feature_codes[feature] = self.features_number
        self.features_number += 1

    def _select_features(self, X, y):
        self.selector = MulticlassFeatureSelector(
                local=True, method=self.selection_method, min_count=3, nfeatures=self.nfeatures)
        X = self.selector.fit_transform(X, y)
        self._active_feature_codes = self.selector.get_support(indices=True)
        return X

    def _prepare_fragmentors(self):
        '''
        Предвычисление обработчиков парадигм
        '''
        self._paradigm_fragmentors = dict()
        for paradigm, code in self.paradigm_table.items():
            pattern = paradigm.split("#")[0]
            if pattern == "-":
                pattern = paradigm.split("#")[1]
            self._paradigm_fragmentors[code] = ParadigmFragment(pattern)
        return

    def _get_possible_paradigm_codes(self, word):
        """
        Поиск парадигм, под которые подходит слово
        """
        answer = []
        for code, fragmentor in self._paradigm_fragmentors.items():
            variable_values = fragmentor.extract_variables(word)
            if len(variable_values) > 0:
                answer.append((code, variable_values))
        return answer

    def _extract_classes_from_probs(self, probs):
        """
        Возвращает список классов по их вероятностям
        в случае мультиклассовой классификации
        """
        max_allowed_probs = np.max(probs, axis=1).reshape((probs.shape[0], 1)) * self.max_probs_ratio
        remaining_positions = np.where(probs >= max_allowed_probs)
        answer = [[] for _ in range(probs.shape[0])]
        for row, col in zip(*remaining_positions):
            answer[row].append(col)
        return answer

















