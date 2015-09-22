#-------------------------------------------------------------------------------
# Name:        paradigm_classifier.py
#-------------------------------------------------------------------------------

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm as sksvm
from scipy.sparse import csr_matrix

from paradigm_detector import ParadigmFragment, fit_to_patterns
from feature_selector import MulticlassFeatureSelector

class ParadigmClassifier(BaseEstimator, ClassifierMixin):
    '''
    Класс, применяющий метод опорных векторов для классификации парадигм
    '''
    def __init__(self, paradigm_table, max_length, use_prefixes=False, SVM_params = None,
                 selection_method='ambiguity', nfeatures=None, feature_fraction=1.0):
        self.paradigm_table = paradigm_table
        self.max_length = max_length
        self.use_prefixes = use_prefixes
        self.SVM_params = SVM_params
        self.is_fitted = False
        self.selection_method = selection_method
        self.nfeatures = nfeatures
        self.feature_fraction = feature_fraction

    def fit(self, X, y):
        if self.SVM_params is None:
            self.SVM_params = {}
        self.classifier = sksvm.LinearSVC(**self.SVM_params)
        self._prepare_fragmentors()

        self._collect_features(X)
        X_train = self._preprocess_input(X)
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
        X_train = self._preprocess_input(X)
        X_train = self.selector.transform(X_train)
        scores = self.classifier.decision_function(X_train)
        answer = [None] * len(X)
        for i, (word, score) in enumerate(zip(X, scores)):
            while True:
                best_class_index = score.argmax()
                fragmentor = self._paradigm_fragmentors[self.classes_[best_class_index]]
                variable_values = fragmentor.extract_variables(word)
                if len(variable_values) > 0:
                    answer[i] = best_class_index
                    break
                else:
                    score[best_class_index] = -np.inf
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

    def _preprocess_input(self, X):
        '''
        Преобразует список слов в матрицу X_train
        X_train[i][j] = 1 <=> self.features[i] --- подслово #X[i]#
        '''
        rows, cols = [], []
        for i, word in enumerate(X):
            for length in range(min(self.max_length, len(word))):
                feature = word[-length-1:] + "#"
                feature_code = self.feature_codes.get(feature, None)
                if feature_code is None:
                    feature = "-" * (length + 1) + "#"
                    feature_code = self.feature_codes[feature]
                rows.append(i)
                cols.append(feature_code)
            if self.use_prefixes:
                for length in range(min(self.max_length, len(word))):
                    feature = "#" + word[:(length+1)]
                    feature_code = self.feature_codes.get(feature, None)
                    if feature_code is None:
                        feature = "#" + "-" * (length + 1)
                        feature_code = self.feature_codes[feature]
                    rows.append(i)
                    cols.append(feature_code)
        data = np.ones(shape=(len(rows,)), dtype=float)
        X_train = csr_matrix((data, (rows, cols)),
                             shape=(len(X), self.features_number))
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
        '''
        Поиск парадигм, под которые подходит слово
        '''
        answer = []
        for code, fragmentor in self._paradigm_fragmentors.items():
            variable_values = fragmentor.extract_variables(word)
            if len(variable_values) > 0:
                answer.append((code, variable_values))
        return answer














