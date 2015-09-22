#-------------------------------------------------------------------------------
# Name:        learn_paradigms.py
# Purpose:     автоматическое определение парадигмы слова
#
# Created:     03.09.2015
#-------------------------------------------------------------------------------

import sys
import os
import numpy as np
import sklearn.metrics as skm

from paradigm_classifier import ParadigmClassifier
import sklearn.cross_validation as skcv


max_lengths = [3] # максимальная длина суффиксов/префиксов, используемых в качестве признаков
feature_fractions = [0.05, 0.1, 0.3] # доля признаков, используемых для классификации
fractions = [0.1, 0.3, 0.5, 0.7] # доля данных, попадающих в обучающую выборку

def read_paradigms(infile):
    '''
    Читает файл со списком парадигм
    '''
    answer = dict()
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split()
            if len(splitted_line) != 2:
                continue
            code, pattern = splitted_line
            answer[pattern] = int(code)
    return answer

def read_input(infile):
    '''
    Читает входные данные
    '''
    data, labels = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split()
            if len(splitted_line) < 2:
                continue
            word, code = splitted_line[:2]
            data.append(word)
            labels.append(int(code))
    return data, labels

def cv_mode(paradigms_file, infile, fraction, nfolds=0,
            selection_method=None, feature_fraction=None, nfeatures=None):
    '''
    Определяет качество классификатора с заданными параметрами
    по скользящему контролю на обучающей выборке

    Параметры:
    -----------
    paradigms_file: str, путь к файлу с парадигмами
    infile: str, путь к файлу с обучающей выборкой
    fraction: float, доля обучающей выборки
    nfolds: int, optional(default=0)
        число разбиений, по которым производится усреднение при скользящем контроле
        nfolds=0 --- в обучающую выборку попадает соответствующее число лексем
                     из начала файла
    selection_method: str or None, optional (default=None),
        метод отбора признаков
    feature_fraction: float or None, optional (default=None),
        доля признаков, которые следует оставить при отборе признаков
        (при этом nfeatures должно быть не задано)
    nfeatures: float or None, optional (default=None),
        число признаков, которые следует оставить при отборе признаков
    '''
    pattern_table = read_paradigms(paradigms_file)
    data, labels = read_input(infile)
    svm_params = None
    if selection_method is None:
        selection_method = 'ambiguity'
    if nfolds == 0:
        train_data_length = int(fraction * len(data))
        train_data, test_data = data[:train_data_length], data[train_data_length:]
        train_labels, test_labels = labels[:train_data_length], labels[train_data_length:]
        test_data, test_labels = [test_data], [test_labels]
        train_data, train_labels = [train_data],[train_labels]
        predictions = [None]
    else:
        test_data, train_data = [None] * nfolds, [None] * nfolds
        test_labels, train_labels = [None] * nfolds, [None] * nfolds
        predictions = [None] * nfolds
        for fold in range(nfolds):
            train_data[fold], test_data[fold], train_labels[fold], test_labels[fold] =\
                skcv.train_test_split(data, labels, test_size = 1.0 - fraction,
                                      random_state = 100 * fold + 13)
    # задаём классификатор
    cls = ParadigmClassifier(pattern_table, max_length=max_length, use_prefixes=True,
                             SVM_params=svm_params, selection_method=selection_method,
                             nfeatures=nfeatures, feature_fraction=feature_fraction)
    # применяем классификатор к данным
    for i, (train_sample, train_labels_sample, test_sample, test_labels_sample) in\
            enumerate(zip(train_data, train_labels, test_data, test_labels)):
        cls.fit(train_sample, train_labels_sample)
        predictions[i] = cls.predict(test_sample)
    # вычисляем точность и обрабатываем результаты
    accuracies = [skm.accuracy_score(first, second)
                  for first, second in zip(test_labels, predictions)]
    confusion_matrices = [skm.confusion_matrix(first, second, labels=cls.classes_)
                          for first, second in zip(test_labels, predictions)]
    print("{0}\t{1:<.2f}\t{2}\t{3:<.2f}".format(
            max_length, fraction, cls.nfeatures, 100 * np.mean(accuracies)))
    # сохраняем матрицы ошибок классификации
    confusion_matrices_folder = "confusion_matrices"
    if not os.path.exists(confusion_matrices_folder):
        os.makedirs(confusion_matrices_folder)
    dest = os.path.join(confusion_matrices_folder,
                        "confusion_matrix_{0}_{1:<.2f}_{2:<.2f}.out".format(
                            max_length, fraction, feature_fraction))
    with open(dest, "w") as fout:
        fout.write("{0:<4}".format("") +
                   "".join("{0:>4}".format(label) for label in cls.classes_) + "\n")
        for label, elem in zip(cls.classes_, confusion_matrices[0]):
            nonzero_positions = np.nonzero(elem)
            nonzero_counts = np.take(elem, nonzero_positions)[0]
            nonzero_labels = np.take(cls.classes_, nonzero_positions)[0]
            fout.write("{0:<4}\t".format(label))
            fout.write("\t".join("{0}:{1}".format(*pair)
                                 for pair in sorted(zip(nonzero_labels, nonzero_counts),
                                                    key=(lambda x: x[1]), reverse=True))
                       + "\n")
    return

'''
Запускать learn_paradigms.py cross-validation paradigm_codes.out lemma_paradigm_codes.out ambiguity
'''

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        sys.exit("You should pass mode as first argument")
    mode, args = args[0], args[1:]
    if mode not in ["train", "cross-validation"]:
        sys.exit("Mode should be train or cross-validation")
    if mode =="train":
        if len(args) not in range(3, 6):
            sys.exit("Pass 'train', coding file, train and test files,"
                     "[feature selection algorithm], [number of features to select]")
        paradigm_file, train_file, test_file = args[:3]
        selection_method = args[3] if len(args) > 3 else None
        nfeatures = int(args[4]) if len(args) > 4 else None
        train_test_mode(paradigm_file, train_file, test_file,
                        selection_method=selection_method, nfeatures=nfeatures)
    elif mode == "cross-validation":
        if len(args) not in range(3, 5):
            sys.exit("Pass 'cross-validation', coding file, train file and number of folds,"
                     "[feature selection algorithm]")
        paradigm_file, infile, nfolds = args[:3]
        selection_method = args[3] if len(args) > 3 else None
        nfolds = int(nfolds)
        # nfeatures = int(args[4]) if len(args) > 4 else -1
        global max_length
        for max_length in max_lengths:
            for fraction in fractions:
                for feature_fraction in feature_fractions:
                    cv_mode(paradigm_file, infile, fraction=fraction, nfolds=nfolds,
                            selection_method=selection_method, feature_fraction=feature_fraction)




