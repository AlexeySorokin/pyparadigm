#-------------------------------------------------------------------------------
# Name:        learn_paradigms.py
# Purpose:     автоматическое определение парадигмы слова
#
# Created:     03.09.2015
#-------------------------------------------------------------------------------

import sys
import os
from collections import defaultdict, OrderedDict
from itertools import chain
import getopt

import numpy as np
import sklearn.metrics as skm
import sklearn.cross_validation as skcv
from sklearn.preprocessing import MultiLabelBinarizer
from paradigm_classifier import ParadigmClassifier


max_lengths = [3] # максимальная длина суффиксов/префиксов, используемых в качестве признаков
feature_fractions = [0.3] # доля признаков, используемых для классификации
fractions = [0.1, 0.3] # доля данных, попадающих в обучающую выборку

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

def read_input(infile, multiclass=False):
    """
    Читает входные данные
    """
    data = OrderedDict()
    # в случае, когда одна лексема может иметь несколько парадигм,
    # сохраняем все такие парадигмы
    if not multiclass:
        def add(lst, x):
            if len(lst) == 0:
                lst = [x]
    else:
        def add(lst, x):
            lst.append(x)
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split()
            if len(splitted_line) < 2:
                continue
            word, code = splitted_line[:2]
            code = int(code)
            if word in data:
                add(data[word], code)
            else:
                data[word] = [code]
    return list(data.keys()), list(data.values())

def cv_mode(testing_mode, multiclass, paradigm_file, infile, fraction, nfolds=0,
            selection_method=None, feature_fraction=None, nfeatures=None,
            output_train_dir=None, output_pred_dir=None):
    '''
    Определяет качество классификатора с заданными параметрами
    по скользящему контролю на обучающей выборке

    Параметры:
    -----------
    testing_mode: str ('predict' or  'predict_proba'), режим использования
    multiclass: bool, может ли одно слово иметь несколько парадигм
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
    pattern_table = read_paradigms(paradigm_file)
    data, labels = read_input(infile,multiclass=multiclass)
    classes = sorted(set(chain(*labels)))
    svm_params = None
    if selection_method is None:
        selection_method = 'ambiguity'
    if nfolds == 0:
        train_data_length = int(fraction * len(data))
        train_data, test_data = data[:train_data_length], data[train_data_length:]
        train_labels, test_labels = labels[:train_data_length], labels[train_data_length:]
        test_data, test_labels = [test_data], [test_labels]
        train_data, train_labels = [train_data],[train_labels]
        predictions, prediction_probs = [None], [None]
        classes_by_cls = [None]
    else:
        test_data, train_data = [None] * nfolds, [None] * nfolds
        test_labels, train_labels = [None] * nfolds, [None] * nfolds
        predictions = [None] * nfolds
        prediction_probs = [None] * nfolds
        classes_by_cls = [None] * nfolds
        for fold in range(nfolds):
            train_data[fold], test_data[fold], train_labels[fold], test_labels[fold] =\
                skcv.train_test_split(data, labels, test_size = 1.0 - fraction,
                                      random_state = 100 * fold + 13)
    # задаём классификатор
    cls = ParadigmClassifier(pattern_table, multiclass=multiclass,
                             max_length=max_length, use_prefixes=True,
                             SVM_params=svm_params, selection_method=selection_method,
                             nfeatures=nfeatures, feature_fraction=feature_fraction)
    # сохраняем тестовые данные
    if output_train_dir is not None:
        if not os.path.exists(output_train_dir):
            os.makedirs(output_train_dir)
        for i, (train_sample, train_labels_sample) in\
                enumerate(zip(train_data, train_labels), 1):
            write_data(os.path.join(output_train_dir, "{0}.data".format(i)),
                       train_sample, train_labels_sample)
    # применяем классификатор к данным
    for i, (train_sample, train_labels_sample, test_sample, test_labels_sample) in\
            enumerate(zip(train_data, train_labels, test_data, test_labels)):
        cls.fit(train_sample, train_labels_sample)
        classes_by_cls[i] = cls.classes_
        if testing_mode == 'predict':
            predictions[i] = cls.predict(test_sample)
        elif testing_mode == 'predict_proba':
            prediction_probs[i] = cls.predict_proba(test_sample)
            predictions[i] = cls._extract_classes_from_probs(prediction_probs[i])
    # вычисляем точность и обрабатываем результаты
    transformer = ((lambda x: [y[0] for y in x])
                   if not multiclass else MultiLabelBinarizer(classes).fit_transform)
    firsts = [transformer(first) for first in test_labels]
    seconds = [transformer(second) for second in predictions]
    accuracies = [skm.accuracy_score(first, second)
                  for first, second in zip(firsts, seconds)]
    if not multiclass:
        confusion_matrices = [skm.confusion_matrix(first, second, labels=classes)
                              for first, second in zip(firsts, seconds)]
    print("{0}\t{1:<.2f}\t{2}\t{3:<.2f}".format(
            max_length, fraction, cls.nfeatures, 100 * np.mean(accuracies)))
    # сохраняем результаты классификации
    if output_pred_dir is not None:
        if not os.path.exists(output_pred_dir):
            os.makedirs(output_pred_dir)
        if testing_mode == 'predict':
            for i, (test_sample, pred_labels_sample, true_labels_sample) in\
                    enumerate(zip(test_data, predictions, test_labels), 1):
                write_data(os.path.join(output_pred_dir, "{0}.data".format(i)),
                           test_sample, pred_labels_sample, true_labels_sample)
        elif testing_mode == 'predict_proba':
            for i, (test_sample, pred_probs_sample, labels_sample, cls_classes) in\
                    enumerate(zip(test_data, prediction_probs, test_labels, classes_by_cls), 1):
                write_probs_data(os.path.join(output_pred_dir, "{0}.prob".format(i)),
                                 test_sample, pred_probs_sample, cls_classes, labels_sample)
    # сохраняем матрицы ошибок классификации
    if not multiclass and nfolds <= 1:
        confusion_matrices_folder = "confusion_matrices"
        if not os.path.exists(confusion_matrices_folder):
            os.makedirs(confusion_matrices_folder)
        dest = os.path.join(confusion_matrices_folder,
                            "confusion_matrix_{0}_{1:<.2f}_{2:<.2f}.out".format(
                                max_length, fraction, feature_fraction))
        with open(dest, "w", encoding="utf8") as fout:
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

def write_data(outfile, data, labels, true_labels=None):
    if true_labels is None:
        with open(outfile, "w", encoding="utf8") as fout:
            for elem, label in zip(data, labels):
                fout.write("{0} {1}\n".format(elem, ",".join("{0}".format(x) for x in label)))
    else:
        with open(outfile, "w", encoding="utf8") as fout:
            for elem, true_label, label in zip(data, true_labels, labels):
                fout.write("{0} {1} {2}\n".format(elem,
                                                  ",".join("{0}".format(x) for x in true_label),
                                                  ",".join("{0}".format(x) for x in label)))
    return

def write_probs_data(outfile, data, probs, codes, true_labels):
    with open(outfile, "w", encoding="utf8") as fout:
        for i, (elem, elem_probs) in enumerate(zip(data, probs)):
            nonzero_positions = np.nonzero(elem_probs)
            nonzero_codes = np.take(codes, nonzero_positions)[0]
            nonzero_probs = np.take(elem_probs, nonzero_positions)[0]
            if true_labels is None:
                to_write = "{0} {1}\n".format(
                    elem, " ".join("{0}:{1:<.3f}".format(*pair) for pair in
                                   sorted(zip(nonzero_codes, nonzero_probs),
                                          key=(lambda x:x[1]), reverse=True)))
            else:
                to_write = "{0} {1} {2}\n".format(
                    elem, true_labels[i], ' '.join("{0}:{1:<.3f}".format(*pair) for pair in
                                                   sorted(zip(nonzero_codes, nonzero_probs),
                                                          key=(lambda x:x[1]), reverse=True)))
            fout.write(to_write)
    return

'''
Запускать learn_paradigms.py cross-validation predict paradigm_codes.out lemma_paradigm_codes.out ambiguity
'''

SHORT_OPTIONS = 'mpT:O:'
LONG_OPTIONS = ['multiclass', 'probs', 'train_output=', 'test_output=']

if __name__ == '__main__':
    args = sys.argv[1:]
    # значения по умолчанию для режима тестирования
    # и поддержки множественнных парадигм для одного слова
    testing_mode, multiclass = 'predict', False
    output_train_dir = "train_data/"
    output_pred_dir = "test_results/"
    # опции командной строки
    opts, args = getopt.getopt(args, SHORT_OPTIONS, LONG_OPTIONS)
    for opt, val  in opts:
        if opt in ['-m', '--multiclass']:
            multiclass = True
        elif opt in ['-p', '--probs']:
            testing_mode = 'predict_proba'
        elif opt in ['-T', '--train_output']:
            output_train_dir = val
        elif opt in ['-O', '--test_output']:
            output_pred_dir = val
    # аргументы командной строки
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
        if len(args) not in range(3,5):
            sys.exit("Pass 'cross-validation', coding file, train file, "
                     "number of folds[, feature selection algorithm]")
        paradigm_file, infile, nfolds = args[:3]
        selection_method = args[3] if len(args) > 3 else None
        nfolds = int(nfolds)
        # nfeatures = int(args[4]) if len(args) > 4 else -1
        global max_length
        for max_length in max_lengths:
            for fraction in fractions:
                for feature_fraction in feature_fractions:
                    output_train_dir_ =\
                        os.path.join(output_train_dir,
                                     "{0}_{1:.1f}_{2:.1f}".format(max_length, fraction, feature_fraction))
                    output_pred_dir_ =\
                        os.path.join(output_pred_dir,
                                     "{0}_{1:.1f}_{2:.1f}".format(max_length, fraction, feature_fraction))
                    cv_mode(testing_mode, multiclass=multiclass, paradigm_file=paradigm_file,
                            infile=infile, fraction=fraction, nfolds=nfolds,
                            selection_method=selection_method, feature_fraction=feature_fraction,
                            output_train_dir=output_train_dir, output_pred_dir=output_pred_dir)




