import sys
import os
from itertools import chain, product
from collections import OrderedDict, Counter, defaultdict
import bisect
import getopt

import numpy as np
import sklearn.metrics as skm
import sklearn.cross_validation as skcv

from input_reader import process_codes_file, process_lemmas_file, read_paradigms, read_lemmas
from paradigm_detector import ParadigmFragment, ParadigmSubstitutor
from learn_paradigms import make_lemmas
from lm_paradigm_classifier_ import LMParadigmClassifier

def cv_mode(testing_mode, multiclass, predict_lemmas, paradigm_file, infile,
            fraction, nfolds=0, order=3):
    lemma_descriptions_list = process_lemmas_file(infile)
    data, labels_with_vars = read_lemmas(lemma_descriptions_list, multiclass=multiclass, return_joint=True)
    paradigm_descriptions_list = process_codes_file(paradigm_file)
    paradigm_table, pattern_counts = read_paradigms(paradigm_descriptions_list)
    if predict_lemmas:
        paradigm_handlers = {code: ParadigmSubstitutor(descr)
                             for descr, code in paradigm_table.items()}
    else:
        test_lemmas, pred_lemmas = None, None
    # подготовка для кросс-валидации
    classes = sorted(set(chain(*((x[0] for x in elem) for elem in labels_with_vars))))
    if nfolds == 0:
        train_data_length = int(fraction * len(data))
        train_data, test_data = [data[:train_data_length]], [data[train_data_length:]]
        train_labels_with_vars, test_labels_with_vars =\
            [labels_with_vars[:train_data_length]], [labels_with_vars[train_data_length:]]
        nfolds = 1
    else:
        test_data, train_data = [None] * nfolds, [None] * nfolds
        test_labels_with_vars, train_labels_with_vars = [None] * nfolds, [None] * nfolds
        for fold in range(nfolds):
            train_data[fold], test_data[fold], train_labels_with_vars[fold], test_labels_with_vars[fold] =\
                skcv.train_test_split(data, labels_with_vars, test_size = 1.0 - fraction,
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
    lm_classifier = LMParadigmClassifier(paradigm_table, pattern_counts, lm_order=order)
    for i, (train_sample, train_labels_sample, test_sample, test_labels_sample) in\
            enumerate(zip(train_data, train_labels_with_vars, test_data, test_labels_with_vars)):
        lm_classifier.fit(train_sample, train_labels_sample)
        lm_classifier.test()


SHORT_OPTIONS = 'mplT:O:'
LONG_OPTIONS = ['multiclass', 'probs', 'lemmas', 'train_output=', 'test_output=']

if __name__ == "__main__":
    args = sys.argv[1:]
    # значения по умолчанию для режима тестирования
    # поддержки множественнных парадигм для одного слова
    # и индикатора отделения окончания перед классификацией
    testing_mode, multiclass = 'predict', False
    predict_lemmas = False
    output_train_dir = None
    output_pred_dir = None
    # опции командной строки
    opts, args = getopt.getopt(args, SHORT_OPTIONS, LONG_OPTIONS)
    for opt, val  in opts:
        # поддердка множественных парадигм
        if opt in ['-m', '--multiclass']:
            multiclass = True
        # дополнительно к классам возвращаются их вероятности
        elif opt in ['-p', '--probs']:
            testing_mode = 'predict_proba'
        # измеряется только качество лемматизации
        elif opt in ['-l', '--lemmas']:
            predict_lemmas = True
        # директория для сохранения обучающей выборки
        elif opt in ['-T', '--train_output']:
            output_train_dir = val
        # директория для сохранения результатов классификации
        elif opt in ['-O', '--test_output']:
            output_pred_dir = val
    # аргументы командной строки
    if len(args) < 1:
        sys.exit("You should pass mode as first argument")
    mode, args = args[0], args[1:]
    if mode not in ["train", "cross-validation"]:
        sys.exit("Mode should be train or cross-validation")
    if mode =="train":
        # ПОКА НЕ РЕАЛИЗОВАНО
        raise NotImplementedError()
    elif mode == "cross-validation":
        if len(args) != 5:
            sys.exit("Pass 'cross-validation', coding file, train file, "
                     "train_data_fraction, number of folds, order of language model")
        paradigm_file, infile = args[:2]
        train_data_fractions = args[2]
        # доля данных в обучающей выборке
        train_data_fractions = list(float(x) if float(x) > 0 else 1.0
                                    for x in train_data_fractions.split(','))
        nfolds, order = map(int, args[3:])
        # doing classification with different parameters
        for fraction in train_data_fractions:
            if output_train_dir is not None:
                output_train_dir_ =\
                    os.path.join(output_train_dir, "{0:.1f}".format(fraction))
                if not os.path.exists(output_train_dir_):
                    os.makedirs(output_train_dir_)
            else:
                output_train_dir_ = None
            if output_pred_dir is not None:
                output_pred_dir_ =\
                    os.path.join(output_pred_dir, "{0:.1f}".format(fraction))
                if not os.path.exists(output_pred_dir_):
                    os.makedirs(output_pred_dir_)
            else:
                output_pred_dir_ = None
            cv_mode(testing_mode, multiclass=multiclass, predict_lemmas=predict_lemmas,
                    paradigm_file=paradigm_file, infile=infile,
                    fraction=fraction, nfolds=nfolds,
                    order=order)




