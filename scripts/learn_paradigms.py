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
import sklearn.tree as skt

from input_reader import process_codes_file, process_lemmas_file, read_paradigms, read_lemmas
from paradigm_classifier import ParadigmClassifier, extract_classes_from_probs
from paradigm_classifier import JointParadigmClassifier, CombinedParadigmClassifier
from paradigm_detector import ParadigmFragment, ParadigmSubstitutor
from transformation_classifier import TransformationsHandler

import statprof
from tools import redirect_stdout

import warnings
warnings.filterwarnings("ignore")

np.seterr(invalid='ignore')

def cv_mode(testing_mode, multiclass, predict_lemmas, find_flection,
            paradigm_file, infile, max_length, fraction, nfolds=0,
            selection_method=None, feature_fraction=None,
            output_train_dir=None, output_pred_dir=None):
    '''
    Определяет качество классификатора с заданными параметрами
    по скользящему контролю на обучающей выборке

    Параметры:
    -----------
    testing_mode: str ('predict' or  'predict_proba'), режим использования
    multiclass: bool, может ли одно слово иметь несколько парадигм
    find_flection:  bool, выполняется ли предварительный поиск флексии
        для того, чтобы использовать в качестве признаков суффиксы основы,
        а не всего слова. Оказалось, что качество ухудшается
    paradigm_file: str, путь к файлу с парадигмами
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
    output_train_dir: str or None, optional(default=None),
        директория для сохранения тестовых данных,
        в случае output_train_dir=None сохранение не производится
    output_pred_dir: str or None, optional(default=None),
        директория для сохранения результатов классификации,
        в случае output_train_dir=None сохранение не производится
    '''
    # чтение входных файлов
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
    if selection_method is None:
        selection_method = 'ambiguity'
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
    paradigm_classifier = ParadigmClassifier(paradigm_table)
    paradigm_classifier_params = {'multiclass': multiclass, 'find_flection': find_flection,
                                  'max_length': max_length, 'use_prefixes': True,
                                  'classifier_params': None, 'selection_method': selection_method,
                                  'nfeatures': feature_fraction, 'smallest_prob': 0.01}
    transformation_handler = TransformationsHandler(paradigm_table, pattern_counts)
    transformation_classifier_params = {'select_features': 'ambiguity',
                                        'selection_params': {'nfeatures': 0.1, 'min_count': 2}}
    # statprof.start()
    cls = JointParadigmClassifier(paradigm_classifier, transformation_handler,
                                  paradigm_classifier_params, transformation_classifier_params)
    # cls = CombinedParadigmClassifier(paradigm_classifier, transformation_handler,
    #                                  paradigm_classifier_params, transformation_classifier_params)
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
                              test_words=test_words, predicted_probs=prediction_probs_for_output)
        print("{0}\t{1:<.2f}\t{2}\t{3:<.2f}\t{4:<.2f}\t{5:<.2f}".format(
            max_length, fraction, cls.paradigm_classifier.nfeatures,
            100 * np.mean(label_precisions), 100 * np.mean(variable_precisions),
            100 * np.mean(form_precisions)))
    else:
        label_precisions, variable_precisions, lemma_precisions, form_precisions =\
            output_accuracies(classes, test_labels_with_vars, predictions,
                              multiclass, test_lemmas, pred_lemmas,
                              outfile=output_pred_dir, paradigm_descrs=descrs_by_codes,
                              test_words=test_words, predicted_probs=prediction_probs_for_output,
                              save_confusion_matrices=True)
        print("{0}\t{1:<.2f}\t{2}\t{3:<.2f}\t{4:<.2f}\t{5:<.2f}\t{6:<.2f}".format(
            max_length, fraction, cls.paradigm_classifier.nfeatures,
            100 * np.mean(label_precisions), 100 * np.mean(variable_precisions),
            100 * np.mean(lemma_precisions), 100 * np.mean(form_precisions)))
    # statprof.stop()
    # with open("statprof_{0:.1f}_{1:.1f}.stat".format(fraction, feature_fraction), "w") as fout:
    #     with redirect_stdout(fout):
    #         statprof.display()
    # вычисляем точность и обрабатываем результаты
    # for curr_test_values, curr_pred_values in zip(test_values_with_codes, pred_values_with_codes):
    #     print(len(curr_test_values), len(curr_pred_values))
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


def make_lemmas(paradigm_handlers, codes_with_var_values):
    answer = []
    for elem in codes_with_var_values:
        answer.append([paradigm_handlers[code].make_first_form(var_values)
                       for code, var_values in elem])
    return answer


def make_full_paradigms(paradigm_handlers, codes_with_var_values):
    answer = []
    for elem in codes_with_var_values:
        answer.append([paradigm_handlers[code]._substitute_words(
            var_values, return_principal=False)
                       for code, var_values in elem])
    return answer

def output_accuracies(classes, test_labels_with_vars, pred_labels_with_vars, multiclass,
                      test_lemmas=None, pred_lemmas=None, outfile=None, test_words=None,
                      paradigm_descrs=None, predicted_probs=None, save_confusion_matrices=True):
    """
    Вычисляет различные показатели точности классификации
    """
    nfolds = len(test_labels_with_vars)
    test_labels = [[[x[0] for x in elem] for elem in test_sample]
                   for test_sample in test_labels_with_vars]
    pred_labels = [[[x[0] for x in elem] for elem in pred_sample]
                   for pred_sample in pred_labels_with_vars]
    transformer = ((lambda x: [y[0] for y in x])
                   if not multiclass else MultiLabelBinarizer(classes).fit_transform)
    transformed_test_labels = [transformer(first) for first in test_labels]
    transformed_pred_labels = [transformer(second) for second in pred_labels]
    micro_precisions = [skm.f1_score(first, second, average='micro')
                        for first, second in zip(transformed_test_labels,
                                                 transformed_pred_labels)]
    test_values_with_codes = [[] for i in range(nfolds)]
    for i, sample in enumerate(test_labels_with_vars):
        for elem in sample:
            test_values_with_codes[i].append(["_".join((str(code), ) + tuple(var_values))
                                                  for code, var_values in elem])
    pred_values_with_codes = [[] for i in range(nfolds)]
    for i, sample in enumerate(pred_labels_with_vars):
        for elem in sample:
            pred_values_with_codes[i].append(["_".join((str(code), ) + tuple(var_values))
                                                  for code, var_values in elem])
    variable_precisions = [skm.f1_score(first, second, average='binary')
                           for first, second in zip(test_values_with_codes, pred_values_with_codes)]
    if test_lemmas:
        lemma_precisions = [skm.f1_score(first, second, average='binary')
                            for first, second in zip(test_lemmas, pred_lemmas)]
    paradigmers_by_classes = {label: ParadigmSubstitutor(paradigm_descrs[label]) for label in classes}
    pred_word_forms = [[] for i in range(nfolds)]
    for i, sample in enumerate(pred_labels_with_vars):
        for elem in sample:
            code, var_values = elem[0]
            pred_word_forms[i].append(
                paradigmers_by_classes[code]._substitute_words(var_values, return_principal=False))
    test_word_forms = [[] for i in range(nfolds)]
    for i, sample in enumerate(test_labels_with_vars):
        for elem in sample:
            code, var_values = elem[0]
            test_word_forms[i].append(
                paradigmers_by_classes[code]._substitute_words(var_values, return_principal=False))
    form_accuracies = [0.0] * nfolds
    for i, (pred_forms_sample, test_forms_sample) in\
            enumerate(zip(pred_word_forms, test_word_forms)):
        total, correct = 0, 0
        for first, second in zip(pred_forms_sample, test_forms_sample):
            total += len(first)
            correct += sum(int(x==y) for x, y in zip(first, second))
        form_accuracies[i] = correct / total
    if outfile:
        if not paradigm_descrs or not test_words:
            print("Cannot output comparison results without paradigm descriptions")
        for fold, (test_sample, pred_sample) in\
                enumerate(zip(test_labels_with_vars, pred_labels_with_vars), 1):
            curr_outfile = os.path.join(outfile, "fold_{0}.out".format(fold))
            current_words = test_words[fold-1]
            with open(curr_outfile, "w", encoding="utf8") as fout:
                for i, elem in enumerate(test_sample):
                    word = test_words[fold-1][i]
                    for code, var_values in elem:
                        fout.write("{0}\t{1}\n".format(word, paradigm_descrs[code]))
                    if predicted_probs:
                        curr_answer = predicted_probs[fold-1][i]
                        for (code, var_values), prob in zip(*curr_answer):
                            fout.write("{0}\t{1:.2f}\n".format(paradigm_descrs[code], 100 * prob))
                    else:
                        curr_answer = pred_elem
                        for (code, var_values), in zip(*curr_answer):
                            fout.write("{0}\n".format(paradigm_descrs[code]))
                    fout.write("\n")
            curr_outfile = os.path.join(outfile, "fold_{0}_incorrect.out".format(fold))
            with open(curr_outfile, "w", encoding="utf8") as fout:
                for i, (test_elem, pred_elem) in enumerate(zip(test_sample, pred_sample)):
                    if all(test_code_with_values in pred_elem
                           for test_code_with_values in test_elem):
                        continue
                    word = current_words[i]
                    for code, var_values in test_elem:
                        fout.write("{0}\t{1}\n".format(word, paradigm_descrs[code]))
                    if predicted_probs:
                        curr_answer = predicted_probs[fold-1][i]
                        for (code, var_values), prob in zip(*curr_answer):
                            fout.write("{0}\t{1:.2f}\n".format(paradigm_descrs[code], 100 * prob))
                    else:
                        curr_answer = pred_elem
                        for (code, var_values), in zip(curr_answer):
                            fout.write("{0}\n".format(paradigm_descrs[code]))
                    fout.write("\n")
            if test_lemmas:
                curr_test_lemmas, curr_pred_lemmas = test_lemmas[fold-1], pred_lemmas[fold-1]
                curr_outfile = os.path.join(outfile, "fold_{0}_lemmas.out".format(fold))
                with open(curr_outfile, "w", encoding="utf8") as fout:
                    for word, first, second in zip(current_words, curr_test_lemmas, curr_pred_lemmas):
                        fout.write('{0}\t{1}\n'.format(word, " ".join(first)))
                        fout.write('{1}\n\n'.format(word, " ".join(second)))
                curr_outfile = os.path.join(outfile, "fold_{0}_lemmas_incorrect.out".format(fold))
                with open(curr_outfile, "w", encoding="utf8") as fout:
                    for word, first, second in zip(current_words, curr_test_lemmas, curr_pred_lemmas):
                        if any(lemma not in second for lemma in first):
                            fout.write('{0}\t{1}\n'.format(word, " ".join(first)))
                            fout.write('{1}\n\n'.format(word, " ".join(second)))
            if save_confusion_matrices:
                confusion_outfile = os.path.join(outfile, "confusion_fold_{0}.out".format(fold))
                confusion_matrix = skm.confusion_matrix(np.ravel(test_labels[fold-1]),
                                                        np.ravel(pred_labels[fold-1]), classes)
                with open(confusion_outfile, "w", encoding="utf8") as fout:
                    for i, string in enumerate(confusion_matrix):
                        fout.write('{}\nCorrect: {}\tTotal: {}\n'.format(
                            paradigm_descrs[classes[i]], string[i], np.sum(string)))
                        for j, val in sorted(enumerate(string), key=(lambda x: x[1]), reverse=True):
                            if j == i:
                                continue
                            if val == 0:
                                break
                            fout.write("{}\t{}\n".format(paradigm_descrs[classes[j]], val))
                        fout.write("\n")
    if test_lemmas:
        return micro_precisions, variable_precisions, lemma_precisions, form_accuracies
    else:
        return micro_precisions, variable_precisions, form_accuracies


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

SHORT_OPTIONS = 'mplfT:O:'
LONG_OPTIONS = ['multiclass', 'probs', 'lemmas', 'flection', 'train_output=', 'test_output=']

# добавить в качестве параметра язык (чтобы уметь определять возможные позиции лемм в парадигме

if __name__ == '__main__':
    args = sys.argv[1:]
    # значения по умолчанию для режима тестирования
    # поддержки множественнных парадигм для одного слова
    # и индикатора отделения окончания перед классификацией
    testing_mode, multiclass, find_flection = 'predict', False, False
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
        # индикатор отделения окончания
        # не рекомендован к использованию, т.к. ухудшает качество
        elif opt in ['-f', '--flection']:
            find_flection = True
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
        if len(args) not in range(3, 6):
            sys.exit("Pass 'train', coding file, train and test files,"
                     "[feature selection algorithm], [number of features to select]")
        paradigm_file, train_file, test_file = args[:3]
        selection_method = args[3] if len(args) > 3 else None
        nfeatures = int(args[4]) if len(args) > 4 else None
        raise NotImplementedError()
        # train_test_mode(paradigm_file, train_file, test_file,
        #                 selection_method=selection_method, nfeatures=nfeatures)
    elif mode == "cross-validation":
        if len(args) not in [6, 7]:
            sys.exit("Pass 'cross-validation', coding file, train file, "
                     "max_feature_length, feature_fraction, train_data_fraction,"
                     "number of folds, [, feature selection algorithm]")
        paradigm_file, infile = args[:2]
        max_feature_lengths, feature_fractions, train_data_fractions = args[2:5]
        nfolds = args[5]
        selection_method = args[6] if len(args) > 6 else None
        # converting parameters
        # максимальная длина признака
        max_feature_lengths = list(int(x) for x in max_feature_lengths.split(','))
        # доля признаков при отборе
        feature_fractions = list(float(x) if float(x) > 0 else 1.0
                                 for x in feature_fractions.split(','))
        # доля данных в обучающей выборке
        train_data_fractions = list(float(x) if float(x) > 0 else 1.0
                                    for x in train_data_fractions.split(','))
        nfolds = int(nfolds)
        # doing classification with different parameters
        for max_length in max_feature_lengths:
            for fraction in train_data_fractions:
                for feature_fraction in feature_fractions:
                    if output_train_dir is not None:
                        output_train_dir_ =\
                            os.path.join(output_train_dir,
                                         "{0}_{1:.1f}_{2:.1f}".format(max_length, fraction, feature_fraction))

                    else:
                        output_train_dir_ = None
                    if output_pred_dir is not None:
                        output_pred_dir_ =\
                            os.path.join(output_pred_dir,
                                         "{0}_{1:.1f}_{2:.1f}".format(max_length, fraction, feature_fraction))
                        if not os.path.exists(output_pred_dir_):
                            os.makedirs(output_pred_dir_)
                    else:
                        output_pred_dir_ = None
                    cv_mode(testing_mode, multiclass=multiclass, predict_lemmas=predict_lemmas,
                            find_flection=find_flection, paradigm_file=paradigm_file, infile=infile,
                            max_length=max_length, fraction=fraction, nfolds=nfolds,
                            selection_method=selection_method, feature_fraction=feature_fraction,
                            output_train_dir=output_train_dir_, output_pred_dir=output_pred_dir_)




