import sys
import copy
import getopt
from collections import OrderedDict
from itertools import chain, product

from sklearn.base import clone

from pyparadigm import LcsSearcher, extract_tables, output_paradigms, vars_to_string
from paradigm_classifier import ParadigmClassifier, JointParadigmClassifier
from transformation_classifier import TransformationsHandler
from paradigm_detector import ParadigmSubstitutor

def read_input_file(infile, group_all=False):
    print("Reading...")
    answer = []
    curr_lemma, current_tag_values, current_forms = None, [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split("\t")
            if len(splitted_line) != 3:
                return None
            lemma, tag_string, form = splitted_line
            tags_with_values_string = tag_string.split(",")
            tags_with_values = OrderedDict()
            for s in tags_with_values_string:
                tag, values_string = s.split('=')
                tags_with_values[tag] = values_string
            if curr_lemma is not None and (not group_all or lemma != curr_lemma):
                answer.append((curr_lemma, current_tag_values, current_forms))
                current_tag_values, current_forms = [], []
            curr_lemma = lemma
            current_tag_values.append(tags_with_values)
            current_forms.append(form)
    if curr_lemma is not None:
        answer.append((curr_lemma, current_tag_values, current_forms))
    return answer

def extract_problem_codes(triples):
    problem_codes, data_for_problems = OrderedDict(), []
    problems_number = 0
    for lemma, tags_with_values, word in triples:
        problem_descr = (tuple(tags_with_values.keys()), tuple(tags_with_values.values()))
        problem_code = problem_codes.get(problem_descr)
        if problem_code is None:
            problem_code = problem_codes[problem_descr] = problems_number
            problems_number += 1
        data_for_problems.append((lemma, problem_code, word))
    return problem_codes, data_for_problems

def extract_transform_codes(lcs_searcher, pairs):
    paradigm_codes, paradigms_number = OrderedDict(), 0
    answer = []
    paradigms_with_vars = lcs_searcher.calculate_paradigms(
        [([lemma] + forms) for lemma, forms in pairs], count_paradigms=True)
    for (lemma, forms), (descr, var_values) in zip(pairs, paradigms_with_vars):
        for i, form in enumerate(forms, 1):
            transform_descr = descr[0] + '#' + descr[i]
            transform_code = paradigm_codes.get(transform_descr)
            if transform_code is None:
                transform_code = paradigm_codes[transform_descr] = paradigms_number
                paradigms_number += 1
            answer.append((lemma, transform_code, var_values, form))
    return paradigm_codes, answer

def test_output(problem_codes, transform_codes, joint_data):
    max_code = -1
    problem_descrs = list(problem_codes.keys())
    transform_descrs = list(transform_codes.keys())
    with open("res.out", "w", encoding="utf8") as fout:
        for lemma, problem_code, transform_code, _, word in joint_data:
            if transform_code > max_code:
                fout.write("{0}\t{1}\t{2}\t{3}\n".format(
                    lemma, ",".join(problem_descrs[problem_code][1]),
                    transform_descrs[transform_code], word))
                max_code = transform_code

def arrange_data_by_problems(joint_data, codes_number=None, has_answers=True):
    if codes_number is None:
        codes_number = max(elem[1] for elem in joint_data) + 1
    train_data_by_problems = [([], [], []) for _ in range(codes_number)]
    for i, elem in enumerate(joint_data):
        lemma, problem_code = elem[:2]
        train_data_by_problems[problem_code][0].append(i)
        train_data_by_problems[problem_code][1].append(lemma)
        if has_answers:
            train_data_by_problems[problem_code][2].append((elem[2], elem[3]))
    return train_data_by_problems


class SigmorphonGuesser:

    def __init__(self, cls, gap=1, initial_gap=0):
        self.gap = gap
        self.initial_gap = initial_gap
        self.cls = cls
        self.problem_codes = OrderedDict()

    def extract_data(self, data, has_codes=False, has_answers=True):
        data_for_problems = []
        problems_number = self.problems_number if has_codes else 0
        for elem in data:
            lemma, current_tags_with_values = elem[:2]
            problem_descrs =\
                [(tuple(tags_with_values.keys()), tuple(tags_with_values.values()))
                 for tags_with_values in current_tags_with_values]
            problem_codes = []
            for problem_descr in problem_descrs:
                problem_code = self.problem_codes.get(problem_descr)
                if problem_code is None:
                    problem_code = self.problem_codes[problem_descr] = problems_number
                    problems_number += 1
                    if has_codes:
                        print("Wrong problem descr {}".format(problem_descr))
                problem_codes.append(problem_code)
            to_append = ((lemma, problem_codes, elem[2])
                         if has_answers else (lemma, problem_codes))
            data_for_problems.append(to_append)
        return data_for_problems

    def make_paradigms_from_data(self, data):
        """
        Строит абстрактные парадигмы, применяется для тестирования
        """
        # data: список элементов вида (лемма, список тэгов, список словоформ)
        # не надо разделять данные по проблемам до разделения парадигм на пары
        data_for_problems = self.extract_data(data, has_codes=False, has_answers=True)
        self.problems_number = len(self.problem_codes)
        self.lcs_seacrher = LcsSearcher(self.gap, self.initial_gap, count_gaps=True)
        self.transform_codes, lemmas_with_transform_codes = extract_transform_codes(
            self.lcs_seacrher, [(x[0], x[2]) for x in data_for_problems])
        problems_in_data = [code for elem in data_for_problems for code in elem[1]]
        self.paradigmers = [ParadigmSubstitutor(descr) for descr in self.transform_codes]
        joint_data = [(lemma, problem_code, paradigm_code, var_values, word)
                      for problem_code, (lemma, paradigm_code, var_values, word)
                      in zip(problems_in_data, lemmas_with_transform_codes)]
        return joint_data

    def fit(self, data):
        joint_data = self.make_paradigms_from_data(data)
        transformations_handler = TransformationsHandler(self.transform_codes)
        transformation_classifier_params = {'select_features': 'ambiguity',
                                            'selection_params': {'nfeatures': 0.25, 'min_count': 2}}
        self.classifiers = [None] * self.problems_number
        # НЕМЕЦКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': True,
        #                      'max_prefix_length': 2, 'suffixes_to_delete': ['en'],
        #                      'has_letter_classifiers': False}
        # ИСПАНСКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': False}
        # АРАБСКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': True,
        #                      'max_prefix_length': 4, 'max_length': 4,
        #                      'has_letter_classifiers': None, 'to_memorize_affixes': 0}
        # ГРУЗИНСКИЙ
        classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': True,
                             'max_prefix_length': 4, 'has_letter_classifiers': 'suffix',
                             'to_memorize_affixes': 0}
        # ФИНСКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': False}
        # РУССКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': False,
        #                      'max_prefix_length': 3, 'suffixes_to_delete': ['ся', 'сь'],
        #                      'to_memorize_affixes': 2, 'has_letter_classifiers': False}
        # НАВАХО
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': True,
        #                      'max_prefix_length': 5, 'max_length': 3, 'has_letter_classifiers': 'prefix',
        #                      'to_memorize_affixes': -3}
        # ТУРЕЦКИЙ
        # classifier_params = {'paradigm_table': self.transform_codes, 'use_prefixes': False}
        classifier_params['min_feature_count'] = 3
        classifier_params['nfeatures'] = 0.1
        for i in range(self.problems_number):
            self.classifiers[i] = JointParadigmClassifier(
                ParadigmClassifier(**classifier_params), transformations_handler,
                dict(), transformation_classifier_params)
        data_by_problems = arrange_data_by_problems(
            joint_data, self.problems_number, has_answers=True)
        for i, (_, curr_X, curr_y) in enumerate(data_by_problems):
            self.classifiers[i].fit(curr_X, [[x] for x in curr_y])
            # print("Classifier {0} of {1} fitted".format(i+1, self.problems_number))
        return self

    def predict(self, data, return_by_problems=False, return_descr=False):
        # data_for_problems = [(lemma, [problem_code]), ...]
        data_for_problems = self.extract_data(data, has_codes=True, has_answers=False)
        # data_for_problems = [(lemma, problem_code), ...]
        data_for_problems = [(lemma, codes[0]) for lemma, codes in data_for_problems]
        data_by_problems = arrange_data_by_problems(data_for_problems, has_answers=False)
        current_problems_number = len(data_by_problems)
        if return_by_problems:
            answers = [[] for _ in range(current_problems_number)]
        else:
            answers = [None] * len(data)
        for i, (indexes, curr_X, _) in enumerate(data_by_problems):
            # print("Predicting classifier {} of {}".format(i+1, self.problems_number))
            if len(curr_X) == 0:
                continue
            if i >= self.problems_number:
                # проблема не встречалась в обучающей выборке, потому выдаём тождественный ответ
                curr_words = [lemma for lemma in curr_X]
            else:
                # print("Classifier {} predicting...".format(i+1))
                cls = self.classifiers[i]
                curr_answers = [x[0] for x in cls.predict(curr_X)]
                curr_descrs = []
                curr_words = []
                for lemma, (transform_code, var_values) in zip(curr_X, curr_answers):
                    if transform_code is not None:
                        fragmentor = self.paradigmers[transform_code]
                        curr_descrs.append(fragmentor.descr)
                        curr_words.append(fragmentor._substitute_words(var_values)[1])
                    else:
                        # не удалось найти ни одного подходящего класса
                        # здесь надо будет повызывать другие классификаторы
                        # пока возвращает тождественный ответ
                        print("No transforms for lemma {}, problem {}".format(
                            lemma, list(self.problem_codes.keys())[i]))
                        curr_descrs.append(['#', '#'])
                        curr_words.append(lemma)
            if return_descr:
                curr_answer = list(zip(curr_descrs, curr_words))
            else:
                curr_answer = curr_words
            if return_by_problems:
                answers[i] = list(zip(indexes, curr_answer))
            else:
                for j, word in zip(indexes, curr_answer):
                    answers[j] = word
        return answers


def get_descr_string(problem_descr):
    if isinstance(problem_descr, dict):
        problem_descr = problem_descr.items()
    else:
        problem_descr = zip(*problem_descr)
    return ",".join("{0}={1}".format(*elem) for elem in problem_descr)


def output_inflection_data(data_for_output, outfile):
    with open(outfile, "w", encoding="utf8") as fout:
        for problem_descr, problem_data in data_for_output:
            for lemma, transform_descr, word, var_values in problem_data:
                fout.write("{}\t{}\t{}\t{}\t{}\n".format(
                    lemma, get_descr_string(problem_descr), word,
                    transform_descr, vars_to_string(lemma, var_values)))
            if len(problem_data) > 0:
                fout.write("\n")

SHORT_OPTS = 'g:i:Ia'
LONG_OPTS = ['gap=', 'initial_gap=', 'incorrect', 'group_all']
MODES = ['make_paradigms', 'guess_inflection', 'test_inflection']

if __name__ == "__main__":
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, SHORT_OPTS, LONG_OPTS)
    gap, initial_gap, output_only_incorrect, group_all = 1, 0, False, False
    for opt, val in opts:
        if opt in ['-g', '--gap']:
            gap = int(val)
        elif opt in ['-i', '--initial_gap']:
            initial_gap = int(val)
        elif opt in ['-I', '--incorrect']:
            output_only_incorrect = True
        elif opt in ['-a', '--group_all']:
            group_all = True
        else:
            raise ValueError("Unrecognized option: {0}".format(opt))
    if len(args) < 1 or args[0] not in MODES:
        sys.exit("First argument must be one of {0})".format(' '.join(MODES)))
    mode, args = args[0], args[1:]
    if mode == 'make_paradigms':
        if len(args) != 4:
            sys.exit("Pass train file, outfile for inflection, "
                     "outfile for paradigms, outfile for paradigm_stats")
        train_file, inflection_outfile, outfile, stats_outfile = args
    elif mode == 'guess_inflection':
        if len(args) != 3:
            sys.exit("Pass train file, test file and output file")
        train_file, test_file, outfile = args
    elif mode == 'test_inflection':
        if len(args) != 3:
            sys.exit("Pass train file, dev file, and output file")
        train_file, test_file, outfile = args
    input_data = read_input_file(train_file, group_all=group_all)
    basic_classifier = JointParadigmClassifier if mode == 'guess_inflection' else None
    form_guesser = SigmorphonGuesser(cls=basic_classifier, gap=gap, initial_gap=initial_gap)
    if mode == 'make_paradigms':
        # joint_data = [(lemma, problem_code, paradigm_code, var_values, word),...]
        joint_data = form_guesser.make_paradigms_from_data(input_data)
        paradigm_descrs_by_codes = list(form_guesser.transform_codes.keys())
        problem_descrs_by_codes = list(form_guesser.problem_codes.keys())
        data_by_problems = arrange_data_by_problems(
            joint_data, form_guesser.problems_number, has_answers=True)
        data_for_output = []
        for descr, (curr_indexes, _, _) in zip(problem_descrs_by_codes, data_by_problems):
            curr_data_for_output = []
            for i in curr_indexes:
                lemma, _, paradigm_code, var_values, word = joint_data[i]
                curr_data_for_output.append((lemma, paradigm_descrs_by_codes[paradigm_code],
                                             word, var_values))
            data_for_output.append((descr, curr_data_for_output))
        output_inflection_data(data_for_output, inflection_outfile)
        # сохраняем статистику по парадигмам
        data_for_table_extraction =\
            [(lemma, [lemma, word], paradigm_descrs_by_codes[paradigm_code], var_values)
             for lemma, _, paradigm_code, var_values, word in joint_data]
        # data_for_output = [(lemma, [lemma, word], paradigm_repr, var_values), ...]
        data_for_output = extract_tables(data_for_table_extraction)
        output_paradigms(data_for_output, outfile, stats_outfile)
    elif mode == 'guess_inflection':
        form_guesser.fit(input_data)
        test_data = read_input_file(test_file, group_all=False)
        answers = form_guesser.predict([(x[0], x[1]) for x in test_data])
        with open(outfile, "w", encoding="utf8") as fout:
            for (lemma, descrs, _), word in zip(test_data, answers):
                fout.write("{0}\t{1}\t{2}\n".format(lemma, get_descr_string(descrs[0]), word))
    elif mode == 'test_inflection':
        form_guesser.fit(input_data)
        test_data = read_input_file(test_file, group_all=False)
        correct_paradigms_with_vars = form_guesser.lcs_seacrher.calculate_paradigms(
            [(lemma, words[0]) for lemma, _, words in test_data], count_paradigms=False)
        answers = form_guesser.predict([(x[0], x[1]) for x in test_data],
                                       return_by_problems=True, return_descr=True)
        with open(outfile, "w", encoding="utf8") as fout:
            for problem_answers in answers:
                current_problem_output = []
                curr_answers_number = 0
                curr_correct_answers_number = 0
                for i, (descr, word) in problem_answers:
                    lemma, problem_descrs, correct_words = test_data[i]
                    correct_word = correct_words[0]
                    correct_descr, _ = correct_paradigms_with_vars[i]
                    curr_answers_number += 1
                    curr_correct_answers_number += int(word == correct_word)
                    if word != correct_word or not output_only_incorrect:
                        current_problem_output.append(
                            (lemma, get_descr_string(problem_descrs[0]), '#'.join(correct_descr),
                             correct_word, '#'.join(descr), word))
                        # fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                        #     lemma, get_descr_string(problem_descrs[0]), '#'.join(correct_descr),
                        #     correct_word, '#'.join(descr), word))
                if len(current_problem_output) > 0:
                    current_problem_descr = current_problem_output[0][1]
                    fout.write("{}\tВсего: {}\tПравильно: {}({:.2f})\n".format(
                        current_problem_descr, curr_answers_number, curr_correct_answers_number,
                        100 * (curr_correct_answers_number / curr_answers_number)))
                    for elem in current_problem_output:
                        fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*elem))
                    fout.write("\n")
