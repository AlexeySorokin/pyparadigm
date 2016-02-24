import sys
import copy
import getopt
from collections import OrderedDict
from itertools import chain, product

from sklearn.base import clone

from pyparadigm import LcsSearcher, extract_tables, output_paradigms
from paradigm_classifier import ParadigmClassifier, JointParadigmClassifier
from transformation_classifier import TransformationsHandler
from paradigm_detector import ParadigmSubstitutor

def read_input_file(infile):
    print("Reading...")
    answer = []
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
                # if values_string.startswith('{'):
                #     values_string = values_string[1:-1]
                # values = values_string.split('/')
                # tags_with_values[tag] = values
                tags_with_values[tag] = values_string
            answer.append((lemma, tags_with_values, form))
            # answer.extend(((lemma, OrderedDict(zip(tags_with_values.keys(), elem)), form)
            #                for elem in product(*tags_with_values.values())))
    return answer

def extract_problem_codes(triples):
    problem_codes, data_for_problems = OrderedDict(), []
    problems_number = 0
    for lemma, tags_with_values, word in triples:
        problem_descr = (tuple(tags_with_values.keys()), tuple(tags_with_values.values()))
        problem_code = problem_codes.get(problem_descr)
        if problem_code is None:
            problem_code = problem_codes[problem_descr]= problems_number
            problems_number += 1
        data_for_problems.append((lemma, problem_code, word))
    return problem_codes, data_for_problems

def extract_transform_codes(lcs_searcher, pairs):
    paradigm_codes, paradigms_number = OrderedDict(), 0
    answer = []
    for j, (lemma, form) in enumerate(pairs, 1):
        paradigm_descr, var_values = lcs_searcher.calculate_paradigm(lemma, [lemma, form])[0]
        # print(lemma, form, paradigm_descr)
        paradigm_descr = '#'.join(paradigm_descr)
        paradigm_code = paradigm_codes.get(paradigm_descr)
        if paradigm_code is None:
            paradigm_code = paradigm_codes[paradigm_descr] = paradigms_number
            paradigms_number += 1
        answer.append((lemma, paradigm_code, var_values, form))
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
        codes_number = max(x[1] for x in joint_data) + 1
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
        problems_number = 0
        for elem in data:
            lemma, tags_with_values = elem[:2]
            problem_descr = (tuple(tags_with_values.keys()), tuple(tags_with_values.values()))
            problem_code = self.problem_codes.get(problem_descr)
            if problem_code is None:
                if not has_codes:
                    problem_code = self.problem_codes[problem_descr] = problems_number
                    problems_number += 1
                else:
                    raise IndexError("Wrong problem descr {}".format(problem_descr))
            to_append = (lemma, problem_code, elem[2]) if has_answers else (lemma, problem_code)
            data_for_problems.append(to_append)
        return data_for_problems

    def make_paradigms_from_data(self, data):
        """
        Строит абстрактные парадигмы, применяется для тестирования
        """
        data_for_problems = self.extract_data(data, has_codes=False, has_answers=True)
        self.problems_number = len(self.problem_codes)
        lcs_seacrher = LcsSearcher(self.gap, self.initial_gap)
        self.transform_codes, lemmas_with_transform_codes = extract_transform_codes(
            lcs_seacrher, [(x[0], x[2]) for x in data_for_problems])
        self.paradigmers = [ParadigmSubstitutor(descr) for descr in self.transform_codes]
        joint_data = [(lemma, data_for_problems[i][1], paradigm_code, var_values, word)
                      for i, (lemma, paradigm_code, var_values, word)
                      in enumerate(lemmas_with_transform_codes)]
        return joint_data

    def fit(self, data):
        # data_for_problems = self.extract_data(data, has_codes=False, has_answers=True)
        # self.problems_number = len(self.problem_codes)
        # lcs_seacrher = LcsSearcher(self.gap, self.initial_gap)
        # self.transform_codes, lemmas_with_transform_codes = extract_transform_codes(
        #     lcs_seacrher, [(x[0], x[2]) for x in data_for_problems])
        # self.paradigmers = [ParadigmSubstitutor(descr) for descr in self.transform_codes]
        # joint_data = [(lemma, data_for_problems[i][1], paradigm_code, var_values, word)
        #               for i, (lemma, paradigm_code, var_values, word)
        #               in enumerate(lemmas_with_transform_codes)]
        joint_data = self.make_paradigms_from_data(data)
        transformations_handler = TransformationsHandler(self.transform_codes)
        transformation_classifier_params = {'select_features': 'ambiguity',
                                            'selection_params': {'nfeatures': 0.1, 'min_count': 2}}
        self.classifiers = [None] * self.problems_number
        for i in range(self.problems_number):
            self.classifiers[i] = JointParadigmClassifier(
                ParadigmClassifier(self.transform_codes), transformations_handler,
                dict(), transformation_classifier_params)
        data_by_problems = arrange_data_by_problems(
            joint_data, self.problems_number, has_answers=True)
        for i, (_, curr_X, curr_y) in enumerate(data_by_problems):
            self.classifiers[i].fit(curr_X, [[x] for x in curr_y])
            print("Classifier {0} of {1} fitted".format(i+1, self.problems_number))

    def predict(self, data, return_by_problems=False):
        data_for_problems = self.extract_data(data, has_codes=True, has_answers=False)
        data_by_problems = arrange_data_by_problems(data_for_problems, has_answers=False)
        if return_by_problems:
            answers = [[] for _ in range(self.problems_number)]
        else:
            answers = [None] * len(data)
        for i, (indexes, curr_X, _) in enumerate(data_by_problems):
            print("Predicting classifier {} of {}".format(i+1, self.problems_number))
            if len(curr_X) == 0:
                continue
            cls = self.classifiers[i]
            curr_answers = [x[0] for x in cls.predict(curr_X)]
            curr_words = []
            for lemma, (transform_code, var_values) in zip(curr_X, curr_answers):
                if transform_code is not None:
                    fragmentor = self.paradigmers[transform_code]
                    curr_words.append(fragmentor._substitute_words(var_values)[1])
                else:
                    # не удалось найти ни одного подходящего класса
                    # здесь надо будет повызывать другие классификаторы
                    # пока возвращает тождественный ответ
                    curr_words.append(lemma)
            if return_by_problems:
                answers[i] = list(zip(indexes, curr_words))
            else:
                for j, word in zip(indexes, curr_words):
                    answers[j] = word
        return answers

def get_descr_string(problem_descr):
    return ",".join("{0}={1}".format(*elem) for elem in problem_descr.items())


SHORT_OPTS = 'g:i:'
LONG_OPTS = ['gap=', 'initial_gap=']
MODES = ['make_paradigms', 'guess_inflection', 'test_inflection']

if __name__ == "__main__":
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, SHORT_OPTS, LONG_OPTS)
    gap, initial_gap = 1, 0
    for opt, val in opts:
        if opt in ['-g', '--gap']:
            gap = int(val)
        elif opt in ['-i', '--initial_gap']:
            initial_gap = int(val)
        else:
            raise ValueError("Unrecognized option: {0}".format(opt))
    if len(args) < 1 or args[0] not in MODES:
        sys.exit("First argument must be one of {0})".format(' '.join(MODES)))
    mode, args = args[0], args[1:]
    if mode == 'make_paradigms':
        if len(args) != 3:
            sys.exit("Pass train file, outfile for paradigms, outfile for paradigm_stats")
        train_file, outfile, stats_outfile = args
    elif mode == 'guess_inflection':
        if len(args) != 3:
            sys.exit("Pass train file, test file and output file")
        train_file, test_file, outfile = args
    elif mode == 'test_inflection':
        if len(args) != 3:
            sys.exit("Pass train file, dev file, and output file")
        train_file, test_file, outfile = args
    input_data = read_input_file(train_file)
    basic_classifier = JointParadigmClassifier if mode == 'guess_inflection' else None
    form_guesser = SigmorphonGuesser(cls=basic_classifier, gap=gap, initial_gap=initial_gap)
    if mode == 'make_paradigms':
        # joint_data = [(lemma, problem_code, paradigm_code, var_values, word),...]
        joint_data = form_guesser.make_paradigms_from_data(input_data)
        # data_for_output = [(lemma, [lemma, word], paradigm_repr, var_values), ...]
        paradigm_descrs_by_codes = list(form_guesser.transform_codes.keys())
        data_for_table_extraction =\
            [(lemma, [lemma, word], paradigm_descrs_by_codes[paradigm_code], var_values)
             for lemma, _, paradigm_code, var_values, word in joint_data]
        data_for_output = extract_tables(data_for_table_extraction)
        output_paradigms(data_for_output, outfile, stats_outfile)
    elif mode == 'guess_inflection':
        form_guesser.fit(input_data)
        test_data = read_input_file(test_file)
        answers = form_guesser.predict([(x[0], x[1]) for x in test_data])
        with open(outfile, "w", encoding="utf8") as fout:
            for (lemma, descr, _), word in zip(test_data, answers):
                fout.write("{0}\t{1}\t{2}\n".format(lemma, get_descr_string(descr), word))
    elif mode == 'test_inflection':
        form_guesser.fit(input_data)
        test_data = read_input_file(test_file)
        answers = form_guesser.predict([(x[0], x[1]) for x in test_data], return_by_problems=True)
        with open(outfile, "w", encoding="utf8") as fout:
            for problem_answers in answers:
                for i, word in problem_answers:
                    lemma, descr, correct_word = test_data[i]
                    fout.write("{0}\t{1}\t{2}\t{3}\n".format(
                        lemma, get_descr_string(descr), correct_word, word))
                fout.write("\n")

