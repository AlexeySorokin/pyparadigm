import sys
import copy
import getopt
from collections import OrderedDict, defaultdict
from itertools import chain, product

from sklearn.base import clone

from graph_utilities import calculate_affixes_to_remove
from pyparadigm import LcsSearcher, extract_tables, output_paradigms, vars_to_string
from paradigm_classifier import ParadigmClassifier, JointParadigmClassifier
from transformation_classifier import TransformationsHandler
from paradigm_detector import ParadigmSubstitutor
from counts_collector import LanguageModel


global LANGUAGE

## ФУНКЦИИ ДЛЯ ЧТЕНИЯ ВХОДА ###

def make_tag_values(tag_string):
    tags_with_values_string = tag_string.split(",")
    tags_with_values = OrderedDict()
    for s in tags_with_values_string:
        tag, values_string = s.split('=')
        tags_with_values[tag] = values_string
    return tags_with_values


def read_input_file(infile, mode=1, group_all=False):
    if mode not in [1, 2, 3]:
        raise ValueError("Reading mode should be one of 1, 2, 3.")
    print("Reading...")
    answer = []
    curr_lemma, current_tag_values, current_forms = None, [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted_line = line.split("\t")
            if mode == 1:
                if len(splitted_line) != 3:
                    return None
                lemma, tag_string, form = splitted_line
                tags_with_values = make_tag_values(tag_string)
                if curr_lemma is not None and (not group_all or lemma != curr_lemma):
                    answer.append((curr_lemma, current_tag_values, current_forms))
                    current_tag_values, current_forms = [], []
                curr_lemma = lemma
                current_tag_values.append(tags_with_values)
                current_forms.append(form)
            elif mode == 2:
                if len(splitted_line) != 4:
                    return None
                source_tag, source_form, target_tag, target_form = splitted_line
                source_tag_with_values = make_tag_values(source_tag)
                target_tag_with_values = make_tag_values(target_tag)
                answer.append(([source_tag_with_values], source_form,
                               [target_tag_with_values], target_form))
            elif mode == 3:
                raise NotImplementedError
    if mode == 1 and curr_lemma is not None:
        answer.append((curr_lemma, current_tag_values, current_forms))
    return answer

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


class InflectionDataHandler:
    """
    Класс для предварительной обработки данных в заданиях 1 и 2.
    Извлекает коды заданий, коды преобразований
    """

    def __init__(self, gap=1, initial_gap=0):
        self.gap = gap
        self.initial_gap = initial_gap
        self.problem_codes = OrderedDict()
        self.transform_codes = OrderedDict()

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

    def extract_transform_codes(self, pairs):
        paradigms_number = len(self.transform_codes)
        answer = []
        paradigms_with_vars = self.lcs_searcher.calculate_paradigms(
            [([lemma] + forms) for lemma, forms in pairs], count_paradigms=True)
        for (lemma, forms), (descr, var_values) in zip(pairs, paradigms_with_vars):
            for i, form in enumerate(forms, 1):
                transform_descr = descr[0] + '#' + descr[i]
                transform_code = self.transform_codes.get(transform_descr)
                if transform_code is None:
                    transform_code = self.transform_codes[transform_descr] = paradigms_number
                    paradigms_number += 1
                answer.append((lemma, transform_code, var_values, form))
        return answer

    def make_paradigms_from_data(self, data):
        """
        Строит абстрактные парадигмы, применяется для тестирования
        """
        # data: список элементов вида (лемма, список тэгов, список словоформ)
        data_for_problems = self.extract_data(data, has_codes=False, has_answers=True)
        self.problems_number = len(self.problem_codes)
        self.lcs_searcher = LcsSearcher(self.gap, self.initial_gap, count_gaps=True)
        lemmas_with_transform_codes =\
            self.extract_transform_codes([(x[0], x[2]) for x in data_for_problems])
        problems_in_data = [code for elem in data_for_problems for code in elem[1]]
        self.paradigmers = [ParadigmSubstitutor(descr) for descr in self.transform_codes]
        joint_data = [(lemma, problem_code, paradigm_code, var_values, word)
                      for problem_code, (lemma, paradigm_code, var_values, word)
                      in zip(problems_in_data, lemmas_with_transform_codes)]
        return joint_data


def get_classifier_params(language):
    # НЕМЕЦКИЙ
    if language == 'german':
        classifier_params = {'use_prefixes': False, 'max_prefix_length': 0,
                             'has_letter_classifiers': None}
    # ИСПАНСКИЙ
    elif language == 'spanish':
        classifier_params = {'use_prefixes': False}
    # АРАБСКИЙ
    elif language == 'arabic':
        classifier_params = {'use_prefixes': True, 'max_prefix_length': 4, 'max_length': 4,
                             'has_letter_classifiers': None, 'to_memorize_affixes': 0}
    # ГРУЗИНСКИЙ
    elif language == 'georgian':
        classifier_params = {'use_prefixes': True, 'max_prefix_length': 3,
                             'has_letter_classifiers': 'suffix', 'to_memorize_affixes': 0}
    # ФИНСКИЙ
    elif language == 'finnish':
        classifier_params = {'use_prefixes': False}
        # joint_classifier_params =\
        #         {'has_language_model': True, 'lm_file': self.lm_file,
        #          'has_joint_classifier': False, 'max_lm_coeff': 1.25}
    # РУССКИЙ
    elif language == 'russian':
        classifier_params = {'use_prefixes': False, 'max_prefix_length': 3,
                             'to_memorize_affixes': 2, 'has_letter_classifiers': 'suffix'}
    # НАВАХО
    elif language == 'navajo':
        classifier_params = {'use_prefixes': True, 'max_prefix_length': 5, 'max_length': 3,
                             'has_letter_classifiers': 'prefix', 'to_memorize_affixes': -3}
    # ТУРЕЦКИЙ
    elif language == 'turkish':
        classifier_params = {'use_prefixes': False,  'has_letter_classifiers': None}
    else:
        classifier_params = {'use_prefixes': False}
    return classifier_params


def get_affixes_removal_params(language, mode):
    if language == 'finnish':
        params = {'remove_affixes': 'suffix', 'max_affix_length': 10}
    elif language == 'german':
        params = {'remove_affixes': None}
    elif language == 'navajo':
        params = {'remove_affixes': 'prefix', 'max_affix_length': 10}
    elif language == 'russian':
        params = {'remove_affixes': 'suffix'}
        # params = {'remove_affixes': 'suffix', 'max_affix_length': 8}
    elif language == 'turkish':
        params = {'remove_affixes': 'suffix', 'max_affix_length': 8}
    elif language == 'arabic':
        params = {'remove_affixes': 'suffix' if mode == 'form' else None}
    elif language == 'georgian':
        params = {'remove_affixes': 'suffix', 'max_affix_length': 8}
    else:
        params = {'remove_affixes': None}
    return params

def save_language_model(data, outfile):
    language_model_to_save = LanguageModel(order=self.lm_order)
    language_model_to_save.train(data)
    language_model_to_save.make_WittenBell_smoothing()
    language_model_to_save.make_arpa_file(outfile)
    return


class BasicSigmorphonGuesser:
    """
    Базовый класс для порождения словоформ и лемм
    """

    def __init__(self, language=None, gap=1, initial_gap=0):
        self.language = language
        self.data_handler = InflectionDataHandler(gap, initial_gap)

    @property
    def transform_codes(self):
        return self.data_handler.transform_codes

    @property
    def problems_number(self):
        return self.data_handler.problems_number

    @property
    def problem_codes(self):
        return self.data_handler.problem_codes

    @property
    def paradigmers(self):
        return self.data_handler.paradigmers

    @property
    def lcs_searcher(self):
        return self.data_handler.lcs_searcher

    def make_paradigms_from_data(self, data):
        return self.data_handler.make_paradigms_from_data(data)


class AffixesRemover:

    def __init__(self, remove_affixes=None, max_affix_length=-1, affix_key='pos',
                 affix_threshold=0.1, min_affix_count=3):
        self.remove_affixes = remove_affixes
        self.max_affix_length = max_affix_length
        self.affix_key = affix_key
        self.affix_threshold = affix_threshold
        self.min_affix_count = min_affix_count
        self.affixes_to_remove = {'prefix': defaultdict(list), 'suffix': defaultdict(list)}
        self._initialize()

    def _initialize(self):
        if self.remove_affixes is None:
            self.task_extractor = self.second_task_extractor = None
            return self
        elif isinstance(self.remove_affixes, str):
            self.remove_affixes = [self.remove_affixes]
        for affix_type in self.remove_affixes:
            if affix_type not in ['prefix', 'suffix']:
                raise ValueError("affix_type should be 'prefix' or 'suffix'")
        if self.affix_key == 'pos':
            self.task_extractor = lambda x: x['pos']
            self.second_task_extractor = lambda x: x[1][0]
        elif self.affix_key == 'task':
            self.task_extractor = get_descr_string
            self.second_task_extractor = get_descr_string
        else:
            raise ValueError("Affix_key must be 'pos' or 'task'.")
        return self

    def train(self, data):
        self.affixes_to_remove = {'prefix': defaultdict(list), 'suffix': defaultdict(list)}
        if self.remove_affixes is not None:
            data_for_affixes_removal =\
                [(elem[0], self.task_extractor(x)) for elem in data for x in elem[1]]
            self.affixes_to_remove = {'prefix': defaultdict(list), 'suffix': defaultdict(list)}
            for affix_type in self.remove_affixes:
                self.affixes_to_remove[affix_type] = calculate_affixes_to_remove(
                    data_for_affixes_removal, affix_type, self.max_affix_length,
                    self.affix_threshold, self.min_affix_count)
        return self

    def get_affixes(self, problem_descr):
        if self.remove_affixes is not None:
            task_for_affixes = self.second_task_extractor(problem_descr)
            prefixes = self.affixes_to_remove['prefix'][task_for_affixes]
            suffixes = self.affixes_to_remove['suffix'][task_for_affixes]
        else:
            prefixes, suffixes = [], []
        return prefixes, suffixes

class SigmorphonFormGuesser(BasicSigmorphonGuesser):

    def __init__(self, cls, language=None, gap=1, initial_gap=0,
                 remove_affixes=None, max_affix_length=-1, affix_key='pos',
                 affix_threshold=0.1, min_affix_count=3,
                 fit_lm=False, lm_file=None, lm_order=3, save_lm_file=None):
        super().__init__(language=language, gap=gap, initial_gap=initial_gap)
        self.cls = cls
        self.affixes_remover = AffixesRemover(remove_affixes, max_affix_length, 'pos',
                                              affix_threshold, min_affix_count)
        self.fit_lm = fit_lm
        self.lm_file = lm_file
        self.lm_order = lm_order
        self.save_lm_file = save_lm_file


    def fit(self, data):
        if fit_lm:
            if self.lm_file is None:
                if self.save_lm_file is None:
                    raise ValueError("Either lm_file or save_lm_file should be given")
                save_language_model([list(x[2][0]) for x in data], self.save_lm_file)
                self.lm_file = self.save_lm_file
            joint_classifier_params =\
                {'has_language_model': True, 'lm_file': self.lm_file,
                 'has_joint_classifier': True, 'max_lm_coeff': 2.0}
        else:
            joint_classifier_params =\
                {'has_language_model': False, 'has_joint_classifier': False}
        self.affixes_remover.train([elem[:1] for elem in data])
        joint_data = self.make_paradigms_from_data(data)
        transformations_handler = TransformationsHandler(self.transform_codes)
        transformation_classifier_params = {'select_features': 'ambiguity',
                                            'selection_params': {'nfeatures': 0.25, 'min_count': 2}}
        self.classifiers = [None] * self.problems_number
        classifier_params = get_classifier_params(self.language)
        # classifier_params['paradigm_table'] = self.transform_codes
        classifier_params['min_feature_count'] = 3
        classifier_params['nfeatures'] = 0.1
        for i, problem_descr in enumerate(self.problem_codes):
            curr_classifier_params = copy.copy(classifier_params)
            prefixes, suffixes = self.affixes_remover.get_affixes(problem_descr)
            curr_classifier_params['prefixes_to_remove'] = prefixes
            curr_classifier_params['suffixes_to_remove'] = suffixes
            self.classifiers[i] = JointParadigmClassifier(
                ParadigmClassifier(self.transform_codes), transformations_handler,
                curr_classifier_params, transformation_classifier_params, **joint_classifier_params)
        data_by_problems = arrange_data_by_problems(
            joint_data, self.problems_number, has_answers=True)
        for i, (_, curr_X, curr_y) in enumerate(data_by_problems):
            # if i != 7:
            #     continue
            if i % 20 == 0:
                print("Classifier {} fitting...".format(i+1))
            self.classifiers[i].fit(curr_X, [[x] for x in curr_y])
        return self

    def predict(self, data, return_by_problems=False, return_descr=False):
        # data_for_problems = [(lemma, [problem_code]), ...]
        data_for_problems =\
            self.data_handler.extract_data(data, has_codes=True, has_answers=False)
        # data_for_problems = [(lemma, problem_code), ...]
        data_for_problems = [(lemma, codes[0]) for lemma, codes in data_for_problems]
        data_by_problems = arrange_data_by_problems(data_for_problems, has_answers=False)
        current_problems_number = len(data_by_problems)
        if return_by_problems:
            answers = [[] for _ in range(current_problems_number)]
        else:
            answers = [None] * len(data)
        for i, (indexes, curr_X, _) in enumerate(data_by_problems):
            # if i != 16:
            #     continue
            if len(curr_X) == 0:
                continue
            if i >= self.problems_number:
                # проблема не встречалась в обучающей выборке, потому выдаём тождественный ответ
                curr_descrs = [['1', '1'] for lemma in curr_X]
                curr_words = [lemma for lemma in curr_X]
            else:
                if i % 20 == 0:
                    print("Classifier {} predicting...".format(i+1))
                cls = self.classifiers[i]
                curr_answers = [x[0] for x in cls.predict(curr_X)]
                curr_descrs, curr_words = [], []
                for lemma, (transform_code, var_values) in zip(curr_X, curr_answers):
                    if transform_code is not None:
                        fragmentor = self.paradigmers[transform_code]
                        curr_descrs.append(fragmentor.descr)
                        # print(lemma, fragmentor.descr, var_values)
                        curr_words.append(fragmentor._substitute_words(var_values)[1])
                    else:
                        # не удалось найти ни одного подходящего класса
                        # здесь надо будет повызывать другие классификаторы
                        # пока возвращает тождественный ответ
                        print("No transforms for lemma {}, problem {}".format(
                            lemma, list(self.problem_codes.keys())[i]))
                        curr_descrs.append(['1', '1'])
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




class SigmorphonFormTransformer(BasicSigmorphonGuesser):

    def __init__(self, word_cls, lemma_cls, joint_classifier=None,
                 language=None, gap=1, initial_gap=0,
                 form_affix_removal_params=None, lemma_affix_removal_params=None,
                 fit_lm=False, lm_file=None, lm_order=3, save_lm_file=None):
        super().__init__(language=language, gap=gap, initial_gap=initial_gap)
        self.word_cls = word_cls
        self.lemma_cls = lemma_cls
        self.joint_classifier = joint_classifier
        self.form_affix_removal_params = form_affix_removal_params
        self.lemma_affix_removal_params = lemma_affix_removal_params
        self.fit_lm = fit_lm
        self.lm_file = lm_file
        self.lm_order = lm_order
        self.save_lm_file = save_lm_file

    def _initialize_affix_removal_params(self):
        if self.form_affix_removal_params is None:
            self.form_affix_removal_params = dict()
        if self.lemma_affix_removal_params is None:
            self.lemma_affix_removal_params = dict()
        self.form_affix_remover =\
            AffixesRemover(affix_key='task', **self.form_affix_removal_params)
        self.lemma_affix_remover =\
            AffixesRemover(affix_key='pos', **self.lemma_affix_removal_params)
        return self

    def fit(self, data):
        self._initialize_affix_removal_params()
        reversed_data =\
            [((elem[2][0], elem[1], [elem[0]]) + elem[2:]) for elem in data]
        self.form_affix_remover.train(reversed_data)
        joint_data = self.make_paradigms_from_data(data)
        joint_reversed_data = self.make_paradigms_from_data(reversed_data)
        transformations_handler = TransformationsHandler(self.transform_codes)
        transformation_classifier_params = {'select_features': 'ambiguity',
                                            'selection_params': {'nfeatures': 0.25, 'min_count': 2}}
        classifier_params = get_classifier_params(self.language)
        classifier_params['paradigm_table'] = self.transform_codes
        classifier_params['min_feature_count'] = 3
        classifier_params['nfeatures'] = 0.25
        reversed_classifier_params = copy.copy(classifier_params)
        reversed_classifier_params['max_length'] = 6
        if not reversed_classifier_params.get('use_prefixes', False):
            reversed_classifier_params['use_prefixes'] = True
            reversed_classifier_params['max_prefix_length'] = 3
        self.direct_classifiers = [None] * self.problems_number
        self.reversed_classifiers = [None] * self.problems_number
        data_by_problems = arrange_data_by_problems(
            joint_data, self.problems_number, has_answers=True)
        reversed_data_by_problems = arrange_data_by_problems(
            joint_reversed_data, self.problems_number, has_answers=True)
        for i, problem_descr in enumerate(self.problem_codes):
            # классификаторы лемма-словоформа
            direct_classifier_params = copy.copy(classifier_params)
            prefixes, suffixes =\
                self.lemma_affix_remover.get_affixes(problem_descr)
            direct_classifier_params['prefixes_to_remove'] = prefixes
            direct_classifier_params['suffixes_to_remove'] = suffixes
            self.direct_classifiers[i] = JointParadigmClassifier(
                ParadigmClassifier(**direct_classifier_params), transformations_handler,
                dict(), transformation_classifier_params)
            _, curr_X, curr_y = data_by_problems[i]
            self.direct_classifiers[i].fit(curr_X, [[label] for label in curr_y])
            # классификаторы словоформа-лемма
            # удаляем суффиксы, здесь это особенно важно (???)
            curr_classifier_params = copy.copy(reversed_classifier_params)
            prefixes, suffixes =\
                self.form_affix_remover.get_affixes(problem_descr)
            curr_classifier_params['prefixes_to_remove'] = prefixes
            curr_classifier_params['suffixes_to_remove'] = suffixes
            self.reversed_classifiers[i] = JointParadigmClassifier(
                ParadigmClassifier(**curr_classifier_params),
                transformations_handler, dict(), transformation_classifier_params)
            _, curr_X, curr_y = reversed_data_by_problems[i]
            self.reversed_classifiers[i].fit(curr_X, [[label] for label in curr_y])
            if i % 20 == 0:
                print("Classifiers {} fitted".format(i+1))
        return self

    def _predict_probs(self, data, reverse):
        classifiers = self.reversed_classifiers if reverse else self.direct_classifiers
        data_for_problems =\
            self.data_handler.extract_data(data, has_codes=True, has_answers=False)
        # data_for_problems = [(lemma, problem_code), ...]
        data_for_problems = [(lemma, codes[0]) for lemma, codes in data_for_problems]
        data_by_problems = arrange_data_by_problems(data_for_problems, has_answers=False)
        current_problems_number = len(data_by_problems)
        answer = [None] * len(data)
        for i, (indexes, curr_X, _) in enumerate(data_by_problems):
            if len(curr_X) == 0:
                continue
            if i >= self.problems_number:
                # проблема не встречалась в обучающей выборке, потому выдаём тождественный ответ
                curr_words_with_probs = [[(lemma, 1.0)] for lemma in curr_X]
            else:
                if i % 20 == 0:
                    print("Classifier {} predicting...".format(i+1))
                cls = classifiers[i]
                curr_labels_with_probs = cls.predict_probs(curr_X)
                curr_words_with_probs = []
                for lemma, lemma_labels_with_probs in zip(curr_X, curr_labels_with_probs):
                    codes_and_var_values, probs = lemma_labels_with_probs
                    lemma_word_probs = defaultdict(float)
                    for (transform_code, var_values), prob in zip(codes_and_var_values, probs):
                        if transform_code is not None:
                            fragmentor = self.paradigmers[transform_code]
                            word = fragmentor._substitute_words(var_values)[1]
                        else:
                            # не удалось найти ни одного подходящего класса
                            # здесь надо будет повызывать другие классификаторы
                            # пока возвращает тождественный ответ
                            print("No transforms for lemma {}, problem {}".format(
                                lemma, list(self.problem_codes.keys())[i]))
                            word = lemma
                        lemma_word_probs[word] += prob
                    curr_words_with_probs.append(sorted(lemma_word_probs.items(),
                                                        key=(lambda x: x[1]), reverse=True))
            for j, elem in zip(indexes, curr_words_with_probs):
                answer[j] = elem
        return answer


    def predict(self, data):
        reversed_data = [(elem[1], elem[0]) for elem in data]
        lemmas_with_probabilities = self._predict_probs(reversed_data, reverse=True)
        data_for_inflection = [(lemma, elem[2]) for possible_lemmas, elem
                               in zip(lemmas_with_probabilities, data)
                               for lemma, _ in possible_lemmas]
        words_with_probabilities = self._predict_probs(data_for_inflection, reverse=False)
        answer = []
        start = 0
        for lemmas_with_probs in lemmas_with_probabilities:
            lemmas_number = len(lemmas_with_probs)
            end = start + lemmas_number
            current_answer = defaultdict(float)
            for (_, first_prob), words_with_probs in\
                    zip(lemmas_with_probs, words_with_probabilities[start:end]):
                for word, second_prob in words_with_probs:
                    current_answer[word] += first_prob * second_prob
            start = end
            answer.append(sorted(current_answer.items(), key=lambda x:x[1], reverse=True)[0][0])
        return answer


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

SHORT_OPTS = 'g:i:Iam:o:s:'
LONG_OPTS = ['gap=', 'initial_gap=', 'incorrect', 'group_all',
             'model', 'order', 'save_model']
MODES = ['make_paradigms', 'guess_inflection', 'test_inflection', 'guess_reinflection', 'test_reinflection']

if __name__ == "__main__":
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, SHORT_OPTS, LONG_OPTS)
    global LANGUAGE
    gap, initial_gap, output_only_incorrect, group_all = 1, 0, False, False
    language_model_infile, lm_order, language_model_save_file = None, None, None
    for opt, val in opts:
        if opt in ['-g', '--gap']:
            gap = int(val)
        elif opt in ['-i', '--initial_gap']:
            initial_gap = int(val)
        elif opt in ['-I', '--incorrect']:
            output_only_incorrect = True
        elif opt in ['-a', '--group_all']:
            group_all = True
        elif opt in ['-m', '--model']:
            language_model_infile = val
        elif opt in ['-o', '--order']:
            lm_order = int(val)
        elif opt in ['-s', '--save_model']:
            language_model_save_file = val
        else:
            raise ValueError("Unrecognized option: {0}".format(opt))
    if len(args) < 1 or args[0] not in MODES:
        sys.exit("First argument must be one of {0})".format(' '.join(MODES)))
    mode, args = args[0], args[1:]
    if mode == 'make_paradigms':
        if len(args) != 5:
            sys.exit("Pass language, train file, outfile for inflection, "
                     "outfile for paradigms, outfile for paradigm_stats")
        language, train_file, inflection_outfile, outfile, stats_outfile = args
    elif mode == 'guess_inflection':
        if len(args) != 4:
            sys.exit("Pass train file, test file and output file")
        language, train_file, test_file, outfile = args
    elif mode == 'test_inflection':
        if len(args) != 4:
            sys.exit("Pass train file, dev file, and output file")
        language, train_file, test_file, outfile = args
    elif mode == 'guess_reinflection':
        if len(args) != 5:
            sys.exit("Pass train file, dev file, task-1 train file and output file")
        language, train_file, test_file, additional_train_file, outfile = args
    elif mode == 'test_reinflection':
        raise NotImplementedError
    if mode in ['guess_inflection', 'test_inflection', 'guess_reinflection', 'test_reinflection']:
        basic_classifier = JointParadigmClassifier
    else:
        basic_classifier = None
    if language_model_infile is not None or language_model_save_file is not None:
        fit_lm = True
    else:
        fit_lm = False
    if mode in ['make_paradigms', 'guess_inflection', 'test_inflection']:
        input_data = read_input_file(train_file, group_all=group_all)
        if mode != 'make_paradigms':
            affix_removal_params = get_affixes_removal_params(language, 'lemma')
            affix_removal_params['affix_key'] = 'pos'
        else:
            affix_removal_params = dict()
        form_guesser = SigmorphonFormGuesser(
            cls=basic_classifier, language=language, gap=gap, initial_gap=initial_gap,
            lm_file=language_model_infile, fit_lm=fit_lm, lm_order=lm_order,
            save_lm_file=language_model_save_file, **affix_removal_params)
    elif mode in ['guess_reinflection', 'test_reinflection']:
        first_task_data = read_input_file(additional_train_file, group_all=False)
        affix_removal_params = get_affixes_removal_params(language, 'form')
        lemma_affix_removal_params = get_affixes_removal_params(language, 'lemma')
        form_guesser = SigmorphonFormTransformer(
            word_cls=basic_classifier, lemma_cls=basic_classifier,
            language=language, gap=gap, initial_gap=initial_gap,
            form_affix_removal_params=affix_removal_params)
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
        correct_paradigms_with_vars = form_guesser.lcs_searcher.calculate_paradigms(
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
    elif mode == 'guess_reinflection':
        form_guesser.fit(first_task_data)
        test_data = read_input_file(test_file, mode=2)
        answers = form_guesser.predict([tuple(x[:3]) for x in test_data])
        with open(outfile, "w", encoding="utf8") as fout:
            for (first_descr, lemma, second_descr, _), word in zip(test_data, answers):
                fout.write("{0}\t{1}\t{2}\t{3}\n".format(get_descr_string(first_descr[0]), lemma,
                                                    get_descr_string(second_descr[0]), word))

