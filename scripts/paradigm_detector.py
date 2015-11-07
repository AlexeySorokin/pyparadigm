import numpy as np
import copy
import re
from itertools import chain, product
from collections import OrderedDict

import utility
import common


def make_flection_paradigm(paradigm):
    """
    Превращает парадигмы, заканчивающиеся переменной, в '1',
    а заканчивающиеся константой w --- в '1+w'
    """
    first = paradigm.split('#')[0]
    first = first.split('+')
    if first[-1].isdigit():
        return '1'
    else:
        return ('1+' if len(first) >= 2 else '') + first[-1]

def get_flection_length(pattern):
    """
    Вычисляет длину окончания в парадигме
    """
    pattern = pattern.split('+')
    if pattern[-1].isdigit() or pattern in ['-']:
        return 0
    else:
        return len(pattern[-1])


class ParadigmFragment:
    '''
    Класс, предназначенный для операций с описаниями фрагментов парадигм
    вида 1+о+2
    '''

    def __init__(self, descr):
        """
        Создаёт фрагментор по описанию

        Аргументы:
        -----------
        descr: str, описание парадигмы в формате, использованном в Ahlberg et al., 2014

        Атрибуты:
        ----------
        variable_indexes: list of ints, индексы переменных
        const_fragments: list of strs, список максимальных постоянных фрагментов
            между переменными, а также перед и после них,
            len(const_fragments) = len(variable_indexes) + 1
        """
        self.variable_indexes, variable_index = OrderedDict(), 0
        self.const_fragments = []
        regexp = "^"  # регулярное выражение, соответствующее шаблону
        var_fragments_count = 0
        start_positions_in_regexp = []
        splitted_descr = descr.split("+")
        last_constant_fragment = ""
        for i, part in enumerate(splitted_descr):
            if part.isdigit():  # переменная
                code = int(part)
                if code not in self.variable_indexes:
                    self.variable_indexes[code] = variable_index
                    variable_index += 1
                else:
                    raise ValueError("Variable should be present only once")
                var_fragments_count += 1
                # сохраняем текущий постоянный фрагмент
                self.const_fragments.append(last_constant_fragment)
                regexp += last_constant_fragment
                last_constant_fragment = ""
                # обновляем регулярное выражение
                start_positions_in_regexp.append((len(regexp) if i > 0 else 0))
                regexp += '.'
            else:
                last_constant_fragment += part
                # обновляем регулярное выражение
                if var_fragments_count > 0:
                    regexp += ".*"
                var_fragments_count = 0
        # сохраняем константный фрагмент после последней переменной
        self.const_fragments.append(last_constant_fragment)
        if var_fragments_count > 0:
            regexp += ".*"
        regexp += "{0}$".format(last_constant_fragment)
        self._regexp = re.compile(regexp)
        # регулярные выражения для участков шаблона
        # например, для шаблона 1+о+2+а self._fragments_regexps
        # содержит выражения о..*а$ и a$
        self._fragments_regexps = [re.compile('(?<={0})(?={1})'.format(fragment, regexp[i:]))
                                   for fragment, i in zip(self.const_fragments,
                                                          start_positions_in_regexp)]
        # заготовка для будущей подстановки значений переменных
        self._for_substitution = '{}'.join(self.const_fragments)
        self._precompute_differences()

    def substitute(self, var_values):
        '''
        Аргументы:
        ----------
        var_values: dictionary or list
        словарь вида {<номер переменной>: <значение>}
        или список значений переменных, начиная с 1-ой

        Возвращает:
        -----------
        Слово
        '''
        if isinstance(var_values, dict):
            var_values = [var_values[code] for code in self.variable_indexes]
        return self._for_substitution.format(*var_values)

    def extract_variables(self, word):
        """
        Извлекает значения переменных, возможные для данного слова
        """
        if not self._regexp.match(word):
            return []
        # в слове нет переменных
        if len(self.variable_indexes) == 0:
            return [[]]
        # находим позиции переменных
        variable_positions = [None] * len(self.variable_indexes)
        start_pos = len(self.const_fragments[0])
        variable_positions[0] = [start_pos]
        # переписать под поиск с конца
        for i, (const_fragment, regexp) in enumerate(zip(self.const_fragments[1:],
                                                         self._fragments_regexps[1:]), 1):
            start_pos += (len(const_fragment) + 1)
            # находит возможные позиции начал константных фрагментов,
            # определяя позиции, где могут начинаться соответствующие регулярные выражения
            matches = list(regexp.finditer(word, pos=start_pos))
            if len(matches) > 0:
                variable_positions[i] = [m.start() for m in matches]
                start_pos = variable_positions[i][0]
            else:
                return []
        # извлекаем возрастающие последовательности индексов
        variable_position_seqs = utility.extract_ordered_sequences(variable_positions + [[len(word)]],
                                                                   self._differences,
                                                                   strict_min=False)
        answer = []
        for seq in variable_position_seqs:
            answer.append([word[seq[i]:(seq[i+1] - len(part))]
                           for i, part in enumerate(self.const_fragments[1:])])
        return answer

    def fits_to_pattern(self, lemma):
        """
        Проверяет, подходит ли лемма под парадигму, не вычисляя переменных
        """
        return bool(self._regexp.match(lemma))

    def _precompute_differences(self):
        """
        Предвычисляет разницу между позициями соседних переменных
        """
        self._differences = [len(fragment) + 1 for fragment in self.const_fragments]
        if len(self._differences) > 1:
            self._differences[0] -= 1
        return


def fit_lemma_to_patterns(word, patterns):
    """
    Находит парадигмы, под которые подходит лемма
    """
    answer = []
    for pattern in patterns:
        fragmentor = ParadigmFragment(pattern)
        variable_values = fragmentor.extract_variables(word)
        if len(variable_values) > 0:
            answer.append((pattern, variable_values))
    return answer

class Paradigm:
    """
    Класс для описания морфологических парадигм

    Аргументы:
    ----------
    categories: list of strs, список категорий
    descr: str, описание парадигмы в формате <шаблон_1>#...#<шаблон_n>,
        где n --- число категорий

    Атрибуты:
    ----------
    paradigm_fragments: list of ParadigmFragment
        список фрагменторов для категорий
    """
    def __init__(self, categories, descr):
        splitted_descr = descr.split('#')
        if len(splitted_descr) == 0:
            raise ValueError("Description should be non-empty")
        if len(categories) != len(splitted_descr):
            raise ValueError("Description and categories list should have equal length.")
        self._category_indexes = OrderedDict()
        for i, cat in enumerate(categories):
            self._category_indexes[cat] = i
        self.descr = splitted_descr
        self._lemma_descr = self.descr[0]
        self.paradigm_fragments = [ParadigmFragment(x) for x in self.descr]
        self._lemma_paradigm_fragment = ParadigmFragment(self._lemma_descr)

    @property
    def categories(self):
        """
        Возвращает список категорий
        """
        return list(self._category_indexes.keys())

    def unique_forms_number(self, return_lemma=True):
        """
        Возвращает число уникальных шаблонов в описании парадигмы
        """
        if not hasattr(self, '_unique_forms_indexes_'):
            # предвычисляем позиции, в которых уникальные шаблоны
            # встречаются первый раз
            self._precompute_unique_forms_indexes()
        answer = len(self._unique_forms_indexes_)
        if answer == 0:
            return 0
        # если шаблон для леммы оказался уникальным (это происходит только в случае,
        # если он не является шаблоном никакой другой формы)
        if not return_lemma and self._unique_forms_indexes_[0] == 0:
            answer -= 1
        return answer

    def fit_word_to_form(self, cat, word):
        """
        Проверяет, может ли быть слово form формой категории cat для данной парадигмы
        """
        try:
            cat_index = self._category_indexes[cat]
        except KeyError:
            return []
        descr, fragmentor = self.descr[cat_index], self.paradigm_fragments[cat_index]
        answer = []
        for variable_values in fragmentor.extract_variables(word):
            lemma = self._lemma_paradigm_fragment.substitute(variable_values)
            answer.append((lemma, descr, variable_values))
        return answer

    def fit_lemma_to_form(self, lemma):
        variable_values = self._lemma_paradigm_fragment.extract_variables(lemma)
        return [(self._lemma_descr, elem) for elem in variable_values]

    def lemmatize(self, word):
        """
        Подбирает возможную лемму для слова при условии,
        что оно является одной из форм данной парадигмы
        """
        answer = []
        for cat, descr, fragmentor in zip(self.categories, self.descr, self.paradigm_fragments):
            for variable_values in fragmentor.extract_variables(word):
                lemma = self._lemma_paradigm_fragment.substitute(variable_values)
                answer.append((lemma, cat, descr, variable_values))
        return answer

    def substitute(self, var_values, return_lemma=True):
        """
        Подставляет вместо переменных значения var_values

        Аргументы:
        ----------
        var_values: list, список значений переменных
        return_lemma: bool, optional(default=True), следует ли возвращать значение леммы

        Возвращает:
        ------------
        answer: dict, словарь вида {<категория>: <словоформа данной категории>}
        """
        # ПЕРЕПИСАТЬ НА УНИКАЛЬНЫЕ ФОРМЫ
        answer = OrderedDict()
        start = 0 if return_lemma else 1
        for cat, fragmentor in zip(self.categories[start:],
                                   self.paradigm_fragments[start:]):
            answer[cat] = fragmentor.substitute(var_values)
        return answer

    def _substitute_words(self, var_values, return_lemma=True):
        """
        Подставляет вместо переменных значения var_values

        Аргументы:
        ----------
        var_values: list, список значений переменных
        return_lemma: bool, optional(default=True), следует ли возвращать значение леммы

        Возвращает:
        ------------
        answer: list, список словоформ в соответствии с порядком на категориях
        """
        start = 0 if return_lemma else 1
        return [fragmentor.substitute(var_values)
                for fragmentor in self.paradigm_fragments[start:]]

    def _make_all_forms(self, lemma, return_lemma=True):
        """
        Вычисляет словоформы данной парадигмы для заданной леммы

        Аргументы:
        ----------
        lemma: str, лемма
        return_lemma: bool, optional(default=True), следует ли возвращать значение леммы

        Возвращает:
        -----------
        answer: list of lists, каждый элемент в answer прелставляет собой список словоформ
        """
        variable_values = self._lemma_paradigm_fragment.extract_variables(lemma)
        return [self._substitute_words(elem, return_lemma=return_lemma)
                for elem in variable_values]

    def make_full_paradigm(self, lemma, return_lemma=True):
        """
        Вычисляет словоформы данной парадигмы для заданной леммы

        Аргументы:
        ----------
        lemma: str, лемма
        return_lemma: bool, optional(default=True), следует ли возвращать значение леммы

        Возвращает:
        -----------
        answer: list of dicts, каждый элемент в answer прелставляет собой словарь пар
            вида {<категория>: <словоформа данной категории>}
        """
        variable_values = self._lemma_paradigm_fragment.extract_variables(lemma)
        return [self.substitute(elem, return_lemma=return_lemma)
                for elem in variable_values]

    def extract_patterns(self, return_lemma=True):
        """
        Возвращает список шаблонов, входящих в парадигмы

        Аргументы:
        ----------
        return_lemma: bool, optional(default=True), следует ли возвращать шаблон для леммы

        Возвращает:
        -----------
        patterns: list, список шаблонов, входящих в self.descr
        """
        if not hasattr(self, '_patterns_'):
            # предвычисляем шаблоны
            self._precompute_patterns()
        return self._patterns_[(0 if return_lemma else 1)]

    def get_pattern_counts(self, return_lemma=True):
        """
        Возвращает, сколько раз каждый шаблон входит в парадигму

        Аргументы:
        ----------
        return_lemma: bool, optional(default=True), следует ли считать шаблон для леммы

        Возвращает:
        -----------
        patterns: dict, список пар {<шаблон>: <число вхождений шаблона в парадигму>}
        """
        if not hasattr(self, '_pattern_counts_'):
            self._precompute_patterns()
            self._pattern_counts_ = [list(map(len, self._patterns_[0])),
                                    list(map(len, self._patterns_[1]))]
        return self._pattern_counts_[(0 if return_lemma else 1)]

    def make_unique_forms(self, lemma, return_lemma=True):
        """
        Возвращает уникальные словоформы в описании парадигмы в порядке их первого появления
        """
        if not hasattr(self, '_unique_forms_indexes_'):
            self._precompute_unique_forms_indexes()
        start = 0 if (return_lemma or self._unique_forms_indexes_[0] != 0) else 1
        variable_values = self._lemma_paradigm_fragment.extract_variables(lemma)
        return [[self.paradigm_fragments[i].substitute(var_values)
                 for i in self._unique_forms_indexes_[start:]]
                for var_values in variable_values]

    def _precompute_unique_forms_indexes(self):
        """
        Предвычисляет индексы уникальных форм в self.descr
        """
        self._unique_forms_indexes_ = []
        used_descrs = set()
        for i, descr in enumerate(self.descr[1:], 1):
            if descr not in used_descrs:
                used_descrs.add(descr)
                self._unique_forms_indexes_.append(i)
        # 0 окажется в индексах уникальных форм
        # только если лемма не совпадает ни с одной из форм
        if self.descr[0] not in used_descrs:
            self._unique_forms_indexes_ = [0] + self._unique_forms_indexes_
        return self

    def _precompute_patterns(self):
        """
        Предвычисляет словарь вида {<шаблон>: <категории, которым соответствует данный шаблон>}
        """
        answer = OrderedDict()
        for descr, cat in zip(self.descr[1:], self.categories[1:]):
            if descr in answer:
                answer[descr].append(cat)
            else:
                answer[descr] = [cat]
        self._patterns_ = [copy.copy(answer), answer]
        self._patterns_[0][self._lemma_descr].append(self.categories[0])
        return self

class ParadigmProcessor:
    """
    Класс для одновременной работы с несколькими парадигмами

    Аргументы:
    ----------
    categories: list of strs, список категорий
    descrs: list of strs, список описаний парадигм в формате <шаблон_1>#...#<шаблон_n>,
        где n --- число категорий

    Атрибуты:
    ----------
    paradigmers: list of Paradigm, список обработчиков парадигм из self.descrs
    """
    def __init__(self, categories, descrs):
        self.categories = categories
        self.descrs = descrs
        self.paradigmers = [Paradigm(categories, descr) for descr in descrs]
        self._paradigmers_by_descrs = dict(zip(self.descrs, self.paradigmers))

    def lemmatize(self, form):
        """
        Вычисляет все варианты лемматизации формы в соответствии
        с одной из парадигм из self.descrs

        Аргументы:
        ----------
        form: str, форма, для которой производится лемматизация

        Возвращает:
        -----------
        answer: список вида (лемма, категория, описание парадигмы, значения переменных)
        """
        return list(chain.from_iterable(((lemma, cat, "#".join(paradigmer.descr), var_values)
                                         for (lemma, cat, _, var_values) in paradigmer.lemmatize(form))
                                        for paradigmer in self.paradigmers))

    def get_full_paradigms(self, form, return_lemma=True):
        """
        Вычисляет все варианты лемматизации формы в соответствии
        с одной из парадигм из self.descrs

        Аргументы:
        ----------
        form: str, форма, для которой производится лемматизация
        return_lemma: следует ли возвращать лемму при возврате парадигмы

        Возвращает:
        -----------
        answer: список вида (лемма, категория, описание парадигмы, парадигма)
        """
        lemmatization = self.lemmatize(form)
        answer = []
        for lemma, cat, descr, var_values in lemmatization:
            paradigmer = self._paradigmers_by_descrs[descr]
            paradigm = paradigmer._substitute(var_values, return_lemma=return_lemma)
            if not return_lemma:  # убираем описание леммы
                descr = descr[descr.find("#")+1:]
            answer.append((lemma, cat, descr, paradigm))
        return answer


def test_paradigm_processor():
    paradigm_descr = "1+ь#1+и#1+и#1+ей#1+и#1+ям#1+ь#1+и#1+ью#1+ями#1+и#1+ях"
    tags = [",".join(elem) for elem in common.get_categories_marks('RU')]
    paradigm_processor = Paradigm(tags ,paradigm_descr)
    print(paradigm_processor.make_full_paradigm('свирель'))
    print(paradigm_processor.make_full_paradigm('свирели'))

def test_extract_variables():
    descr = "1+ар+2+в+3"
    word = "варварство"
    fragment = ParadigmFragment(descr)
    for elem in fragment.extract_variables(word):
        print(",".join(elem))
    descr = "1+и"
    word = "честью"
    fragment = ParadigmFragment(descr)
    print(fragment._regexp.pattern)
    print(fragment._regexp.match(word))
    print(fragment.fits_to_pattern(word))

def test_unique_forms():
    categories = [common.LEMMA_KEY] + [",".join(x) for x in common.get_categories_marks('RU')]
    descr = "ребёночек#ребёночек#детки#ребёночка#деток#ребёночку#деткам#ребёночка#деток#ребёночком#детками#ребёночке#детках"
    paradigmer = Paradigm(categories, descr)
    paradigmer.make_unique_forms("ребёночек", return_lemma=False)

if __name__ == '__main__':
    # test_extract_variables()
    test_unique_forms()