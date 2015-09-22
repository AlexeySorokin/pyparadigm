# -*- coding: utf-8 -*-

import sys

from functools import reduce
from itertools import product, chain
from collections import defaultdict, OrderedDict
import numpy as np
import re

from utility import extract_ordered_sequences

class WordGraph:

    @classmethod
    def wordtograph(cls, word):
        '''
        Преобразует слово в ДКА, распознающий все его подслова
        '''
        word_length = len(word)
        trans = dict() # словарь для переходов
        '''
        positions_to_the_right[a] - позиции символа a справа от текущей позиции
        '''
        positions_to_the_right = defaultdict(tuple)
        for i, symbol in list(enumerate(word))[::-1]:
            '''
            Идём по слову справа налево, обновляя positions_to_the_right и добавляя переходы
            '''
            positions_to_the_right[symbol] += (i,)
            for other_symbol, positions in positions_to_the_right.items():
                # минимальный элемент в positions --- последний
                trans[(i, other_symbol)] = (positions[-1]+1, positions)
        graph = cls(trans)
        return graph

    @classmethod
    def get_lcs_candidates(cls, words):
        word_graphs = [WordGraph.wordtograph(x) for x in words]
        intersection_graph = reduce(lambda x, y: x.intersect(y), word_graphs)
        lcs_candidates = intersection_graph.longestwords
        return lcs_candidates

    def __init__(self, transitions):
        self.alphabet = {symbol for (state, symbol) in transitions}
        self.states = {state for (state, symbol) in transitions} | set([i[0] for i in transitions.values()])
        self.transitions = transitions
        self.revtrans = {}
        for (state, sym) in self.transitions:
            if self.transitions[(state, sym)][0] in self.revtrans:
                self.revtrans[self.transitions[(state, sym)][0]] += [(state, sym, self.transitions[(state, sym)][1])]
            else:
                self.revtrans[self.transitions[(state, sym)][0]] = [(state, sym, self.transitions[(state, sym)][1])]

    def __getattr__(self, attr):
        if attr == 'longestwords':
            self._maxpath()
            return self.longestwords
        raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))

    def intersect(self, other):
        alphabet = self.alphabet & other.alphabet
        stack = [(0, 0)]
        statemap = {(0, 0): 0}
        nextstate = 1
        trans = {}
        while len(stack) > 0:
            (asource, bsource) = stack.pop()
            for sym in alphabet:
                if (asource, sym) in self.transitions and (bsource, sym) in other.transitions:
                    atarget = self.transitions[(asource, sym)][0]
                    btarget = other.transitions[(bsource, sym)][0]
                    if (atarget,btarget) not in statemap:
                        statemap[(atarget, btarget)] = nextstate
                        nextstate += 1
                        stack.append((atarget, btarget))
                    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
                    trans[(statemap[(asource, bsource)], sym)] = (statemap[(atarget, btarget)], flatten([self.transitions[(asource, sym)][1], other.transitions[(bsource, sym)][1]]))
        return WordGraph(trans)

    def _backtrace(self, maxsources, maxlen, state, tempstring, indexes):
        if state not in self.revtrans: # странный способ проверять на достижение начального состояния
            tempstring.reverse()
            indexes.reverse()
            # В списке надо хранить пары виды (строка, индекс)
            self.longestwords.append(("".join(tempstring), indexes))
            return
        for (backstate, symbol, idx) in self.revtrans[state]:
            if maxlen[backstate] == maxlen[state] - 1:
                self._backtrace(maxsources, maxlen, backstate, tempstring + [symbol], indexes+[idx])

    def _maxpath(self):
        tr = {}
        for (state, sym) in self.transitions:
            if state not in tr:
                tr[state] = set()
            tr[state].update({self.transitions[(state, sym)][0]})
        S = {0}
        maxlen = {}
        maxsources = {}
        for i in self.states:
            maxlen[i] = 0
            maxsources[i] = {}
        step = 1
        while len(S) > 0:
            Snew = set()
            for state in S:
                if state in tr:
                    for target in tr[state]:
                        if maxlen[target] < step:
                            maxsources[target] = {state}
                            maxlen[target] = step
                            Snew.update({target})
                        elif maxlen[target] == step:
                            maxsources[target] |= {state}
            S = Snew
            step += 1
        endstates = [key for key, val in maxlen.items() if val == max(maxlen.values())]
        self.longestwords = []
        for w in endstates:
            self._backtrace(maxsources, maxlen, w, [],[])

##########################################

def vars_to_string(baseform, varlist):
    """Input: baseform, variable list
       Output: string of type '0=baseform,1=var1,2=var2...'"""
    varlist = tuple(varlist)
    answer =  ",".join((("{0}={1}".format(i, word))
                        for i, word in enumerate((baseform,) + varlist)))
    return answer



def extract_tables(tables):
    """
    Input: list of tables
    Output: Dictionary of paradigm tables with the list of their members.
    """
    words_by_vartables = defaultdict(set)
    for lemma, table, forms_with_vars, var_spans in tables:
        forms_with_vars = tuple(forms_with_vars)
        to_add = (lemma, tuple(var_spans))
        words_by_vartables[forms_with_vars].add(to_add)
    return words_by_vartables

##########################################

def extract_best_lcss(lcs_candidates, forms, method='Hulden'):
    '''
    Возвращает наилучшие общие последовательности на основе метода method
    по списку общих подпоследовательностей и списка исходных слов
    '''
    # вначале приводим последовательности индексов в приличный вид
    # lcs_sequences = [('песк', [[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]]),
    #                  ('песо', [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 4]])]
    lcs_sequences = [(lcs, extract_index_sequences(indexes)) for lcs, indexes in lcs_candidates]
    # lcss = ['песк'], lcss_indexes = [[[[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]]]]
    lcss, lcss_indexes = extract_best_sequence(lcs_sequences, forms, method)
    # best_lcss = [('песк',  [[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]])]
    best_lcss = list(chain.from_iterable([[(lcs, indexes) for indexes in lcs_indexes]
                                          for lcs, lcs_indexes in zip(lcss, lcss_indexes)]))
    return best_lcss

def extract_index_sequences(indexes):
    '''
    Преобразует индексы, хранящиеся в автомате, в нужный формат
    '''
    parsed_indexes = list(zip(*indexes))
    # a = []
    # for item in parsed_indexes:
    #    a.append(list(map(sorted, item)))
    # parsed_indexes = a
    parsed_indexes = [list(map(sorted, elem)) for elem in parsed_indexes]
    index_sequences = list(map(extract_ordered_sequences, parsed_indexes))
    return index_sequences

def extract_best_sequence(lcs_sequences, table, method='Hulden'):
    '''
    Извлекает наилучшую из наиболее длинных общих последовательностей

    Аргументы:
    ----------
    lcs_sequences: list, список наибольших общих подпоследовательностей
    table: list, список словоформ
    method: str ('Hulden', остальные пока не реализованы),
    метод извлечения наилучшей подпоследовательности

    Возвращает:
    -----------
    lcss: list, список наилучших lcs
    table_indexes: list of lists,
    список списков наилучших наборов индексов для каждой из наилучших lcs
    '''
    best_score = None
    best_lcss, best_indexes = [], []
    for lcs, indexes in lcs_sequences:
        best_lcs_indexes, score = extract_best_indexes(indexes, table, method)
        if best_score is None or score < best_score:
            best_lcss = [lcs]
            best_indexes = [best_lcs_indexes]
            best_score = score
        elif score == best_score:
            best_lcss.append(lcs)
            best_indexes.append(best_lcs_indexes)
        else:
            print(table)
            pass
    return best_lcss, best_indexes

def get_gap_scores(args):
    '''
    Аргументы:
    ----------
    args: list, список позиций разрывов lcs в строках

    Возвращает:
    -----------
    gap_positions_number: int, число непрерывных фрагментов в выравнивании для lcs
    total_gaps_number: int, суммарное число разрывов в выравнивании для lcs
    '''
    gap_positions_number = len(reduce((lambda x, y: x|y),
                                      (set(elem) for elem in args),
                                      set()))
    total_gaps_number = sum(len(elem) for elem in args)
    return (gap_positions_number, total_gaps_number)

def extract_best_indexes(indexes, table, method='Hulden'):
    if method == 'Hulden':
        # indexes = [[[0,1,2,3]], [[0,1,2,4]], [[0,1,2,4]]
        # вместо списков индексов из indexes
        # gap_indexes хранит позиции разрывов в этих списках
        # gap_indexes = [[[]], [[2]], [[2]]]
        gap_indexes = [list(map(find_gap_positions, elem)) for elem in indexes]
        # gap_indexes_combibations хранит все возможные наборы позиций разрыва
        # вместе с номерами списков, откуда взяты эти позиции
        # gap_indexes_combinations = [((0, []), (0, [2]), (0, [2]))]
        gap_indexes_combinations = list(product(*(enumerate(x) for x in gap_indexes)))
        # enumerated_index_combinations хранит парами номера списков индексов
        # в списке возможных вариантов позиций разрывов для каждой словоформы
        # и номера разрывов в этих списках
        # enumerated_index_combinations = [([0,0,0], [[], [2], [2]])]
        enumerated_index_combinations =\
            [tuple(map(list, zip(*elem))) for elem in gap_indexes_combinations]
        # enumerated_index_combinations =\
        #     [tuple(map(list, zip(*elem))) for elem in
        #      product(*(enumerate(x) for x in gap_indexes))]
        combinations_positions_with_scores = [(elem[0], get_gap_scores(elem[1]))
                                              for elem in enumerated_index_combinations]
        _, best_score = min(combinations_positions_with_scores, key=(lambda x: x[1]))
        best_combinations_positions =\
            np.where([(x[1] == best_score) for x in combinations_positions_with_scores])
        best_indexes_positions = [combinations_positions_with_scores[elem[0]][0]
                                  for elem in best_combinations_positions]
        best_indexes = [[form_indexes[i] for form_indexes, i in zip(indexes, elem)]
                        for elem in best_indexes_positions]
    else:
        raise NotImplementedError
    return best_indexes, best_score

def find_gap_positions(lst):
    '''
    Находит те позиции i в упорядоченном списке lst, что lst[i+1] > lst[i] + 1
    '''
    if len(lst) <= 1:
        return []
    gap_positions = []
    prev = lst[0]
    for i, curr in enumerate(lst[1:]):
        if curr > prev + 1:
            gap_positions.append(i)
        prev = curr
    return gap_positions

def compute_paradigm(table, indexes, var_beginnings):
    '''
    Аргументы:
    ----------
    table: list
    список словоформ, для которого вычисляется парадигма
    indexes: list of lists
    список списков, содержащих позиции наибольшей общей последовательности
    var_beginnings: list
    позиции наибольшей общей последовательности, в которых начинаются переменные

    Возвращает:
    -----------
    paradigms: list
    список словоформ, в которых переменные заменены на индексы

    Пример:
    --------
    Вход:
    table=['песок', 'песком', 'песков'],
    indexes=[[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]],
    var_beginnings=[0, 3, 4]

    Выход:
    --------
    paradigm = ['1+о+2', '1+2+ом', '1+2+ов']
    '''
    paradigm = []
    vars_number = len(var_beginnings) - 1
    for form, form_indexes in zip(table, indexes):
        var_spans = []
        curr = var_beginnings[0]
        for next in var_beginnings[1:]:
            var_spans.append((form_indexes[curr], form_indexes[next-1] + 1))
            curr = next
        var_spans.append((len(form), None))
        fragments_list = []
        if var_spans[0][0] > 0:
            fragments_list.append(form[:var_spans[0][0]])
        for i in range(1, vars_number + 1):
            fragments_list += [str(i), form[var_spans[i-1][1]:var_spans[i][0]]]
        form_with_vars = "+".join(fragments_list)
        if form_with_vars.endswith("+"):
            form_with_vars = form_with_vars[:-1]
        form_with_vars = re.sub("[+]+", "+", form_with_vars)
        paradigm.append(form_with_vars)
    return paradigm

# вход/выход

ru_cases = ['им', 'род', 'дат', 'вин', 'тв', 'пр']
ru_numbers = ['ед', 'мн']

la_cases = ['ном', 'ген', 'дат', 'акк', 'абл', 'вок']
la_numbers = ['ед', 'мн']

def make_categories_marks(language):
    '''
    Возвращает допустимые грамматические описания в зависимости от языка
    '''
    if language == 'RU':
        marks = list(map(tuple, product(ru_cases, ru_numbers)))
    elif language == 'LA':
        marks = list(map(tuple, product(la_cases, la_numbers)))
    return marks

def read_input(infile, language, method='first'):
    '''
    Аргументы:
    ----------
    infile: файл с записями вида
    1 <лемма_1>
    <форма_11>,<лемма_1>,<морфологические_показатели_1>
    <форма_12>,<лемма_1>,<морфологические_показатели_2>
    ...
    2 <лемма_2>
    <форма_21>,<лемма_2>,<морфологические_показатели_1>
    <форма_22>,<лемма_2>,<морфологические_показатели_2>
    ...
    method: str, 'first' or 'all' (default='first')
    метод добавления форм в парадигму
    first --- добавляется только первая словоформа для данной граммемы
    all --- добавляются все словоформы

    Возвращает:
    ----------
    tables: список вида [(лемма_1, формы_1), (лемма_2, формы_2), ...]
    '''
    # сразу создаём функцию, которая проверяет, следует ли добавлять форму
    # в список. Так делаем, чтобы if вызывался только 1 раз
    marks = make_categories_marks(language)
    if method == 'first':
        add_checker = (lambda x: (x is not None) and len(x) == 0)
        def table_adder(lemma, table_dict):
            forms = [(elem[0] if len(elem) > 0 else '-') for elem in table_dict.values()]
            # возвращает список, потому что в случае 'all'
            # придётся возвращать несколько парадигм, то есть список
            return [(lemma, forms)]
    elif method == 'all':
        add_checker = (lambda x: x is not None)
        # пока не очень понимаю, что возвращать в этом случае,
        # поэтому оставлю без реализации
        raise NotImplementedError
    else:
        sys.exit("Method must be 'first' or 'all'.")
    tables = []
    current_table_dict = OrderedDict((mark, []) for mark in marks)
    has_forms = False # индикатор того, нашлись ли у слова словоформы
    with open(infile, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip() # всегда удаляйте лишние пробелы по краям строчек
            line = line.strip('\ufeff') # удалим метку кодировки
            splitted_line = line.split(',') # используйте говорящие имена переменных
            if len(splitted_line) == 1:
                # строчка вида i_1 <лемма_1>
                if has_forms:
                    tables += table_adder(lemma, current_table_dict)
                    current_table_dict = OrderedDict((mark, []) for mark in marks)
                    has_forms = False
                splitted_line = splitted_line[0].split("\t")
                # lemma = splitted_line[-1]
                index, lemma = splitted_line
                if int(index) % 500 == 1:
                    print(index)
            else:
                form, form_lemma = splitted_line[:2]
                mark = tuple(splitted_line[2:])
                if form not in ['—', '-'] and form_lemma == lemma:
                    if add_checker(current_table_dict.get(mark, None)):
                        current_table_dict[mark].append(form)
                        has_forms = True
        if has_forms:
            tables += table_adder(lemma, current_table_dict)
    return tables

def output_paradigms(tables_by_paradigm, outfile, short_outfile=None):
    with open(outfile, "w", encoding='utf-8') as fout:
        if short_outfile is not None:
            try:
                fshort = open(short_outfile, "w", encoding='utf-8')
            except IOError as err:
                print("I/O error({0}): {1}".format(err.errno, err.strerror))
                short_outfile = None
        count = 1
        for paradigm, var_values in sorted(tables_by_paradigm.items(),
                                           key=(lambda x:len(x[1])),
                                           reverse=True):
            paradigm_str = "#".join(paradigm) + "\n"
            fout.write(paradigm_str)
            if short_outfile is not None:
                fshort.write("{0:<6}".format(count) + paradigm_str)
                first_elem_str = vars_to_string(*(sorted(var_values)[0]))
                fshort.write("{0:<6}{1}\n".format(len(var_values), first_elem_str))
                count += 1
            fout.write("\n".join((vars_to_string(*elem) for elem in var_values)))
            fout.write("\n")
        if short_outfile is not None:
            fshort.close()


# красиво преобразуем файл

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 4:
        sys.exit("Pass input file, language code, output file and output stats file")
    infile, language, outfile, stats_outfile = args
    # infile = "lat_nouns.txt"
    tables = read_input(infile, language)

    filtered_tables = []
    for num, (lemma, table) in enumerate(tables):
        if num % 100 == 1:
            print(num)
        # получаем lcs вместе с индексами
        # lemma = 'песок', tables = ['песок', 'песком', 'песков']
        correct_indices, correct_table = [], []
        for i, form in enumerate(table):
            if form not in ['—', '-']: # прочерки бывают разные...
                correct_indices.append(i)
                correct_table.append(form)
        # lcs_candidates = [('песк', [[(0,), (0,), (0,)], [(1,), (1,), (1,)], [(2,), (2,), (2,)], [(4,), (3,), (3,)]]),\
        #                   ('песо', [[(0,), (0,), (0,)], [(1,), (1,), (1,)], [(2,), (2,), (2,)], [(3,), (4,), (4,)]]),]
        lcs_candidates = WordGraph.get_lcs_candidates(correct_table)
        # best_lcss = [('песк',  [[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3]])]
        best_lcss = extract_best_lcss(lcs_candidates, correct_table)

        for lcs, lcs_indexes in best_lcss:
            gap_positions = reduce((lambda x,y:(x|y)),
                                   map(set, map(find_gap_positions, lcs_indexes)), set())
            # gap_positions = [2]
            gap_positions = sorted(gap_positions)
            # var_beginnings = [0, 3, 4]
            var_beginnings = [0] + [i+1 for i in gap_positions] + [len(lcs)]
            # lcs_vars = ['пес', 'к']
            lcs_vars = [lcs[i:j] for i,j in zip(var_beginnings[:-1], var_beginnings[1:])]
            # paradigm_repr = ['1+о+2', '1+2+ом', '1+2+ов']
            paradigm_repr = compute_paradigm(correct_table, lcs_indexes, var_beginnings)
            # вспоминаем, что парадигма могла быть дефектной
            final_paradigm_repr = ['-' for form in table]
            for i, form in zip(correct_indices, paradigm_repr):
                final_paradigm_repr[i] = form
            filtered_tables.append([lemma, table, final_paradigm_repr, lcs_vars])

    tables_by_paradigm = extract_tables(filtered_tables)
    output_paradigms(tables_by_paradigm, outfile, stats_outfile)