import sys

from functools import reduce
from itertools import product, chain
from collections import defaultdict, OrderedDict
import numpy as np
import re

from utility import extract_ordered_sequences, find_optimal_cover_elements
from graph_utilities import Graph
from common import LEMMA_KEY, get_categories_marks, get_possible_lemma_marks


class LcsSearcher:
    """
    Класс для поиска наилучшей общей подпоследовательности

    Атрибуты:
    ---------
    gap, int or None(optional, default=None):
        максимальный пробел между элементами LCS в середине слова,
        gap=None --- допустим любой пробел
    initial_gap, int or None(optional, default=None):
        максимальный пропуск до первого элемента LCS,
        initial_gap=None --- допустим любой пробел
    method, str('Hulden', optional, default='Hulden'):
        метод для определения наилучшей LCS
    """

    def __init__(self, gap=None, initial_gap=None, method='Hulden', count_gaps=True):
        self.gap = gap
        self.initial_gap = initial_gap
        self.method = method
        self.count_gaps = count_gaps
        self.paradigm_counts = defaultdict(int)

    def process_table(self, words):
        """
        Вычисляет LCS для words
        """
        self._make_automaton(words)
        best_lcss = self._find_best_subsequence()
        # надо распределить индексы по омонимичным формам
        final_best_lcss = []
        for lcs, indexes in best_lcss:
            new_indexes = [None] * len(words)
            for word, word_indexes in zip(self.word_weights_.keys(), indexes):
                for form_index in self._form_indexes_by_word[word]:
                    new_indexes[form_index] = word_indexes
            final_best_lcss.append((lcs, new_indexes))
        if len(final_best_lcss) == 0:
            final_best_lcss = [("", [[] for word in words])]
        return final_best_lcss


    def calculate_paradigms(self, tables, count_paradigms=False):
        """
        Предсказывает парадигму для каждой таблицы склонения в tables
        В случае, если предсказано несколько парадигм, отбирает самую частотную
        """
        candidate_paradigms_with_vars =\
            [self.calculate_all_paradigms(table) for table in tables]
        # считаем парадигмы
        if count_paradigms:
            for paradigms_with_vars in candidate_paradigms_with_vars:
                for paradigm, _ in paradigms_with_vars:
                    self.paradigm_counts[paradigm] += 1
        # отбираем наиболее частотные
        answer = []
        for elem in candidate_paradigms_with_vars:
            answer.append(max(elem, key=(lambda x:self.paradigm_counts[x[0]])))
        return answer

    def calculate_all_paradigms(self, table):
        correct_indices, correct_table = _make_correct_table(table)
        best_lcss = self.process_table(correct_table)
        paradigms_with_vars = []
        for lcs, lcs_indexes in best_lcss:
            # gap_positions = [2]
            gap_positions = reduce((lambda x, y: (x | y)),
                                   map(set, map(find_gap_positions, lcs_indexes)), set())
            gap_positions = sorted(gap_positions)
            # var_beginnings = [0, 3, 4]
            if lcs != "":
                var_beginnings = [0] + [i for i in gap_positions] + [len(lcs)]
            else:
                var_beginnings = [0]
            # lcs_vars = ['пес', 'к']
            lcs_vars = [lcs[i:j] for i, j in zip(var_beginnings[:-1], var_beginnings[1:])]
            # paradigm_repr = ['1+о+2', '1+2+ом', '1+2+ов']
            paradigm_repr = compute_paradigm(correct_table, lcs_indexes, var_beginnings)
            # вспоминаем, что парадигма могла быть дефектной
            final_paradigm_repr = ['-' for form in table]
            for i, form in zip(correct_indices, paradigm_repr):
                final_paradigm_repr[i] = form
            paradigms_with_vars.append((tuple(final_paradigm_repr), lcs_vars))
        return paradigms_with_vars

    def _make_automaton(self, words):
        """
        Строит ациклический ДКА для всех общих подпоследовательностей
        """
        # словарь для частот словоформ в таблице,
        # чтобы не хранить несколько раз один и тот же автомат
        self.word_weights_ = OrderedDict()
        self._form_indexes_by_word = defaultdict(list)
        for i, word in enumerate(words):
            self._form_indexes_by_word[word].append(i)
            if word in self.word_weights_:
                self.word_weights_[word] += 1
            else:
                self.word_weights_[word] = 1
        word_graphs = [WordGraph.word_to_graph(x) for x in self.word_weights_]
        # вычисляем автомат для общих подпоследовательностей
        self.common_subseq_automaton_ = reduce(lambda x, y: x.intersect(y), word_graphs)
        return self

    def _find_best_subsequence(self):
        '''
        Ищет наиболее длинное слово, принимаемое self.common_subseq_automaton_
        и удовлетворяющее условиям на gap и initial_gap
        '''
        if self.gap is None and self.initial_gap is None:
            # сразу ищем наиболее длинные подпоследовательности
            lcs_candidates = self.common_subseq_automaton_.find_longest_words()
        else:
            gap = (self.gap if self.gap is not None
                   else max(len(word) for word in self.word_weights_))
            initial_gap = (self.initial_gap if self.initial_gap is not None
                           else max(len(word) for word in self.word_weights_))
            lcs_candidates = []
            # вначале находим самые длинные подпоследовательности
            # и пытаемся найти удовлетворяющую требованиям на разрывы
            for lcs, possible_indexes in self.common_subseq_automaton_.find_longest_words():
                new_indexes = [[] for _ in possible_indexes]
                # form_indexes --- варианты набора индексов для словоформы
                for i, form_indexes in enumerate(possible_indexes):
                    # перебираем возможные наборы индексов и оставляем
                    # только те, где максимальный разрыв между соседними
                    # элементами не превышает self.gap, а начальный разрыв
                    # не превышает self.initial_gap
                    for seq in form_indexes:
                        if len(seq) == 1:
                            if seq[0] <= initial_gap:
                                new_indexes[i].append(seq)
                            continue
                        if ((gap + 1 >= max(((seq[j + 1] - v)
                                             for (j, v) in enumerate(seq[:-1]))))
                            and seq[0] <= initial_gap):
                            new_indexes[i].append(seq)
                # если для всех словоформ нашёлся подходящий набор индексов
                # то сохраняем подпоследовательность
                if all(((len(elem) > 0) for elem in new_indexes)):
                    lcs_candidates.append((lcs, new_indexes))
            if len(lcs_candidates) == 0:
                # строим автоматы для индексов отдельно по каждой словоформе
                coordinate_automata = _make_coordinate_automata(self.common_subseq_automaton_)
                coordinate_automata = [elem.make_unambigious_automaton(gap=gap, initial_gap=initial_gap)
                                       for elem in coordinate_automata]
                words_by_coordinates = \
                    [elem.find_longest_words() for elem in coordinate_automata]
                if any(len(x)==0 for x in words_by_coordinates):
                    lcs_candidates = []
                else:
                    lengths = set(len(elem[0][0]) for elem in words_by_coordinates)
                    # пытаемся найти общее слово среди слов максимальной длины,
                    # принимаемых покоординатными автоматами
                    if len(lengths) == 1:
                        common_words = extract_common_words(words_by_coordinates)
                        word_length = list(lengths)[0] - 1
                    else:
                        common_words = []
                        word_length = min(lengths)
                    # ищем общие слова, уменьшая их возможную длину
                    while len(common_words) == 0 and word_length > 0:
                        words_by_coordinates = [elem.words_of_fixed_length(word_length)
                                                for elem in coordinate_automata]
                        common_words = extract_common_words(words_by_coordinates)
                        word_length -= 1
                    lcs_candidates = common_words
        best_lcss = self._find_best_lcss(lcs_candidates)
        return best_lcss

    def _find_best_lcss(self, candidates):
        """
        Возвращает наилучшие общие последовательности на основе метода method
        по списку общих подпоследовательностей и списка исходных слов
        """
        # candidates = [('песк', [[(0, 1, 2, 4)], [(0, 1, 2, 3)], [(0, 1, 2, 3)]]),
        #               ('песо', [[(0, 1, 2, 3)], [(0, 1, 2, 4)], [(0, 1, 2, 4)]])]
        # вначале приводим последовательности индексов в приличный вид
        # lcs_sequences = [('песк', [[(0, 1, 2, 4)], [(0, 1, 2, 3)], [(0, 1, 2, 3)]]),
        #                  ('песо', [[(0, 1, 2, 3)], [(0, 1, 2, 4)], [(0, 1, 2, 4))])]
        if self.method == 'Hulden':
            func = self.calculate_Hulden_gap_scores
        else:
            candidates = chain.from_iterable((((lcs, tuple(indexes))
                                               for indexes in product(*lcs_indexes_list))
                                              for lcs, lcs_indexes_list in candidates))
            raise NotImplementedError
        best_score, best_indexes, best_lcss = None, [], []
        for lcs, indexes in candidates:
            score, score_indexes = func(indexes)
            if best_score is None or score < best_score:
                best_lcss = [(lcs, elem) for elem in score_indexes]
                best_score = score
            elif score == best_score:
                best_lcss.extend((lcs, elem) for elem in score_indexes)
        return best_lcss

    def calculate_Hulden_gap_scores(self, indexes):
        """
        Аргументы:
        ----------
        indexes: list, список индексов элементов общей подпоследовательности

        Возвращает:
        -----------
        gap_positions_number: int, число непрерывных фрагментов в выравнивании для lcs
        total_gaps_number: int, суммарное число разрывов в выравнивании для lcs
        """
        gap_positions = [[find_gap_positions(x, add_initial=True) for x in elem]
                         for elem in indexes]
        optimal_gap_positions_sets, optimal_cover_size =\
                find_optimal_cover_elements(gap_positions)
        total_best_score, best_indexes_combinations = None, []
        for optimal_gap_positions in optimal_gap_positions_sets:
            best_indexes_by_coordinate = [None] * len(indexes)
            current_total_score = 0
            for i, (coordinate_indexes, coordinate_gap_positions, weight) in\
                    enumerate(zip(indexes, gap_positions, self.word_weights_.values())):
                current_best_coordinate_indexes, best_score = [], None
                if len(coordinate_indexes) > 0:
                    for elem, elem_gap_positions in\
                            zip(coordinate_indexes, coordinate_gap_positions):
                        # пропускаем множества, содержащие неоптимальные индексы
                        if any(x not in optimal_gap_positions for x in elem_gap_positions):
                            continue
                        if self.count_gaps:
                            score = len(elem_gap_positions)
                        else:
                            score = 0
                        if best_score is None or score < best_score:
                            best_score = score
                            current_best_coordinate_indexes = [elem]
                        elif best_score == score:
                            current_best_coordinate_indexes.append(elem)
                best_indexes_by_coordinate[i] = current_best_coordinate_indexes
                current_total_score += weight * best_score
            indexes_combinations = [list(elem) for elem in product(*best_indexes_by_coordinate)]
            if total_best_score is None or current_total_score < total_best_score:
                total_best_score = current_total_score
                best_indexes_combinations = indexes_combinations
            elif current_total_score == total_best_score:
                best_indexes_combinations.extend(map(list, product(*best_indexes_by_coordinate)))
        return (optimal_cover_size, total_best_score), best_indexes_combinations

class WordGraph:
    """
    Класс для ациклических ДКА, где все состояния являются завершающими
    """

    @classmethod
    def word_to_graph(cls, word):
        """
        Преобразует слово в ДКА, распознающий все его подслова
        """
        word_length = len(word)
        states = set(range(word_length + 1))  # состояния
        trans = {_: dict() for _ in states}  # словарь для переходов
        # positions_to_the_right[a] - позиции символа a справа от текущей позиции
        positions_to_the_right = defaultdict(tuple)
        for i, symbol in list(enumerate(word))[::-1]:
            '''
            Идём по слову справа налево, обновляя positions_to_the_right и добавляя переходы
            '''
            positions_to_the_right[symbol] += (i,)
            for other_symbol, positions in positions_to_the_right.items():
                # минимальный элемент в positions --- последний
                trans[i][other_symbol] = [(positions[-1] + 1, (positions,))]
        graph = cls(states, trans, start_state=0, dim=1)
        return graph

    def __init__(self, states, transitions, start_state, dim):
        """
        Создаёт недетерминированный конечный автомат
        с переходами из transitions

        states: iterable, множество состояний
        transitions: dict, множество переходов
            каждый переход имеет вид state: state_transitions, где
            state \in states, transitions --- словарь вида symbol: data,
            data --- список вида [(state_1, indexes_1), ... (state_m, indexes)m)],
            перечисляющий все переходы из данного состояния по данному символу
        start_state: object, стартовое состояние
        dim: int, число координат в индексах переходов
        """
        self.states_number = len(states)
        self.dim = dim
        self._make_transitions(start_state, states, transitions)  # перекодируем состояния
        self._make_reverse_transitions()  # также сохраняем обратные переходы

    def __getattr__(self, attr):
        if attr == 'longest_words':
            return self._longest_words
        raise AttributeError("%r object has no attribute %r" % (self.__class__, attr))

    def intersect(self, other):
        """
        Функция, строящая пересечение данного автомата
        с другим автоматом other
        """
        # выполняем обход в глубину параллельно по обоим автоматам,
        # проходя лишь по переходам, имеющимся и там, и там
        new_start_state = (self.start_state, other.start_state)
        new_states_stack = [new_start_state]
        new_dim = self.dim + other.dim
        processed_states = set()
        new_trans = dict()
        while len(new_states_stack) > 0:
            source_state_pair = new_states_stack.pop()
            if source_state_pair in processed_states:
                continue  # состояние уже обработано
            first_source_transitions = self.transitions[source_state_pair[0]]
            second_source_transitions = other.transitions[source_state_pair[1]]
            new_trans[source_state_pair] = dict()
            current_transitions = new_trans[source_state_pair]
            for symbol, first_data in first_source_transitions.items():
                if symbol in second_source_transitions:
                    current_transitions[symbol] = []
                    second_data = second_source_transitions[symbol]
                    for ((first_dest, first_indexes), (second_dest, second_indexes)) \
                            in product(first_data, second_data):
                        new_states_stack.append((first_dest, second_dest))
                        # ТЕПЕРЬ ИНДЕКС --- ВСЕГДА СПИСОК КОРТЕЖЕЙ
                        current_transitions[symbol].append(
                            ((first_dest, second_dest), (first_indexes + second_indexes)))
            processed_states.add(source_state_pair)
        return WordGraph(processed_states, new_trans, new_start_state, new_dim)

    def _make_transitions(self, start_state, states, transitions):
        """
        Перекодирует состояния натуральными числами и сохраняет переходы
        """
        # сохраняем индексы исходных состояний
        self._states_map = list(states)
        state_codes = {state: i for i, state in enumerate(self._states_map)}
        self.start_state = state_codes[start_state]
        self.transitions = [dict() for _ in range(self.states_number)]
        for state in states:
            state_code = state_codes[state]
            current_transitions = self.transitions[state_code]
            for symbol, data in transitions[state].items():
                current_transitions[symbol] = \
                    [(state_codes[other], indexes) for other, indexes in data]
        return

    def _make_reverse_transitions(self):
        """
        Задаёт обратные переходы после того, как заданы прямые
        """
        self.revtrans = [dict() for _ in range(self.states_number)]
        for state, state_transitions in enumerate(self.transitions):
            for symbol, data in state_transitions.items():
                for (other, indexes) in data:
                    # если уже есть обратные переходы из other по symbol
                    if symbol in self.revtrans[other]:
                        self.revtrans[other][symbol].append((state, indexes))
                    else:
                        self.revtrans[other][symbol] = [(state, indexes)]
        return

    def find_longest_words(self):
        """
        Находит все пути максимальной длины из начального состояния

        Возвращает:
        -----------
        answer: list
            список пар вида (word, indexes), где word --- слово,
            а indexes --- соответствующий набор индексов
        """
        if not hasattr(self, 'transition_graph_'):
            self._make_transition_graph()
        paths = self.transition_graph_.find_longest_paths(self.start_state)
        return list(chain.from_iterable(self._extract_indexes_from_path(path)
                                        for path in paths))

    def words_of_fixed_length(self, k):
        """
        Находит слова длины k, принимаемые автоматом, вместе с их индексами
        """
        if not hasattr(self, 'transition_graph_'):
            self._make_transition_graph()
        maximal_paths = self.transition_graph_.find_maximal_paths(self.start_state)
        answer = []
        for path in maximal_paths:
            # пути упорядочены по убыванию длин, поэтому последующие точно не подойдут
            if len(path) < k + 1:
                break
            path = path[:(k + 1)]
            answer.extend(self._extract_indexes_from_path(path))
        return answer

    def make_unambigious_automaton(self, gap, initial_gap):
        """
        Возвращает граф, получаемый из автомата преобразованием
        пар (состояние, индекс последнего прочитанного символа)
        в новое состояние и соответствующием заданием переходов.
        При этом невозможны переходы вида
        (s_1, i_1) -> (a, (s_2, i_2)) для i_2 - i_1 > gap + 1
        и s_0 -> (a, (s, i)) для i > initial_gap.
        """
        new_start_state = self.start_state
        new_transitions = {self.start_state: dict()}
        new_states_queue = []  # очередь для порождения новых состояний
        # Добавляем в очередь все пары (состояние, индекс),
        # достижимые из начального состояния
        for symbol, data in self.transitions[self.start_state].items():
            for state, indexes in data:
                # indexes = [(x_11, ..., x_1k), ..., (x_m1, ..., x_ml)],
                # а нам нужны наборы (x_1i_1, ..., x_mi_m)
                for index in map(tuple, product(*indexes)):
                    if (np.array(index) > initial_gap).any():
                        continue
                    pair = ((state, index))
                    # компоненты индекса по каждой координате должны быть кортежом
                    index = tuple((x,) for x in index)
                    if pair not in new_transitions:
                        new_states_queue.append(pair)
                        new_transitions[pair] = dict()
                    if symbol in new_transitions[self.start_state]:
                        new_transitions[self.start_state][symbol].append((pair, index))
                    else:
                        new_transitions[self.start_state][symbol] = [(pair, index)]
        # добавляем все необходимые состояния с помощью очереди
        for pair in new_states_queue:
            state, index = pair
            state_transitions = self.transitions[state]
            current_pair_transitions = dict()
            for symbol, data in state_transitions.items():
                for (other, indexes) in data:
                    for other_index in map(tuple, product(*indexes)):
                        diff = np.array(other_index) - np.array(index)
                        if (diff > 0).all() and (diff <= gap + 1).all():
                            # разрыв между позициями последовательных символов
                            # не превышает gap для всех координат
                            new_pair = (other, other_index)
                            if new_pair not in new_transitions:
                                new_states_queue.append(new_pair)
                                new_transitions[new_pair] = dict()
                            # компоненты индекса по каждой координате должны быть кортежом
                            other_index = tuple((x,) for x in other_index)
                            if symbol in current_pair_transitions:
                                current_pair_transitions[symbol].append((new_pair, other_index))
                            else:
                                current_pair_transitions[symbol] = [(new_pair, other_index)]
            new_transitions[pair] = current_pair_transitions
        return WordGraph(new_transitions.keys(), new_transitions, new_start_state, self.dim)

    def _make_edge_labels(self):
        """
        Создаёт словарь вида (пара вершин): метки рёбер между данными вершинами
        """
        self.edge_labels_ = dict()
        for state, state_transitions in enumerate(self.transitions):
            for symbol, data in state_transitions.items():
                for (other, indexes) in data:
                    pair = (state, other)
                    if pair not in self.edge_labels_:
                        self.edge_labels_[pair] = []
                    self.edge_labels_[pair].append((symbol, indexes))
        return

    def _make_transition_graph(self):
        """
        Добавляет в поле self.transition_graph_
        граф, полученный из автомата удалением меток с рёбер
        """
        if not hasattr(self, 'edge_labels_'):
            self._make_edge_labels()
        self.transition_graph_ = Graph(self.edge_labels_.keys())
        return self

    def _extract_indexes_from_path(self, path):
        """
        Возвращает слова, принимаемые на данном пути,
        вместе с соответствующими индексами
        """
        if len(path) <= 1:
            return [("", [])]
        if not hasattr(self, 'edge_labels_'):
            self._make_edge_labels()
        state_pairs = [(elem, path[i + 1]) for i, elem in enumerate(path[:-1])]
        # edge_labels_on_path = [[('п', ((0,), (0,), (0,)))], [('е', ((1,), (1,), (1,)))],
        #                        [('с', ((2,), (2,), (2,)))], [('к', ((4,), (3,), (3,)))]]
        edge_labels_on_path = [self.edge_labels_[state_pair] for state_pair in state_pairs]
        words_with_indexes = [zip(*elem) for elem in product(*edge_labels_on_path)]
        # words_with_indexes = [('песк', [((0,), (0,), (0,)), ((1,), (1,), (1,)),
        #                                 ((2,), (2,), (2,)), ((4,), (3,), (3,))])]
        words_with_indexes = [("".join(first), list(second))
                              for first, second in words_with_indexes]
        answer = [None] * len(words_with_indexes)
        for i, (word, data) in enumerate(words_with_indexes):
            lists = [elem for elem in zip(*data)]
            word_indexes = [extract_ordered_sequences(elem) for elem in lists]
            # word_indexes = [list(map(tuple, product(*elem))) for elem in lists]
            # word_indexes = [[(0, 1, 2, 4)], [(0, 1, 2, 3)], [(0, 1, 2, 3)]]
            answer[i] = (word, word_indexes)
        return answer

def _make_coordinate_automata(automaton):
    """
    Строит автоматы c теми же переходами, что и в self.common_subseq_automaton_,
    с индексами, берущимися отдельно по каждой координате.
    """
    # вначале строим таблицы для покоординатных автоматов
    new_states = set(range(automaton.states_number))
    new_start_state = automaton.start_state
    new_transitions = [None] * automaton.dim
    for i in range(automaton.dim):
        new_transitions[i] = [dict() for _ in range(automaton.states_number)]
    for state, state_transitions in enumerate(automaton.transitions):
        for symbol, data in state_transitions.items():
            for i, coordinate_transitions in enumerate(new_transitions):
                coordinate_transitions[state][symbol] = []
            for other, indexes in data:
                for i, (coordinate_transitions, coordinate_indexes) \
                        in enumerate(zip(new_transitions, indexes)):
                    coordinate_transitions[state][symbol].append((other, (coordinate_indexes,)))
    new_automata = [WordGraph(new_states, trans, new_start_state, dim=1)
                    for trans in new_transitions]
    return new_automata

def extract_common_words(words_by_coordinates):
    """
    Получает на вход набор из m списков вида
    [(w_{j1}, i_j(w_{j1})), ..., (w_{jk_j}, i_j(w_{jk_j}))]
    и возвращает список вида [(w, [is(w)]], содержащий только те слова w,
    которые встречаются во всех списках, вместе с новыми индексами i(w).
    В is(w) входят все кортежи (x_1, ..., x_m), где x_j --- элемент i_j(w).
    """
    words_by_coordinates_dicts = [dict() for _ in words_by_coordinates]
    for words_dict, words_with_indexes in zip(words_by_coordinates_dicts,
                                              words_by_coordinates):
        for word, word_indexes in words_with_indexes:
            if len(word_indexes) != 1:
                raise TypeError("Only one-dimensional indexes are accepted.")
            if word in words_dict:
                words_dict[word].extend(map(tuple, word_indexes[0]))
            else:
                words_dict[word] = [tuple(x) for x in word_indexes[0]]
    tuplify = (lambda x: x if isinstance(x, tuple) else (x,))
    common_words_data = \
        reduce((lambda first, second: {key: (tuplify(value) + tuplify(second[key]))
                                       for key, value in first.items() if key in second}),
               words_by_coordinates_dicts)
    common_words_indexes = list((word, list(indexes))
                                for word, indexes in common_words_data.items())
    return common_words_indexes


##########################################

def vars_to_string(baseform, varlist):
    """Input: baseform, variable list
       Output: string of type '0=baseform,1=var1,2=var2...'"""
    varlist = tuple(varlist)
    answer = ",".join((("{0}={1}".format(i, word))
                       for i, word in enumerate((baseform,) + varlist)))
    return answer


def extract_tables(tables):
    """
    Input: list of tables
    Output: Dictionary of paradigm tables with the list of their members.
    """
    words_by_vartables = defaultdict(list)
    for lemma, table, forms_with_vars, var_spans in sorted(tables, key=(lambda x: x[0])):
        forms_with_vars = tuple(forms_with_vars) if isinstance(forms_with_vars, list) else forms_with_vars
        to_add = (lemma, tuple(var_spans))
        words_by_vartables[forms_with_vars].append(to_add)
    return words_by_vartables

##########################################

def _make_correct_table(table):
    """
    Удаляет из таблицы пропуски

    Возвращает:
    -----------
    correct_indices: list of ints --- индексы словоформ, не равных пропуску,
    correct_table: list of strs --- таблица корректных слвооформ
    """
    correct_indices, correct_table = [], []
    for i, form in enumerate(table):
        if form not in ['—', '-']:  # прочерки бывают разные...
            correct_indices.append(i)
            correct_table.append(form)
    return correct_indices, correct_table


def find_gap_positions(lst, add_initial=False):
    """
    Находит те позиции i в упорядоченном списке lst, что lst[i+1] > lst[i] + 1

    Аргументы:
    ----------
    lst: list, упорядоченный список целых чисел
    add_initial: bool, индикатор учёта начального разрыва

    Возвращает:
    -----------
    gap_positions: list, набор позиций i, таких что lst[i+1] > lst[i] + 1
    """
    if len(lst) == 0:
        return []
    gap_positions = []
    if add_initial and lst[0] > 0:
        gap_positions.append(0)
    prev = lst[0]
    for i, curr in enumerate(lst[1:], 1):
        if curr > prev + 1:
            gap_positions.append(i)
        prev = curr
    return gap_positions


def compute_paradigm(table, indexes, var_beginnings):
    """
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
    """
    paradigm = []
    vars_number = len(var_beginnings) - 1
    for form, form_indexes in zip(table, indexes):
        var_spans = []
        curr = var_beginnings[0]
        for next in var_beginnings[1:]:
            var_spans.append((form_indexes[curr], form_indexes[next - 1] + 1))
            curr = next
        var_spans.append((len(form), None))
        fragments_list = []
        if var_spans[0][0] > 0:
            fragments_list.append(form[:var_spans[0][0]])
        for i in range(1, vars_number + 1):
            fragments_list += [str(i), form[var_spans[i - 1][1]:var_spans[i][0]]]
        form_with_vars = "+".join(fragments_list)
        if form_with_vars.endswith("+"):
            form_with_vars = form_with_vars[:-1]
        form_with_vars = re.sub("[+]+", "+", form_with_vars)
        paradigm.append(form_with_vars)
    return paradigm


# вход/выход

def read_input(infile, language, method='first'):
    """
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
    """
    # сразу создаём функцию, которая проверяет, следует ли добавлять форму
    # в список. Так делаем, чтобы if вызывался только 1 раз
    marks = get_categories_marks(language)
    if method == 'first':
        add_checker = (lambda x: (x is not None) and len(x) == 0)
        def table_adder(lemma, table_dict):
            # вычисляем формы для леммы
            table_dict[LEMMA_KEY] = [lemma]
            forms = [(elem[0] if len(elem) > 0 else '-') for elem in table_dict.values()]
            # возвращает список, потому что в случае 'all'
            # придётся возвращать несколько парадигм, то есть список
            return [(lemma, forms)]
    elif method == 'all':
        add_checker = (lambda x: x is not None)
        def table_adder(lemma, table_dict):
            # вычисляем формы для леммы
            table_dict[LEMMA_KEY] = [lemma]
            form_variants_list = list(table_dict.values())
            for i, form_variants in enumerate(form_variants_list):
                if len(form_variants) == 0:
                    form_variants_list[i] = ['-']
            # либо для каждой формы одинаковое число вариантов,
            # либо для некоторых 1, а для других одно и то же k>1
            form_variants_counts = set(len(x) for x in form_variants_list)
            if len(form_variants_counts) == 2:
                min_count, max_count = sorted(form_variants_counts)
                if min_count == 1:
                    for i, form_variants in enumerate(form_variants_list):
                        if len(form_variants) == 1:
                            form_variants_list[i] *= max_count
                else:
                    form_variants_list = [[elem[0]] for elem in form_variants_list]
            elif len(form_variants_counts) >= 2:
                form_variants_list = [[elem[0]] for elem in form_variants_list]
            return [(lemma, list(elem)) for elem in zip(*form_variants_list)]
    else:
        sys.exit("Method must be 'first' or 'all'.")
    tables = []
    table_keys = [LEMMA_KEY] + marks
    current_table_dict = OrderedDict((mark, []) for mark in table_keys)
    has_forms = False  # индикатор того, нашлись ли у слова словоформы
    with open(infile, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            line = line.strip('\ufeff')  # удалим метку кодировки
            splitted_line = line.split(',')
            if len(splitted_line) == 1:
                # строчка вида i_1 <лемма_1>
                if has_forms:
                    tables += table_adder(lemma, current_table_dict)
                    current_table_dict = OrderedDict((mark, []) for mark in table_keys)
                    has_forms = False
                splitted_line = splitted_line[0].split("\t")
                # lemma = splitted_line[-1]
                index, lemma = splitted_line
                if int(index) % 500 == 1:
                    print(index)
            else:
                form, form_lemma = splitted_line[:2]
                if language != 'RU_verbs':
                    mark = tuple(splitted_line[2:])
                else:
                    mark = ",".join(splitted_line[2:])
                if form not in ['—', '-', ''] and form_lemma == lemma:
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
                                           key=(lambda x: (len(x[1]), x[0])),
                                           reverse=True):
            if isinstance(paradigm, str):
                paradigm_str = paradigm
            else:
                paradigm_str = "#".join(paradigm)
            fout.write(paradigm_str + "\n")
            if short_outfile is not None:
                fshort.write("{0:<6}".format(count) + paradigm_str + "\n")
                first_elem_str = vars_to_string(*(sorted(var_values)[0]))
                fshort.write("{0:<6}{1}\n".format(len(var_values), first_elem_str))
                count += 1
            fout.write("\n".join((vars_to_string(*elem) for elem in var_values)))
            fout.write("\n")
        if short_outfile is not None:
            fshort.close()

def _extract_lemma_forms(language, table_dict, lemma):
    '''
    Вычисляет начальную форму для слова на основе форм для категорий
    Для русского или латинского форма извлекается из Nom,Sg, потом из Nom,Pl,
    если обе эти ячейки пусты, то возвращается лемма
    '''
    answer = [lemma]
    for key in get_possible_lemma_marks(language):
        forms = table_dict[key]
        if len(forms) > 0:
            answer = forms
            break
    if lemma not in answer:
        lemma.append(answer)
    return answer

def test():
    # first = WordGraph.word_to_graph('строка')
    # second = WordGraph.word_to_graph('штора')
    # other = first.intersect(second)
    # print(other.longest_words(gap=1))

    words = ['mainehikas', 'kaampi']
    lcs_searcher = LcsSearcher(gap=1, initial_gap=0)
    best_lcss = lcs_searcher.process_table(words)
    print(best_lcss)

    # words = ['песок', 'песком', 'песков']
    # # words = ['брассист', 'растр']
    # lcs_searcher = LcsSearcher(gap=None)
    # best_lcss = lcs_searcher.process_table(words)
    return


if __name__ == "__main__":
    # test()
    args = sys.argv[1:]
    if len(args) == 0:
        test()
    if len(args) != 7:
        sys.exit("Pass input file, language code, paradigm_extraction_method, "
                 "maximal gap in lcs, maximal initial gap, "
                 "output file and output stats file")
    infile, language, method, gap, initial_gap, outfile, stats_outfile = args
    gap, initial_gap = map(int, (gap, initial_gap))
    if gap < 0:
        gap = None
    if initial_gap < 0:
        initial_gap = None
    tables = read_input(infile, language, method=method)
    paradigms_with_vars = []
    lcs_searcher = LcsSearcher(gap=gap, initial_gap=initial_gap)
    for num, (lemma, table) in enumerate(tables):
        if num % 100 == 0:
            print(num, lemma)
        paradigms_with_vars.append(lcs_searcher.calculate_all_paradigms(table))
    tables_by_paradigm = extract_tables(list(chain.from_iterable(
        [(lemma, table, paradigm_repr, lcs_vars)
         for paradigm_repr, lcs_vars in elem]
        for (lemma, table), elem in zip(tables, paradigms_with_vars))))
    output_paradigms(tables_by_paradigm, outfile, stats_outfile)
