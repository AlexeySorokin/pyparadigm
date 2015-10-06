import numpy as np
import bisect
import re
import itertools

import utility

class ParadigmFragment:
    '''
    Класс, предназначенный для операций с описаниями фрагментов парадигм
    вида 1+о+2
    '''

    def __init__(self, descr):
        self.variable_indexes, used_variables = [], set()
        self.const_fragments_indexes = []
        splitted_descr = descr.split("+")
        self.fragments = splitted_descr[:]
        for i, part in enumerate(splitted_descr):
            if part.isdigit():
                code = int(part)
                if code not in used_variables:
                    self.variable_indexes.append((code, i))
                    self.fragments[i] = "" # на всякий случай обнулим
                else:
                    raise ValueError("Variable should be present only once")
            else:
                self.const_fragments_indexes.append(i)
        self.max_var_number = max(x for x, _ in self.variable_indexes)

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
        if not isinstance(var_values, dict):
            var_values = {i: val for i, val in enumerate(var_values, 1)}
        fragments = self.fragments[:] # чтобы не портить значений
        for index, pos in self.variable_indexes:
            try:
                fragments[pos] = var_values[index]
            except KeyError:
                raise KeyError(index)
        return "".join(fragments)

    def extract_variables(self, word):
        """
        Извлекает значения переменных, возможные для данного слова
        """
        # вначале находим позиции постоянных фрагментов
        const_fragments_positions = self._find_constant_fragments_positions(word)
        # нужно найти все возможные способы разбить на непустые переменные
        variable_values = []
        for elem in const_fragments_positions:
            variable_segments_boundaries =\
                    self._find_variable_segments_boundaries(elem, len(word))
            variable_positions =\
                    self._find_variable_positions(variable_segments_boundaries)
            fragment_positions = [None] * len(self.fragments)
            for index, pos in zip(self.const_fragments_indexes, elem):
                fragment_positions[index] = pos
            for variant in variable_positions:
                for (_, index), pos in zip(self.variable_indexes, variant):
                    fragment_positions[index] = pos
                # добавляем позицию для конца слова, чтобы закрыть последний фрагмент
                fragment_positions.append(len(word))
                variable_values.append({code: word[fragment_positions[i]:
                                                   fragment_positions[i+1]]
                                        for code, i in self.variable_indexes})
        return variable_values

    def _find_constant_fragments_positions(self, word):
        '''
        Находит начальные позиции константных фрагментов в слове
        '''
        fragment_positions = [None] * (len(self.const_fragments_indexes) + 1)
        differences = [0] * (len(self.const_fragments_indexes) + 1)
        # позиция предыдущей переменной и длина предыдущего фрагмента
        # для вычисления минимального различия между позициями соседних фрагментов
        prev_pos, prev_fragment_length = -1, 0
        for i, pos in enumerate(self.const_fragments_indexes):
            differences[i] = pos - prev_pos  - 1 + prev_fragment_length
            fragment = self.fragments[pos]
            matches = list(re.finditer(u'(?={0})'.format(fragment), word))
            if len(matches) > 0:
                fragment_positions[i] = [m.start() for m in matches]
            else:
                return []
            prev_pos, prev_fragment_length = pos, len(fragment)
        fragment_positions[-1] = [len(word)]
        differences[-1] = len(self.fragments) - prev_pos - 1 + prev_fragment_length
        answer = utility.extract_ordered_sequences(fragment_positions, differences, strict_min=False)
        return [elem[:-1] for elem in answer]

    def _find_variable_segments_boundaries(self, positions, length):
        '''
        Находит диапазоны для переменных фрагментов
        в зависимости от позиций константных фрагментов
        '''
        start_index, start_pos = 0, 0
        answer = []
        for index, pos in zip(self.const_fragments_indexes, positions):
            if pos > start_pos or index > start_index:
                answer.append((start_pos, index - start_index, pos))
            start_index = index + 1
            start_pos = pos + len(self.fragments[index])
        index, pos = len(self.fragments), length
        if pos > start_pos or index > start_index:
            answer.append((start_pos, index - start_index, pos))
        return answer

    @staticmethod
    def _find_variable_positions(boundaries):
        '''
        находит позиции переменных на основе границ сегментов
        '''
        answer = []
        variable_positions_by_segment = [utility.generate_monotone_sequences(*elem)
                                         for elem in boundaries]
        answer = list(itertools.chain.from_iterable(elem)
                      for elem in itertools.product(*variable_positions_by_segment))
        return answer

def fit_to_patterns(word, patterns):
    '''
    Находит парадигмы, под которые подходит лемма
    '''
    answer = []
    for pattern in patterns:
        fragmentor = ParadigmFragment(pattern)
        variable_values = fragmentor.extract_variables(word)
        if len(variable_values) > 0:
            answer.append((pattern, variable_values))
    return answer


if __name__ == '__main__':
    fragment = ParadigmFragment(u"1+ос+2")
    variable_values = fragment.extract_variables(u'достопримечательность')
    for elem in variable_values:
        print(",".join("{0}={1}".format(var, value)
                       for var, value in sorted(elem.items(),
                                                key=(lambda x:int(x[0])))))


