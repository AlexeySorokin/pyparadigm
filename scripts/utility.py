import sys
import bisect

import numpy as np


def extract_ordered_sequences(lists, min_differences=None, max_differences=None,
                              strict_min=True, strict_max=False):
    '''
    Аргументы:
    ----------
    lists: list of lists
    список упорядоченных списков L1, ..., Lm
    min_differences: array-like, shape=(m, ) or None(default=None)
    набор [d1, ..., dm] минимальных разностей
    между соседними элементами порождаемых списков
    min_differences: array-like, shape=(m, ) or None(default=None)
    набор [d'1, ..., d'm] минимальных разностей
    между соседними элементами порождаемых списков

    Возвращает:
    -----------
    генератор списков [x1, ..., xm], x1 \in L1, ..., xm \in Lm
    d'1 >= x1 >= d1, x1 + d2' >= x2 > x1 + d2, ..., x1 + d(m-1)' >= xm > x(m-1) + dm
    если min_differences=None, то возвращаются просто
    строго монотонные последовательности
    '''
    m = len(lists)
    if m == 0:
        return []
    if min_differences is None:
        min_differences = np.zeros(dtype=float, shape=(m,))
        min_differences[0] = -np.inf
    else:
        min_differences = np.array(min_differences)
        if min_differences.shape != (m,):
            raise ValueError("Lists and min_differences must have equal length")
    if max_differences is None:
        max_differences = np.empty(dtype=float, shape=(m,))
        max_differences.fill(np.inf)
    else:
        max_differences = np.array(max_differences)
        if max_differences.shape != (m,):
            raise ValueError("Lists and min_differences must have equal length")
    if (np.any(max_differences[1:] <= min_differences[1:]) or
        (max_differences[0] < min_differences[0])):
        return []
    list_lengths = [len(elem) for elem in lists]
    if any(x == 0 for x in list_lengths):
        return []
    min_indexes = [find_first_larger_indexes([x + d for x in lists[i]], lists[i+1], strict_min)
                   for i, d in enumerate(min_differences[1:])]
    max_indexes = [find_first_larger_indexes([x + d for x in lists[i]], lists[i+1], strict_max)
                   for i, d in enumerate(max_differences[1:])]
    answer = []
    # находим минимальную позицию первого элемента
    startpos = bisect.bisect_left(lists[0], min_differences[0])
    if startpos == len(lists[0]):
        return []
    endpos = bisect.bisect_right(lists[0], max_differences[0])
    sequence, index_sequence = [lists[0][startpos]], [startpos]
    # TO BE MODIFIED
    while len(sequence) > 0:
        '''
        обходим все монотонные последовательности в лексикографическом порядке
        '''
        i, li = len(index_sequence) - 1, index_sequence[-1]
        if i < m - 1 and min_indexes[i][li] < max_indexes[i][li]:
            '''
            добавляем минимально возможный элемент в последовательность
            '''
            next_index = min_indexes[i][li]
            i += 1
            index_sequence.append(next_index)
            sequence.append(lists[i][next_index])
        else:
            if i == m - 1:
                '''
                если последовательность максимальной длины
                то обрезаем последний элемент
                чтобы сохранить все возможные варианты для последней позиции
                '''
                index_sequence.pop()
                sequence.pop()
                i -= 1
                curr_max_index = max_indexes[i][index_sequence[-1]] if i >= 0 else endpos
                while li < curr_max_index:
                    answer.append(sequence + [lists[-1][li]])
                    li += 1
                if i < 0:
                    break
            '''
            пытаемся перейти к следующей
            в лексикографическом порядке последовательности
            '''
            index_sequence[-1] += 1
            while i > 0 and index_sequence[-1] == max_indexes[i-1][index_sequence[-2]]:
                '''
                увеличиваем последний индекс
                если выходим за границы, то
                укорачиваем последовательность и повторяем процедуру
                '''
                index_sequence.pop()
                i -= 1
                index_sequence[-1] += 1
            if i == 0 and index_sequence[0] == endpos:
                break
            sequence = sequence[:i] + [lists[i][index_sequence[-1]]]
    return answer

def find_first_larger_indexes(first, second, strict=True):
    '''
    Аргументы:
    ---------
    first, second: array,
    упорядоченные массивы длины m и n типа type
    strict: bool,
    индикатор строгости неравенства

    Возвращает:
    indexes: array, shape=(m,)
    упорядоченный массив длины m, indexes[i] равен минимальному j,
    такому что first[i] < second[j] (в случае strict=True)
    или first[i] <= second[j] (в случае strict=False)
    '''
    pred = (lambda x,y: x < y) if strict else (lambda x,y: x <= y)
    return _find_first_indexes(first, second, pred)

def _find_first_indexes(first, second, pred):
    '''
    Аргументы:
    ---------
    first, second: array,
    упорядоченные массивы длины m и n типа type
    pred: type->(type->bool),
    предикат, монотонный по второму аргументу и антимонотонный по первому

    Возвращает:
    indexes: array, shape=(m,)
    упорядоченный массив длины m, indexes[i] равен минимальному j,
    такому что pred(first[i], second[j])=True или n, если такого i нет
    '''
    m, n = len(first), len(second)
    i, j = 0, 0
    indexes = np.empty(dtype=int, shape=(m,))
    while i < m and j < n:
        if pred(first[i], second[j]):
            indexes[i] = j
            i += 1
        else:
            j += 1
    if i < m:
        indexes[i:] = n
    return indexes

def generate_monotone_sequences(first, length, upper, min_differences=None,
                                max_differences=None):
    lists = [np.arange(first, upper) for _ in range(length)]
    if len(lists) == 0:
        return []
    if min_differences is not None:
        min_differences = np.array(min_differences)
        if min_differences.shape != (length-1,):
            raise ValueError("Min_differences must be of shape (length - 1, )")
    else:
        min_differences = np.zeros(shape=(length,), dtype = int)
        min_differences[0] = first
    if max_differences is not None:
        max_differences = np.array(max_differences)
        if max_differences.shape != (length-1,):
            raise ValueError("Min_differences must be of shape (length - 1, )")
    else:
        max_differences = np.empty(shape=(length,), dtype = int)
        max_differences.fill(upper - first)
        max_differences[0] = first
    return extract_ordered_sequences(lists, min_differences, max_differences)



if __name__ == "__main__":
    l1 = [1, 4, 8]
    l2 = [3, 7, 8]
    l3 = [2, 4, 7, 9]
    lists = [l1, l2, l3]
    min_differences = [1, 1, 0]
    max_differences = [5, 4, 3]
    for elem in extract_ordered_sequences(lists, min_differences, max_differences):
        print(elem)
    print("")
    for elem in generate_monotone_sequences(3, 3, 8):
        print(elem)
