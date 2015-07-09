import numpy as np

def extract_ordered_sequences(lists):
    '''
    Аргументы:
    ----------
    lists: list of lists
    список упорядоченных списков L1, ..., Lm

    Возвращает:
    -----------
    генератор списков [x1, ..., xm],
    x1 < x_2 < ... < xm, x1 \in L1, ..., xm \in Lm
    '''
    m = len(lists)
    if m == 0:
        return []
    list_lengths = [len(elem) for elem in lists]
    if any(x == 0 for x in list_lengths):
        return []
    indexes = [find_first_larger_indexes(lists[i], lists[i+1]) for i in range(m-1)]
    answer = []
    sequence, index_sequence = [lists[0][0]], [0]
    while len(sequence) > 0:
        '''
        обходим все монотонные последовательности в лексикографическом порядке
        '''
        i, li = len(index_sequence) - 1, index_sequence[-1]
        if i < m - 1 and indexes[i][li] < list_lengths[i+1]:
            '''
            добавляем минимально возможный элемент в последовательность
            '''
            next_index = indexes[i][li]
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
                while li < list_lengths[-1]:
                    answer.append(sequence + [lists[-1][li]])
                    li += 1
                if i < 0:
                    break
            '''
            пытаемся перейти к следующей
            в лексикографическом порядке последовательности
            '''
            index_sequence[-1] += 1
            while index_sequence[-1] == list_lengths[i] and i > 0:
                '''
                увеличиваем последний индекс
                если выходим за границы, то
                укорачиваем последовательность и повторяем процедуру
                '''
                index_sequence.pop()
                i -= 1
                index_sequence[-1] += 1
            if i == 0 and index_sequence[0] == list_lengths[0]:
                break
            sequence = sequence[:i] + [lists[i][index_sequence[-1]]]
    return answer

def find_first_larger_indexes(first, second):
    '''
    Аргументы:
    ---------
    first, second: array
    упорядоченные массивы длины m и n

    Возвращает:
    indexes: array, shape=(m,)
    упорядоченный массив длины m, indexes[m] равен минимальному i,
    такому что second[i] < first[j], или n, если такого i нет
    '''
    m, n = len(first), len(second)
    i, j = 0, 0
    indexes = np.empty(dtype=int, shape=(m,))
    while i < m and j < n:
        if second[j] <= first[i]:
            j += 1
        else:
            indexes[i] = j
            i += 1
    if i < m:
        indexes[i:] = n
    return indexes

if __name__ == "__main__":
    l1 = [1]
    l2 = [0, 3, 5, 7]
    l3 = [2, 4, 7]
    lists = [l1, l2, l3]
    for elem in extract_ordered_sequences(lists):
        print(elem)




