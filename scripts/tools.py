import sys
import numpy as np
from contextlib import contextmanager

@contextmanager  # для перенаправления stdout в файл в процессе работы
def redirect_stdout(new_target):
    old_target, sys.stdout = sys.stdout, new_target  # replace sys.stdout
    try:
        yield new_target  # run some code with the replaced stdout
    finally:
        sys.stdout = old_target  # restore to the previous value


def partition_by_lengths(slicable, segment_lengths):
    end = 0
    for length in segment_lengths:
        start, end = end, end + length
        yield slicable[start:end]


def counts_to_probs(counts, classes_number):
    """
    Преобразует словарь счётчиков в массив вероятностей

    counts: dict, словарь счётчиков,
    classes_number: int, число классов

    Возвращает:
    ------------
    answer: array of floats, shape = (classes_number,), массив частот
    """
    answer = np.zeros(shape=(classes_number,), dtype=float)
    for label, count in counts.items():
        answer[label] += 1
    answer /= np.sum(answer)
    return answer