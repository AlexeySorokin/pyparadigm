"""
Файл, содержащий функции для чтения входных файлов при классификации парадигм
"""
import sys
from collections import defaultdict, OrderedDict

from local_paradigm_transformer import LocalTransformation, descr_to_transforms


def process_lemmas_file(infile):
    """
    Читает файл с таблицей лемм
    """
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            # не делаем strip
            splitted = line.split()
            if len(splitted) != 3:
                continue
            lemma, code, variables_data = splitted
            code = int(code)
            var_values = [elem.split('=')[1] for elem in variables_data.split(',')]
            answer.append((lemma, code, var_values))
    return answer

def process_codes_file(infile):
    """
    Читает файл с таблицей кодов
    """
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted = line.split()
            if len(splitted) != 3:
                continue
            code, descr, count = splitted
            code, count = int(code), int(count)
            answer.append((code, descr, count))
    return answer

def read_paradigms(descriptions_list):
    """
    Обрабатывает содержимое файла со списком парадигм,
    полученное в функции process_codes_file
    """
    answer, counts = dict(), dict()
    for code, pattern, count in descriptions_list:
        answer[pattern] = code
        counts[code] = count
    new_answer = OrderedDict()
    new_counts = OrderedDict()
    # парадигмы и счётчики оказываются отсортированы
    # по числу словоформ, подходящих под парадигму
    for pattern, code in sorted(answer.items(), key=(lambda x: x[1])):
        new_answer[pattern] = code
        new_counts[code] = counts[code]
    return new_answer, new_counts

def read_lemmas(lemma_descriptions_list, multiclass=False, return_joint=False):
    """
    Извлекает информацию из описаний лемм,
    полученных с помощью функции process_lemmas_file
    """
    data = OrderedDict()
    # в случае, когда одна лексема может иметь несколько парадигм,
    # сохраняем все такие парадигмы
    if not multiclass:
        def add(lst, x):
            if len(lst) == 0:
                lst.append(x)
    else:
        def add(lst, x):
            lst.append(x)
    for lemma, code, var_values in lemma_descriptions_list:
        if lemma in data:
            add(data[lemma], (code, var_values))
        else:
            data[lemma] = [(code, var_values)]
    if return_joint:
        return list(data.keys()), list(data.values())
    else:
        return (list(data.keys()),
                [list(x[0] for x in elem) for elem in data.values()],
                [list(x[1] for x in elem) for elem in data.values()])

def read_transformations_(paradigms_by_codes, paradigm_counts=None):
    """
    Аргументы:
    ----------
    paradigm_codes: dict, словарь вида (парадигма, код)
    paradigm_counts: dict or None (optional, default=None), словарь вида (код, счётчик)

    Возвращает:
    ----------
    transformations_list: list, список трансформаций
    transformations_codes: dict, словарь вида (трансформация, код трансформации)
    transformations_counts: dict, словарь вида (код трансформации, счётчик)
    transformations_by_paradigms: dict, словарь вида (код парадигмы, коды входящих в ней локальных трансформаций)
    """
    if paradigm_counts is None:
        paradigm_counts = defaultdict(int)
    transformations_list, transformation_codes = [None], dict()
    transformation_counts, transformations_number = defaultdict(int), 1
    transformations_by_paradigms = dict()
    for code, descr in paradigms_by_codes.items():
        paradigm_count = paradigm_counts[code]
        transformations = descr_to_transforms(descr)
        for trans in transformations:
            trans_code = transformation_codes.get(trans, None)
            if trans_code is None:
                trans_code = transformation_codes[trans] = transformations_number
                transformations_number += 1
                transformations_list.append(trans)
            transformation_counts[trans_code] += paradigm_count
        transformations_by_paradigms[code] = [transformation_codes[trans]
                                              for trans in transformations]
    transformations_by_strings = defaultdict(list)
    for transformation, code in transformation_codes.items():
        substr = transformation.trans[0]
        transformations_by_strings[substr].append(code)
    return (transformations_list, transformation_codes,
            transformation_counts, transformations_by_paradigms), transformations_by_strings


def read_transformations(paradigm_codes_data):
    """
    Получает на вход список парадигм вида
    (36, 1+е+2+ь#1+е+2+ь#1+2+и#1+2+я#1+2+ей#1+2+ю#1+2+ям#1+е+2+ь#1+2+и#1+2+ем#1+2+ями#1+2+е#1+2+ях,	12)
    и извлекает оттуда кодировку трансформаций
    """
    # вначале для каждого слова извлечь трансформации
    # NB! расставить переменные
    # потом сохранить их в нужные контейнеры
    paradigms_by_codes = OrderedDict()
    codes_by_paradigms = dict()
    counts_by_code = dict()
    transformations_list, transformation_codes = [], dict()
    transformation_counts, transformations_number = defaultdict(int), 1
    transformations_by_paradigms = dict()
    for code, descr, count in paradigm_codes_data:
        paradigms_by_codes[code] = descr
        codes_by_paradigms[descr] = code
        counts_by_code[code] = count
        transformations = descr_to_transforms(descr)
        for trans in transformations:
            if trans not in transformation_codes:
                transformation_codes[trans] = transformations_number
                transformations_number += 1
                transformations_list.append(trans)
            transformation_counts[trans] += count
        transformations_by_paradigms[code] = [transformation_codes[trans]
                                              for trans in transformations]
    transformation_counts = {transformation_codes[trans]: count
                             for trans, count in transformation_counts.items()}
    transformations_by_strings = defaultdict(list)
    for transformation, code in transformation_codes.items():
        substr = transformation.trans[0]
        transformations_by_strings[substr].append(code)
    return ((paradigms_by_codes, codes_by_paradigms, counts_by_code),
            (transformations_list, transformation_codes,
             transformation_counts, transformations_by_paradigms),
            transformations_by_strings)
