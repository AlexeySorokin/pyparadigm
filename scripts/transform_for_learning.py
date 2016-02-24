#-------------------------------------------------------------------------------
# Name:        transform_for_learning.py
# Purpose:     преобразование статистики парадигм к виду, удобному для обучения
#
# Created:     03.09.2015
#-------------------------------------------------------------------------------

import sys
import re
from collections import defaultdict, OrderedDict
import getopt

'''
Использование:
transform_for_learning.py paradigm_stats paradigms word_paradigms lemmas_outfile paradigm_codes_outfile
Файлы paradigms paradigm_stats --- выдача pyparadigm,
Word_paradigms --- исходный файл с парадигмами по словам, используется только для задания порядка слов
'''

def make_form_patterns(current_paradigm):
    main_part = current_paradigm[current_paradigm.find('#'):]
    if len(main_part) > 0:
        main_part = main_part[1:]
    splitted_paradigm = main_part.split('#')
    form_paradigms = OrderedDict()
    form_counts = defaultdict(int)
    for pattern in splitted_paradigm:
        if pattern in ['-', '']:
            continue
        form_counts[pattern] += 1
        if pattern in form_paradigms:
            continue
        form_paradigm = "{0}#{1}".format(pattern, main_part)
        form_paradigms[pattern] = form_paradigm
    return form_paradigms, form_counts

OPTIONS = 'w'
LONG_OPTIONS = ['words']

if __name__ == '__main__':
    mode = 'lemmas'
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, OPTIONS, LONG_OPTIONS)
    for opt, _ in opts:
        if opt in ['-w', '--words']:
            mode = 'words'
    if len(args) != 5:
        sys.exit("Pass infile, stats file, file with ordering, outfile for lemmas and outfile for paradigm codes")
    infile, stats_file, words_file, lemmas_outfile, paradigm_codes_outfile = args
    paradigms, word_paradigms, form_paradigms = dict(), defaultdict(list), defaultdict(list)
    forms_by_lemmas = defaultdict(list)
    code = 0
    if mode == 'lemmas':
        # файл со статистиками используется только в случае обработки лемм
        with open(stats_file, 'r', encoding="utf8") as fin:
            while True:
                line = fin.readline()
                if line == "":
                    break
                splitted_line = line.strip().split()
                if len(splitted_line) != 2:
                    if fin:
                        fin.readline()
                    continue
                _, paradigm = splitted_line
                line = fin.readline()
                if line == "":
                    break
                splitted_line = line.strip().split()
                if len(splitted_line) != 2:
                    continue
                count = int(splitted_line[0])
                paradigms[paradigm] = {'code': code, 'count': count}
                code += 1
    with open(infile, "r", encoding="utf8") as fin:
        if mode == 'words':
            current_paradigm_code = 0
        for line in fin:
            line = line.strip()
            if '#' in line:
                current_paradigm = line
                if mode == 'lemmas':
                    current_paradigm_code = paradigms[current_paradigm]['code']
                else:
                    pos = current_paradigm.find('#')
                    current_paradigm_patterns, current_form_counts =\
                        make_form_patterns(current_paradigm)
            elif line.startswith("0"):
                matches = re.findall("[0-9]+[=][^,]+", line)
                lemma = matches[0].split('=')[1]
                rest = matches[1:]
                if mode == 'lemmas':
                    rest = ",".join(rest)
                    word_paradigms[lemma].append((current_paradigm_code, rest))
                else:
                    rest_splitted = [elem.split('=') for elem in rest]
                    current_forms = []
                    for form, descr in current_paradigm_patterns.items():
                        count = current_form_counts[form]
                        descr_to_use = descr
                        for code, value in rest_splitted:
                            form = re.sub("(?<![0-9]){0}(?![0-9])".format(code), value, form)
                            descr_to_use = re.sub("(?<![0-9]){0}(?![0-9])".format(code), value, descr_to_use)
                        form = form.replace('+', '')
                        descr_to_use = descr_to_use.replace('+', '')
                        current_forms.append(form)
                        form_paradigms[form].append((descr, ",".join(rest)))
                        if descr not in paradigms:
                            paradigms[descr] = {'code': current_paradigm_code, 'count': count}
                            current_paradigm_code += 1
                        else:
                            paradigms[descr]['count'] += count
                    forms_by_lemmas[lemma].append(current_forms)
    # если mode='words', упорядочиваем парадигмы в порядке убывания мощности
    if mode == 'words':
        for i, (descr, elem) in enumerate(sorted(paradigms.items(),
                                                 key=(lambda x:x[1]['count']),
                                                 reverse=True)):
            paradigms[descr]['code'] = i
    word_order = defaultdict(list)
    # читаем порядок лемм
    with open(words_file, "r", encoding = "utf8") as fin:
        for i, line in enumerate(fin):
            line = line.strip()
            line = line.strip("\ufeff")
            splitted_line = line.split(',')
            if len(splitted_line) == 1:
                code, lemma = line.split("\t")
                word_order[lemma] = int(code)
    # сохраняем парадигмы
    with open(paradigm_codes_outfile, "w", encoding="utf8") as fout:
        for paradigm, elem in sorted(paradigms.items(), key=(lambda x: x[1]['code'])):
            code, count = elem['code'], elem['count']
            fout.write("{0:<6}{1}\t{2}\n".format(code, paradigm, count))
    # сохраняем парадигмы для слов
    with open(lemmas_outfile, "w", encoding="utf8") as fout:
        if mode == 'lemmas':
            for lemma, lemma_info in sorted(word_paradigms.items(),
                                            key=(lambda x: word_order[x[0]])):
                for code, rest in lemma_info:
                    fout.write("{0}\t{1}\t{2}\n".format(lemma, code, rest))
        else:
            for lemma, lemma_forms in sorted(forms_by_lemmas.items(),
                                             key=(lambda x: word_order[x[0]])):
                for forms in lemma_forms:
                    for form in forms:
                        for descr, rest in form_paradigms[form]:
                            fout.write("{0}\t{1}\t{2}\n".format(form,
                                                                paradigms[descr]['code'],
                                                                rest))






