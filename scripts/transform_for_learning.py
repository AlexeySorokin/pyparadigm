#-------------------------------------------------------------------------------
# Name:        transform_for_learning.py
# Purpose:     преобразование статистики парадигм к виду, удобному для обучения
#
# Created:     03.09.2015
#-------------------------------------------------------------------------------

import sys
import re
from collections import defaultdict

'''
Использование:
transform_for_learning.py paradigms word_paradigms lemmas_outfile paradigm_codes_outfile
Файл paradigms --- выдача pyparadigm,
Word_paradigms --- исходный файл с парадигмами по словам, используется только для задания порядка слов
'''

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 4:
        sys.exit("Pass infile, file with ordering, outfile for lemmas and outfile for paradigm codes")
    infile, words_file, lemmas_outfile, paradigm_codes_outfile = args
    paradigms, paradigms_number = dict(), 0
    word_paradigms = defaultdict(list)
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if '#' in line:
                current_paradigm = line
                if current_paradigm not in paradigms:
                    paradigms[current_paradigm] = current_paradigm_code = paradigms_number
                    paradigms_number += 1
                else:
                    current_paradigm_code = paradigms[current_paradigm]
            elif line.startswith("0"):
                matches = re.search("0=([^,]*),(.*)", line)
                if matches is not None:
                    lemma = matches.groups()[0]
                    rest = matches.groups()[1]
                    word_paradigms[lemma].append((current_paradigm_code, rest))
    word_order = defaultdict(list)
    with open(words_file, "r", encoding = "utf8") as fin:
        for line in fin:
            line = line.strip()
            line = line.strip("\ufeff")
            splitted_line = line.split(',')
            if len(splitted_line) == 1:
                code, lemma = line.split("\t")
                word_order[lemma] = int(code)
    with open(paradigm_codes_outfile, "w", encoding="utf8") as fout:
        for paradigm, code in sorted(paradigms.items(), key=(lambda x: x[1])):
            fout.write("{0:<4}{1}\n".format(code, paradigm))
    with open(lemmas_outfile, "w", encoding="utf8") as fout:
        for lemma, lemma_info in sorted(word_paradigms.items(),
                                        key=(lambda x: word_order[x[0]])):
            for code, rest in lemma_info:
                fout.write("{0}\t{1}\t{2}\n".format(lemma, code, rest))








