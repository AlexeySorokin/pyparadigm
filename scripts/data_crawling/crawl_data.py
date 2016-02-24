# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Тестирование синтаксиса urllib
#-------------------------------------------------------------------------------

import sys
import re

import urllib.request as urr
import urllib.parse as urp
import urllib.error as ure
import bs4

from html.parser import HTMLParser
html_parser = HTMLParser()

h3_regexp = re.compile(r"Морфологические и синтаксические свойства.*")
split_regexp = u"(<h1.*?/h1>)"
cell_split_regexp = u"(?:,[ ]*)|(?:[ ]*//[ ]*)|(?:[ ]*\n[ ]*)"

RU_cases = {"Им.":"им", "Р.":"род", "Д.":"дат", "В.":"вин", "Тв.":"тв","Пр.": "пр"}
LA_cases = {"Ном.":"ном", "Ген.":"ген", "Дат.":"дат", "Акк.":"акк", "Абл.":"абл","Вок.": "вок"}

RU_participle_keys = ['Пр. действ. наст.', 'Пр. действ. прош.', 'Деепр. наст.',
                      'Деепр. прош.', 'Пр. страд. наст.', 'Пр. страд. прош.']

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 2:
        sys.exit("Pass the number of words to read and the language")
    words_number, task = int(args[0]), args[1]
    if task not in ["RU_nouns", "RU_verbs", "RU_adjectives", "LA"]:
        sys.exit("Language must be RU or LA")
    if task == "RU_nouns":
        language_string = u"Русский"
        cases = RU_cases
        infile = "../../data/dictionaries/noun_counts_rnc.csv"
        outfile = "rus_noun_paradigms_1.txt"
        table_width = 250
    elif task == "RU_verbs":
        language_string = u"Русский"
        cases = RU_cases
        infile = "../../../data/dictionaries/words_by_pos/rnc_v.out"
        outfile = "rus_verb_paradigms.txt"
        pos_string = "глагол"
    elif task == "RU_adjectives":
        language_string = u"Русский"
        cases = RU_cases
        infile = "../../../data/dictionaries/words_by_pos/rnc_a.out"
        outfile = "rus_adj_paradigms.txt"
        pos_string = "прилагательное"
    else:
        language_string = u"Латинский"
        cases = LA_cases
        infile = "../../data/dictionaries/sorted_latin_nouns.txt"
        outfile = "latin_noun_paradigms_1.txt"
        table_width = 210
    default_path = u"https://ru.wiktionary.org/wiki/"
    errors_path ="errors_1"
    i = 0
    ferr = open(errors_path, "w", encoding="utf8")
    with open(infile, "r", encoding="utf8") as fin:
        with open(outfile, "w", encoding="utf8") as fout:
            for line in fin:
                if i >= words_number:
                    break
                if i % 20 == 1:
                    print(i)
                line = line.strip()
                lemma = line.split("\t")[0]
                # if lemma != 'подать':
                #     continue
                if lemma in ['быть', 'внимать']:
                    continue
                path = default_path + urp.quote(lemma)
                request = urr.Request(path)
                try:
                    response = urr.urlopen(request)
                except ure.HTTPError:
                    print("Not found: {0}".format(lemma))
                    continue
                contents = response.read().decode("utf8")
                if task in ["RU_verbs", "RU_adjectives"]:
                    contents_soup = bs4.BeautifulSoup(contents, "html.parser")
                    # условие с text=re.compile плохо работает при наличии пробелов
                    # поэтому делаем "в лоб"
                    h3_soups, h3_soup = contents_soup.find_all("h3"), None
                    h3_morpho_soups = []
                    for elem in h3_soups:
                        if elem.text.startswith("Морфологические и синтаксические свойства"):
                            h3_morpho_soups.append(elem)
                    for elem in h3_morpho_soups:
                        # elem уже точно не None
                        siblings = elem.find_next_siblings()
                        table_soup = None
                        for sibling in siblings:
                            if sibling.name == "h3":
                                break
                            if sibling.name != "p":
                                continue
                            if sibling.text.lower().startswith(pos_string):
                                table_soup = elem.find_next_sibling()
                                if table_soup.name != 'table':
                                    table_soup = None
                                break
                        else:
                            table_soup = None
                        if (table_soup is None):
                            continue
                        if task == "RU_verbs":
                            trs = table_soup.find_all("tr")[1:]
                            basic_trs, other_trs = trs[:6], trs[6:]
                            tds = [tr.find_all("td")[1:] for tr in basic_trs]
                            if len(tds) == 0 or len(tds[0]) != 3:
                                continue
                            present_forms = [re.sub('[△*]', '', elem[0].text).split("\n") for elem in tds]
                            past_forms = []
                            for elem in tds:
                                for form in elem[1].text.split("\n"):
                                    form = re.sub('[△*]', '', form)
                                    if form not in ["", '-', '—'] and form not in past_forms:
                                        past_forms.append(form)
                            imperative_forms = [re.sub('[△*]', '', tds[1][2].text).split("\n"),
                                                re.sub('[△*]', '', tds[4][2].text).split("\n")]
                            participle_forms = [['-']] * 6
                            for elem in other_trs:
                                tds = elem.find_all("td")
                                if len(tds) != 2:
                                    continue
                                key, value = tds
                                key, value = key.text, value.text
                                value = re.sub('[*△]', '', value).split(', ')
                                for j, ref_key in enumerate(RU_participle_keys):
                                    if key == ref_key:
                                        participle_forms[j] = value
                                        break
                            inflection = present_forms + [[x] for x in past_forms] +\
                                imperative_forms + participle_forms
                            fout.write('{}\t{}\n'.format(lemma,
                                                         ";".join(",".join(elem) for elem in inflection)))
                        elif task == "RU_adjectives":
                            trs = table_soup.find_all("tr")[2:]
                            sys.exit()
                        i += 1
                        break
                    if (table_soup is None):
                        print(lemma)
                        ferr.write(lemma + "\n")
                        continue
    ferr.close()
                # splits = re.split(split_regexp, contents)
                # for index in range(1, len(splits), 2):
                #     span = bs4.BeautifulSoup(splits[index]).span
                #     if span is None:
                #         continue
                #     piece_content = span.get_text()
                #     if piece_content == language_string:
                #         break
                # if index >= len(splits):
                #     print("Not Russian: {0}".format(lemma))
                #     continue
                # index += 1
                #
                # soup = bs4.BeautifulSoup(splits[index])
                # table = soup.find("table", attrs={"width": table_width})
                # if table is None:
                #     ferr.write(lemma + "\n")
                #     continue
                # headings = [th.get_text()[:2]
                #             for th in table.find("tr").find_all("th")[1:]]
                # trs = table.find_all("tr")[1:]
                # to_write = []
                # for tr in trs:
                #     td = tr.find("td")
                #     if td is None:
                #         continue
                #     case = cases.get(td.get_text())
                #     if case is None:
                #         continue
                #     word_forms = [re.split(cell_split_regexp, x.get_text()) for x in tr.find_all("td")[1:]]
                #     for word_forms_list, number in zip(word_forms, headings):
                #         # ferr.write(str(word_forms_list) + "\n")
                #         for word in word_forms_list:
                #             word = word.replace('*', '')
                #             to_write.append((word, case, number))
                # if len(to_write) > 0:
                #     i += 1
                #     fout.write("{0}\t{1}\n".format(i, lemma))
                #     for word, case, number in to_write:
                #         fout.write(",".join([word, lemma, case, number]) + "\n")




