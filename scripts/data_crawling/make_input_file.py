import sys
import re


verb_categories = ['инфинитив', 'наст,1л,ед', 'наст,2л,ед', 'наст,3л,ед', 'наст,1л,мн', 'наст,2л,мн', 'наст,3л,мн',
                   'пр,муж', 'пр,жен', 'пр,ср', 'пр,мн', 'прич,действ,наст', 'прич,действ,пр',
                   'деепр,наст', 'деепр,прош', 'прич,страд,наст', 'прич,страд,пр']


def make_input_file(infile, categories, outfile, add_basic=True):
    with open(infile, "r", encoding="utf8") as fin,\
            open(outfile, "w", encoding="utf8") as fout:
        for i, line in enumerate(fin, 1):
            if i % 100 == 0:
                print(i)
            line = line.strip()
            splitted = line.split('\t')
            if len(splitted) != 2:
                continue
            lemma, forms = splitted
            forms = re.sub('[ ]*/[ ]*', ',', forms)
            forms = forms.replace('—', '-')
            splitted_forms = []
            for elem in forms.split(';'):
                if '(' not in elem:
                    splitted_forms.append(elem.split(','))
                elif ' (' in elem:
                    elem = elem.replace(' (', ',').replace(')', '')
                    splitted_forms.append(elem.split(','))
                else:
                    # умерев(ши)
                    splitted_elem = elem.split(',')
                    curr_splitted_forms = []
                    for x in splitted_elem:
                        if '(' not in x:
                            curr_splitted_forms.append(x)
                        else:
                            first, second = re.match('(.+)\((.+)\)$', x).groups()
                            curr_splitted_forms.extend((first, first+second))
                    splitted_forms.append(curr_splitted_forms)
            forms = splitted_forms
            if add_basic:
                forms = [[lemma]] + forms
            fout.write("{}\t{}\n".format(i, lemma))
            for category_forms, category in zip(forms, categories):
                for form in category_forms:
                    fout.write("{},{},{}\n".format(form, lemma, category))


if __name__ == '__main__':
    args = sys.argv[1:]
    infile, pos, outfile = args
    if pos == 'verb':
        categories = verb_categories
    else:
        raise NotImplementedError
    make_input_file(infile, categories, outfile)
