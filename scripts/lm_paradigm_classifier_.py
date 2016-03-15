import sys
import os
import subprocess
import copy

from itertools import chain, product

from cython import address

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from kenlm import LanguageModel, Model, State
from pynlpl.lm.lm import ARPALanguageModel
from transformation_classifier import TransformationsHandler
from local_paradigm_transformer import LocalTransformation

class LMParadigmClassifier(BaseEstimator, ClassifierMixin):
    """
    Пытаемся классифицировать парадигмы с помощью языковых моделей
    """
    def __init__(self, paradigm_codes, paradigm_counts, lm_order=3,
                 lm_type="kenlm", multiclass=False, tmp_folder="saved_models"):
        self.paradigm_codes = paradigm_codes
        self.paradigm_counts = paradigm_counts
        self.lm_order = lm_order
        self.lm_type = lm_type
        self.tmp_folder = tmp_folder
        self.multiclass = multiclass
        self.lm = None
        self.filename_count = 1
        self._initialize()

    def _initialize(self):
        self.transformations_handler = TransformationsHandler(self.paradigm_codes,
                                                              self.paradigm_counts)
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)


    def fit(self, X, y):
        lemmas_with_codes_and_vars = list(chain.from_iterable(
            [(lemma, code, values) for code, values in label]
            for lemma, label in zip(X, y)))
        strings_for_lm_learning = \
            self.transformations_handler._extract_transformations_for_lm_learning(
                lemmas_with_codes_and_vars)
        self.infile = os.path.join(self.tmp_folder,
                                   "saved_models_{0}.sav".format(self.filename_count))
        with open(self.infile, "w") as fout:
            for seq in strings_for_lm_learning:
                fout.write(" ".join(map(str, seq)) + "\n")
        self.outfile = os.path.join(self.tmp_folder,
                                    "saved_models_{0}.arpa".format(self.filename_count))
        with open(self.infile, "r") as fin, open(self.outfile, "w") as fout:
            subprocess.call(["/data/sorokin/tools/kenlm/bin/lmplz",
                             "-o", str(self.lm_order), "-S", "4G"], stdin=fin, stdout=fout)
        if self.lm_type == "pynlpl":
            self.lm = ARPALanguageModel(self.outfile, base_e=False)
        elif self.lm_type == "kenlm":
            self.lm = Model(self.outfile)
        return self

    @property
    def transformation_codes(self):
        return self.transformations_handler.transformation_codes

    @property
    def transformations_by_strings(self):
        return self.transformations_handler.transformations_by_strings

    @property
    def transformations(self):
        return self.transformations_handler.transformations

    def get_best_continuous_score(self, word):
        total_score = 0
        best_variant, best_score = None, -np.inf
        if self.lm_type == "kenlm":
            state = State()
            self.lm.BeginSentenceWrite(state)
        else:
            history = ('<s>',)
        for i, symbol in enumerate(word):
            prefix, suffix = word[:i], word[i:]
            for code in self.transformations_by_strings.get(suffix, []):
                if self.lm_type == "kenlm":
                    # curr_state изменяется в функции BaseScore, поэтому копируем
                    curr_state, out_state = copy.copy(state), State()
                    code_score = self.lm.BaseScore(curr_state, str(code), out_state)
                    end_score = self.lm.EndSentenceBaseScore(out_state, State())
                elif self.lm_type == "pynlpl":
                    code_score = self.lm.scoreword(str(code), history)
                    new_history = history + (code,)
                    end_score = self.lm.scoreword('</s>', new_history)
                score = (total_score + code_score + end_score) # / (i + 1)
                # print("{3} {0} {1} {2:.2f}".format(
                #         " ".join(prefix), self.transformations[code], score, code))
                if score > best_score:
                    best_variant = list(prefix) + [code]
                    best_score = score
                    print("{3} {0} {1} {2:.2f}".format(
                        " ".join(prefix), self.transformations[code], score, code))
            if self.lm_type == "kenlm":
                out_state = State()
                score = self.lm.BaseScore(state, symbol, out_state)
                state = out_state
            elif self.lm_type == "pynlpl":
                score = self.lm.scoreword(symbol, history)
                history += (symbol,)
            total_score += score
        curr_state, word_score = out_state, total_score
        for code in sorted(self.transformations_by_strings[""]):
            if self.lm_type == "kenlm":
                state, out_state = copy.copy(curr_state), State()
                code_score = self.lm.BaseScore(curr_state, str(code), out_state)
                end_score = self.lm.EndSentenceBaseScore(out_state, state)
            elif self.lm_type == "pynlpl":
                code_score = self.lm.scoreword(str(code), history)
                new_history = history + (str(code), )
                end_score = self.lm.scoreword('</s>', new_history)
            score = (total_score + code_score + end_score) # / (len(word) + 1)
            # print("{0} {1} {2:.2f}".format(" ".join(word),
            #                                "#".join(self.transformations[code].trans), score))
            if score > best_score:
                best_variant, best_score = list(word) + [code], score
                print("{3} {0} {1} {2:.2f}".format(
                        " ".join(word), self.transformations[code], score, code))
        answer = []
        for elem in best_variant:
            if isinstance(elem, str):
                answer.append([elem] * 13)
            else:
                answer.append(self.transformations[elem].trans)
        print("#".join("".join(elem) for elem in zip(*answer)))


    def test(self):
        """
        Тестируем интерфейс языковых моделей
        """
        word = "лыжня"
        self.get_best_continuous_score(word)
