import sys
import os
import subprocess
from itertools import chain, product

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import kenlm

from transformation_classifier import TransformationsHandler

class LMParadigmClassifier(BaseEstimator, ClassifierMixin):
    """
    Пытаемся классифицировать парадигмы с помощью языковых моделей
    """
    def __init__(self, paradigm_codes, paradigm_counts, lm_order=3,
                 multiclass=False, tmp_folder="saved_models"):
        self.paradigm_codes = paradigm_codes
        self.paradigm_counts = paradigm_counts
        self.lm_order = lm_order
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
        self.lm = kenlm.LanguageModel(self.outfile)
        return self