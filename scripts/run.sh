#!/bin/bash
python3.3 setup.py build_ext --inplace
python3.3 -u lm_paradigm_classifier_main.py cross-validation