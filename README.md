# pyparadigm
A project for automatic construction of morphological paradigms

**1. Assigning words to paradigms according to full inflection table.**
python pyparadigm.py <inflection_tables> <LANGUAGE_CODE> <words_by_paradigms_file> <paradigm_codes_file>  
**<inflection_tables>:** file with paradigms in the format as in data/Latin/latin_noun_paradigms.txt  
**<LANGUAGE_CODE>:** code of the language (only LA for 'Latin' and RU for 'Russian' are supported yet)  
**<words_by_paradigms_file>:** output file containing words and paradigms in the format as in data/Latin/nouns_by_paradigms.txt  
**<paradigm_stats_file>:** output file containing paradigms with one member for each paradigm

**2. Automatic detection of paradigms**
1. python transform_for_learning.py <words_by_paradigm_file> <inflection_tables> <outfile_for_lemmas> <outfile_for_paradigm_codes>   
   *Transforms the output of pyparadigm to the format used in paradigms learning*  
   **<words_by_paradigm_file>:** first output file of step    **<inflection_tables>:** input file of step 1, is required only for ordering.  
   **<lemmas_with_codes>:** output file with lemmas and paradigm codes for future learning  
   **<paradigms_with_codes>:** output file with paradigms and their codes

2a. python learn_paradigms.py cross-validation <paradigms_with_codes> <lemmas_with_codes> <folds_number> [<feature_selection_method>]  
   *Performs cross-validation testing of paradigms learning algorithms. By this moment, the module is in testing phase, therefore the percentage of training data, the fraction of features to select and maximal length of suffix features used during learning are set in the program code. In the release version they will also be command line arguments.*  
   **<paradigms_with_codes>:** second outfile of the previous step,
   **<lemmas_with_codes>:** first outfile of the previous step,
   **<folds_number>:** number of folds in cross-validation
   **<feature_selection_method>** feature selection algorithm ('ambiguity' or 'log_odds', default and preferred is 'ambiguity')
   
    
