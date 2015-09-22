# pyparadigm
A project for automatic construction of morphological paradigms

**1. Assigning words to paradigms according to full inflection table.**
python pyparadigm.py &lt; inflection_tables&gt;  &lt; LANGUAGE_CODE&gt;  &lt; words_by_paradigms_file&gt;  &lt; paradigm_codes_file&gt;   
**&lt; inflection_tables&gt; :** file with paradigms in the format as in data/Latin/latin_noun_paradigms.txt  
**&lt; LANGUAGE_CODE&gt; :** code of the language (only LA for 'Latin' and RU for 'Russian' are supported yet)  
**&lt; words_by_paradigms_file&gt; :** output file containing words and paradigms in the format as in data/Latin/nouns_by_paradigms.txt  
**&lt; paradigm_stats_file&gt; :** output file containing paradigms with one member for each paradigm

**2. Automatic detection of paradigms**
1. python transform_for_learning.py &lt; words_by_paradigm_file&gt;  &lt; inflection_tables&gt;  &lt; outfile_for_lemmas&gt;  &lt; outfile_for_paradigm_codes&gt;    
   *Transforms the output of pyparadigm to the format used in paradigms learning*  
   **&lt; words_by_paradigm_file&gt; :** first output file of step    **&lt; inflection_tables&gt; :** input file of step 1, is required only for ordering.  
   **&lt; lemmas_with_codes&gt; :** output file with lemmas and paradigm codes for future learning  
   **&lt; paradigms_with_codes&gt; :** output file with paradigms and their codes

2a. python learn_paradigms.py cross-validation &lt; paradigms_with_codes&gt;  &lt; lemmas_with_codes&gt;  &lt; folds_number&gt;  [&lt; feature_selection_method&gt; ]  
   *Performs cross-validation testing of paradigms learning algorithms. By this moment, the module is in testing phase, therefore the percentage of training data, the fraction of features to select and maximal length of suffix features used during learning are set in the program code. In the release version they will also be command line arguments.*  
   **&lt; paradigms_with_codes&gt; :** second outfile of the previous step,  
   **&lt; lemmas_with_codes&gt; :** first outfile of the previous step,  
   **&lt; folds_number&gt; :** number of folds in cross-validation,  
   **&lt; feature_selection_method&gt; :** feature selection algorithm ('ambiguity' or 'log_odds', default and preferred is 'ambiguity')
