# pyparadigm
A project for automatic construction of morphological paradigms

**1. Assigning words to paradigms according to full inflection table.**  
python3 pyparadigm.py  &lt;inflection_tables&gt;  &lt;LANGUAGE_CODE&gt; &lt;maximal_gap&gt; &lt;maximal_initial_gap&gt;  &lt;table_processing_mode&gt; &lt;words_by_paradigms_file&gt;  &lt;paradigm_codes_file&gt;   
**&lt;inflection_tables&gt; :** file with paradigms in the format as in data/Latin/latin_noun_paradigms.txt  
**&lt;LANGUAGE_CODE&gt; :** code of the language (LA for 'Latin', RU for 'Russian', FI for 'Finnish' and RU_verbs for Russian verbs are supported)  
**&lt;maximal_gap&gt; :** maximal gap length in lcs method  
**&lt;maximal_initial_gap&gt; :** maximal initial gap length in lcs method  
**&lt;paradigm_processing_mode&gt; :** paradigm processing method, &laquo;first&raquo; uses only first variant in case of multiple word forms for one word, &laquo;all&raquo; considers all the variants  
**&lt;words_by_paradigms_file&gt; :** output file containing words and paradigms in the format as in data/Latin/nouns_by_paradigms.txt  
**&lt;paradigm_stats_file&gt; :** output file containing paradigms with one member for each paradigm

**2. Automatic detection of paradigms**  
1. python transform_for_learning.py &lt;words_by_paradigm_file&gt; &lt;paradigm_stats_file&gt; &lt;inflection_tables&gt;  &lt;outfile_for_lemmas&gt;  &lt;outfile_for_paradigm_codes&gt;    
*Transforms the output of pyparadigm to the format used in paradigms learning*  
**&lt;words_by_paradigm_file&gt; :** first output file of step 1  
**&lt;paradigm_stats_file&gt; :** second output file of step 2  
**&lt;inflection_tables&gt; :** input file of step 1, is required only for ordering.  
**&lt;lemmas_with_codes&gt; :** output file with lemmas and paradigm codes for future learning  
**&lt;paradigms_with_codes&gt; :** output file with paradigms and their codes

2a. python learn_paradigms.py [-p] [-m] [-T &lt;train_data_dir&gt;] [-O &lt;test_output_dir&gt;] cross-validation &lt;paradigms_with_codes&gt;  &lt;lemmas_with_codes&gt; &lt;max_feature_lengths&gt;  &lt;feature_fractions&gt; &lt;train_data_fractions&gt; &lt;folds_number&gt;  [&lt;feature_selection_method&gt; ]  
   **-p :** if True, class probabilities are also predicted. Default value is False,   
   **-m :** if True, multiple paradigms for one lemma can be predicted. Default value is False,  
   **train_data_dir :** a directory to output train data splits,  
   **test_output_dir :** a directory to output classification results,  
   **&lt;paradigms_with_codes&gt; :** second outfile of the previous step,  
   **&lt;lemmas_with_codes&gt; :** first outfile of the previous step,  
   **&lt;max_feature_lengths&gt; :** comma-separated list of maximal feature lengths,  
   **&lt;train_data_fractions&gt; :** comma-separated list of train data fractions,  
   **&lt;feature_fraction&gt; :** comma-separated list of feature fractions. This parameter determines the proportion of features which are selected on data preprocessing step.   
   **&lt;folds_number&gt; :** number of folds in cross-validation,  
   **&lt;feature_selection_method&gt; :** feature selection algorithm ('ambiguity' or 'log_odds', default and preferred is 'ambiguity')
