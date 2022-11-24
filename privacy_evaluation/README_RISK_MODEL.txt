Usage: synthetic_risk_model_mem.py [model] [exp_id] [beta] [train_filename] [test_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_mem.py baseline 1 0.05 train_uw test_uw syn_ _uw _ Results_Synthetic_UW/

1. [model]: name of data generation model. Selected from ['baseline', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'real'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [beta]: the threshold for the false positive rate. A real number in [0, 1]. Default: '0.05'. Try: '0.1'.
4. [train_filename]: the filename of the training file. Default: 'train_uw'.
5. [test_filename]: the filename of the training file. Default: 'test_uw'.
6. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
7. [suffix_syn]: the suffix of the synthetic filename. Default: '_uw'.
8. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
9. [output_directory]: output directory. Default: 'Results_Synthetic_UW/'.



Usage: synthetic_risk_model_attr.py [model] [exp_id] [x] [y] [original_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_attr.py baseline 1 0 8 train_uw syn_ _uw _ Results_Synthetic_UW/

1. [model]: name of data generation model. Selected from ['baseline', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'real'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [x]: 10 to x is the number of neighbours. A integer larger than -1. Default: '0'. Try: '1'.
4. [y]: 2 to y is the number of sensitive attributes A integer larger than -1. Default: '8'. Try: '10'.
5. [original_filename]: the filename of the original patient file. Default: 'train_uw'.
6. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
7. [suffix_syn]: the suffix of the synthetic filename. Default: '_uw'.
8. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
9. [output_directory]: output directory. Default: 'Results_Synthetic_UW/'.



Usage: synthetic_risk_model_reid.py [model] [exp_id] [theta] [original_filename] [pop_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_reid.py baseline 1 0.05 train_uw pop_uw syn_ _uw _ Results_Synthetic_UW/

1. [model]: name of data generation model. Selected from ['baseline', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'real'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [theta]: ratio of the correctly inferred attributed in a successful attack. A real number in [0, 1]. Default: '0.05'. Try: '0.001'.
4. [original_filename]: the filename of the original patient file. Default: 'train_uw'.
5. [pop_filename]: the filename of the population file with demographics (QIDs) only. Default: 'pop_uw'.
6. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
7. [suffix_syn]: the suffix of the synthetic filename. Default: '_uw'.
8. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
9. [output_directory]: output directory. Default: 'Results_Synthetic_UW/'.



Usage: synthetic_risk_model_uw_nnaa.py [model] [exp_id] [train_filename] [test_filename] [prefix_syn] [infix_syn] [output_directory]

Example: synthetic_risk_model_uw_nnaa.py iwae 1 train_uw test_uw syn_ _uw _ Results_Synthetic_UW/

1. [model]: name of data generation model. Selected from ['iwae', 'medgan', 'medbgan', 'emrwgan', 'medwgan', 'dpgan', 'real']. Default: 'real'.
2. [exp_id]: No. of the experiment. Selected from ['1', '2', '3']. Default: '1'.
3. [train_filename]: the filename of the training file. Default: 'train_uw'.
4. [test_filename]: the filename of the test file. Default: 'test_uw'.
5. [prefix_syn]: the prefix of the synthetic filename. Default: 'syn_'.
6. [suffix_syn]: the suffix of the synthetic filename. Default: '_uw'.
7. [infix_syn]: the suffix of the synthetic filename in the middle of [model_name] and [exp_id]. Default: '_'.
8. [output_directory]: output directory. Default: 'Results_Synthetic_UW/'.
