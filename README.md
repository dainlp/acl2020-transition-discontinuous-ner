This repository has a pytorch implementation of transition-based model for discontinuous NER, introduced in our ACL 2020 paper:

Xiang Dai, Sarvnaz Karimi, Ben Hachey, and Cecile Paris. 2020. An Effective Transition-based Model for Discontinuous NER. In ACL, Seattle, Washington.

* CADEC dataset can be downloaded at: https://data.csiro.au/dap/landingpage?pid=csiro:10948&v=3&d=true
* ShARe data can be downloaded at: https://physionet.org/


Once you download the dataset, you can use script data/cadec/build_data_for_transition_discontinuous_ner.sh to build the dataset.
* Sample data can be found at data/sample directory.


The script used to train the model on CADEC can be found in code/xdai/ner/transition_discontinuous/cadec.sh

