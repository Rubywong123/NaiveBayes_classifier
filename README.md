# EMNLP Assignment 1

### code structure

* <font color=red> Pay attention </font> that the directory of data(named "data") should be placed in the same directory as .sh and .py files, and it should contain four files, namely `sst_train.csv`, `sst_test.csv`, `yelp_train.csv`, `yelp_test.csv`.

* The structure of working directory should be:

\-main.py

\-model.py

\-config.py

\-cleandata.py

\-utils.py

\-run.sh

\-data

|\-\-\-\-sst_train.csv

|\-\-\-\-sst_test.csv

|\-\-\-\-yelp_train.csv

|\-\-\-\-yelp_test.csv

`main.py` contains the main routine of the procedure, which includes loading data, pre-processing data, training model and evaluation.

`model.py` contains the DIY model, and many sub-functions defined in it.

`config.py` contains the operation of getting options by using `argparse.ArgumentParser`. `--dataset` and `--alpha` could be defined by users in shell.

`cleandata.py` contains functions of doing data pre-processing.

`utils.py` defines metrics and other basic functions.



### Dependency

Python 3.8.9(64-bit)

NLTK 3.5

numpy 1.20.2

### Run it now!

Please run the shell to check the program by typing as follows:

* If you are MAC user, then:

```
bash run_mac.sh {sst, yelp}
```

* Otherwise:

```
bash run.sh {sst, yelp}
```

The argument(chosen dataset) for .sh file will be passed to the program. By default, it'll run on `sst-5`.



