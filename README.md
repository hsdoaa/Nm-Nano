# Nm-Nano
Nm-Nano: A framework for predicting 2´-O-Methylation (Nm) Sites in Nanopore RNA Sequencing Data

# Getting Started and pre-requisites
The following softwares and modules should be installed before using  Nm-Nano

python 3.6.10

minimpa2 (https://github.com/lh3/minimap2)

Nanopolish (https://github.com/jts/nanopolish)

samtools (http://www.htslib.org/)

numpy 1.18.1

pandas 1.0.1

sklearn 0.22.2.post1

tensorflow 2.0.0

keras 2.3.1 (using Tensorflow backend)


# Running  Nm-Nano:

In order to run  Nm-Nano, the user has to choose one of Nm-nano machine learning models (xgboost or Random forest with embeeding) and run one of following python commands:

python test_xgboost.py      #To run the xgboost model

Or 

python RF_embedding.py           #To run the RF with embeeding model


# Note:
- The user should include the benchmark dataset in the same path of xgboost.py and RF_embedding.py

- All files required to generate the benchmark dataset is included in the generate_benchmark folder. For testing Nm-nano framework, we include a small benchmark dataset sample for hela cell line. However, the user is free to generate a benchmark benchmark dataset for any other cell lines based on the instructions mentioned in README file in generate_benchmark folder.
