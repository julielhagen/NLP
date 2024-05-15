# NLP
This project examine how silver data measure up to gold data across two language families, namely the Germanic and Slavic ones.

It aims to compare the quality and performance of human annotated and automatically generated data from Wikipedia. We ask if silver data is a contender to replace gold data for building Language Model that can identify NER-tags.

The data from Wikipedia and the all of the test set are provided. The silver training and test data lies in the folder ‘Data’. The gold annotations, for both of the explored languages, lie in the folder ‘annotations’.

We also included pickled models for each version of BERT, where the data is tokenized.

It was not possible to upload the already trained models to GITHUB due to size. If you need to reproduce the results, the file ‘wikiann_bert’ has to run and retrained. This require a powerful computer and we suggest connecting to an HPC if your own computer is not powerfull enough.

The ‘baseline’ file is our first draft of a model, and was trained on another dataset.
