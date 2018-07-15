# Fujitsu AI-NLP Challenge

## Description:
	Find the most relevant answer from a set of answer for a particular question. For more information please walthrough this link https://openinnovationgateway.com/ai-nlp-challenge/challenge.php


Note: We executed this project on python3.5


## FYIP:

    Project Execution Main File is fujitsu_answer_sentence_selection.py

    We split this project in two mode: 1) Data generation mode where we create x_train and y_train
    from given training  data and save it as .txt file in data folder namely x_train.txt, y_train.txt,
    x_dev.txt, y_dev.txt. If you want to run this project to create x_train and y_train again
    so uncomment create_x_y() and save_x_y() function in fujitsu_answer_sentence_selection.py

    2) Second Mode is where we dont need to create x_train and y_train just load them and execute

    3) main/evaluation.py is the file to get MRR score as per fujitsu standard

## Project Directory Structure:

```
QA_Fujitsu_CNN/
   │
   ├- data/
   │    |-- SelQA-ass-result.example.json
   │    ├- SelQA-ass-dev.json
   │    ├- SelQA-ass-test.json
   │    └- SelQA-ass-train.json
   |    |-- vocab.txt
   │    ├- stopwords.txt
   │    ├- sentdata.txt
   │    └- x_train.txt
   |    |-- y_train.txt
   │    ├- x_dev.txt
   │    └- y_dev.txt
   │
   ├- model/
   │   │
   │   ├- this will contain various trained tensorflow model/
   │
   ├- helper/
   │   │
   │   ├- data_helper.py
   │   |-- dependency_helper.py
   │   │-- Evaluation.py
   │   │-- FeatureEngineering.py
   │   └- pre_processing.py
   │
   ├- main/
   │   ├- evaluation.py (FujitSu Standard Evaluation)
   │   └- fujitsu_answer_sentence_selection.py
   |
   └- README.MD
```
