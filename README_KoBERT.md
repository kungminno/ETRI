# [KoBERT By SKTBrain](https://github.com/SKTBrain/KoBERT)

## How to Install
* Install KoBERT as a python package

<pre><code>!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'</code></pre>

## Requirement
* see [requirement.txt](https://github.com/kungminno/ETRI/blob/main/KoBERT/requirements.txt)

## Class and function definitions
* BERTDataset Class
* BERTClassifier Class
* FocalLoss Class

* calc_accuracy
* test_models
* new_softmax
* predict

*****

## Model Learning

*****

## Learned model Test

*****

## Learned model Prediction
### Introduction
This code reads a CSV file containing text and true labels, and then uses a pre-trained model to predict labels for the text. The predicted labels are stored in a list and then used to generate a classification report using the scikit-learn library.

### Dependencies
The following libraries are required to run this code:
* torch
* csv
* scikit-learn

### Usage
1. Load the pre-trained model by providing the path of the saved model file.
2. Read a CSV file containing text and true labels.
3. Loop over the rows in the CSV file, and use the pre-trained model to predict labels for each text.
4. Store the true and predicted labels in separate lists.
5. Generate a classification report using the scikit-learn library, and provide the list of target names.

### Output
The output of this code is a classification report, which contains precision, recall, F1-score, and support for each class. The classification report provides an evaluation of how well the model is performing on each label.

### Note
The learned model used in the prediction can be downloaded from the following link.

* [Focal_KoBERT_e10_b64_fold_5.pt](https://drive.google.com/file/d/1-ksRR8nnxkIb3AG0k345_udQjcAN3_ga/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_state_dict.pt](https://drive.google.com/file/d/10-wtu9ZRTyrf9ptGilPplQxllN-sc_wv/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_all.tar](https://drive.google.com/file/d/106QgpX75WSZI0rO1QxIDrxtWHx9ghmOZ/view?usp=share_link)
