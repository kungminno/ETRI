# KoBERT fine-tuning
* [KoBERT By SKTBrain](https://github.com/SKTBrain/KoBERT)
  * [How to Install](#how-to-install)
  * [Requirement](#requirement)
  * [Class and function definitions](#class-and-function-definitions)

*****

### How to Install
* Install KoBERT as a python package

<pre><code>!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'</code></pre>

### Requirement
* see [requirement.txt](https://github.com/kungminno/ETRI/blob/main/KoBERT/requirements.txt)

### Class and function definitions
* [BERTDataset](#bertdataset)
* [BERTClassifier](#bertclassifier)
* [FocalLoss](#focalloss)

* [calc_accuracy](#calc_accuracy)
* [test_models](#test_models)
* [new_softmax](#new_softmax)
* [predict](#predict)

### BERTDataset 
### BERTClassifier
### FocalLoss
### calc_accuracy
### test_models
### new_softmax
### predict

*****

## Model Learning
<pre><code>max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10  
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5</code></pre>

*****

## Learned model Test
### Introduction

This code loads and preprocesses a test dataset, and then tests the accuracy of multiple pre-trained models on the test dataset. The model with the highest accuracy is then selected as the best model.

### Dependencies

The following libraries are required to run this code:
* pandas
* torch
* transformers (assuming tok is an instance of BertTokenizer)

### Usage

1. Provide the file path of the test dataset CSV file.
2. Read the CSV file into a pandas dataframe and preprocess it by dropping null values and empty rows.
3. Convert the test dataset into a list of texts and create a BERTDataset instance.
4. Create a data loader for the test dataset.
5. Test the accuracy of multiple pre-trained models using the test_models function, which is not shown in this code snippet.
6. Select the model with the highest accuracy by finding its index in the list of model file paths.
7. Get the filepath of the best model.

### Output
The output of this code is the filepath of the pre-trained model with the highest accuracy on the test dataset.

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
*The learned model used in the prediction we developed can be downloaded from the following link.*

* [Focal_KoBERT_e10_b64_fold_5.pt](https://drive.google.com/file/d/1-ksRR8nnxkIb3AG0k345_udQjcAN3_ga/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_state_dict.pt](https://drive.google.com/file/d/10-wtu9ZRTyrf9ptGilPplQxllN-sc_wv/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_all.tar](https://drive.google.com/file/d/106QgpX75WSZI0rO1QxIDrxtWHx9ghmOZ/view?usp=share_link)
