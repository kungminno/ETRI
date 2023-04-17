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
#### Introduction
이 클래스는 PyTorch의 nn.Module 클래스를 상속받아 생성된 클래스입니다.

#### Class Parameters
* <code>alpha</code>는 클래스 간 불균형을 보정하기 위한 가중치입니다. 
* <code>gamma</code>는 어려운 샘플에 대한 가중치를 증가시키는 파라미터입니다. 
* <code>reduction</code> 매개 변수는 출력값의 reduce 방법을 정의합니다.

#### Method
<code>forward()</code> 메소드는 입력값과 대상값을 받아 사용됩니다. 
* <code>BCE_loss</code>는 이진 교차 엔트로피 손실을 계산합니다. 
* <code>pt</code>는 softmax 확률의 역수로 정의되어, 확률이 작을수록 커집니다. 
* <code>F_loss</code>는 focal loss를 계산하기 위한 공식입니다.

#### Output
<code>reduction</code> 매개 변수에 따라 F_loss의 출력값이 달라집니다. 
* <code>'mean'</code>으로 설정하면 손실값의 평균이 반환됩니다.
* <code>'sum'</code>으로 설정하면 손실값의 합이 반환됩니다.
* 설정하지 않으면, 원본 손실 값인 F_loss가 반환됩니다.


### calc_accuracy
#### Introduction
이 함수는 입력값 X와 대상값 Y를 받아 모델의 정확도를 계산하는 함수입니다.

#### Function Workflow
1. max_vals와 max_indices는 입력값 X에서 가장 큰 값을 찾아서, 그 값과 해당 인덱스를 반환합니다. 이때, 행 방향으로 가장 큰 값을 찾아서 반환하도록 설정합니다.
2. max_indices와 대상값 Y를 비교하여 같은 값이 있는 경우, 그 값을 모두 합칩니다. 
3. data.cpu().numpy() 함수를 사용하여 계산 결과를 넘파이 배열로 변환합니다.
4. max_indices.size()[0]으로 나눠서 정확도를 계산합니다.

#### Output
결과값은 <code>train_acc</code>로 반환되며, 모델의 정확도를 나타냅니다.


### test_models
#### Introduction
이 함수는 사전에 학습된 BERTClassifier 모델을 테스트 데이터셋에서 테스트하는 함수입니다. 함수는 상태 사전 파일에서 모델을 로드하고, 테스트 데이터셋에서 추론을 수행하며, 모델의 손실과 정확도를 계산하고 정확도 값을 반환합니다.

#### Function Parameters
<code>state_dict_filepath</code>: 학습된 BERTClassifier 모델의 파라미터를 담고 있는 상태 사전 파일의 경로입니다.

#### Function Workflow
1. 상태 사전 파일에서 BERTClassifier 모델을 로드합니다.
2. 모델을 평가 모드로 설정합니다.
3. 손실 값을 계산하는 FocalLoss 함수를 정의합니다.
4. 테스트 데이터셋을 배치별로 반복하며 각 배치에서 추론을 수행합니다.
5. 테스트 데이터셋에서 모델의 손실과 정확도를 계산합니다.
6. 계산된 손실과 정확도 값을 각각의 리스트에 추가합니다.
7. 테스트 손실과 정확도 값을 출력 후 테스트 정확도 값을 반환합니다.

#### Output
결과값은 <code>test_acc</code>로 반환되며, 테스트 정확도를 나타냅니다.


### new_softmax
#### Introduction
이 함수는 입력으로 받은 배열에 대해 소프트맥스 함수를 계산하여 각 요소의 확률값을 반환하는 함수입니다.

#### Function Workflow
1. 입력된 배열의 최대값을 찾고, 각 요소엣 최대값을 뺀 후 exp 함수를 적용하여 각 요소의 값을 계산합니다. 이 과정은 overflow를 방지하기 위해 수행됩니다.
2. 모든 요소에 대해 exp 함수를 적용하여 얻은 값들의 합을 구합니다.
3. 이를 이용하여 소프트맥스 함수를 적용하고, 최종적으로 각 요소의 확률 값을 반환합니다. 이때, 반환되는 확률 값은 0~100 범위 내의 값으로 반환합니다.

#### Output
입력 배열의 요소 수와 같은 크기의 numpy 배열이며, 소숫점 셋째 자리에서 반올림된 각 요소의 확률 값들로 이루어집니다.


### predict
#### Introduction
이 함수는 문장과 해당 문장의 실제 레이블을 입력으로 받아 각 감정 클래스의 예측된 레이블과 확률을 반환하는 함수입니다.

#### Function Workflow
1. 입력 문장과 실제 레이블로 데이터셋을 생성하고, BERT 토크나이저를 사용하여 입력 텍스트를 토크나이즈합니다.
2. 감정 분류를 위해 사전 훈련된 BERT 모델에 토큰화된 입력을 전달합니다.
3. BERT 모델은 각 감정 클래스에 대한 로짓의 목록을 출력하고, 이를 소프트맥스 함수를 사용하여 확률로 변환합니다.
4. 예측된 레이블은 가장 높은 확률을 가진 감정 클래스를 선택하여 결정합니다.

#### Output
* <code>y_true</code>: 입력 문장의 실제 레이블 리스트
* <code>y_pred</code>: 예측된 레이블 리스트
* <code>probabilities</code>: 각 감정 클래스의 예측된 확률의 리스트

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

이 코드는 테스트 데이터셋을 로드하고 전처리한 다음, 여러 미리 학습된 모델들의 정확도를 테스트합니다. 가장 높은 정확도를 가진 모델이 최고의 모델로 선택됩니다.

### Dependencies

이 코드를 실행하려면 다음 라이브러리가 필요합니다:
* pandas
* torch
* transformers (assuming tok is an instance of BertTokenizer)

### Usage
1. 테스트 데이터셋 CSV 파일의 파일 경로를 제공합니다.
2. CSV 파일을 데이터프레임으로 읽어들이고 null 값 및 빈 행을 삭제하여 전처리합니다.
3. 텍스트 리스트로 테스트 데이터셋을 변환하고 BERTDataset 인스턴스를 생성합니다.
4. 테스트 데이터셋을 위한 데이터 로더를 만듭니다.
5. test_models 함수를 사용하여 여러 미리 학습된 모델들의 정확도를 테스트합니다.
6. 가장 높은 정확도를 가진 모델의 인덱스를 모델 파일 경로의 리스트에서 찾아 선택 후 그 모델의 파일 경로를 얻습니다.

### Output
이 코드의 출력물은 테스트 데이터셋에서 가장 높은 정확도를 가진 미리 학습된 모델의 파일 경로입니다.

*****

## Learned model Prediction
### Introduction
이 코드는 텍스트와 true label을 포함하는 CSV 파일을 읽어들이고, 미리 학습된 모델을 사용하여 텍스트의 predict label을 생성합니다. predict label은 리스트에 저장되며, 그 다음 scikit-learn 라이브러리를 사용하여 classification report를 생성합니다.

### Dependencies
이 코드를 실행하려면 다음 라이브러리가 필요합니다:
* torch
* csv
* scikit-learn

### Usage
1. 저장된 모델 파일의 경로를 제공하여 미리 학습된 모델을 불러옵니다.
2. 텍스트와 true label을 포함하는 CSV 파일을 읽어들입니다.
3. CSV 파일의 각 행에 대해 반복하며, 각 텍스트에 대해 미리 학습된 모델을 사용하여 predict label을 생성합니다.
4. true label과 predict label을 별도의 리스트에 저장합니다.
5. scikit-learn 라이브러리를 사용하여 classification report를 생성하고, 대상 이름의 리스트(the list of target names)를 제공합니다.

### Output
이 코드의 출력물은 classification report 입니다. 이 report는 각 클래스의 precision, recall, F1-score 및 support을 제공합니다. classification report는 모델이 각 레이블에서 얼마나 잘 수행되고 있는지를 평가하는 데 사용됩니다.

### Note
*<code>Learned model Prediction</code>에서 사용되는 우리 개발한 학습 모델은 다음 링크에서 다운로드할 수 있습니다.*

* [Focal_KoBERT_e10_b64_fold_5.pt](https://drive.google.com/file/d/1-ksRR8nnxkIb3AG0k345_udQjcAN3_ga/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_state_dict.pt](https://drive.google.com/file/d/10-wtu9ZRTyrf9ptGilPplQxllN-sc_wv/view?usp=share_link)

* [Focal_KoBERT_e10_b64_fold_5_all.tar](https://drive.google.com/file/d/106QgpX75WSZI0rO1QxIDrxtWHx9ghmOZ/view?usp=share_link)
