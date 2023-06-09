# KoBERT fine-tuning
* [KoBERT By SKTBrain](https://github.com/SKTBrain/KoBERT)
  * [How to Install](#how-to-install)
  * [Requirement](#requirement)
  * [Class and function definitions](#class-and-function-definitions)
* [Model Train](#model-train)
* [Model Test](#model-test)

*****

### How to Install
* Install KoBERT as a python package

<pre><code>!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'</code></pre>

### Requirement
* see [requirement.txt](https://github.com/kungminno/ETRI/blob/main/KoBERT/requirements.txt)

### Note
> *데이터셋 경로 설정: 데이터셋을 로드하기 위해 올바른 파일 경로를 설정해야 합니다.*<br>
> *모델 및 결과 저장 경로 설정: 학습된 모델과 결과를 저장할 경로를 설정해야 합니다.*


### Dependencies
* <code>torch</code>
* <code>torch.nn</code>
* <code>torch.optim</code>
* <code>numpy</code>
* <code>pandas</code>
* <code>matplotlib</code>
* <code>csv</code>
* <code>os</code>
* <code>json</code>
* <code>transformers</code>
* <code>gluonnlp</code>


### Class and function definitions
* [BERTDataset](#bertdataset)
* [BERTClassifier](#bertclassifier)

* [calc_accuracy](#calc_accuracy)
* [test_models](#test_models)

### BERTDataset
#### Introduction
이 클래스는 BERT 모델의 입력 형식에 맞게 데이터셋을 변환하는 클래스입니다.

#### Class Parameters
* <code>dataset</code>: 데이터셋
* <code>sent_idx</code>: 문장 인덱스
* <code>label_idx</code>: 레이블 인덱스
* <code>bert_tokenizer</code>: BERT tokenizer
* <code>max_len</code>: 최대 시퀀스 길이
* <code>pad</code>: 패딩 값
* <code>pair</code>: 문장 쌍 여부


#### Method
* <code>getitem</code>: 데이터셋에서 인덱스 i에 해당하는 데이터(문장과 레이블을 묶은)를 반환, 반환값은 튜플 형태
* <code>len</code>: 데이터셋의 샘플 수를 반환

#### Output
BERT 모델에 입력될 수 있는 형식으로 변환된 문장 정보(token_id, valid_length, segment_id)와 해당 문장의 레이블 정보를 튜플 형태로 반환하고, 데이터셋의 샘플 수를 반환합니다.


### BERTClassifier
#### Introduction
이 클래스는 BERT 모델을 이용하여 분류 작업을 수행하는 클래스입니다.

#### Class Parameters
* <code>bert</code>: 입력값을 처리하는 BERT 모델
* <code>dr_rate</code>: 드롭아웃 비율을 설정할 수 있는 인자
* <code>classifier</code>: 최종 출력값을 구하기 위한 선형 레이어

#### Method
* <code>gen_attention_mask</code>: 입력 토큰 시퀀스와 각 토큰의 유효 길이를 이용하여 어텐션 마스크를 생성하는 함수
* <code>forward</code>: 입력값을 받아 BERT 모델을 통과시켜 분류 결과를 반환하는 함수
  1. forward 함수는 입력값으로 <code>token_ids</code>, <code>valid_length</code>, <code>segment_ids</code>를 받으며, gen_attention_mask 함수를 이용하여 attention_mask를 생성합니다. 이때 segment_ids는 BERT 모델의 세그먼트 ID를 나타내며, attention_mask는 입력값의 길이만큼 1로 채워진 텐서입니다.
  2. BERT 모델을 통과시켜 [CLS] 토큰의 출력값인 pooler를 구합니다. 만약 dr_rate가 존재한다면, dropout을 적용하고 classifier 레이어를 통과시켜 최종 출력값을 구합니다. 만약 dr_rate가 없다면, pooler를 그대로 classifier 레이어에 통과시켜 최종 출력값을 구합니다.

#### Output
최종 출력값은 분류하고자 하는 클래스 수(num_classes)와 동일한 차원을 가지는 텐서입니다.


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
3. 손실 값을 계산하는 Sparse Categorical Cross Entropy 함수를 정의합니다.
4. 테스트 데이터셋을 배치별로 반복하며 각 배치에서 추론을 수행합니다.
5. 테스트 데이터셋에서 모델의 손실과 정확도를 계산합니다.
6. 계산된 손실과 정확도 값을 각각의 리스트에 추가합니다.
7. 테스트 손실과 정확도 값을 출력 후 테스트 정확도 값을 반환합니다.

#### Output
결과값은 <code>test_acc</code>로 반환되며, 테스트 정확도를 나타냅니다.


*****

## Model Train
```python
max_len = 64            # 입력 시퀀스의 최대 길이
batch_size = 64         # 한 번에 처리할 데이터 개수
warmup_ratio = 0.1      # 학습률 스케줄러의 warmup 비율
num_epochs = 10         # 전체 학습 데이터에 대한 학습 횟수
max_grad_norm = 1       # 기울기 클리핑을 위한 최대 기울기 값
log_interval = 200      # 로그 출력 간격
learning_rate = 5e-5    # 학습률
```

### Introduction
이 코드는 PyTorch를 기반으로 작성되어 있으며, KoBERT 모델을 사용하여 한국어 텍스트 데이터를 이용하여 감정 분석을 수행합니다.

### Usage
1. 필요한 라이브러리와 모델을 불러옵니다.
2. 학습에 사용할 데이터셋이 포함된 CSV 파일을 불러옵니다. 결측값이 있는 데이터는 삭제하고, 각 텍스트와 레이블을 리스트 형태로 변환하여 학습 데이터를 준비합니다.
3. K-Fold 교차 검증을 사용하기 위해 K를 5로 설정하여 5-Fold 교차 검증을 수행합니다.
4. 각 Fold에 대해 모델을 학습하고 검증합니다. 이 때, 옵티마이저로는 AdamW를 사용하며, 손실 함수로는 Sparse Categorical Cross Entropy를 사용합니다. 또한, 스케줄러를 사용하여 학습률을 조절합니다.
5. 각 Fold에 대한 학습이 완료되면, 해당 모델을 파일로 저장합니다.
6. 학습 과정 중 손실(loss) 및 정확도(accuracy)의 변화를 그래프로 시각화하여 결과를 확인합니다.

### Output
이 코드는 다음과 같은 결과를 출력합니다.
1. 각 epoch 및 batch에 대해 학습 정확도, 학습 손실, 검증 정확도, 검증 손실이 출력됩니다.
2. 각 Fold에 대해 학습된 모델의 가중치를 파일로 저장합니다. 이 파일들은 나중에 다시 불러와 모델을 평가하거나 새로운 데이터에 대한 예측을 수행하는 데 사용할 수 있습니다. 파일의 형식은 다음과 같습니다:
    * <code>모델 파일(.pt)</code>: 전체 모델 구조와 가중치를 포함합니다.
    * <code>상태 사전 파일(_state_dict.pt)</code>: 모델의 가중치만 포함합니다.
    * <code>전체 체크포인트 파일(_all.tar)</code>: 모델의 가중치와 옵티마이저의 상태를 포함합니다.
3. 학습 및 검증 과정에서 손실과 정확도의 변화를 그래프로 시각화합니다.

*****

## Model Test
### Introduction
이 코드는 테스트 데이터셋을 불러와 각각의 모델에 대한 정확도를 계산한 후, 가장 높은 정확도를 가진 최적의 모델을 찾는 과정입니다.

### Usage
1. 테스트 데이터셋이 포함된 CSV파일을 불러옵니다.
2. CSV 파일을 데이터프레임으로 읽어들여 결측값을 제거하고 전처리를 수행한 후, 테스트 데이터셋의 텍스트르 리스트로 만듭니다.
3. 'BERTDataset' 클래스를 사용하여 테스트 데이터셋을 변환하고, 모델 테스트에 사용되는 데이터 로더를 생성합니다.
4. 미리 학습된 각 모델의 가중치 파일에 대해 'test_models()' 함수를 호출하여 모델의 정확도를 계산하고 결과르 리스트에 저장합니다.
5. 가장 높은 정확도를 가진 최적의 모델의 가중치 파일 경로를 찾습니다.

### Output
이 코드는 가장 높은 정확도를 가진 최적의 모델의 가중치 파일 경로를 출력합니다.

### Note
*<code>Model Test</code>에서 사용되는 우리 개발한 학습 모델은 다음 링크에서 다운로드할 수 있습니다.*

* [Sparse_KoBERT_e10_b64_fold_4.pt](https://drive.google.com/file/d/1-QTGV6Im6NZjIrCm1NYW73u8gfEbGSp3/view?usp=share_link)

* [Sparse_KoBERT_e10_b64_fold_4_state_dict.pt](https://drive.google.com/file/d/1-SuwQgRnYFvGrZYljSvOd3Q--XCQ4HwJ/view?usp=share_link)

* [Sparse_KoBERT_e10_b64_fold_4_all.tar](https://drive.google.com/file/d/1-VHb1PVXqnEVeGqg_bHbdubEsww7zvw0/view?usp=share_link)
