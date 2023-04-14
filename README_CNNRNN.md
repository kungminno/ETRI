CNNRNN 사용법
=================


### CNNRNN 주의 사항

 - <span style='background-color:#fff5b1'>파일 경로 지정 필요</span>
 - Used library:  	
    -   python==3.9
    -   tensorflow==2.9.1
    -   keras==2.9.0
    -   librosa==0.9.2
    -   matplot==0.1.9
    -   matplotlib==3.5.2
    -   numpy==1.22.4
    -   opencv-python==4.6.0.66
    -   pandas==1.3.5
    -   scikit-learn==1.0.2
    -   scipy==1.7.3
    -   tqdm==4.64.0

### data_CNNRNN: CNNRNN 데이터의 정제/통합/정리/변환 등 데이터 전처리 과정 및 결과 파일
 
 1. data_CNNRNN의 흐름:
    - 각 annotation폴더에 있는 Session.csv데이터를 불러와 각 Session별로 8:2 비율로 train과 test로 나누고 json 파일로 저장 ('train.json', 'test.json')
    - 위에 json 파일을 불러와 오디오 데이터 전처리에 편하게 수정 후 다시 저장 ('train_audio.json', 'test_audio.json')
    - 음성 데이터에서 mel spectrogram 추출 후, 각 spectrogram의 이름/저장위치/감정label 정보를 담은 파일로 저장 ('data_train.pickle', 'data_test.pickle')
    - 위의 데이터에서 spectrogram을 읽어 array 형식으로 데이터 삽입 후 파일 저장 ('train_final.pickle', 'test_final.pickle')


 2. data_CNNRNN에서의 파일 설명: 
	- train, test			: 각 Session 별 8:2 비율로 저장한 데이터(.json)
	- train_audio, test_audio	: train_audio, test_audio 파일을 전처리 하기 편하게 변환시킨 데이터(.json)
	- data_train, data_test		: train_audio, test_audio 파일을 데이터프레임 형식으로 변환 후, mel spectrogram 추출. 해당 spectrogram의 정보(파일명(.png), 저장 위치, label)를 담은 딕셔너리 형식 파일(.pickle)
        - ex
		> data_train['id'][0] = 'Sess01_impro04_F013'  
    	>data_train['path'][0] = './Sess01_impro04_F013.png'  
    	>data_train['emotion'][0] = 1  

	- train_final, test_final	: data_train, data_test에서 ['path']에 존재하는 이미지를 불러와 딕셔너리 값으로 변경한 딕셔너리 형식 파일(.pickle)
        - ex
		> train_final['id'][0] = 'Sess01_impro04_F013'  
    	> train_final['path'][0] = [[[  1,   0,   0],[  2,   0,   0], [  2,   0,   0]...,[  2,   0,   0],[  2,   0,   0],[  2,   0,   0]] ...,   
   	 	> train_final['emotion'][0] = 1  


### main_CNNRNN: CNNRNN 모델 학습 파일
 1. main_CNNRNN의 흐름:
	- ResNet50 + LSTM bulid model 함수 선언
	- Train dataset load
	- 5-fold cross-validation 적용해서 모델 학습(이 모델을 다 돌려보지 않아도 기존의 완성된 모델 load해서 사용 가능)
	- Test dataset load
	- 3에서 돌린 모델 or 미리 완성해 load한 모델을 사용해 test dataset predict

CNNRNN parameter values 
> Learning Rate		: 1e-4  
> Batch size			: 16    
> Epoch				: 60
> Optimization Function	: Adam(beta_1=0.9, beta_2=0.999)    


 2. main_CNNRNN에서의 파일 설명:
	- train_final, test_final	: CNNRNN 모델 학습에 사용될 train dataset과 test dataset(.pickle)
	- CNN_RNN_fold_epoch60_e4	: 완성된 CNNRNN 모델 파일(.h5)

