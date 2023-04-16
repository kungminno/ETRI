# data_to_csv.ipynb

*****

    코드 실행 시 사용자 환경에 맞는 데이터셋 경로 설정이 필요합니다.

## main
> 이 코드는 KEMDy19 데이터셋을 이용하여 json 파일과 csv 파일을 생성하는 파이썬 프로그램입니다.

1. KEMDy19 데이터셋의 annotation 폴더에 있는 모든 csv 파일을 읽어들입니다.
2. 각 csv 파일마다 column_extraction 함수를 이용하여 데이터프레임과 인덱스 배열을 추출합니다.
3. 추출된 데이터프레임과 인덱스 배열을 이용하여 train 데이터와 test 데이터로 분할합니다.
4. process_emotion 함수를 이용하여 각 데이터의 감정 정보를 train_final과 test_final 딕셔너리에 저장합니다.
5. 저장된 train_final과 test_final 딕셔너리를 이용하여 train.json과 test.json 파일을 생성합니다.
6. 생성된 train.json과 test.json 파일을 이용하여 train.csv와 test.csv 파일을 생성합니다.


> 각 함수의 역할은 다음과 같습니다.

1. column_extraction 함수는 csv 파일에서 필요한 데이터만 추출하여 데이터프레임과 인덱스 배열을 반환합니다.
2. process_emotion 함수는 추출된 데이터프레임에서 감정 정보를 추출하여 딕셔너리에 저장합니다.
3. save_json 함수는 딕셔너리를 json 파일로 저장합니다.
4. json_to_csv 함수는 json 파일을 csv 파일로 변환합니다.


## column_extraction
> 이 함수는 KEMDy19 데이터셋의 csv 파일에서 필요한 열만 추출하여 데이터프레임과 인덱스 배열을 반환하는 파이썬 함수입니다.

1. csv 파일을 pandas 라이브러리를 이용하여 읽어들입니다.
2. 데이터프레임에서 'Segment ID'와 'Total Evaluation' 열만 추출 후 첫 번째 행은 삭제합니다.
3. 'Segment ID' 열에서 파일 이름과 일치하지 않는 데이터와 'Total Evaluation' 열에서 중복된 데이터를 삭제합니다.
4. 데이터프레임의 인덱스를 재설정하고, 행을 무작위로 섞어 인덱스 배열을 생성합니다.
5. 추출된 데이터프레임과 인덱스 배열을 반환합니다.


## clean_file_content
> 이 함수는 파일 내용에서 불필요한 문자를 제거하고 다중 공백을 제거하여 문자열을 반환하는 파이썬 함수입니다.

1. 파일 내용에서 제거할 불필요한 문자들을 unwanted_chars 변수에 지정합니다.
2. 반복문을 이용하여 각각의 불필요한 문자들을 파일 내용에서 제거합니다.
3. re.sub() 함수를 이용하여 연속된 공백을 하나의 공백으로 변환합니다.
4. 변경된 파일 내용을 반환합니다.

## process_emotion
> 이 함수는 주어진 DataFrame에서 선택한 인덱스의 음성 파일에 대한 정보를 추출하고, 감정 레이블에 따라 사전(dictionary)에 정보를 저장하는 파이썬 함수입니다.

1. DataFrame에서 주어진 인덱스(idx_array)를 이용하여 각각의 텍스트 및 음성 파일에 대한 정보를 추출합니다.
2. 추출한 정보를 이용하여 파일 경로 및 파일 내용을 가져옵니다.
3. 감정 레이블(total_eval)이 emotions 리스트에 있는 경우, 해당 음성 파일에 대한 정보를 dictionary에 저장합니다. 저장되는 정보는 Emotion, Label, WavPath, Text 입니다.
4. 감정 레이블(total_eval)이 emotions 리스트에 없는 경우 해당 파일은 처리되지 않습니다. 
