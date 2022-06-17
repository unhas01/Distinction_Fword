# Distinguish_Fword
## 1. 서론
캡스톤 디자인 과목 수강 중 진행했던 프로젝트 파트 중 하나입니다.    

YouTube에서 한국어로 구성된 댓글 중 욕설(Fword)이 포함된 댓글들을 구별하는 AI 모델입니다.  

사용한 Network는 RNN의 알고리즘 중 하나인 LSTM을 사용합니다.   

[LSTM 이론 참고](https://wegonnamakeit.tistory.com/7)

---
## 2. 학습 데이터 (data.csv)
**데이터 수집**

- [스마일게이트 데이터 셋](https://github.com/smilegate-ai/korean_unsmile_dataset)  
- [한국어 욕설 데이터 셋](https://github.com/2runo/Curse-detection-data)  
- YouTube 댓글 크롤링  

| label_word | label|
|---|---|
| `긍정` | 0 |
| `부정` | 1 |
| `욕설` | 2 |    

약 **24500개**의 문장 데이터 (각 label은 **약 8000개씩** 균등하게 분포되어있음)

-------
## 3. Korean_NLP_Distinguish_Fword.ipynb

###### 작업은 Colab서 진행했습니다.
- **전체 데이터를 Train / Test로 분리 (8:2)** 
```python
sentence_train, sentence_test, label_train, label_test = train_test_split(data['Sentence'], data['label'], test_size = 0.2, shuffle = False)  
# data.csv는 Shuffle이 되어있는 파일입니다.
```
- **전처리** 
	- 특수문자 & 이모티콘 제거
	```python
	temp_list=[]

	for i in train_df.Sentence:
  		s = ' '.join(re.compile('[가-힣]+').findall(i))
		temp_list.append(s)
	```
    - 한글자 단어 제거 (그, 너 , 또, ...)
    ```python
    temp_list=[]

	for i in train_df.Sentence:
  		li = []
  		for index in range(len(i.split())):
	    	if len(i.split()[index]) > 1:
            	li.append(i.split()[index])
  
  		s = " ".join(li)
  		temp_list.append(s)
    ```
    - 문장의 토큰 갯수가 5개 이상 25개 이하의 문장만 저장
    ```python
    def word_split(x):
    	return len(x.split())

	train_df['word_count'] = train_df['Sentence'].apply(word_split)
	train_df = train_df[train_df.word_count>=5]
    train_df = train_df[train_df.word_count<=25]
    ```
- **초성, 중성, 종성으로 분리**
	- 대학교 -> ㄷㅐㅎㅏㄱ교
	- 아이폰 -> ㅇㅏㅇㅣㅍㅗㄴ
	```python
    for i in train_df.Sentence:
    	i = j2hcj(h2j(i)) + ' '
  		temp_list.append(i)
    ```
 - **임베딩** 
 	- FastText 사용을 위한 txt파일로 변환
 	```python
    train_df.Sentence.to_csv('data_train.txt')
	test_df.Sentence.to_csv('data_test.txt')
    ```
    - FastText 모델 학습
    
    |parameter| 의미|
    |---|---|
    |input| txt 파일|
    |model| `skipgram` or `CBOW`|
    |lr| learning rate|
    |dim| dimesion|
    |ws| window size|
    |minn| 최소 character|
    |wordNgrams| 1 ~ 6 값 사용 |
    
    ```python
    ft_model = fasttext.train_unsupervised(input='/content/data_train.txt', model = 'skipgram', lr = 0.05, dim = 100, ws = 5, epoch = 50, minn = 1, wordNgrams = 5)
    ```
	
    -  FastText 모델 저장
    ```python
    ft_model.save_model('ft_model.bin')
    ```
    
    - FastText 모델을 사용해서 벡터화
    ```python
    train_vec = []
	sentence_number = 25  

	for sen in tqdm(train_df.Sentence.values):
  		word_list_vec = []
  		sen_split = sen.split()
  		for w_index in range(sentence_number):
    		if w_index < len(sen_split):
      			word_list_vec.append(ft_model[sen_split[w_index]])
    		else:
      			word_list_vec.append(np.array([0]*100))
  		word_list_vec = np.array(word_list_vec)
  		train_vec.append(word_list_vec)
    ```
    
 - **label One-Hot-Encoding**
 	```python
 	y_train = pd.get_dummies(train_df['label']).values
 	```
   
 - **LSTM model**
 	- activation = **`sigmoid`** 사용
 	- loss 함수 = **`categorical_crossentropy`** 사용
 	```python
    model = Sequential()
	model.add(LSTM(units=10, input_shape=(25, 100)))
	model.add(Dense(3, activation='sigmoid'))
	model.compile( optimizer = tf.keras.optimizers.RMSprop( learning_rate = 0.001 ), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    ```
    
    - `model.summary()`   
    ![캡처](https://user-images.githubusercontent.com/87689191/174227385-794d3e4f-865d-4c19-a446-72627725bb94.PNG)
    
    - `fit()`
    ```python
    model.fit(X_train, y_train, epochs = 30, callbacks=tf.keras.callbacks.EarlyStopping('val_loss',patience=5),validation_split=0.2)
    ```
    ![1](https://user-images.githubusercontent.com/87689191/174228601-517df977-9bd1-4c61-9905-fc144f83a252.PNG)
    
    - `evaluate()`
    ```python
    loss, acc = model.evaluate(X_test, y_test)
    ```
    ![2](https://user-images.githubusercontent.com/87689191/174228709-48bb6a3e-d0e7-4041-a839-ce3941c56a42.PNG)
    
    
 
 ---
 ## 4. 결과물
 - **FastText model**  : [ft_model.bin](https://github.com/unhas01/Distinguish_Fword/blob/master/ft_model.bin)
 - **LSTM model** : [Kor_NLP_LSTM.h5](https://github.com/unhas01/Distinguish_Fword/blob/master/Kor_NLP_LSTM.h5)
 
 
---
## 5. 마무리
Train data에 대해서 `accuracy`는 94%가 나오고 Test data에 대해서도 `accuracy`도 91%로 낮지 않은 정확도가 나온다.  
하지만 정확도에 비해 실제 문장을 입력해서 결과값을 테스트 해보면 91%의 정확도처럼 느껴지진 않는다.   

**생각한 문제점**
- 데이터 부족
- label 선정 과정 속에서의 편향  

우선 데이터가 부족하다고 생각한다. 처음 약 24000개에서 8:2로 나누면 19000개로 학습을 하는데 심지어 19000개에서도 위에서 언급된 전처리 과정을 거치면 14000개로 준다.   
그리고 AI허브에서 가져온 말뭉치 대화 데이터 셋은 YouTube에서 흔하게 사용하는 말투, 느낌과 거리가 멀다고 생각한다.  
그래서 YouTube에서 댓글을 크롤링을 통해 데이터를 모았는데 3개의 label의 개수를 비슷하게 맞춰야 할 필요가 있다. 크롤링을 통해 얻은 데이터들은 라벨링(labeling)을 수작업으로 해서 여기서의 문제도 있다고 생각한다.  사람마다의 욕설의 판별 기준이 다르기도 하고 많은 데이터를 수작업으로 라벨링하는 단계에서 실수한 부분도 있을거라 생각한다.    
결과적으로 느낀점은 많은 데이터를 모으는 과정은 쉽지 않다고 느낍니다.

----
### ※ Reference
[단어 임베딩과 LSTM을 활용한 비속어 판별 방법 / 조선대학교 산업기술창업대학원 소프트웨어융합공학과 이명호](https://oak.chosun.ac.kr/bitstream/2020.oak/2036/2/%EB%8B%A8%EC%96%B4%20%EC%9E%84%EB%B2%A0%EB%94%A9%EA%B3%BC%20LSTM%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%B9%84%EC%86%8D%EC%96%B4%20%ED%8C%90%EB%B3%84%20%EB%B0%A9%EB%B2%95.pdf)
