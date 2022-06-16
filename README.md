# Distinguish_Fword
### 1. 서론
YouTube에서 한국어로 구성된 댓글 중 욕설(Fword)이 포함된 댓글들을 구별하는 AI 모델입니다.  

사용한 Network는 RNN의 알고리즘 중 하나인 LSTM을 사용합니다.   

[LSTM 참고](https://wegonnamakeit.tistory.com/7)

---
### 2. 학습 데이터 (data.csv)

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
### 3. Korean_NLP_Distinguish_Fword.ipynb

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



### ※ Reference
[단어 임베딩과 LSTM을 활용한 비속어 판별 방법 / 조선대학교 산업기술창업대학원 소프트웨어융합공학과 이명호](https://oak.chosun.ac.kr/bitstream/2020.oak/2036/2/%EB%8B%A8%EC%96%B4%20%EC%9E%84%EB%B2%A0%EB%94%A9%EA%B3%BC%20LSTM%EC%9D%84%20%ED%99%9C%EC%9A%A9%ED%95%9C%20%EB%B9%84%EC%86%8D%EC%96%B4%20%ED%8C%90%EB%B3%84%20%EB%B0%A9%EB%B2%95.pdf)
