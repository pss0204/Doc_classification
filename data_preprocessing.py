import json
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 데이터 로드
with open('News_Category_Dataset_v3_balanced.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 전처리 함수
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    # 불용어 제거 (간단하게 몇 개만 예시)
    stopwords = {'the', 'is', 'and', 'in', 'to', 'a'}
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

# 데이터 전처리
texts = [preprocess(item['short_description']) for item in data]
labels = [item['category'] for item in data]

# 단어 집합 생성
vocab = set(word for text in texts for word in text)
word2idx = {word: idx for idx, word in enumerate(vocab)}

# 데이터 벡터화 (Bag of Words)
def vectorize(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in word2idx:
            vector[word2idx[word]] += 1
    return vector

X = [vectorize(text) for text in texts]
y = labels

# 학습용, 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 