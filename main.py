import json
import re
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. 데이터 로드 및 전처리
with open('News_Category_Dataset_v3.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stopwords = {'the', 'is', 'and', 'in', 'to', 'a'}
    tokens = [word for word in tokens if word not in stopwords]
    return tokens

texts = [preprocess(item['short_description']) for item in data]
labels = [item['category'] for item in data]

vocab = set(word for text in texts for word in text)
word2idx = {word: idx for idx, word in enumerate(vocab)}

def vectorize(text):
    vector = [0] * len(vocab)
    for word in text:
        if word in word2idx:
            vector[word2idx[word]] += 1
    return vector

X = [vectorize(text) for text in texts]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Naive Bayes 구현
class NaiveBayes:
    def __init__(self):
        self.classes = set()
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.vocab = set()
        self.total_words = defaultdict(int)
    
    def train(self, X, y):
        for features, label in zip(X, y):
            self.classes.add(label)
            self.class_counts[label] += 1
            for idx, count in enumerate(features):
                if count > 0:
                    self.class_word_counts[label][idx] += count
                    self.total_words[label] += count
                    self.vocab.add(idx)
        self.total_docs = len(y)
    
    def predict(self, X):
        predictions = []
        for features in X:
            class_scores = {}
            for cls in self.classes:
                log_prob = math.log(self.class_counts[cls] / self.total_docs)
                for idx, count in enumerate(features):
                    if count > 0:
                        word_prob = (self.class_word_counts[cls].get(idx, 0) + 1) / (self.total_words[cls] + len(self.vocab))
                        log_prob += count * math.log(word_prob)
                class_scores[cls] = log_prob
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions

nb = NaiveBayes()
nb.train(X_train, y_train)
nb_pred = nb.predict(X_test)

# 3. 추가 알고리즘 구현
# 결정 트리
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# 신경망
nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)

# 4. 성능 평가
def evaluate(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

models = {
    'Naive Bayes': nb_pred,
    'Decision Tree': dt_pred,
    'SVM': svm_pred,
    'Neural Network': nn_pred
}

results = {}
for name, pred in models.items():
    results[name] = evaluate(y_test, pred, name)

# 시각화
labels = list(results.keys())
accuracy = [results[name][0] for name in labels]
precision = [results[name][1] for name in labels]
recall = [results[name][2] for name in labels]
f1 = [results[name][3] for name in labels]

x = range(len(labels))
width = 0.2

plt.figure(figsize=(12, 8))
plt.bar(x, accuracy, width, label='Accuracy')
plt.bar([p + width for p in x], precision, width, label='Precision')
plt.bar([p + width*2 for p in x], recall, width, label='Recall')
plt.bar([p + width*3 for p in x], f1, width, label='F1-Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks([p + width*1.5 for p in x], labels)
plt.legend()
plt.show() 