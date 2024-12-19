import math
from collections import defaultdict

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
                # 로그 확률 사용
                log_prob = math.log(self.class_counts[cls] / self.total_docs)
                for idx, count in enumerate(features):
                    if count > 0:
                        # 단어가 클래스에 등장한 확률 + 라플라스 스무딩
                        word_prob = (self.class_word_counts[cls].get(idx, 0) + 1) / (self.total_words[cls] + len(self.vocab))
                        log_prob += count * math.log(word_prob)
                class_scores[cls] = log_prob
            # 가장 높은 확률의 클래스 선택
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        return predictions 