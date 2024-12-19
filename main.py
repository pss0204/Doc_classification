import json
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = []
with open("News_Category_Dataset_v3.json", "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# Inspect the dataset
print("Dataset Head:")
print(df.head())

# Ensure necessary columns exist
if not all(col in df.columns for col in ["category", "headline"]):
    raise ValueError("Dataset must contain 'category' and 'headline' columns.")

# Data Preprocessing
df = df.dropna(subset=["category", "headline"])
df["headline"] = df["headline"].str.lower()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["headline"], df["category"], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes Implementation
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocab = set()

    def train(self, X, y):
        class_counts = defaultdict(int)
        word_counts = defaultdict(lambda: defaultdict(int))

        for text, label in zip(X, y):
            class_counts[label] += 1
            for word in text.split():
                self.vocab.add(word)
                word_counts[label][word] += 1

        total_docs = len(X)
        self.class_probs = {cls: count / total_docs for cls, count in class_counts.items()}

        for cls in class_counts:
            total_words = sum(word_counts[cls].values())
            for word in self.vocab:
                self.word_probs[cls][word] = (
                    (word_counts[cls][word] + 1) / (total_words + len(self.vocab))
                )

    def predict(self, X):
        predictions = []
        for text in X:
            class_scores = {}
            for cls in self.class_probs:
                class_scores[cls] = math.log(self.class_probs[cls])
                for word in text.split():
                    if word in self.vocab:
                        class_scores[cls] += math.log(self.word_probs[cls][word])

            predictions.append(max(class_scores, key=class_scores.get))
        return predictions

# Initialize Naive Bayes classifier
nb_classifier = NaiveBayesClassifier()

# Initialize classifiers
classifiers = {
    'Naive Bayes': nb_classifier,
    'SVM': LinearSVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Train and evaluate each classifier
results = {}
for name, clf in classifiers.items():
    if name == 'Naive Bayes':
        clf.train(X_train, y_train)
        y_pred = clf.predict(X_test)
    else:
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'report': report
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Detailed Report:")
    print(report)
