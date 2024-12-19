from naive_bayes import NaiveBayes
from data_preprocessing import X_train, X_test, y_train, y_test
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Naive Bayes 예측
nb = NaiveBayes()
nb.train(X_train, y_train)
nb_pred = nb.predict(X_test)

# 평가 함수
def evaluate(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# 각 모델 평가
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

plt.figure(figsize=(10,6))
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