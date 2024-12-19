from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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