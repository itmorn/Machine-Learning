#coding=utf-8
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
# print type(iris)
# print iris
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(iris.data, iris.target)
print clf.predict([[ 4.9,  3. ,  1.4,  0.2]])
# scores = cross_val_score(clf, iris.data, iris.target)
# print scores.mean()



