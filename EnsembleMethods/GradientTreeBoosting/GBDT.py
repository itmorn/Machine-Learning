# coding=utf-8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

iris = load_iris()

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                 max_depth=1, random_state=0).fit(iris.data, iris.target)
                                 
print clf.predict([[ 4.9,  3. ,  1.4,  0.2]])          
