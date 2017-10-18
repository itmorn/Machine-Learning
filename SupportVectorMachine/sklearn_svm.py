#coding=utf-8
import numpy as np
from numpy import *
from sklearn.svm import SVC

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

X ,y = loadImages('trainingDigits')
clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0)
print clf.fit(X, y) 
'''
C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，
这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
  　　0 – 线性：u'v
 　　 1 – 多项式：(gamma*u'*v + coef0)^degree
  　　2 – RBF函数：exp(-gamma|u-v|^2)
  　　3 –sigmoid：tanh(gamma*u'*v + coef0)
degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability ：是否采用概率估计？.默认为False
shrinking ：是否采用shrinking heuristic方法，默认为true
tol ：停止训练的误差值大小，默认为1e-3
cache_size ：核函数cache缓存大小，默认为200
class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)
verbose ：允许冗余输出？
max_iter ：最大迭代次数。-1为无限制。
decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3
random_state ：数据洗牌时的种子值，int值
主要调节的参数有：C、kernel、degree、gamma、coef0。
'''

#训练
rightCount = 0
for i in range(len(X)):
    if clf.predict([X[i]]) == [y[i]] :
        rightCount += 1
print "测试正确率: " + str((1.0*rightCount/len(X))*100) + "%"


#测试
X, y = loadImages('testDigits')
rightCount = 0
for i in range(len(X)):
    if clf.predict([X[i]]) == [y[i]] :
        rightCount += 1
print "预测正确率: " + str((1.0*rightCount/len(X))*100) + "%"



'''
结果：

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
测试正确率: 99.0049751244%
预测正确率: 98.9247311828%

'''








