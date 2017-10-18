

import svmMLiA
from numpy import *

# dataArr,labelArr = svmMLiA.loadDataSet('testSet.txt')
# 
# print labelArr
# 
# # b,alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
# b,alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
# 
# print alphas
# print b
# 
# shape(alphas[alphas>0])
# 
# for i in range(100):
#     if alphas[i]>0.0: print dataArr[i],labelArr[i]
# 
# ws=svmMLiA.calcWs(alphas,dataArr,labelArr)
# print ws
# 
# datMat=mat(dataArr)
# 
# print datMat[0]*mat(ws)+b
# print labelArr[0]
# 
# print datMat[1]*mat(ws)+b
# print labelArr[1]
# 
# print datMat[2]*mat(ws)+b
# print labelArr[2]


svmMLiA.testDigits(('rbf', 20))
















