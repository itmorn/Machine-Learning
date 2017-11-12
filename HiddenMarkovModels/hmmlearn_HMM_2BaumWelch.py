#coding=utf-8
import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)
X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1]])
model2.fit(X2)
print model2.startprob_
print
print model2.transmat_
print
print model2.emissionprob_


'''
result:
[  8.67510767e-11   1.00000000e+00   4.41182118e-12] #初始概率分布

[[  1.75337963e-01   6.41371396e-01   1.83290641e-01]  #状态转移概率分布
 [  4.49292494e-01   3.28996871e-05   5.50674606e-01]
 [  1.96457695e-01   5.94861793e-01   2.08680511e-01]]

[[ 0.18339428  0.81660572] #观测概率分布
 [ 0.99896576  0.00103424]
 [ 0.1505637   0.8494363 ]]


'''



