#coding=utf-8
import numpy as np
from hmmlearn import hmm
import math
states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

ball_picked = [0,1,0]
seen = np.array([ball_picked]).T

states_predict = model.predict(seen)

a = model.score(seen) #score函数返回的是以自然对数为底的对数概率值
print math.exp(a)





