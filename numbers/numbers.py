#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

f = np.genfromtxt("mnist_train.csv", delimiter=',')

##### PREPROCESSING ######
# 60kR x 1C labels
correct_answer = f[:,0:1]
correct_answer = np.squeeze(correct_answer)
# 60kR x 785C col1 bias unit = 1; col2-785 = data; rows shuffled
x = f[:,1:] / 255 #scale data
x = np.c_[np.ones(60000),x] #add bias unit
np.random.shuffle(x) #shuffles rows only
# 10R x 785C one row per perceptron. data range -.05-.05
w_original = np.random.random_sample((10,785))
w_original = (w_original - .5) / 10 
# 50R x 3C each col per learning rate
accuracy = np.empty([50,3])

y = np.ndarray([60000,10])
t = np.ndarray([60000,10])

#learn rate = 10
n = .1
#for training learning rate = .1 .001 .00001.. j 0 to 2
 #learning_rate /= 100
 
w = np.ndarray([10,785])
w = np.copy(w_original)
 
for epoch in range (0, 50):
  
  ##### ACTIVATE #####
  # 60kR x 10C each col is perceptron output. 
  for p in range (0, 10):
    y[:,p:(p+1)] = np.sum(x * w[p:(p+1),:], axis=1)[..., np.newaxis]
  #record accuracy statistics
  highest_output =  np.argmax(y, axis=1)
  check_correct = highest_output == correct_answer
  accuracy[epoch,0] = check_correct.sum() / 60000
  #finish activation
  y[y > 0] = 1
  y[y <= 0] = 0
  ##### LEARN #####
  #calculate t for use in gradient descent
  c = correct_answer[..., np.newaxis]
  for p in range(0,10):
    t[:,p:(p+1)] = np.where(np.logical_and(c == p, y[:,p:(p+1)] == 1), 1, t[:,p:(p+1)])
    t[:,p:(p+1)] = np.where(np.logical_and(c != p, y[:,p:(p+1)] == 1), 0, t[:,p:(p+1)])
    t[:,p:(p+1)] = np.where(np.logical_and(c == p, y[:,p:(p+1)] == 0), 0, t[:,p:(p+1)])
    t[:,p:(p+1)] = np.where(np.logical_and(c != p, y[:,p:(p+1)] == 0), 1, t[:,p:(p+1)])
  #apply the algorithm!
  for p in range(0,10):
    w[p:(p+1),:] = w[p:(p+1),:] + np.sum(n*(t[:,p:(p+1)]-y[:,p:(p+1)])*x, axis=0)

#file_arr = np.genfromtxt("mnist_validation.csv", delimiter=',')

plt.plot(accuracy)
