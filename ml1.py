  import numpy as np
import pandas as pd

global dataset
global set_size
global train_size
global test_size
global intercept
global lmbda
global learn_rate
global iterations
lmbda = 0.01
dataset = pd.read_csv('datafile.csv', sep=';', decimal=',')
dataset = np.array(dataset)

set_size = dataset.shape
train_size = set_size[0] * 0.8
train_size = int(train_size)

test_size = set_size[0] - train_size
intercept = np.ones(set_size[0]).reshape(-1, 1)
dataset = np.hstack((intercept, dataset))

np.random.shuffle(dataset)

training_X = dataset[:train_size, :-1]
training_Y = dataset[:train_size, -1].reshape(-1, 1)
test_X = dataset[train_size:, :-1]
test_Y = dataset[train_size:, -1].reshape(-1, 1)

# gradient descent solution


def MSE(y_calc, y_known):
    return np.sum((y_calc - y_known) * (y_calc - y_known)) / len(y_calc)


def LossFnGD(weight_arr, X, y):
    y_old = np.matmul(X, weight_arr)
    loss = np.sum((y_old - y) * (y_old - y)) / (test_size * 2)
    L2 = lmbda * np.dot(weight_arr, np.transpose(weight_arr)) / (2 * test_size)
    return loss + L2


def Gradient(weight_arr, X, y):
    y_old = np.matmul(X, weight_arr)
    grad = (np.transpose(X).dot(y_old - y)) / test_size
    L2grad = lmbda * weight_arr / test_size
    return grad


learn_rate = 0.00005
iterations = 200000
weight = np.random.normal(0, 1, (test_X.shape[1], 1))

for iteration in range(iterations):
    weight = weight - learn_rate * Gradient(weight, training_X, training_Y)
    if iteration % 10000 == 0:
        print('iteration', iteration)
        y_iter = np.dot(test_X, weight)
        print('MSE for iteration is', MSE(y_iter, test_Y))

y_final = np.dot(test_X, weight)
print('MSE for gradient descent after 2000000 iterations is', MSE(y_final, test_Y))

# closed form solution
A = np.transpose(training_X).dot(training_X) + lmbda * np.eye((training_X.shape[1]))
B = np.transpose(training_X).dot(training_Y)
weight = np.linalg.pinv(A).dot(B)
y_final = test_X.dot(weight)
y_training = training_X.dot(weight)
print('MSE for Closed form is', MSE(y_final, test_Y))

# When we solve the mle equation the formula that we get is same as that of the close form expression
# Hence, the code would be the same as in the closed form solution, and hence the same error
