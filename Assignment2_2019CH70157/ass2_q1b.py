import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("q1train.xlsx")
X = df[['Aptitude', 'Verbal']]
Y = df[['Label']]

y_train = Y.to_numpy(copy = False)
x_train = X.to_numpy(copy = False)
#x_train = np.column_stack([np.ones(shape=(70, 1)), x_train])

input_size = 2
hidden_size = 3
output_size = 1

def sig(x):
    return 1/(1+np.exp(-x))

def sig_grad(x):
    a = 1/(1+np.exp(-x))
    a = a*(1-a)
    return a

w_a = np.array([[0.12,0.23,0.34],[0.45,0.56,0.67],[0.78,0.81,0.94]])
w_b = np.array([[0.21,0.32,0.43,0.54]])
#w_a = np.ones((3,3))
#w_b = np.ones((1,4))

parameters = np.concatenate([w_a.ravel(), w_b.ravel()])

def cost(param,in_size,hid_size,out_size,x,y):
    #Reshape param back
    w1 = np.reshape(param[:hid_size*(in_size+1)],(hid_size,(in_size+1)))
    w2 = np.reshape(param[hid_size*(in_size+1):],(out_size,(hid_size+1)))
    m = y.size
    J = 0
    w1_derivative = np.zeros(w1.shape)
    w2_derivative = np.zeros(w2.shape)

    a1 = np.concatenate([np.ones((m,1)),x],axis=1)
    a2 = sig(a1.dot(w1.T))
    a2 = np.concatenate([np.ones((a2.shape[0],1)),a2],axis=1)
    a3 = sig(a2.dot(w2.T))

    y_mat = y.reshape(-1)
    y_mat = np.eye(out_size)[y_mat]
    tempa = w1
    tempb = w2
    J = (-1/m)*np.sum((np.log(a3) * y_mat) + np.log(1 - a3) * (1 - y_mat))
    del_3 = a3 - y_mat
    del_2 = del_3.dot(w2)[:, 1:] * sig_grad(a1.dot(w1.T))
    Delta1 = del_2.T.dot(a1)
    Delta2 = del_3.T.dot(a2)

    w1_derivative = (1 / m) * Delta1
    w1_derivative[:, 1:] = w1_derivative[:, 1:]
    w2_derivative = (1 / m) * Delta2
    w2_derivative[:, 1:] = w2_derivative[:, 1:]
    grad = np.concatenate([w1_derivative.ravel(), w2_derivative.ravel()])
    return J, grad

def predict(w1,w2,x):
    out_size = w2.shape[0]
    p = np.zeros((x.shape[0],1))
    #x = np.concatenate([np.ones((out_size,1)),x], axis = 1)
    x = np.column_stack([np.ones(shape=(70, 1)), x])
    a2 = sig(x.dot(w1.T))
    a2 = np.concatenate([np.ones((a2.shape[0],1)),a2], axis = 1)
    #p = np.argmax(sig(a2.dot(w2.T)),axis=1)
    p = sig(a2.dot(w2.T))
    return p

p = predict(w_a,w_b,x_train)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(p == y_train) * 100))
print(p)
print(p.shape)