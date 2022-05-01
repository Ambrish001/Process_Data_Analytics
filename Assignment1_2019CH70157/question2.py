import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

df1 = pd.read_excel("D:\Download\College_Academics\SEM_6\CLL788\Assignments\Ass_1\q1.xlsx")
X_ini = df1['Population in 10,000\'s']
Y = df1['Profit In Lakhs(Rs)']
X = np.stack([np.ones(Y.size), X_ini], axis=1)


def compute_cost(x, y, theta):
    m = len(y)
    J = 1 / (2 * m) * (np.sum(np.square(np.dot(x, theta.T) - y)))
    return J


def plot_graph(x, y, theta, x_label, y_label, name):
    plt.scatter(x, y, label="Profit in Lakhs", color="green")
    x1 = np.stack([np.ones(Y.size), x], axis=1)
    plt.plot(x, np.dot(x1, theta.T), label=name, color="red")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.show()


def plot_cost(j, name):
    plt.plot(j[10:])
    plt.title(name)
    plt.xlabel("No. of Iterations")
    plt.ylabel("Cost")
    plt.show()


# ----------------- Part a ----------------------------
def batch_descent(x, y, learning_rate):
    m = len(y)
    theta_orig = np.array([-1.0, 1.0])
    theta = theta_orig.copy()
    J_list = []
    error = 100
    while error > 0.000001:
        theta = theta - learning_rate * ((np.dot(x, theta.T) - y).dot(x)) / m
        J_list.append(compute_cost(x, y, theta))
        if len(J_list) >= 2:
            error = np.absolute(J_list[-1] - J_list[-2])
    return theta, J_list


ta = time.time()
Theta_batch, J_batch = batch_descent(X, Y, learning_rate=0.001)
tb = time.time()
print("Theta values for Batch Gradient Descent :",Theta_batch)
print("Intercept value for Batch Gradient Descent :",Theta_batch[0])
print("Slope value for Batch Gradient Descent :",Theta_batch[1])
print("Final Cost : ",J_batch[-1])
print("Time taken for Batch Gradient Descent : ",tb-ta)
plot_cost(J_batch,"Batch Gradient Descent")
print("\n\n\n")
plot_graph(X_ini,Y,Theta_batch,"Population in 10ks","Profit in Lakhs","Batch Gradient Descent")
print("\n\n")


def stochastic_descent(x, y, learning_rate):
    m = len(y)
    theta = np.array([-1.0, 1.0])
    J_list = []
    for i in range(1000):
        for j in range(m):
            theta = theta - learning_rate * ((np.dot(x, theta.T) - y)[j] * (x[j])) / m
            J_list.append(compute_cost(x, y, theta))
    return theta, J_list


tc = time.time()
Theta_stochastic, J_stochastic = stochastic_descent(X, Y, learning_rate=0.001)
td = time.time()
print("Theta values for Stochastic Gradient Descent :",Theta_stochastic)
print("Intercept value for Stochastic Gradient Descent :",Theta_stochastic[0])
print("Slope value for Stochastic Gradient Descent :",Theta_stochastic[1])
print("Final Cost",J_stochastic[-1])
print("Time taken for Stochastic Descent",td-tc)
plot_cost(J_stochastic,"Stochastic Gradient Descent")
print("\n\n\n")
plot_graph(X_ini,Y,Theta_stochastic,"Population in 10ks","Profit in Lakhs","Stochastic Gradient Descent")
print("\n\n")


def random_batch(df,size):
    m = len(df)
    batches = []
    new_df = df.iloc[np.random.permutation(len(df))]
    x_mix = new_df['Population in 10,000\'s']
    y_mix = new_df['Profit In Lakhs(Rs)']
    x_mix = np.stack([np.ones(m), x_mix], axis=1)
    num_batch = m // size

    for i in range(num_batch):
        a = i*size
        b = (i+1)*size
        x_mini = x_mix[a:b]
        y_mini = y_mix[a:b]
        mini_batch = (x_mini, y_mini)
        batches.append(mini_batch)

    if (m % size) != 0:
        a = num_batch*size
        x_mini = x_mix[a:]
        y_mini = y_mix[a:]
        mini_batch = (x_mini, y_mini)
        batches.append(mini_batch)
    return batches


def mini_batch_descent(df, x, y, learning_rate, size):
    theta_orig = np.array([-1.0, 1.0])
    theta = theta_orig.copy()
    J_list = []
    for i in range(10000):
        for batch in random_batch(df, size):
            batch_x = batch[0]
            batch_y = batch[1]
            j = compute_cost(batch_x, batch_y, theta)
            theta = theta - learning_rate * ((np.dot(batch_x, theta.T) - batch_y).dot(batch_x)) / size
            J_list.append(j)
    return theta, J_list

te = time.time()
Theta_mini_batch, J_mini_batch = mini_batch_descent(df1,X,Y,learning_rate=0.001,size=10)
tf = time.time()
print("Theta values for Mini-Batch Gradient Descent :",Theta_mini_batch)
print("Intercept value for Stochastic Gradient Descent :",Theta_mini_batch[0])
print("Slope value for Stochastic Gradient Descent :",Theta_mini_batch[1])
print("Final Cost",J_mini_batch[-1])
print("Time taken for Mini batch Descent",tf-te)
plot_cost(J_mini_batch,"Mini Batch Gradient Descent")
print("\n\n\n")
plot_graph(X_ini,Y,Theta_mini_batch,"Population in 10ks","Profit in Lakhs","Mini Batch Gradient Descent")
print("\n\n")


def least_square_closed(x, y):
    m = len(x)
    theta = np.linalg.inv(np.dot(x.T, x)).dot(np.dot(x.T, y))
    J = 1 / (2 * m) * np.sum(np.square(np.dot(x, theta.T) - y))
    return theta, J

tg = time.time()
Theta_lsc, J_lsc = least_square_closed(X, Y)
th = time.time()
print("Theta values for Least Square Closed Form :",Theta_lsc)
print("Intercept value for Least Square Closed Form :",Theta_lsc[0])
print("Slope value for Least Square Closed Form :",Theta_lsc[1])
print("Final Cost ",J_lsc)
print("Time taken for Least Square Closed",th-tg)
plot_graph(X_ini,Y,Theta_lsc,"Population in 10ks","Profit in Lakhs","Closed Least Square")
print("\n")

# ---------------------- Part b -----------------------------

def lwr(x, y, q_pt, bandw, num_iter, learning_rate):
    m = len(x)
    theta = np.array([-1.0, 1.0])
    w = np.exp(np.square(x - q_pt) / (-2 * (bandw ** 2)))
    J_list = []
    for i in range(num_iter):
        for j in range(m):
            theta = theta - learning_rate * w[j] * ((np.dot(x, theta.T) - y)[j] * (x[j])) / m
            J_list.append(1 / (2 * m) * (np.sum(np.square(np.dot(x, theta.T) - y))))
    return theta, J_list


ti = time.time()
Theta_lwr, J_lwr = lwr(X[:4], Y[:4], 7.576, 0.5, 4, 0.001)
tj = time.time()
print("Theta values for Locally Weighted Linear Regression :",Theta_lwr)
print("Intercept value for Locally Weighted Linear Regression :",Theta_lwr[0])
print("Slope value for Locally Weighted Linear Regression :",Theta_lwr[1])
print("Final Cost ",J_lwr[-1])
print("Time taken for Locally Weighted Linear Regression",tj-ti)
plot_cost(J_lwr,"Locally Weighted Linear Regression")
print("\n\n\n")
plot_graph(X_ini,Y,Theta_lwr,"Population in 10ks","Profit in Lakhs","Locally weighted Linear Regression")
print("\n\n")

# ---------------------- Part c -----------------------------
def plot_line_graph(x,y,inter,slope,xname,yname,title):
    plt.scatter(x, y, label="Profit in Lakhs", color="green")
    b = np.linspace(5,22.5,100)
    a = slope * b + inter
    plt.plot(b,a)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.show()

def ridge_regression(x, y):
    m = len(x)
    ridge_reg = Ridge(alpha=0.5)
    ridge_reg.fit(x, y)
    theta = np.array([ridge_reg.intercept_, ridge_reg.coef_])
    J = 1 / (2 * m) * np.sum(np.square(np.dot(x, theta.T) - y))
    return theta, J

tk = time.time()
Theta_ridge, J_ridge = ridge_regression(X, Y)
tl = time.time()
print("Theta values for Ridge Regression :",Theta_ridge)
print("Intercept value for Ridge Regression :",Theta_ridge[0])
print("Slope value for Ridge Regression :",Theta_ridge[1][1])
print("Final Cost",J_ridge[-1])
print("Time taken for Ridge Regression",tl-tk)
plot_line_graph(X_ini,Y,Theta_ridge[0],Theta_ridge[1][1],"Population in 10ks","Profit in Lakhs","Ridge Regression")
print("\n\n")

def lasso_regression(x, y):
    m = len(x)
    lasso_reg = Lasso(alpha=0.5)
    lasso_reg.fit(x, y)
    theta = np.array([lasso_reg.intercept_, lasso_reg.coef_])
    J = 1 / (2 * m) * np.sum(np.square(np.dot(x, theta.T) - y))
    return theta, J

tm = time.time()
Theta_lasso, J_lasso = lasso_regression(X, Y)
tn = time.time()
print("Theta values for Lasso Regression :",Theta_lasso)
print("Intercept value for Lasso Regression :",Theta_lasso[0])
print("Slope value for Lasso Regression :",Theta_lasso[1][1])
print("Final Cost",J_lasso[-1])
print("Time taken for Lasso Regression",tn-tm)
plot_line_graph(X_ini,Y,Theta_lasso[0],Theta_lasso[1][1],"Population in 10ks","Profit in Lakhs","Lasso Regression")
print("\n\n")



def elastic_net_regression(x, y):
    m = len(x)
    elastic_net_reg = ElasticNet(alpha=0.5)
    elastic_net_reg.fit(x, y)
    theta = np.array([elastic_net_reg.intercept_, elastic_net_reg.coef_])
    J = 1 / (2 * m) * np.sum(np.square(np.dot(x, theta.T) - y))
    return theta, J


to = time.time()
Theta_elastic, J_elastic = elastic_net_regression(X, Y)
tp = time.time()
print("Theta values for Elastic Net Regression :", Theta_elastic)
print("Intercept value for Elastic Net Regression :", Theta_elastic[0])
print("Slope value for Elastic Net Regression :", Theta_elastic[1][1])
print("Final Cost", J_elastic[-1])
print("Time taken for Elastic Net Regression", tp - to)
plot_line_graph(X_ini, Y, Theta_elastic[0],Theta_elastic[1][1], "Population in 10ks", "Profit in Lakhs", "Elastic Net Regression")
