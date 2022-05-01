import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

train_data = pd.read_excel("q2train.xlsx")
test_data = pd.read_excel("q2test.xlsx")

x_train = train_data[['Aptitude', 'Verbal']]
y_train = train_data[['Label']]
x_test = test_data[['Aptitude', 'Verbal']]
X_tr = np.column_stack([np.ones(shape=(70, 1)), x_train])
X_te = np.column_stack([np.ones(shape=(30, 1)), x_test])
theta = np.array([[0.1, 0.1, 0.1]])
error = 100
J_list = [10, 20]  # Initializing with random values
alpha = 0.001
m = train_data.shape[0]
epsilon = 1e-8


def sigmo(x):
    return 1 / (1 + np.exp(-x))


def compute_cost(x, y):
    J = -(y * np.log(epsilon + sigmo(x)) + (1 - y) * np.log(1 - sigmo(x) + epsilon)).mean()
    return J[0]


while error > 1e-7:
    j = compute_cost(np.dot(X_tr, theta.T), y_train)
    J_list.append(j)
    error = np.absolute(J_list[-1] - J_list[-2])
    theta = theta - (alpha / m) * (np.dot((sigmo(np.dot(X_tr, theta.T)) - y_train).T, X_tr))

J_list.pop(0)
J_list.pop(1)
print(theta)


y_test = np.dot(X_te, theta.T)
result = sigmo(y_test)

for i in range(X_te.shape[0]):
    if result[i] >= 0.7:
        result[i] = 1
    else:
        result[i] = 0

np.savetxt("output1.txt", result)
plt.plot(J_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

plt.figure()
plt.title("Result")
plt.xlabel("Aptitude")
plt.ylabel("Verbal")
dfa = pd.DataFrame(np.hstack((x_test,result)))
sns.scatterplot(data = dfa, x = dfa.iloc[:,0], y = dfa.iloc[:,1],hue = 2)
plt.show()