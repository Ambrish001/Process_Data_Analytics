import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("q1train.xlsx")
X = df[['Aptitude', 'Verbal']]
Y = df[['Label']]

w = np.array([[0.1,0.1,0.1]])
y_tr = Y.to_numpy(copy = False)
x_tr = X.to_numpy(copy = False)
x_tr = np.column_stack([np.ones(shape=(70, 1)), x_tr])

f = sns.pairplot(x_vars=["Aptitude"], y_vars=["Verbal"], data=df, hue="Label", height=5)
f.fig.suptitle("Training Data")
X = np.array(df[['Aptitude','Verbal']].copy()).reshape(-1,2)
y = np.array(df[['Label']].copy()).reshape(-1,1)
y = np.where(y==0,-1,1)
plt.show()

for i in range(y_tr.shape[0]):
    if y_tr[i][0] == 0:
        y_tr[i][0] = -1

num_epochs = 500
for i in range(num_epochs):
    for j in range(70):
        if (np.dot(x_tr[j],w.T)>0 and y[j][0]<0) or (np.dot(x_tr[j],w.T)<0 and y[j][0]>0):
            w = w + 0.01*x_tr[j]*y[j][0]

print(w)

q1test = pd.read_excel("q1test.xlsx")
X = np.array(q1test[['Aptitude','Verbal']].copy()).reshape(-1,2)
X = np.column_stack([np.ones(shape=(30, 1)), X])
y_n = np.dot(X,w.T)
y_n = np.where(y_n >= 9, 1,0)
y_n = pd.DataFrame(y_n)
result = pd.DataFrame(np.hstack((X,y_n)))
result.to_excel("Result_1a.xlsx")
line_x = np.linspace(0,100,100)
line_y = (9 + 4.9 - 0.205*line_x)/0.032
result.columns = ['Bias','Aptitude', 'Verbal', 'Label']
g = sns.pairplot(x_vars=["Aptitude"], y_vars=["Verbal"], hue="Label",data=result, height=5)
plt.plot(line_x,line_y,'-r',label = "Sep Line")
g.fig.suptitle("Test Data")
plt.show()