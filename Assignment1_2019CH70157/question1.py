import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

df1 = pd.read_excel("data_1.xlsx")
df3 = pd.read_excel("data_3.xlsx")
x_df1 = df1['x']
y_df1 = df1['y']
x_df3 = df3['x']
y_df3 = df3['y']


def make_histogram(xaxis, xname, title):
    plt.hist(xaxis, bins=20, edgecolor='black')
    plt.xlabel(xname)
    plt.title(title)
    plt.show()


def joint_histogram(df, title):
    sns.histplot(df)
    plt.title(title)
    plt.show()


def make_scatter_plot(x_axis, y_axis, xname, yname, title):
    plt.scatter(x_axis, y_axis, s=50, edgecolor='black', linewidth=1)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.show()


def make_heatmap(df_name, title):
    sns.heatmap(df_name)
    plt.title(title)
    plt.show()


def hey_heat(df_name):
    fig = px.density_heatmap(df_name, x="x", y="y")
    fig.show()


def make_boxplot(df_name, title):
    plt.boxplot(df_name)
    plt.title(title)
    plt.show()


# -------------------------Part 1------------------------------


make_histogram(df1['x'], "X_data_1", "Histogram1_Data_1")
make_histogram(df1['y'], "Y_data_1", "Histogram2_Data_1")
joint_histogram(df1, "Data_1")
make_scatter_plot(df1['x'], df1['y'], "X_data_1", "Y_data_1", "Scatter_Data_1")
make_heatmap(df1, "HeatMap_Data_1")
make_boxplot(df1, "BoxPlot_Data_1")
hey_heat(df1)
print("\n\n--------------------------x-------------------------------\n\n")
# -------------------------Part 2-------------------------------


make_histogram(df3['x'], "X_data_3", "Histogram1_Data_3")
make_histogram(df3['y'], "Y_data_3", "Histogram2_Data_3")
joint_histogram(df3, "Data_3")
make_scatter_plot(df3['x'], df3['y'], "X_data_3", "Y_data_3", "Scatter_Data_3")
make_heatmap(df3, "HeatMap_Data_3")
make_boxplot(df3, "BoxPlot_Data_3")
hey_heat(df3)
print("\n\n--------------------------x-------------------------------\n\n")

# ----------------------------Part 3---------------------------

print(df1.describe())
print(df3.describe())
print("\n\n--------------------------x-------------------------------\n\n")

# ----------------------------Part 4---------------------------

x_a = df3['x'].mean()
sdx = df3['x'].std()

y_a = df3['y'].mean()
sdy = df3['y'].std()

x_m = df3['x'].median()
x_mad = df3['x'].mad()
y_m = df3['y'].median()
y_mad = df3['y'].mad()

std_outlier_x = [a for a in df3['x'] if ((a > x_a + 3 * sdx) or (a < x_a - 3 * sdx))]
std_outlier_y = [a for a in df3['y'] if ((a > y_a + 3 * sdy) or (a < y_a - 3 * sdy))]

mad_outlier_x = [a for a in df3['x'] if ((0.6745 * np.absolute(a - x_m) / x_mad) > 3)]
mad_outlier_y = [a for a in df3['y'] if ((0.6745 * np.absolute(a - y_m) / y_mad) > 3)]

print("Outliers detected through STD approach:")
for i in range(len(df3['x'])):
    if df3['x'][i] in std_outlier_x or df3['y'][i] in std_outlier_y:
        print("Index : ", i, " x: ", df3['x'][i], " y: ", df3['y'][i])

print("Outliers detected through MAD approach:")
for i in range(len(df3['x'])):
    if df3['x'][i] in mad_outlier_x or df3['y'][i] in mad_outlier_y:
        print("Index : ", i, " x: ", df3['x'][i], " y: ", df3['y'][i])
