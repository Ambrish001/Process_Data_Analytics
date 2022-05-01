import numpy as np
import pandas as pd
from pprint import pprint

df1 = pd.read_excel("play_tennis.xlsx")
epsilon = 1e-8


def calc_entropy(df):
    grp = df.keys()[-1]
    entropy = 0
    target = df[grp].unique()
    for i in target:
        loss = df[grp].value_counts()[i]
        s = len(df[grp])
        loss /= (s + epsilon)
        entropy -= loss * np.log(loss)
    return entropy


def calc_entropy_name(df, name):
    grp = df.keys()[-1]
    entropy = 0
    values = df[name].unique()
    target = df[grp].unique()
    for i in values:
        answer = 0
        loss_value = 0
        for j in target:
            loss_target = len(df[name][df[name] == i][df[grp] == j])
            loss_value = len(df[name][df[name] == i])
            loss_target /= (loss_value + epsilon)
            answer -= loss_target * np.log(loss_target + epsilon)
        loss_entropy = loss_value / len(df)
        entropy -= loss_entropy * answer
    return np.absolute(entropy)


def find_max(df):
    Info_Gain = []
    for i in df.keys()[:-1]:
        Info_Gain.append(calc_entropy(df) - calc_entropy_name(df, i))
    return df.keys()[:-1][np.argmax(Info_Gain)]


def extract_minitable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def construct_tree(df, tree=None):
    # target = df.keys()[-1]
    node = find_max(df)
    col_value = np.unique(df[node])
    if tree is None:
        tree = {node: {}}
        for i in col_value:
            subtable = extract_minitable(df, node, i)
            col, count = np.unique(subtable['PlayTennis'], return_counts=True)
            if len(count) == 1:
                tree[node][i] = col[0]
            else:
                tree[node][i] = construct_tree(subtable)
    return tree


local_tree = construct_tree(df1)
pprint(local_tree)

# Random value for testing
testing_list = {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': 'Strong'}


def test(lis):
    if lis['Outlook'] == 'Overcast':
        print("Yes")
    elif lis['Outlook'] == 'Rainy':
        if lis['Windy'] == 'Weak':
            print("Yes")
        else:
            print("No")
    elif lis['Outlook'] == 'Sunny':
        if lis['Humidity'] == 'Normal':
            print("Yes")
        else:
            print("No")
    else:
        print("No")

#For printing result of any test on tree, use the line below
#test(testing_list)
