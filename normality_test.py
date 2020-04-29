import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
import pylab
# from matplotlib import pyplot as plt

if __name__ == '__main__':

    semester1 = '2016-1'

    weeks = ['Week0', 'Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6', 'Week7', 'Week8']

    input_data1_bd1 = pd.read_csv('roc_values/' + semester1 + '/bd1.csv', index_col=0)
    input_data1_bd2 = pd.read_csv('roc_values/' + semester1 + '/bd2.csv', index_col=0)
    input_data1_bd3 = pd.read_csv('roc_values/' + semester1 + '/bd3.csv', index_col=0)
    input_data1_bd4 = pd.read_csv('roc_values/' + semester1 + '/bd4.csv', index_col=0)
    input_data1_bd5 = pd.read_csv('roc_values/' + semester1 + '/bd5.csv', index_col=0)
    input_data1_bd6 = pd.read_csv('roc_values/' + semester1 + '/bd6.csv', index_col=0)
    input_data1_bd7 = pd.read_csv('roc_values/' + semester1 + '/bd7.csv', index_col=0)
    input_data1_bd8 = pd.read_csv('roc_values/' + semester1 + '/bd8.csv', index_col=0)
    input_data1_bd9 = pd.read_csv('roc_values/' + semester1 + '/bd9.csv', index_col=0)
    input_data1_bd10 = pd.read_csv('roc_values/' + semester1 + '/bd10.csv', index_col=0)
    input_data1_bd11 = pd.read_csv('roc_values/' + semester1 + '/bd11.csv', index_col=0)
    input_data1_bd12 = pd.read_csv('roc_values/' + semester1 + '/bd12.csv', index_col=0)
    input_data1_bd13 = pd.read_csv('roc_values/' + semester1 + '/bd13.csv', index_col=0)

    input_data1_bd1 = input_data1_bd1[weeks]
    input_data1_bd2 = input_data1_bd2[weeks]
    input_data1_bd3 = input_data1_bd3[weeks]
    input_data1_bd4 = input_data1_bd4[weeks]
    input_data1_bd5 = input_data1_bd5[weeks]
    input_data1_bd6 = input_data1_bd6[weeks]
    input_data1_bd7 = input_data1_bd7[weeks]
    input_data1_bd8 = input_data1_bd8[weeks]
    input_data1_bd9 = input_data1_bd9[weeks]
    input_data1_bd10 = input_data1_bd10[weeks]
    input_data1_bd11 = input_data1_bd11[weeks]
    input_data1_bd12 = input_data1_bd12[weeks]
    input_data1_bd13 = input_data1_bd13[weeks]

    # 2016-1
    print("\n" + str(stats.shapiro(input_data1_bd12.loc['adaboost'])[1]))
    print(str(stats.shapiro(input_data1_bd2.loc['adaboost'])[1]))
    print(str(stats.shapiro(input_data1_bd5.loc['adaboost'])[1]))
    print(str(stats.shapiro(input_data1_bd9.loc['adaboost'])[1]))
    print(str(stats.shapiro(input_data1_bd12.loc['mlp'])[1]))

    print(str(ks_2samp(input_data1_bd12.loc['adaboost'], input_data1_bd2.loc['adaboost'])))
    print(str(ks_2samp(input_data1_bd12.loc['adaboost'], input_data1_bd5.loc['adaboost'])))
    print(str(ks_2samp(input_data1_bd12.loc['adaboost'], input_data1_bd9.loc['adaboost'])))
    print(str(ks_2samp(input_data1_bd12.loc['adaboost'], input_data1_bd12.loc['mlp'])))

    stats.probplot(input_data1_bd12.loc['mlp'], dist="norm", plot=pylab)
    pylab.show()


    # # 2016-2
    # print("\n" + str(stats.shapiro(input_data1_bd2.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd5.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd9.loc['knn'])[1]))
    # print(str(stats.shapiro(input_data1_bd12.loc['knn'])[1]))
    # print(str(stats.shapiro(input_data1_bd1.loc['random_forest'])[1]))

    # # 2017-1
    # print("\n" + str(stats.shapiro(input_data1_bd2.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd5.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd6.loc['knn'])[1]))
    # print(str(stats.shapiro(input_data1_bd8.loc['knn'])[1]))
    # print(str(stats.shapiro(input_data1_bd10.loc['random_forest'])[1]))

    # # 2017-2
    # print("\n" + str(stats.shapiro(input_data1_bd2.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd5.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd13.loc['adaboost'])[1]))
    # print(str(stats.shapiro(input_data1_bd8.loc['random_forest'])[1]))
    # print(str(stats.shapiro(input_data1_bd10.loc['random_forest'])[1]))