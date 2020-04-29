import pandas as pd
from scipy import stats

if __name__ == '__main__':

    semester1 = '2017-2'
    semester2 = '2017-2'

    # semesters = ['2016-1', '2016-2', '2017-1', '2017-2']
    weeks = ['Week0', 'Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6', 'Week7', 'Week8']

    input_data1_bd1 = pd.read_csv('roc_values/' + semester1 + '/bd1_oversample.csv', index_col=0)
    input_data1_bd2 = pd.read_csv('roc_values/' + semester1 + '/bd2_oversample.csv', index_col=0)
    input_data1_bd3 = pd.read_csv('roc_values/' + semester1 + '/bd3_oversample.csv', index_col=0)
    input_data1_bd4 = pd.read_csv('roc_values/' + semester1 + '/bd4_oversample.csv', index_col=0)
    input_data1_bd5 = pd.read_csv('roc_values/' + semester1 + '/bd5_oversample.csv', index_col=0)
    input_data1_bd6 = pd.read_csv('roc_values/' + semester1 + '/bd6_oversample.csv', index_col=0)
    input_data1_bd7 = pd.read_csv('roc_values/' + semester1 + '/bd7_oversample.csv', index_col=0)
    input_data1_bd8 = pd.read_csv('roc_values/' + semester1 + '/bd8_oversample.csv', index_col=0)
    input_data1_bd9 = pd.read_csv('roc_values/' + semester1 + '/bd9_oversample.csv', index_col=0)
    input_data1_bd10 = pd.read_csv('roc_values/' + semester1 + '/bd10_oversample.csv', index_col=0)
    input_data1_bd11 = pd.read_csv('roc_values/' + semester1 + '/bd11_oversample.csv', index_col=0)
    input_data1_bd12 = pd.read_csv('roc_values/' + semester1 + '/bd12_oversample.csv', index_col=0)
    input_data1_bd13 = pd.read_csv('roc_values/' + semester1 + '/bd13_oversample.csv', index_col=0)

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

    input_data2_bd1 = pd.read_csv('roc_values/' + semester2 + '/bd1_quest_data_oversample.csv', index_col=0)
    input_data2_bd2 = pd.read_csv('roc_values/' + semester2 + '/bd2_quest_data_oversample.csv', index_col=0)
    input_data2_bd3 = pd.read_csv('roc_values/' + semester2 + '/bd3_quest_data_oversample.csv', index_col=0)
    input_data2_bd4 = pd.read_csv('roc_values/' + semester2 + '/bd4_quest_data_oversample.csv', index_col=0)
    input_data2_bd5 = pd.read_csv('roc_values/' + semester2 + '/bd5_quest_data_oversample.csv', index_col=0)
    input_data2_bd6 = pd.read_csv('roc_values/' + semester2 + '/bd6_quest_data_oversample.csv', index_col=0)
    input_data2_bd7 = pd.read_csv('roc_values/' + semester2 + '/bd7_quest_data_oversample.csv', index_col=0)
    input_data2_bd8 = pd.read_csv('roc_values/' + semester2 + '/bd8_quest_data_oversample.csv', index_col=0)
    input_data2_bd9 = pd.read_csv('roc_values/' + semester2 + '/bd9_quest_data_oversample.csv', index_col=0)
    input_data2_bd10 = pd.read_csv('roc_values/' + semester2 + '/bd10_quest_data_oversample.csv', index_col=0)
    input_data2_bd11 = pd.read_csv('roc_values/' + semester2 + '/bd11_quest_data_oversample.csv', index_col=0)
    input_data2_bd12 = pd.read_csv('roc_values/' + semester2 + '/bd12_quest_data_oversample.csv', index_col=0)
    input_data2_bd13 = pd.read_csv('roc_values/' + semester2 + '/bd13_quest_data_oversample.csv', index_col=0)

    input_data2_bd1 = input_data2_bd1[weeks]
    input_data2_bd2 = input_data2_bd2[weeks]
    input_data2_bd3 = input_data2_bd3[weeks]
    input_data2_bd4 = input_data2_bd4[weeks]
    input_data2_bd5 = input_data2_bd5[weeks]
    input_data2_bd6 = input_data2_bd6[weeks]
    input_data2_bd7 = input_data2_bd7[weeks]
    input_data2_bd8 = input_data2_bd8[weeks]
    input_data2_bd9 = input_data2_bd9[weeks]
    input_data2_bd10 = input_data2_bd10[weeks]
    input_data2_bd11 = input_data2_bd11[weeks]
    input_data2_bd12 = input_data2_bd12[weeks]
    input_data2_bd13 = input_data2_bd13[weeks]

    print(str(stats.mannwhitneyu(input_data1_bd2.loc['adaboost'], input_data2_bd2.loc['adaboost']).pvalue))
    print(str(stats.mannwhitneyu(input_data1_bd5.loc['adaboost'], input_data2_bd5.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data1_bd12.loc['adaboost'], input_data2_bd9.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data1_bd12.loc['adaboost'], input_data2_bd12.loc['mlp']).pvalue))
