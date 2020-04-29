import pandas as pd
from scipy import stats

if __name__ == '__main__':
    
    semester = '2017-2'
    weeks = ['Week0', 'Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6', 'Week7', 'Week8']

    input_data_bd1 = pd.read_csv('roc_values/' + semester + '/bd1.csv', index_col=0)
    input_data_bd2 = pd.read_csv('roc_values/' + semester + '/bd2.csv', index_col=0)
    input_data_bd3 = pd.read_csv('roc_values/' + semester + '/bd3.csv', index_col=0)
    input_data_bd4 = pd.read_csv('roc_values/' + semester + '/bd4.csv', index_col=0)
    input_data_bd5 = pd.read_csv('roc_values/' + semester + '/bd5.csv', index_col=0)
    input_data_bd6 = pd.read_csv('roc_values/' + semester + '/bd6.csv', index_col=0)
    input_data_bd7 = pd.read_csv('roc_values/' + semester + '/bd7.csv', index_col=0)
    input_data_bd8 = pd.read_csv('roc_values/' + semester + '/bd8.csv', index_col=0)
    input_data_bd9 = pd.read_csv('roc_values/' + semester + '/bd9.csv', index_col=0)
    input_data_bd10 = pd.read_csv('roc_values/' + semester + '/bd10.csv', index_col=0)
    input_data_bd11 = pd.read_csv('roc_values/' + semester + '/bd11.csv', index_col=0)
    input_data_bd12 = pd.read_csv('roc_values/' + semester + '/bd12.csv', index_col=0)
    input_data_bd13 = pd.read_csv('roc_values/' + semester + '/bd13.csv', index_col=0)

    input_data_bd1 = input_data_bd1[weeks]
    input_data_bd2 = input_data_bd2[weeks]
    input_data_bd3 = input_data_bd3[weeks]
    input_data_bd4 = input_data_bd4[weeks]
    input_data_bd5 = input_data_bd5[weeks]
    input_data_bd6 = input_data_bd6[weeks]
    input_data_bd7 = input_data_bd7[weeks]
    input_data_bd8 = input_data_bd8[weeks]
    input_data_bd9 = input_data_bd9[weeks]
    input_data_bd10 = input_data_bd10[weeks]
    input_data_bd11 = input_data_bd11[weeks]
    input_data_bd12 = input_data_bd12[weeks]
    input_data_bd13 = input_data_bd13[weeks]

    # # 2016-1
    # print(str(stats.mannwhitneyu(input_data_bd12.loc['adaboost'], input_data_bd2.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd12.loc['adaboost'], input_data_bd5.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd12.loc['adaboost'], input_data_bd9.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd12.loc['adaboost'], input_data_bd12.loc['mlp']).pvalue))

    # # 2016-2
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd5.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd9.loc['knn']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd12.loc['knn']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd1.loc['random_forest']).pvalue))

    # # 2017-1
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd5.loc['adaboost']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd6.loc['knn']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd8.loc['knn']).pvalue))
    # print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd10.loc['random_forest']).pvalue))

    # 2017-2
    print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd5.loc['adaboost']).pvalue))
    print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd13.loc['adaboost']).pvalue))
    print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd2.loc['random_forest']).pvalue))
    print(str(stats.mannwhitneyu(input_data_bd2.loc['adaboost'], input_data_bd4.loc['random_forest']).pvalue))