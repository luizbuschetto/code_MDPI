import pandas as pd
from scipy import stats

if __name__ == '__main__':
    
    semesters = ['2016-1', '2016-2', '2017-1', '2017-2']
    weeks = ['Week0', 'Week1', 'Week2', 'Week3', 'Week4', 'Week5', 'Week6', 'Week7', 'Week8']

    results = pd.DataFrame(columns=['semester', 'bd', 'classifier', 'avg', 'median'])

    for semester in semesters:
        input_data_bd1 = pd.read_csv('roc_values/' + semester + '/bd1_oversample.csv', index_col=0)
        input_data_bd2 = pd.read_csv('roc_values/' + semester + '/bd2_oversample.csv', index_col=0)
        input_data_bd3 = pd.read_csv('roc_values/' + semester + '/bd3_oversample.csv', index_col=0)
        input_data_bd4 = pd.read_csv('roc_values/' + semester + '/bd4_oversample.csv', index_col=0)
        input_data_bd5 = pd.read_csv('roc_values/' + semester + '/bd5_oversample.csv', index_col=0)
        input_data_bd6 = pd.read_csv('roc_values/' + semester + '/bd6_oversample.csv', index_col=0)
        input_data_bd7 = pd.read_csv('roc_values/' + semester + '/bd7_oversample.csv', index_col=0)
        input_data_bd8 = pd.read_csv('roc_values/' + semester + '/bd8_oversample.csv', index_col=0)
        input_data_bd9 = pd.read_csv('roc_values/' + semester + '/bd9_oversample.csv', index_col=0)
        input_data_bd10 = pd.read_csv('roc_values/' + semester + '/bd10_oversample.csv', index_col=0)
        input_data_bd11 = pd.read_csv('roc_values/' + semester + '/bd11_oversample.csv', index_col=0)
        input_data_bd12 = pd.read_csv('roc_values/' + semester + '/bd12_oversample.csv', index_col=0)
        input_data_bd13 = pd.read_csv('roc_values/' + semester + '/bd13_oversample.csv', index_col=0)

        # input_data_bd1 = pd.read_csv('roc_values/' + semester + '/bd1_oversample.csv', index_col=0)
        # input_data_bd2 = pd.read_csv('roc_values/' + semester + '/bd2_oversample.csv', index_col=0)
        # input_data_bd3 = pd.read_csv('roc_values/' + semester + '/bd3_oversample.csv', index_col=0)
        # input_data_bd4 = pd.read_csv('roc_values/' + semester + '/bd4_oversample.csv', index_col=0)
        # input_data_bd5 = pd.read_csv('roc_values/' + semester + '/bd5_oversample.csv', index_col=0)
        # input_data_bd6 = pd.read_csv('roc_values/' + semester + '/bd6_oversample.csv', index_col=0)

        # input_data_bd1 = pd.read_csv('roc_values/' + semester + '/bd1_quest_data.csv', index_col=0)
        # input_data_bd2 = pd.read_csv('roc_values/' + semester + '/bd2_quest_data.csv', index_col=0)
        # input_data_bd3 = pd.read_csv('roc_values/' + semester + '/bd3_quest_data.csv', index_col=0)
        # input_data_bd4 = pd.read_csv('roc_values/' + semester + '/bd4_quest_data.csv', index_col=0)
        # input_data_bd5 = pd.read_csv('roc_values/' + semester + '/bd5_quest_data.csv', index_col=0)
        # input_data_bd6 = pd.read_csv('roc_values/' + semester + '/bd6_quest_data.csv', index_col=0)

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

        results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'random_forest', 'avg': input_data_bd1.loc['random_forest'].mean(), 'median': input_data_bd1.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'decision_tree', 'avg': input_data_bd1.loc['decision_tree'].mean(), 'median': input_data_bd1.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'naive_bayes', 'avg': input_data_bd1.loc['naive_bayes'].mean(), 'median': input_data_bd1.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'adaboost', 'avg': input_data_bd1.loc['adaboost'].mean(), 'median': input_data_bd1.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'mlp', 'avg': input_data_bd1.loc['mlp'].mean(), 'median': input_data_bd1.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd1', 'classifier': 'knn', 'avg': input_data_bd1.loc['knn'].mean(), 'median': input_data_bd1.loc['knn'].median()}, ignore_index=True) 
        
        results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'random_forest', 'avg': input_data_bd2.loc['random_forest'].mean(), 'median': input_data_bd2.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'decision_tree', 'avg': input_data_bd2.loc['decision_tree'].mean(), 'median': input_data_bd2.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'naive_bayes', 'avg': input_data_bd2.loc['naive_bayes'].mean(), 'median': input_data_bd2.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'adaboost', 'avg': input_data_bd2.loc['adaboost'].mean(), 'median': input_data_bd2.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'mlp', 'avg': input_data_bd2.loc['mlp'].mean(), 'median': input_data_bd2.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd2', 'classifier': 'knn', 'avg': input_data_bd2.loc['knn'].mean(), 'median': input_data_bd2.loc['knn'].median()}, ignore_index=True)
        
        results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'random_forest', 'avg': input_data_bd3.loc['random_forest'].mean(), 'median': input_data_bd3.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'decision_tree', 'avg': input_data_bd3.loc['decision_tree'].mean(), 'median': input_data_bd3.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'naive_bayes', 'avg': input_data_bd3.loc['naive_bayes'].mean(), 'median': input_data_bd3.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'adaboost', 'avg': input_data_bd3.loc['adaboost'].mean(), 'median': input_data_bd3.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'mlp', 'avg': input_data_bd3.loc['mlp'].mean(), 'median': input_data_bd3.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd3', 'classifier': 'knn', 'avg': input_data_bd3.loc['knn'].mean(), 'median': input_data_bd3.loc['knn'].median()}, ignore_index=True)
        
        results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'random_forest', 'avg': input_data_bd4.loc['random_forest'].mean(), 'median': input_data_bd4.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'decision_tree', 'avg': input_data_bd4.loc['decision_tree'].mean(), 'median': input_data_bd4.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'naive_bayes', 'avg': input_data_bd4.loc['naive_bayes'].mean(), 'median': input_data_bd4.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'adaboost', 'avg': input_data_bd4.loc['adaboost'].mean(), 'median': input_data_bd4.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'mlp', 'avg': input_data_bd4.loc['mlp'].mean(), 'median': input_data_bd4.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd4', 'classifier': 'knn', 'avg': input_data_bd4.loc['knn'].mean(), 'median': input_data_bd4.loc['knn'].median()}, ignore_index=True)
        
        results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'random_forest', 'avg': input_data_bd5.loc['random_forest'].mean(), 'median': input_data_bd5.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'decision_tree', 'avg': input_data_bd5.loc['decision_tree'].mean(), 'median': input_data_bd5.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'naive_bayes', 'avg': input_data_bd5.loc['naive_bayes'].mean(), 'median': input_data_bd5.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'adaboost', 'avg': input_data_bd5.loc['adaboost'].mean(), 'median': input_data_bd5.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'mlp', 'avg': input_data_bd5.loc['mlp'].mean(), 'median': input_data_bd5.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd5', 'classifier': 'knn', 'avg': input_data_bd5.loc['knn'].mean(), 'median': input_data_bd5.loc['knn'].median()}, ignore_index=True)
        
        results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'random_forest', 'avg': input_data_bd6.loc['random_forest'].mean(), 'median': input_data_bd6.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'decision_tree', 'avg': input_data_bd6.loc['decision_tree'].mean(), 'median': input_data_bd6.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'naive_bayes', 'avg': input_data_bd6.loc['naive_bayes'].mean(), 'median': input_data_bd6.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'adaboost', 'avg': input_data_bd6.loc['adaboost'].mean(), 'median': input_data_bd6.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'mlp', 'avg': input_data_bd6.loc['mlp'].mean(), 'median': input_data_bd6.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd6', 'classifier': 'knn', 'avg': input_data_bd6.loc['knn'].mean(), 'median': input_data_bd6.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'random_forest', 'avg': input_data_bd7.loc['random_forest'].mean(), 'median': input_data_bd7.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'decision_tree', 'avg': input_data_bd7.loc['decision_tree'].mean(), 'median': input_data_bd7.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'naive_bayes', 'avg': input_data_bd7.loc['naive_bayes'].mean(), 'median': input_data_bd7.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'adaboost', 'avg': input_data_bd7.loc['adaboost'].mean(), 'median': input_data_bd7.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'mlp', 'avg': input_data_bd7.loc['mlp'].mean(), 'median': input_data_bd7.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd7', 'classifier': 'knn', 'avg': input_data_bd7.loc['knn'].mean(), 'median': input_data_bd7.loc['knn'].median()}, ignore_index=True)
        
        results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'random_forest', 'avg': input_data_bd8.loc['random_forest'].mean(), 'median': input_data_bd8.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'decision_tree', 'avg': input_data_bd8.loc['decision_tree'].mean(), 'median': input_data_bd6.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'naive_bayes', 'avg': input_data_bd8.loc['naive_bayes'].mean(), 'median': input_data_bd8.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'adaboost', 'avg': input_data_bd8.loc['adaboost'].mean(), 'median': input_data_bd8.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'mlp', 'avg': input_data_bd8.loc['mlp'].mean(), 'median': input_data_bd8.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd8', 'classifier': 'knn', 'avg': input_data_bd8.loc['knn'].mean(), 'median': input_data_bd8.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'random_forest', 'avg': input_data_bd9.loc['random_forest'].mean(), 'median': input_data_bd9.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'decision_tree', 'avg': input_data_bd9.loc['decision_tree'].mean(), 'median': input_data_bd9.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'naive_bayes', 'avg': input_data_bd9.loc['naive_bayes'].mean(), 'median': input_data_bd9.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'adaboost', 'avg': input_data_bd9.loc['adaboost'].mean(), 'median': input_data_bd9.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'mlp', 'avg': input_data_bd9.loc['mlp'].mean(), 'median': input_data_bd9.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd9', 'classifier': 'knn', 'avg': input_data_bd9.loc['knn'].mean(), 'median': input_data_bd9.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'random_forest', 'avg': input_data_bd10.loc['random_forest'].mean(), 'median': input_data_bd10.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'decision_tree', 'avg': input_data_bd10.loc['decision_tree'].mean(), 'median': input_data_bd10.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'naive_bayes', 'avg': input_data_bd10.loc['naive_bayes'].mean(), 'median': input_data_bd10.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'adaboost', 'avg': input_data_bd10.loc['adaboost'].mean(), 'median': input_data_bd10.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'mlp', 'avg': input_data_bd10.loc['mlp'].mean(), 'median': input_data_bd10.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd10', 'classifier': 'knn', 'avg': input_data_bd10.loc['knn'].mean(), 'median': input_data_bd10.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'random_forest', 'avg': input_data_bd11.loc['random_forest'].mean(), 'median': input_data_bd11.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'decision_tree', 'avg': input_data_bd11.loc['decision_tree'].mean(), 'median': input_data_bd11.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'naive_bayes'quest_data_, 'avg': input_data_bd11.loc['naive_bayes'].mean(), 'median': input_data_bd11.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'adaboost', 'avg': input_data_bd11.loc['adaboost'].mean(), 'median': input_data_bd11.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'mlp', 'avg': input_data_bd11.loc['mlp'].mean(), 'median': input_data_bd11.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd11', 'classifier': 'knn', 'avg': input_data_bd11.loc['knn'].mean(), 'median': input_data_bd11.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'random_forest', 'avg': input_data_bd12.loc['random_forest'].mean(), 'median': input_data_bd12.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'decision_tree', 'avg': input_data_bd12.loc['decision_tree'].mean(), 'median': input_data_bd12.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'naive_bayes', 'avg': input_data_bd12.loc['naive_bayes'].mean(), 'median': input_data_bd12.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'adaboost', 'avg': input_data_bd12.loc['adaboost'].mean(), 'median': input_data_bd12.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'mlp', 'avg': input_data_bd12.loc['mlp'].mean(), 'median': input_data_bd12.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd12', 'classifier': 'knn', 'avg': input_data_bd12.loc['knn'].mean(), 'median': input_data_bd12.loc['knn'].median()}, ignore_index=True)

        results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'random_forest', 'avg': input_data_bd13.loc['random_forest'].mean(), 'median': input_data_bd13.loc['random_forest'].median()}, ignore_index=True)
        # results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'decision_tree', 'avg': input_data_bd13.loc['decision_tree'].mean(), 'median': input_data_bd13.loc['decision_tree'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'naive_bayes', 'avg': input_data_bd13.loc['naive_bayes'].mean(), 'median': input_data_bd13.loc['naive_bayes'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'adaboost', 'avg': input_data_bd13.loc['adaboost'].mean(), 'median': input_data_bd13.loc['adaboost'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'mlp', 'avg': input_data_bd13.loc['mlp'].mean(), 'median': input_data_bd13.loc['mlp'].median()}, ignore_index=True)
        results = results.append({'semester': semester, 'bd': 'bd13', 'classifier': 'knn', 'avg': input_data_bd13.loc['knn'].mean(), 'median': input_data_bd13.loc['knn'].median()}, ignore_index=True)

    results.to_csv('statistic_tests/results_avg_median_oversample.csv', index=False)
    # results.to_csv('statistic_tests/results_avg_median_oversample.csv', index=False)
    # results.to_csv('statistic_tests/results_avg_median_quest_data.csv', index=False)