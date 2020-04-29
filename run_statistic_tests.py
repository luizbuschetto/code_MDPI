import pandas as pd
from scipy import stats

if __name__ == '__main__':
    
    semesters = ['2016-1', '2016-2', '2017-1', '2017-2']
    
    for semester in semesters:
        input_data_bd1 = pd.read_csv('roc_values/' + semester + '/bd1.csv', index_col=0)
        input_data_bd2 = pd.read_csv('roc_values/' + semester + '/bd2.csv', index_col=0)
        input_data_bd3 = pd.read_csv('roc_values/' + semester + '/bd3.csv', index_col=0)
        input_data_bd4 = pd.read_csv('roc_values/' + semester + '/bd4.csv', index_col=0)
        input_data_bd5 = pd.read_csv('roc_values/' + semester + '/bd5.csv', index_col=0)
        input_data_bd6 = pd.read_csv('roc_values/' + semester + '/bd6.csv', index_col=0)

        # Removido o SVM
        index_names = ['random_forest', 'decision_tree', 'naive_bayes', 'adaboost', 'mlp', 'knn']

        results_bd1 = pd.DataFrame(0, index=index_names, columns=index_names)
        results_bd2 = pd.DataFrame(0, index=index_names, columns=index_names)
        results_bd3 = pd.DataFrame(0, index=index_names, columns=index_names)
        results_bd4 = pd.DataFrame(0, index=index_names, columns=index_names)
        results_bd5 = pd.DataFrame(0, index=index_names, columns=index_names)
        results_bd6 = pd.DataFrame(0, index=index_names, columns=index_names)

        for idx, classifier1 in enumerate(index_names):
            classifiers2 = []

            for i in range(idx + 1, len(index_names)):
                classifiers2.append(index_names[i])

            for classifier2 in classifiers2:
                # if classifier1 != classifier2:
                results_bd1.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd1.loc[classifier1], input_data_bd1.loc[classifier2]).pvalue
                results_bd2.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd2.loc[classifier1], input_data_bd2.loc[classifier2]).pvalue
                results_bd3.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd3.loc[classifier1], input_data_bd3.loc[classifier2]).pvalue
                results_bd4.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd4.loc[classifier1], input_data_bd4.loc[classifier2]).pvalue
                results_bd5.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd5.loc[classifier1], input_data_bd5.loc[classifier2]).pvalue
                results_bd6.loc[classifier1, classifier2] = stats.mannwhitneyu(input_data_bd6.loc[classifier1], input_data_bd6.loc[classifier2]).pvalue

        bd_names = ['bd1', 'bd2', 'bd3', 'bd4', 'bd5', 'bd6']
        index_inter_names = []

        for bd in bd_names:
            for name in index_names:
                index_inter_names.append(bd + '_' + name)

        results_inter_bds = pd.DataFrame(0, index=index_inter_names, columns=bd_names)

        for classifier in index_names:
                results_inter_bds.loc['bd1_' + classifier, 'bd2'] = stats.mannwhitneyu(input_data_bd1.loc[classifier], input_data_bd2.loc[classifier]).pvalue
                results_inter_bds.loc['bd1_' + classifier, 'bd3'] = stats.mannwhitneyu(input_data_bd1.loc[classifier], input_data_bd3.loc[classifier]).pvalue
                results_inter_bds.loc['bd1_' + classifier, 'bd4'] = stats.mannwhitneyu(input_data_bd1.loc[classifier], input_data_bd4.loc[classifier]).pvalue
                results_inter_bds.loc['bd1_' + classifier, 'bd5'] = stats.mannwhitneyu(input_data_bd1.loc[classifier], input_data_bd5.loc[classifier]).pvalue
                results_inter_bds.loc['bd1_' + classifier, 'bd6'] = stats.mannwhitneyu(input_data_bd1.loc[classifier], input_data_bd6.loc[classifier]).pvalue

                results_inter_bds.loc['bd2_' + classifier, 'bd1'] = stats.mannwhitneyu(input_data_bd2.loc[classifier], input_data_bd1.loc[classifier]).pvalue
                results_inter_bds.loc['bd2_' + classifier, 'bd3'] = stats.mannwhitneyu(input_data_bd2.loc[classifier], input_data_bd3.loc[classifier]).pvalue
                results_inter_bds.loc['bd2_' + classifier, 'bd4'] = stats.mannwhitneyu(input_data_bd2.loc[classifier], input_data_bd4.loc[classifier]).pvalue
                results_inter_bds.loc['bd2_' + classifier, 'bd5'] = stats.mannwhitneyu(input_data_bd2.loc[classifier], input_data_bd5.loc[classifier]).pvalue
                results_inter_bds.loc['bd2_' + classifier, 'bd6'] = stats.mannwhitneyu(input_data_bd2.loc[classifier], input_data_bd6.loc[classifier]).pvalue

                results_inter_bds.loc['bd3_' + classifier, 'bd1'] = stats.mannwhitneyu(input_data_bd3.loc[classifier], input_data_bd1.loc[classifier]).pvalue
                results_inter_bds.loc['bd3_' + classifier, 'bd2'] = stats.mannwhitneyu(input_data_bd3.loc[classifier], input_data_bd2.loc[classifier]).pvalue
                results_inter_bds.loc['bd3_' + classifier, 'bd4'] = stats.mannwhitneyu(input_data_bd3.loc[classifier], input_data_bd4.loc[classifier]).pvalue
                results_inter_bds.loc['bd3_' + classifier, 'bd5'] = stats.mannwhitneyu(input_data_bd3.loc[classifier], input_data_bd5.loc[classifier]).pvalue
                results_inter_bds.loc['bd3_' + classifier, 'bd6'] = stats.mannwhitneyu(input_data_bd3.loc[classifier], input_data_bd6.loc[classifier]).pvalue

                results_inter_bds.loc['bd4_' + classifier, 'bd1'] = stats.mannwhitneyu(input_data_bd4.loc[classifier], input_data_bd1.loc[classifier]).pvalue
                results_inter_bds.loc['bd4_' + classifier, 'bd2'] = stats.mannwhitneyu(input_data_bd4.loc[classifier], input_data_bd2.loc[classifier]).pvalue
                results_inter_bds.loc['bd4_' + classifier, 'bd3'] = stats.mannwhitneyu(input_data_bd4.loc[classifier], input_data_bd3.loc[classifier]).pvalue
                results_inter_bds.loc['bd4_' + classifier, 'bd5'] = stats.mannwhitneyu(input_data_bd4.loc[classifier], input_data_bd5.loc[classifier]).pvalue
                results_inter_bds.loc['bd4_' + classifier, 'bd6'] = stats.mannwhitneyu(input_data_bd4.loc[classifier], input_data_bd6.loc[classifier]).pvalue

                results_inter_bds.loc['bd5_' + classifier, 'bd1'] = stats.mannwhitneyu(input_data_bd5.loc[classifier], input_data_bd1.loc[classifier]).pvalue
                results_inter_bds.loc['bd5_' + classifier, 'bd2'] = stats.mannwhitneyu(input_data_bd5.loc[classifier], input_data_bd2.loc[classifier]).pvalue
                results_inter_bds.loc['bd5_' + classifier, 'bd3'] = stats.mannwhitneyu(input_data_bd5.loc[classifier], input_data_bd3.loc[classifier]).pvalue
                results_inter_bds.loc['bd5_' + classifier, 'bd4'] = stats.mannwhitneyu(input_data_bd5.loc[classifier], input_data_bd4.loc[classifier]).pvalue
                results_inter_bds.loc['bd5_' + classifier, 'bd6'] = stats.mannwhitneyu(input_data_bd5.loc[classifier], input_data_bd6.loc[classifier]).pvalue

                results_inter_bds.loc['bd6_' + classifier, 'bd1'] = stats.mannwhitneyu(input_data_bd6.loc[classifier], input_data_bd1.loc[classifier]).pvalue
                results_inter_bds.loc['bd6_' + classifier, 'bd2'] = stats.mannwhitneyu(input_data_bd6.loc[classifier], input_data_bd2.loc[classifier]).pvalue
                results_inter_bds.loc['bd6_' + classifier, 'bd3'] = stats.mannwhitneyu(input_data_bd6.loc[classifier], input_data_bd3.loc[classifier]).pvalue
                results_inter_bds.loc['bd6_' + classifier, 'bd4'] = stats.mannwhitneyu(input_data_bd6.loc[classifier], input_data_bd4.loc[classifier]).pvalue
                results_inter_bds.loc['bd6_' + classifier, 'bd5'] = stats.mannwhitneyu(input_data_bd6.loc[classifier], input_data_bd5.loc[classifier]).pvalue

        results_bd1.to_csv('statistic_tests/' + semester + '/bd1.csv', index=True, index_label='classifier')
        results_bd2.to_csv('statistic_tests/' + semester + '/bd2.csv', index=True, index_label='classifier')
        results_bd3.to_csv('statistic_tests/' + semester + '/bd3.csv', index=True, index_label='classifier')
        results_bd4.to_csv('statistic_tests/' + semester + '/bd4.csv', index=True, index_label='classifier')
        results_bd5.to_csv('statistic_tests/' + semester + '/bd5.csv', index=True, index_label='classifier')
        results_bd6.to_csv('statistic_tests/' + semester + '/bd6.csv', index=True, index_label='classifier')
        results_inter_bds.to_csv('statistic_tests/' + semester + '/inter_bds.csv', index=True, index_label='classifier_bd')
