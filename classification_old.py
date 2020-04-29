import os
import sys

import pandas as pd
import numpy as np
from itertools import compress

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
# from sklearn import preprocessing

from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

# Feature Selection
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# Resample
from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import TomekLinks

if __name__ == '__main__':

    arguments = sys.argv[1:]
    args_count = len(arguments)

    if args_count != 8:
        raise ValueError('Devem ser passados 8 (oito) argumentos: USE_QUEST_DATA, USE_SMOTE, USE_FEATURE_SELECTION, SEMESTER, NUMBER_OF_FEATURES, NUMBER_OF_WEEKS, DATA_FILE_NAME, FOLDER_NAME')

    # USE_QUEST_DATA = False
    # USE_SMOTE = False
    # USE_FEATURE_SELECTION = False
    # SEMESTER = '2017-2'
    # NUMBER_OF_FEATURES = 10
    # NUMBER_OF_WEEKS = 17
    # DATA_FILE_NAME = 'bd6.csv'
    # FOLDER_NAME = 'bd6'

    USE_QUEST_DATA = int(sys.argv[1])
    USE_SMOTE = int(sys.argv[2])
    USE_FEATURE_SELECTION = int(sys.argv[3])
    SEMESTER = str(sys.argv[4])
    NUMBER_OF_FEATURES = int(sys.argv[5])
    NUMBER_OF_WEEKS = int(sys.argv[6])
    DATA_FILE_NAME = str(sys.argv[7])
    FOLDER_NAME = str(sys.argv[8])

    questions = 0

    if USE_QUEST_DATA:
        if SEMESTER == '2016-1':
            questions = 11
        elif SEMESTER == '2016-2':
            questions = 13
        elif SEMESTER == '2017-1':
            questions = 14
        elif SEMESTER == '2017-2':
            questions = 9

    # input_data = pd.read_csv('data/output/' + SEMESTER + '/' + DATA_FILE_NAME, header=None, skiprows=1)
    input_data = pd.read_csv('data/output/' + SEMESTER + '/' + DATA_FILE_NAME)
    print('\nInput file: ' + DATA_FILE_NAME)

    # Criando uma pasta para os resultados, casa não haja uma
    if not os.path.exists('graphics/' + SEMESTER + '/' + FOLDER_NAME):
        os.mkdir('graphics/' + SEMESTER + '/' + FOLDER_NAME)
        print('Creating ' + SEMESTER + '/' + FOLDER_NAME + ' Folder...')

    input_data = input_data.replace('Reprovado', 0)
    input_data = input_data.replace('Aprovado', 1)
    input_data = input_data.drop(columns=['Nome completo'])

    # "-1": coluna dos Labels
    # "12": features vector para cada semana (neste caso)
    # NUMBER_OF_WEEKS = (len(input_data.columns) - 1) / NUMBER_OF_FEATURES
    # NUMBER_OF_WEEKS = int(NUMBER_OF_WEEKS)

    # Dados para treinamento e teste
    x_test = input_data
    x_test = x_test.drop(columns=['final_result'])

    column_names = list(x_test.columns.values)
    y_test = input_data['final_result']

    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_test = min_max_scaler.fit_transform(x_test)
    # x_test = pd.DataFrame(data=x_test, columns=column_names)

    if USE_SMOTE:
        # Resample - SMOTE -> Oversample
        print('Using SMOTE')
        smote = SMOTE(ratio='minority')
        x_test, y_test = smote.fit_sample(x_test, y_test)
        x_test = pd.DataFrame(data=x_test, columns=column_names)

    # # Resample - TomekLinks -> Undersample
    # tl = TomekLinks(return_indices=True, ratio='majority')
    # x_test, y_test, _ = tl.fit_sample(x_test, y_test)
    # x_test = pd.DataFrame(data=x_test, columns=column_names)

    feature_cols = list(input_data.columns[:-1])

    # Criando os modelos dos classificadores
    random_forest = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=0)
    naive_bayes = BernoulliNB(alpha=0.1)
    decision_tree = DecisionTreeClassifier(criterion='gini', random_state=0)
    ada_boost = AdaBoostClassifier(decision_tree, n_estimators=100, random_state=0)
    MLP = MLPClassifier(max_iter=9999, random_state=0) # Itera até a rede convergir
    svm = SVC(kernel='rbf', random_state=0)
    knn = KNeighborsClassifier(n_neighbors=3)

    # kf = KFold(n_splits=len(input_data))
    kf = KFold(n_splits=len(x_test))

    feature_cols_used = []

    true_negative_rate_MLP = []
    true_negative_rate_random_forest = []
    true_negative_rate_naive_bayes = []
    true_negative_rate_decision_tree = []
    true_negative_rate_ada_boost = []
    true_negative_rate_svm = []
    true_negative_rate_knn = []

    true_positive_rate_MLP = []
    true_positive_rate_random_forest = []
    true_positive_rate_naive_bayes = []
    true_positive_rate_decision_tree = []
    true_positive_rate_ada_boost = []
    true_positive_rate_svm = []
    true_positive_rate_knn = []

    roc_auc_mlp = []
    roc_auc_random_forest = []
    roc_auc_naive_bayes = []
    roc_auc_decision_tree = []
    roc_auc_ada_boost = []
    roc_auc_svm = []
    roc_auc_knn = []

    # Loop de treinamento
    for i in range(0, len(feature_cols) - questions, NUMBER_OF_FEATURES):
        feature_cols_used = []
        for x in range(0, i + questions + NUMBER_OF_FEATURES):
            feature_cols_used.append(feature_cols[x])

        print("\nFeatures: ")
        print(feature_cols_used)

        # for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
        #     X_train = X.loc[np.r_[train_index], :]
        #     y_train = y[train_index]
        #
        #     X_test = X.loc[np.r_[test_index], :]
        #     y_test = y[test_index]
        #
        #     sm = SMOTE(ratio='minority')
        #     X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)
        #     # model = ...  # Choose a model here
        #     # model.fit(X_train_oversampled, y_train_oversampled)
        #     # y_pred = model.predict(X_test)
        #
        #     random_forest.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_random_forest = random_forest.predict(X_test)
        #
        #     naive_bayes.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_naive_bayes = naive_bayes.predict(X_test)
        #
        #     decision_tree.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_decision_tree = decision_tree.predict(X_test)
        #
        #     ada_boost.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_ada_boost = ada_boost.predict(X_test)
        #
        #     MLP.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_MLP = MLP.predict(X_test)
        #
        #     knn.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_knn = knn.predict(X_test)
        #
        #     svm.fit(X_train_oversampled, y_train_oversampled)
        #     y_prediction_svm = svm.predict(X_test)

        # if USE_FEATURE_SELECTION:
        #     print('\nUsing Feature Selection\n')
        #     # Feature Selection
        #     y = y_test
        #     # y = y.values
        #
        #     X = x_test[feature_cols_used].values
        #
        #     svc = SVC(kernel="linear")
        #     rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(n_splits=2),
        #                   scoring='accuracy')
        #     rfecv.fit(X, y)
        #     feature_cols_used = list(compress(feature_cols_used, rfecv.support_))
        #
        #     print("\nFeatures após a seleção: ")
        #     print(feature_cols_used)

        # Fazendo as predições
        y_prediction_MLP = cross_val_predict(MLP, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_random_forest = cross_val_predict(random_forest, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_naive_bayes = cross_val_predict(naive_bayes, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_decision_tree = cross_val_predict(decision_tree, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_ada_boost = cross_val_predict(ada_boost, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_svm = cross_val_predict(svm, x_test[feature_cols_used], y_test, cv=kf)
        y_prediction_knn = cross_val_predict(knn, x_test[feature_cols_used], y_test, cv=kf)

        # Gerando matrizes de confusão
        confusion_MLP = confusion_matrix(y_test, y_prediction_MLP)
        confusion_random_forest = confusion_matrix(y_test, y_prediction_random_forest)
        confusion_naive_bayes = confusion_matrix(y_test, y_prediction_naive_bayes)
        confusion_decision_tree = confusion_matrix(y_test, y_prediction_decision_tree)
        confusion_ada_boost = confusion_matrix(y_test, y_prediction_ada_boost)
        confusion_svm = confusion_matrix(y_test, y_prediction_svm)
        confusion_knn = confusion_matrix(y_test, y_prediction_knn)

        # Pegando os valores das matrizes de confusão
        tn_mlp, fp_mlp, fn_mlp, tp_mlp = confusion_matrix(y_test, y_prediction_MLP).ravel()
        tn_random_forest, fp_random_forest, fn_random_forest, tp_random_forest = confusion_matrix(y_test, y_prediction_random_forest).ravel()
        tn_naive_bayes, fp_naive_bayes, fn_naive_bayes, tp_naive_bayes = confusion_matrix(y_test, y_prediction_naive_bayes).ravel()
        tn_decision_tree, fp_decision_tree, fn_decision_tree, tp_decision_tree = confusion_matrix(y_test, y_prediction_naive_bayes).ravel()
        tn_ada_boost, fp_ada_boost, fn_ada_boost, tp_ada_boost = confusion_matrix(y_test, y_prediction_ada_boost).ravel()
        tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(y_test, y_prediction_svm).ravel()
        tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(y_test, y_prediction_knn).ravel()

        # Gerando curva ROC
        fp_roc_mlp, tp_roc_mlp, _ = roc_curve(y_test, y_prediction_MLP)
        fp_roc_random_forest, tp_roc_random_forest, _ = roc_curve(y_test, y_prediction_random_forest)
        fp_roc_naive_bayes, tp_roc_naive_bayes, _ = roc_curve(y_test, y_prediction_naive_bayes)
        fp_roc_decision_tree, tp_roc_decision_tree, _ = roc_curve(y_test, y_prediction_decision_tree)
        fp_roc_ada_boost, tp_roc_ada_boost, _ = roc_curve(y_test, y_prediction_ada_boost)
        fp_roc_svm, tp_roc_svm, _ = roc_curve(y_test, y_prediction_svm)
        fp_roc_knn, tp_roc_knn, _ = roc_curve(y_test, y_prediction_knn)

        # Gerando "Area Under the Curve (AUC)"
        roc_mlp = auc(fp_roc_mlp, tp_roc_mlp)
        roc_auc_mlp.append(roc_mlp)
        roc_random_forest = auc(fp_roc_random_forest, tp_roc_random_forest)
        roc_auc_random_forest.append(roc_random_forest)
        roc_decision_tree = auc(fp_roc_decision_tree, tp_roc_decision_tree)
        roc_auc_decision_tree.append(roc_decision_tree)
        roc_ada_boost = auc(fp_roc_ada_boost, tp_roc_ada_boost)
        roc_auc_ada_boost.append(roc_ada_boost)
        roc_naive_bayes = auc(fp_roc_naive_bayes, tp_roc_naive_bayes)
        roc_auc_naive_bayes.append(roc_naive_bayes)
        roc_svm = auc(fp_roc_svm, tp_roc_svm)
        roc_auc_svm.append(roc_svm)
        roc_knn = auc(fp_roc_knn, tp_roc_knn)
        roc_auc_knn.append(roc_knn)

        # Printando as matrizes de confusão
        print("\nConfusion Matrix - Naive Bayes")
        print(confusion_naive_bayes)
        print("\nConfusion Matrix - Random Forest")
        print(confusion_random_forest)
        print("\nConfusion Matrix - Decision Tree")
        print(confusion_decision_tree)
        print("\nConfusion Matrix - AdaBoost")
        print(confusion_ada_boost)
        print("\nConfusion Matrix - MLP")
        print(confusion_MLP)
        print("\nConfusion Matrix - SVM")
        print(confusion_svm)
        print("\nConfusion Matrix - kNN")
        print(confusion_knn)

        cm_decision_tree = confusion_decision_tree.astype('float') / confusion_decision_tree.sum(axis=1)[:, np.newaxis]
        cm_naive_bayes = confusion_naive_bayes.astype('float') / confusion_naive_bayes.sum(axis=1)[:, np.newaxis]
        cm_random_forest = confusion_random_forest.astype('float') / confusion_random_forest.sum(axis=1)[:, np.newaxis]
        cm_ada_boost = confusion_ada_boost.astype('float') / confusion_ada_boost.sum(axis=1)[:, np.newaxis]
        cm_mlp = confusion_MLP.astype('float') / confusion_MLP.sum(axis=1)[:, np.newaxis]
        cm_svm = confusion_svm.astype('float') / confusion_svm.sum(axis=1)[:, np.newaxis]
        cm_knn = confusion_knn.astype('float') / confusion_knn.sum(axis=1)[:, np.newaxis]

        tp_decision_tree, tn_decision_tree = cm_decision_tree[0][0],cm_decision_tree[1][1]
        tp_random_forest, tn_random_forest = cm_random_forest[0][0], cm_random_forest[1][1]
        tp_naive_bayes, tn_naive_bayes = cm_naive_bayes[0][0], cm_naive_bayes[1][1]
        tp_ada_boost, tn_ada_boost = cm_ada_boost[0][0], cm_ada_boost[1][1]
        tp_mlp, tn_mlp = cm_mlp[0][0], cm_mlp[1][1]
        tp_knn, tn_knn = cm_knn[0][0], cm_knn[1][1]
        tp_svm, tn_svm = cm_svm[0][0], cm_svm[1][1]

        true_positive_rate_decision_tree.append(tp_decision_tree)
        true_negative_rate_decision_tree.append(tn_decision_tree)

        true_negative_rate_naive_bayes.append(tn_naive_bayes)
        true_positive_rate_naive_bayes.append(tp_naive_bayes)

        true_negative_rate_random_forest.append(tn_random_forest)
        true_positive_rate_random_forest.append(tp_random_forest)

        true_positive_rate_ada_boost.append(tp_ada_boost)
        true_negative_rate_ada_boost.append(tn_ada_boost)

        true_positive_rate_MLP.append(tp_mlp)
        true_negative_rate_MLP.append(tn_mlp)

        true_positive_rate_svm.append(tp_svm)
        true_negative_rate_svm.append(tn_svm)

        true_positive_rate_knn.append(tp_knn)
        true_negative_rate_knn.append(tn_knn)

    # # Plotando os gráficos
    # week = []
    #
    # for t in range(0, len(true_negative_rate_random_forest)):
    #     week.append(t)
    #
    # my_xticks = ["S" + str(x) for x in range(0, NUMBER_OF_WEEKS + 1)]
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)') # Percentual = Taxa de acerto?
    # plt.title("Verdadeiros Negativos")
    #
    # line1, = plt.plot(week, true_negative_rate_random_forest, label='True Negative - Random Forest')
    # line2, = plt.plot(week, true_negative_rate_naive_bayes, label='True Negative - Naive Bayes')
    # line3, = plt.plot(week, true_negative_rate_decision_tree, label='True Negative - Decision Tree')
    # line4, = plt.plot(week, true_negative_rate_ada_boost, label='True Negative - AdaBoost')
    # line5, = plt.plot(week, true_negative_rate_MLP, label='True Negative - MLP')
    # line6, = plt.plot(week, true_negative_rate_svm, label='True Negative - SVM')
    # line7, = plt.plot(week, true_negative_rate_knn, label='True Negative - kNN')
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/verdadeiros_negativos.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Área')
    # plt.title("Área sob a curva ROC x Semanas ")
    #
    # line1, = plt.plot(week, roc_auc_naive_bayes, label='Área sob a curva ROC - Naive Bayes')
    # line2, = plt.plot(week, roc_auc_decision_tree, label='Área sob a curva ROC - Decision Tree')
    # line3, = plt.plot(week, roc_auc_random_forest, label='Área sob a curva ROC - Random Forest')
    # line4, = plt.plot(week, roc_auc_ada_boost, label='Área sob a curva ROC - AdaBoost')
    # line5, = plt.plot(week, roc_auc_mlp, label='Área sob a curva ROC - MLP')
    # line6, = plt.plot(week, roc_auc_svm, label='Área sob a curva ROC - SVM')
    # line7, = plt.plot(week, roc_auc_knn, label='Área sob a curva ROC - kNN')
    #
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    # plt.grid(axis='y', linestyle='-')
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/area_sob_roc.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - MLP")
    # line2, = plt.plot(week, roc_auc_mlp, color='r', label='Curva ROC - MLP')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_mlp.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - Random Forest ")
    # line2, = plt.plot(week, roc_auc_random_forest, color='r', label='Curva ROC - Random Forest')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_random_forest.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - Naive Bayes ")
    # line2, = plt.plot(week, roc_auc_naive_bayes, color='r', label='Curva ROC - Naive Bayes')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_naive_bayes.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - Decision Tree")
    # line2, = plt.plot(week, roc_auc_decision_tree, color='r', label='Curva ROC - Decision Tree')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_decision_tree.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - AdaBoost")
    # line2, = plt.plot(week, roc_auc_ada_boost, color='r', label='Curva ROC - AdaBoost')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_ada_boost.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - SVM")
    # line2, = plt.plot(week, roc_auc_svm, color='r', label='Curva ROC - SVM')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_svm.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("Curva ROC - kNN")
    # line2, = plt.plot(week, roc_auc_knn, color='r', label='Curva ROC - kNN')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/roc_knn.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - MLP ")
    # line2, = plt.plot(week, true_positive_rate_MLP, color='r', label='True Positive - MLP')
    # line1, = plt.plot(week, true_negative_rate_MLP, label='True Negative - MLP')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_mlp.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - Random Forest")
    # line2, = plt.plot(week, true_positive_rate_random_forest, color='r', label='True Positive - Random Forest')
    # line1, = plt.plot(week, true_negative_rate_random_forest, label='True Negative - Random Forest')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_random_forest.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - Naive Bayes")
    # line2, = plt.plot(week, true_positive_rate_naive_bayes, color='r', label='True Positive - Naive Bayes')
    # line1, = plt.plot(week, true_negative_rate_naive_bayes, label='True Negative - Naive Bayes')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_naive_bayes.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - Decision Tree")
    # line2, = plt.plot(week, true_positive_rate_decision_tree, color='r', label='True Positive - Decision Tree')
    # line1, = plt.plot(week, true_negative_rate_decision_tree, label='True Negative - Decision Tree')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_decision_tree.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - AdaBoost")
    # line2, = plt.plot(week, true_positive_rate_ada_boost, color='r', label='True Positive - AdaBoost')
    # line1, = plt.plot(week, true_negative_rate_ada_boost, label='True Negative - AdaBoost')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_ada_boost.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - SVM")
    # line2, = plt.plot(week, true_positive_rate_svm, color='r', label='True Positive - SVM')
    # line1, = plt.plot(week, true_negative_rate_svm, label='True Negative - SVM')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_svm.png')
    # plt.clf()
    #
    # plt.xticks(week, my_xticks)
    # plt.yticks(np.arange(0, 1, 0.1))
    # plt.xlabel('Semanas')
    # plt.ylabel('Taxa de Acerto (%)')
    # plt.title("True Positive & True Negative - kNN")
    # line2, = plt.plot(week, true_positive_rate_knn, color='r', label='True Positive - kNN')
    # line1, = plt.plot(week, true_negative_rate_knn, label='True Negative - kNN')
    # plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
    # plt.grid(axis='y', linestyle='-')
    #
    # plt.savefig('graphics/' + SEMESTER + '/' + FOLDER_NAME + '/vp_vn_knn.png')
    # plt.clf()
    #
    # #with open('graphics/'+ SEMESTER + '/' + FOLDER_NAME + '/output.txt', 'w') as f:
    # print("\n\nTrue Negative and True Positive Vectors: \n")
    # print("Random Forest: ")
    # print("True Negative Vector: ", true_negative_rate_random_forest)
    # print("True Positive Vector: ", true_positive_rate_random_forest)
    #
    # print("\nDecision Tree: ")
    # print("True Negative Vector: ", true_negative_rate_decision_tree)
    # print("True Positive Vector: ", true_positive_rate_decision_tree)
    #
    # print("\nNaive Bayes: ")
    # print("True Negative Vector: ", true_negative_rate_naive_bayes)
    # print("True Positive Vector: ", true_positive_rate_naive_bayes)
    #
    # print("\nAdaBoost: ")
    # print("True Negative Vector: ", true_negative_rate_ada_boost)
    # print("True Positive Vector: ", true_positive_rate_ada_boost)
    #
    # print("\nSVM: ")
    # print("True Negative Vector: ", true_negative_rate_svm)
    # print("True Positive Vector: ", true_positive_rate_svm)
    #
    # print("\nkNN: ")
    # print("True Negative Vector: ", true_negative_rate_knn)
    # print("True Positive Vector: ", true_positive_rate_knn)
    #
    # print("\n\nROC - Area under the curve\n")
    # print("Random Forest = ", roc_auc_random_forest)
    # print("Decision Tree =", roc_auc_decision_tree)
    # print("Naive Bayes = ", roc_auc_naive_bayes)
    # print("AdaBoost =  ", roc_auc_ada_boost)
    # print("MLP = ", roc_auc_mlp)
    # print("SVM = ", roc_auc_svm)
    # print("kNN = ", roc_auc_knn)

    # Generating output CSV with ROC values
    column_names = []

    for week in range(0, NUMBER_OF_WEEKS + 1):
        column_names.append('Week' + str(week))

    index_names = ['random_forest', 'decision_tree', 'naive_bayes', 'adaboost', 'mlp', 'svm', 'knn']

    roc_values = pd.DataFrame(index=index_names, columns=column_names)

    roc_values.loc['random_forest'] = roc_auc_random_forest
    roc_values.loc['decision_tree'] = roc_auc_decision_tree
    roc_values.loc['naive_bayes'] = roc_auc_naive_bayes
    roc_values.loc['adaboost'] = roc_auc_ada_boost
    roc_values.loc['mlp'] = roc_auc_mlp
    roc_values.loc['svm'] = roc_auc_svm
    roc_values.loc['knn'] = roc_auc_knn

    roc_values.to_csv('roc_values/' + SEMESTER + '/' + FOLDER_NAME + '.csv', index=True, index_label='classifier')


