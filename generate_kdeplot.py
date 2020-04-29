import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':

    semester = '2016-2'

    dados = pd.read_csv('/home/luiz/Desktop/' + semester + '.csv')

    # dados = dados.rename(columns={"situacao": "Situation"})

    dados['final_result'] = dados['final_result'].replace('Reprovado', 0)
    dados['final_result'] = dados['final_result'].replace('Aprovado', 1)

    evadidos = dados.copy()
    concluintes = dados.copy()

    evadidos = evadidos[evadidos.final_result == 0]
    concluintes = concluintes[concluintes.final_result == 1]

    # week = 'Week_0'

    weeks = ['Week_0', 'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_5', 'Week_6', 'Week_7', 'Week_8']

    for week in weeks:
        #####################
        # KDEPlots
        #####################
        sns.set(color_codes=True)
        sns.kdeplot(evadidos[week], shade=True, label='Failed')
        sns.kdeplot(concluintes[week], shade=True, label='Approved')
        plt.savefig('/home/luiz/Desktop/graficos/' + semester + '/kdeplots/' + week + '.png')
        plt.close()

        #####################
        # BoxPlots
        #####################
        data = [evadidos[week], concluintes[week]]
        plt.boxplot(data, positions=[1, 2], labels=['Failed', 'Approved'])
        plt.title(week)
        plt.savefig('/home/luiz/Desktop/graficos/' + semester + '/boxplots/' + week + '.png')
        plt.close()
