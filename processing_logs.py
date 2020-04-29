import pandas as pd
import numpy as np
from collections import OrderedDict

class Processing(object):
	""" Essa função faz o processamento das tabelas do moodle
	geralmente as tabelas do moodle nas ultimas versões estão
	no formato.
	Hora ou Data|Nome do completo| Usuario afetado| Contexto do Evento| Componente| Nome de Evento| Descrição| Origem | IP
	"""
	
	def __init__(self, logs, professores, monitores):
		"""
		Função init
		Recebe os logs do moodle em um arquivo csv, nomes dos professores em uma lista
		nome dos monitores em uma lista
		"""

		self._logs = pd.read_csv(logs, delimiter=",")
		self._professores = professores
		self._monitores = monitores

	def preprocessing(self, dataframe):
		"""
		Essa função vai receber um dataframe e vai processar 
		para, o número é a quantidade de iterção por semana
		Nome    Semana1 Semana3 Semana3... Semana(n) 
		Matheus    2      3        4    ...   4
		"""

		# 2016-1 -> Início: 2016/03/14, Final: 2016/07/23
		# 2016-2 -> Início: 2016/08/08, Final: 2016/12/16
		#
		# 2017-1 -> Início: 2017/03/06, Final: 2017/07/08
		# 2017-2 -> Início: 2017/07/31, Final: 2017/12/07

		QUEST_DATA = True
		semester = '2017-2'
		number_of_weeks = 17

		if semester == '2016-1':
			data_inicio = '2016/03/14'
			data_final = '2016/07/23'

		elif semester == '2016-2':
			data_inicio = '2016/08/08'
			data_final = '2016/12/16'

		elif semester == '2017-1':
			data_inicio = '2017/03/06'
			data_final = '2017/07/08'

		elif semester == '2017-2':
			data_inicio = '2017/07/31'
			data_final = '2017/12/07'

		#recebe a coluna nome inteira
		names = self._logs['Nome completo']
		#index fica os nomes
		names = set(names)
		#recebe a coluna hora
		df = dataframe['Hora']
		# print("============ DEBUGANDO 1 ============")
		# print(df.head(10))
		# print("============ END DEBUGANDO 1 ============")

		#tamanho da coluna hora
		lenght = len(df)
		
		#to datetime converte todos as horas para o formato dia/mes/ano 
		df = pd.to_datetime(dataframe['Hora'], format="%d/%m/%Y %H:%M")
		# print("============ DEBUGANDO ============")
		# print(df.head(10))
		# print("============ END DEBUGANDO ============")

		#essa parte de código vai converter para ano/mes/dia 
		tupla = [None]*lenght
		for i in range(lenght):
			m = df[i]
			b = m.strftime("%Y-%m-%d")
			tupla[i] = b
		dataframe['Hora'] = tupla
		gp = dataframe

		# print("============ DEBUGANDO 3 ============")
		# print(gp.head(10))
		# print("============ END DEBUGANDO 3 ============")

		#set o index como da tabela em y a hora 
		dataframe.set_index(dataframe['Hora'], inplace=True)
		
		#deleta a coluna hora pois o index é a hora agr
		del dataframe['Hora']

		"""
		vai gerara um count: que vai agrupar
		todos os nomes iguais que acessou naquela hora.
		10/10/2010 Matheus
		10/10/2010 Matheus
		10/10/2010 Matheus
		10/10/2010 Matheus   count:3
		"""

		grouped2 = pd.DataFrame({'count' : dataframe.groupby( [ "Hora", "Nome completo"] ).size()}).reset_index()
		grouped2_types = pd.DataFrame({'count' : dataframe.groupby( [ "Hora", "Nome completo", "Componente"] ).size()}).reset_index()
		gp = pd.DataFrame({'count' : dataframe.groupby( [ "Hora", "Nome completo"] ).size()}).reset_index()

		#copio para outra, para set index como nome
		gp2 = gp
		gp2 = gp2.set_index(gp2['Nome completo'], inplace=True)


		n = np.array(dataframe['Nome completo'])
		l = len(grouped2)

		#excluindo os nomes de monitores e professores
		for k in range(l):
			
			if grouped2['Nome completo'][k] == 'Vinicius Faria Culmant Ramos':
				grouped2.drop([k], axis=0 ,inplace=True)

			elif grouped2['Nome completo'][k] == '-':
				grouped2.drop([k], axis=0 ,inplace=True)

			elif grouped2['Nome completo'][k] == 'Mihael Zamin Sousa (13203715)':
				grouped2.drop([k], axis=0 ,inplace=True)

			elif grouped2['Nome completo'][k] == 'Daniel Bitencourt Pereira (13104260)':
				grouped2.drop([k], axis=0 ,inplace=True)

			elif grouped2['Nome completo'][k]=='Usuário Administrador':
				grouped2.drop([k], axis=0 ,inplace=True)

			elif grouped2['Nome completo'][k] == 'Cristian Cechinel':
				grouped2.drop([k], axis=0, inplace=True)

		l = len(grouped2_types)

		# excluindo os nomes de monitores e professores
		for k in range(l):

			if grouped2_types['Nome completo'][k] == 'Vinicius Faria Culmant Ramos':
				grouped2_types.drop([k], axis=0, inplace=True)

			elif grouped2_types['Nome completo'][k] == '-':
				grouped2_types.drop([k], axis=0, inplace=True)

			elif grouped2_types['Nome completo'][k] == 'Mihael Zamin Sousa (13203715)':
				grouped2_types.drop([k], axis=0, inplace=True)

			elif grouped2_types['Nome completo'][k] == 'Daniel Bitencourt Pereira (13104260)':
				grouped2_types.drop([k], axis=0, inplace=True)

			elif grouped2_types['Nome completo'][k] == 'Usuário Administrador':
				grouped2_types.drop([k], axis=0, inplace=True)

			elif grouped2_types['Nome completo'][k] == 'Cristian Cechinel':
				grouped2_types.drop([k], axis=0, inplace=True)

		########################## Getting unique activities

		# new_df = dataframe.copy()
		# indexes = pd.Series(range(0, len(new_df)))
		# new_df.set_index(indexes, inplace=True)
		#
		# l = len(new_df)
		#
		# # excluindo os nomes de monitores e professores
		# for k in range(l):
		# 	# print(grouped2['Nome completo'])
		#
		# 	if grouped2['Nome completo'][k] == 'Vinicius Faria Culmant Ramos':
		# 		grouped2.drop([k], axis=0, inplace=True)
		# 	elif grouped2['Nome completo'][k] == '-':
		# 		grouped2.drop([k], axis=0, inplace=True)
		# 	elif grouped2['Nome completo'][k] == 'Mihael Zamin Sousa (13203715)':
		# 		grouped2.drop([k], axis=0, inplace=True)
		# 	elif grouped2['Nome completo'][k] == 'Daniel Bitencourt Pereira (13104260)':
		# 		grouped2.drop([k], axis=0, inplace=True)
		# 	elif grouped2['Nome completo'][k] == 'Usuário Administrador':
		# 		grouped2.drop([k], axis=0, inplace=True)
		# 	elif grouped2['Nome completo'][k] == 'Cristian Cechinel':
		# 		grouped2.drop([k], axis=0, inplace=True)
		#
		# activities_comp = new_df['Componente']
		# activities_event = new_df['Nome do evento']
		#
		# activities_comp = activities_comp.unique()
		# activities_event = activities_event.unique()
		#
		# fmt = '%s'
		#
		# np.savetxt("/home/luiz/Desktop/activities.csv", activities_comp, fmt=fmt, delimiter=",")

		################################################################################

		#print(len(grouped2))
		#i = np.arange(0,767,1)
		grouped2['Hora'] = pd.to_datetime(grouped2['Hora'], format="%Y/%m/%d")
		grouped2_types['Hora'] = pd.to_datetime(grouped2_types['Hora'], format="%Y/%m/%d")
		#print(grouped2)
		#gp2 = grouped2
		# print("============ DEBUGANDO ============")
		# print(grouped2.head(10))
		# print("============ END DEBUGANDO  ============")

		grouped2_types['Cognitive'] = pd.Series(np.zeros(len(grouped2_types)), index=grouped2_types.index)
		grouped2_types['Social'] = pd.Series(np.zeros(len(grouped2_types)), index=grouped2_types.index)
		grouped2_types['Teaching'] = pd.Series(np.zeros(len(grouped2_types)), index=grouped2_types.index)
		grouped2_types['Other'] = pd.Series(np.zeros(len(grouped2_types)), index=grouped2_types.index)

		for idx, item in grouped2_types.iterrows():
			if item['Componente'] == 'Arquivo' or item['Componente'] == 'Envio de arquivos' or item['Componente'] == 'Envio de texto online' \
					or item['Componente'] == 'Laboratório de Avaliação' or item['Componente'] == 'Laboratório de Programação Virtual' \
					or item['Componente'] == 'Lição' or item['Componente'] == 'Página' or item['Componente'] == 'Pesquisa' \
					or item['Componente'] == 'Presença' or item['Componente'] == 'Sistema' or item['Componente'] == 'Tarefa' \
					or item['Componente'] == 'URL' or item['Componente'] == 'Wiki':
				grouped2_types.loc[idx, 'Cognitive'] = item['count']

			elif item['Componente'] == 'Comentários ao envio':
				grouped2_types.loc[idx, 'Teaching'] = item['count']

			elif item['Componente'] == 'Enquete' or item['Componente'] == 'Fórum':
				grouped2_types.loc[idx, 'Social'] = item['count']

			elif item['Componente'] == 'Relatório do usuário' or item['Componente'] == 'Relatório geral' or item['Componente'] == 'Relatório de notas' or item['Componente'] == 'Logs':
				grouped2_types.loc[idx, 'Other'] = item['count']

		grouped2_types = grouped2_types.drop(columns=['Componente'])
		#####

		# gera um iteração entre as tadas do semestre

		my_df = pd.DataFrame(grouped2)
		my_df_types = pd.DataFrame(grouped2_types)
		# my_df = pd.DataFrame({'count': grouped2.groupby(["Hora", "Nome completo", "Social", "Teaching", "Social"]).size()}).reset_index()
		# print("============ DEBUGANDO 4  ============")
		# print(my_df.head(10))
		# print("============ END DEBUGANDO 4  ============")
		# df = pd.DataFrame()
		# df_types = pd.DataFrame()

		print("MUITO IMPORTANTE !!!! COLOCAR A DATA DO SEMESTRE CORRETO")
		df = my_df[(my_df['Hora'] >= data_inicio) & (my_df['Hora'] <= data_final)]
		df_types = my_df_types[(my_df_types['Hora'] >= data_inicio) & (my_df_types['Hora'] <= data_final)]
		#
		# print("============ DEBUGANDO 5  ============")
		# print(df.head(10))
		# print("============ END DEBUGANDO 5  ============")

		#df.to_csv('logs/ItUsuarios.csv',index=True,header=True)		
		#df = file
		#print(df)
		#print(df['Hora'])
		#df = pd.read_csv(io.StringIO(s), parse_dates=True)
		
		#converte a HORA para um periodo de semanas (no caso W-THU conta a semana de segunda a segunda ou
		# terça - terça

		df['Hora']=pd.to_datetime(df['Hora'])
		df_types['Hora']=pd.to_datetime(df_types['Hora'])

		semana = df['Hora'].dt.to_period('W-THU')
		semana = np.unique(semana)
		#cria uma coluna de semana com a coluna data(hora) do csv
		df['Semana'] = df['Hora'].dt.to_period('W-THU')
		df_types['Semana'] = df_types['Hora'].dt.to_period('W-THU')

		df2 = df.copy()
		df3 = df.copy()

		df2['Semana'] = df2['Hora'].dt.to_period('W-THU')
		df3['Semana'] = df3['Hora'].dt.to_period('W-THU')

		#Ira agrupar Semana e faz uma soma por nome completo ao londo da semana
		df_types = df_types.groupby(by=['Semana', 'Nome completo'])['Nome completo', 'count', 'Hora', 'Cognitive', 'Teaching', 'Social', 'Other'].sum()
		df = df.groupby(by=['Semana', 'Nome completo'])['Nome completo', 'count', 'Hora'].sum()
		# df = df.groupby(by=['Semana', 'Nome completo'])['Nome completo', 'count', 'Hora'].sum()
		# Faz a média na semana
		df2 = df2.groupby(by=['Semana', 'Nome completo'])['Nome completo', 'count', 'Hora'].mean()
		df3 = df3.groupby(by=['Semana', 'Nome completo'])['Nome completo', 'count', 'Hora'].median()

		# data2 = df2.to_dict('index')
		# data = df.to_dict('index') # cria um dict ,um arquivo JSON
		# data3 = df3.to_dict('index')
		# #print(data)
		#df = df.groupby(by=['Semana', 'Nome completo'])['count'].median()

		#print(df)
		
		#print(row['count'])
		my_df = pd.DataFrame(df)
		my_df_types = pd.DataFrame(df_types)
		my_df2 = pd.DataFrame(df2)
		my_df3 = pd.DataFrame(df3)


		#print(my_df)
		# gera um arquivo para ler linha a linha
		my_df = my_df_types.copy()
		my_df.to_csv('logs/Iteracao.csv',index=True,header=True)
		my_df2.to_csv('logs/itMedia.csv',index=True,header=True)
		my_df3.to_csv('logs/itMediana.csv',index=True,header=True)


		#print(my_df)

		# Lendo o arquivo para uma lista de linhas
		f = open('logs/Iteracao.csv', mode='r')
		
		lines = f.readlines()
		
		#print(tst)
		del lines[0]
		
		# cria uma lista com nomes e semana
		semanas, nomes = [], []
		#d1 é um dicionario
		d1 = OrderedDict()
		d2 = OrderedDict()
		d3 = OrderedDict()
		d4 = OrderedDict()
		d5 = OrderedDict()


		for l in lines:
		
		    line = l.rstrip().split(',')
		
		    if line[0] not in semanas:
				#inseri na lista as semanas
		        semanas.append(line[0])
				
		        d1[line[0]] = {}
		        d2[line[0]] = {}
		        d3[line[0]] = {}
		        d4[line[0]] = {}
		        d5[line[0]] = {}

		    d1[line[0]][line[1]] = line[2] #, line[3], line[4], line[5]
		    d2[line[0]][line[1]] = line[3] #, line[3], line[4], line[5]
		    d3[line[0]][line[1]] = line[4] #, line[3], line[4], line[5]
		    d4[line[0]][line[1]] = line[5] #, line[3], line[4], line[5]
		    d5[line[0]][line[1]] = line[6] #, line[3], line[4], line[5]

		nomes=[]
		#inseri na lista de nomes os nomes
		[nomes.append(k) for v in d1.values() for k in v.keys() if k not in nomes]
		#lista de data
		data = [] 
		data1 = []
		data2 = []
		data3 = []
		data4 = []

		#fazer um for in nomes ou seja cada nome
		for nome in nomes:
		    n = []
		    teaching = []
		    cognitive = []
		    social = []
		    other = []

		    #vou correr as semanas
		    for semana in semanas:
		    	#se o nome tiver na semanas
		    	#vou inserir em uma lista a quantidade de interações
		        if nome in d1[semana]:
		            n.append(d1[semana][nome])
		            cognitive.append(d2[semana][nome])
		            teaching.append(d3[semana][nome])
		            social.append(d4[semana][nome])
		            other.append(d5[semana][nome])
		        else:
		            n.append(0)
		            cognitive.append(0)
		            teaching.append(0)
		            social.append(0)
		            other.append(0)

		    data.append(n)
		    data1.append(cognitive)
		    data2.append(teaching)
		    data3.append(social)
		    data4.append(other)

		# Construção da tabela final
		columns = []
		for i in range(len(semanas)):
		    columns.append('Week'+str(i))
		df = pd.DataFrame(data, index=nomes, columns=columns)

		columns = []
		for i in range(len(semanas)):
			columns.append('Week_cogn_' + str(i))
		df_cognitive = pd.DataFrame(data1, index=nomes, columns=columns)

		columns = []
		for i in range(len(semanas)):
			columns.append('Week_teac_' + str(i))
		df_teaching = pd.DataFrame(data2, index=nomes, columns=columns)

		columns = []
		for i in range(len(semanas)):
			columns.append('Week_soc_' + str(i))
		df_social = pd.DataFrame(data3, index=nomes, columns=columns)

		columns = []
		for i in range(len(semanas)):
			columns.append('Week_other_' + str(i))
		df_other = pd.DataFrame(data4, index=nomes, columns=columns)

		interaction_count = df.copy()
		#df.index.name = 'Nome'
		df.to_csv('logs/tabelaIteracoesSemanaol.csv',index=True,header=True)

		f2 = open('logs/itMedia.csv', mode='r')
		lines = f2.readlines()
		#print(tst)
		del lines[0]


		semanas, nomes = [], []

		d1 = OrderedDict()
		for l in lines:
		    line = l.rstrip().split(',')
		    if line[0] not in semanas:
		        semanas.append(line[0])
		        d1[line[0]] = {}
		    d1[line[0]][line[1]] = line[2]

		nomes=[]
		[nomes.append(k) for v in d1.values() for k in v.keys() if k not in nomes]

		data = [] 
		for nome in nomes:
		    n = []
		    for semana in semanas:
		        if nome in d1[semana]:
		            n.append(d1[semana][nome])
		        else:
		            n.append(0)
		    data.append(n) 
		# Construção da tabela final
		columns = []
		for i in range(len(semanas)):
		    columns.append('MediaW'+str(i))    
		df2 = pd.DataFrame(data, index=nomes, columns=columns)
		avg_interaction = df2.copy()
		#df.index.name = 'Nome'
		df2.to_csv('logs/tabelaMedia.csv',index=True,header=True)


		# Lendo o arquivo para uma lista de linhas
		f1 = open('logs/itMediana.csv', mode='r')
		lines = f1.readlines()
		#print(tst)
		del lines[0]


		semanas, nomes = [], []

		d1 = OrderedDict()
		for l in lines:
		    line = l.rstrip().split(',')
		    if line[0] not in semanas:
		        semanas.append(line[0])
		        d1[line[0]] = {}
		    d1[line[0]][line[1]] = line[2]

		nomes=[]
		[nomes.append(k) for v in d1.values() for k in v.keys() if k not in nomes]

		data = [] 
		for nome in nomes:
		    n = []
		    for semana in semanas:
		        if nome in d1[semana]:
		            n.append(d1[semana][nome])
		        else:
		            n.append(0)
		    data.append(n) 
		# Construção da tabela final
		columns = []
		for i in range(len(semanas)):
		    columns.append('MedianaW'+str(i))    
		df1 = pd.DataFrame(data, index=nomes, columns=columns)
		median_interaction = df1.copy()
		#df.index.name = 'Nome'
		df1.to_csv('logs/tabelaMediana.csv',index=True,header=True)

		# Implementado por mim
		columns = []
		for i in range(len(semanas)):
			columns.append('Week' + str(i))

		weekly_total = pd.DataFrame(0, index=['0'], columns=columns)
		zero_interaction = pd.DataFrame(0, index=nomes, columns=columns)

		for student, interaction in interaction_count.iterrows():

			total_zero_interaction = 0

			for idx, item in enumerate(interaction):
				if int(item) == 0:
					total_zero_interaction += 1

				weekly_total[columns[idx]] = int(weekly_total[columns[idx]]) + int(interaction[columns[idx]])
				zero_interaction.loc[student, columns[idx]] += total_zero_interaction

		columns = []

		# for i in range(len(semanas)):
		for i in range(number_of_weeks + 1):
			columns.append('Week_' + str(i))
			columns.append('Week_avg_' + str(i))
			columns.append('Week_median_' + str(i))
			columns.append('Week_zero_' + str(i))
			columns.append('Week_diff_' + str(i))
			columns.append('Week_commit_' + str(i))
			columns.append('Week_cogn_' + str(i))
			columns.append('Week_teac_' + str(i))
			columns.append('Week_soc_' + str(i))
			columns.append('Week_other_' + str(i))

		quest_data = pd.read_csv('data/' + semester + '/quest.csv')
		quest_data = quest_data.set_index('Nome completo')
		quest_student_names = quest_data.index.values

		if QUEST_DATA:
			quest_data_columns = quest_data.columns.to_series()

			_final_dataframe = pd.DataFrame(index=quest_student_names, columns=columns)
			final_dataframe = quest_data.join(_final_dataframe)
			final_dataframe = final_dataframe.fillna(0)
		else:
			final_dataframe = pd.DataFrame(0, index=quest_student_names, columns=columns)
			# final_dataframe = final_dataframe.set_index('Nome completo')

		# final_dataframe.index.name = 'Nome completo'

		grades = pd.read_csv('data/' + semester + '/notas.csv', delimiter=",")
		for i in range(len(grades)):
			grades['Nome'][i] = grades['Nome'][i] + ' ' + grades['Sobrenome'][i]

		for index, row in grades.iterrows():
			if str(row['Total do curso (Real)']) == '-':
				grades.drop(index, inplace=True)

		# if 'Bruna do Nascimento Goulart (16102435)' in final_dataframe.index and QUEST_DATA:
		# 	_student = final_dataframe.loc['Bruna do Nascimento Goulart (16102435)', :]
		# 	student = _student.iloc[0, :]
		#
		# 	final_dataframe = final_dataframe.drop(['Bruna do Nascimento Goulart (16102435)'])
		# 	final_dataframe.loc['Bruna do Nascimento Goulart (16102435)'] = student

		# if 'Bruna do Nascimento Goulart (16102435)' in final_dataframe.index:
		# 	final_dataframe = final_dataframe.drop(['Bruna do Nascimento Goulart (16102435)'])

		if 'Alisson Nasario Januario (16207568)' in final_dataframe.index:
			final_dataframe = final_dataframe.drop(['Alisson Nasario Januario (16207568)'])

		for student, interaction in final_dataframe.iterrows():
			# print(str(student))
			for i in range(number_of_weeks + 1):
				final_dataframe.loc[student, 'Week_' + str(i)] = interaction_count.loc[student, 'Week' + str(i)]
				final_dataframe.loc[student, 'Week_avg_' + str(i)] = avg_interaction.loc[student, 'MediaW' + str(i)]
				final_dataframe.loc[student, 'Week_median_' + str(i)] = median_interaction.loc[student, 'MedianaW' + str(i)]
				final_dataframe.loc[student, 'Week_zero_' + str(i)] = zero_interaction.loc[student, 'Week' + str(i)]
				final_dataframe.loc[student, 'Week_commit_' + str(i)] = int(interaction_count.loc[student, 'Week' + str(i)]) / int(weekly_total['Week' + str(i)])

				final_dataframe.loc[student, 'Week_cogn_' + str(i)] = int(float(df_cognitive.loc[student, 'Week_cogn_' + str(i)]))
				final_dataframe.loc[student, 'Week_teac_' + str(i)] = int(float(df_teaching.loc[student, 'Week_teac_' + str(i)]))
				final_dataframe.loc[student, 'Week_soc_' + str(i)] = int(float(df_social.loc[student, 'Week_soc_' + str(i)]))
				final_dataframe.loc[student, 'Week_other_' + str(i)] = int(float(df_other.loc[student, 'Week_other_' + str(i)]))

				iter_count = final_dataframe.loc[student, 'Week_' + str(i)]
				iter_count_type = final_dataframe.loc[student, 'Week_cogn_' + str(i)] + final_dataframe.loc[student, 'Week_teac_' + str(i)] \
								  + final_dataframe.loc[student, 'Week_soc_' + str(i)] + final_dataframe.loc[student, 'Week_other_' + str(i)]

				if int(iter_count) != int(iter_count_type):
					raise ValueError('As quantidades de interações não batem')

				# if i != 18:
				if i != number_of_weeks:
					final_dataframe.loc[student, 'Week_diff_' + str(i)] = (int(interaction_count.loc[student, 'Week' + str(i + 1)]) - int(interaction_count.loc[student, 'Week' + str(i)])) / 2

			mask = (grades['Nome'] == student)
			result = grades.loc[mask, 'Total do curso (Real)']

			if result.empty:
				# final_dataframe.loc[student, 'final_result'] = '-'
				final_dataframe.drop(student, inplace=True)
			else:
				if float(result) >= 6:
					final_dataframe.loc[student, 'final_result'] = 'Aprovado'
				else:
					final_dataframe.loc[student, 'final_result'] = 'Reprovado'

		student_names = final_dataframe.index.values

		for name in student_names:
			if name not in quest_student_names:
				print('Removing: ' + name)
				final_dataframe.drop(name, inplace=True)

		b1_columns = []
		b2_columns = []
		b3_columns = []
		b4_columns = []
		b5_columns = []

		b6_columns = []
		b7_columns = []
		b8_columns = []
		b9_columns = []
		b10_columns = []
		b11_columns = []
		b12_columns = []

		if QUEST_DATA:
			for item in quest_data_columns:
				b1_columns.append(item)
				b2_columns.append(item)
				b3_columns.append(item)
				b4_columns.append(item)
				b5_columns.append(item)
				b6_columns.append(item)
				b7_columns.append(item)
				b8_columns.append(item)
				b9_columns.append(item)
				b10_columns.append(item)
				b11_columns.append(item)
				b12_columns.append(item)

		for week in range(0, number_of_weeks + 1):
			b1_columns.append('Week_' + str(week))

			b2_columns.append('Week_' + str(week))
			b2_columns.append('Week_avg_' + str(week))
			b2_columns.append('Week_median_' + str(week))

			b3_columns.append('Week_' + str(week))
			b3_columns.append('Week_cogn_' + str(week))
			b3_columns.append('Week_teac_' + str(week))
			b3_columns.append('Week_soc_' + str(week))
			b3_columns.append('Week_other_' + str(week))

			b4_columns.append('Week_' + str(week))
			b4_columns.append('Week_avg_' + str(week))
			b4_columns.append('Week_median_' + str(week))
			b4_columns.append('Week_cogn_' + str(week))
			b4_columns.append('Week_teac_' + str(week))
			b4_columns.append('Week_soc_' + str(week))
			b4_columns.append('Week_other_' + str(week))

			b5_columns.append('Week_' + str(week))
			b5_columns.append('Week_avg_' + str(week))
			b5_columns.append('Week_median_' + str(week))
			b5_columns.append('Week_zero_' + str(week))
			b5_columns.append('Week_commit_' + str(week))
			b5_columns.append('Week_diff_' + str(week))

			# Atributos do artigo do Swan
			b6_columns.append('Week_cogn_' + str(week))
			b6_columns.append('Week_teac_' + str(week))
			b6_columns.append('Week_soc_' + str(week))

			b7_columns.append('Week_cogn_' + str(week))
			b7_columns.append('Week_teac_' + str(week))

			b8_columns.append('Week_cogn_' + str(week))
			b8_columns.append('Week_soc_' + str(week))

			b9_columns.append('Week_teac_' + str(week))
			b9_columns.append('Week_soc_' + str(week))

			b10_columns.append('Week_cogn_' + str(week))

			b11_columns.append('Week_teac_' + str(week))

			b12_columns.append('Week_soc_' + str(week))


		b1_columns.append('final_result')
		b2_columns.append('final_result')
		b3_columns.append('final_result')
		b4_columns.append('final_result')
		b5_columns.append('final_result')

		b6_columns.append('final_result')
		b7_columns.append('final_result')
		b8_columns.append('final_result')
		b9_columns.append('final_result')
		b10_columns.append('final_result')
		b11_columns.append('final_result')
		b12_columns.append('final_result')

		if QUEST_DATA:
			final_dataframe.to_csv('data/output/' + semester + '/bd1_quest_data.csv', columns=b1_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd2_quest_data.csv', columns=b2_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd3_quest_data.csv', columns=b3_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd4_quest_data.csv', columns=b4_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd5_quest_data.csv', columns=b5_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd6_quest_data.csv', columns=b6_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd7_quest_data.csv', columns=b7_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd8_quest_data.csv', columns=b8_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd9_quest_data.csv', columns=b9_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd10_quest_data.csv', columns=b10_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd11_quest_data.csv', columns=b11_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd12_quest_data.csv', columns=b12_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd13_quest_data.csv', index=True, index_label='Nome completo')
		else:
			final_dataframe.to_csv('data/output/' + semester + '/bd1.csv', columns=b1_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd2.csv', columns=b2_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd3.csv', columns=b3_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd4.csv', columns=b4_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd5.csv', columns=b5_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd6.csv', columns=b6_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd7.csv', columns=b7_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd8.csv', columns=b8_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd9.csv', columns=b9_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd10.csv', columns=b10_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd11.csv', columns=b11_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd12.csv', columns=b12_columns, index=True, index_label='Nome completo')
			final_dataframe.to_csv('data/output/' + semester + '/bd13.csv', index=True, index_label='Nome completo')
