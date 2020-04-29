import os,sys
from processing_logs import *

if __name__ == '__main__':
	# logs = "logs/logs.csv"
	logs = "data/2017-2/logs.csv"

	#Quantidade de professores 
	#Quantidade de monitores
	#Nome professores
	#Nome monitores

	#n = input(int(n))
	#m = input(int(m))
	professores = ['ZZZZZZZZ', 'Usuário Administrado']
	monitores = ['ZZZZZZZZZZ ZZZZZZZZZZ  ZZZZZZZZZ (11111116)']
	"""
	para sistema online sys.args 
	for i in range(n):
		#Formato Nome 
		#Matheus Francisco Batista Machado
		nome = input()
		professores.append(nome)
 
 	for j in range(m):

		#Formato Nome e ID
		#Matheus Francisco Batista Machado (13232323)
 		nome = input()
 		monitores.append(nome)
	"""


	p = Processing(logs,professores,monitores)
	df = p._logs	
	#print(df.head(5))
	p.preprocessing(df)