import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from sklearn.preprocessing import OneHotEncoder

dados = pd.read_csv("C:/Users/Gustavo/Documents/MeusProjetos/Machine-Learning/Atividade1/Stars.csv")
colunas = dados.columns

#verifica se há células vázias no dataframe dados 
null = pd.isnull(dados)
if True in null.values:
    print("Há células vázias no dataframe")


#preparando a base de dados
dados["Color"] = dados["Color"].str.lower()
dados["Color"] = dados["Color"].str.replace('-', " ")

categoriesColor = list(dados["Color"].unique())
categoriesSpec = list(dados["Spectral_Class"].unique())

encoder = OneHotEncoder(categories= [categoriesColor])
encoder.fit_transform(dados["Color"].values.reshape(-1,1))
buffer = encoder.transform(dados["Color"].values.reshape(-1,1)).toarray()
dados = dados.join(pd.DataFrame(data = buffer, columns = categoriesColor))
dados = dados.drop(columns= ['Color'])

encoder = OneHotEncoder(categories= [categoriesSpec])
encoder.fit_transform(dados["Spectral_Class"].values.reshape(-1,1))
buffer = encoder.transform(dados["Spectral_Class"].values.reshape(-1,1)).toarray()
dados = dados.join(pd.DataFrame(data = buffer, columns = categoriesSpec))
dados = dados.drop(columns= ['Spectral_Class'])

def histograma(dado, title, bin = 50):
    plt.hist(dado, bins = bin)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel('Frequência')
    plt.show()

#histograma(dados["Temperature"], "Temperatura") #não é normal
#histograma(dados["L"], "L") #não é normal
#histograma(dados["R"], "R") #não é normal
histograma(dados["A_M"], "A_M") #é normal

'''
#teste se há dados inconsistentes 
for element in colunas:
    plt.hist(dados[element], bins = 20)
    plt.title(element)
    plt.show()'''

