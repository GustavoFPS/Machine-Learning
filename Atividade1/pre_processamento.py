import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler


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
    mean = np.mean(dado)
    q3 = np.percentile(dado, 75)
    q1 = np.percentile(dado, 25)
    IIQ =  q3 - q1 
    plt.axvline(x=mean, color="red", linestyle="--", label = "Média")
    plt.axvline(x=q3 + 1.5*IIQ, color="g", linestyle="--", label = "Máximo")
    plt.axvline(x=q1 - 1.5*IIQ, color="g", linestyle="--", label = "Mínimo")
    plt.axvline(x= np.percentile(dado, 50), color="black", linestyle="--", label = "Mediana")
    plt.legend()
    plt.show()


histograma(dados["Temperature"], "Temperatura") #não é normal
histograma(dados["L"], "L") #não é normal
histograma(dados["R"], "R") #não é normal
histograma(dados["A_M"], "A_M") #é normal

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()

dados['Temperature'] = scaler.fit_transform(dados[['Temperature']])
dados["L"] = scaler.fit_transform(dados[["L"]])
dados["R"] = scaler.fit_transform(dados[["R"]])

dados["A_M"] = scaler.fit_transform(dados[["A_M"]])


'''
#teste se há dados inconsistentes 
for element in colunas:
    plt.hist(dados[element], bins = 20)
    plt.title(element)
    plt.show()'''

