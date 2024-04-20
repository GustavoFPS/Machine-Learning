import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sys
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


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

go = False
if go == True:
    histograma(dados["Temperature"], "Temperatura") #não é normal
    histograma(dados["L"], "L") #não é normal
    histograma(dados["R"], "R") #não é normal
    histograma(dados["A_M"], "A_M") #é normal

scaler1 = MinMaxScaler()
scaler2 = StandardScaler()

dados['Temperature'] = scaler1.fit_transform(dados[['Temperature']])
dados["L"] = scaler1.fit_transform(dados[["L"]])
dados["R"] = scaler1.fit_transform(dados[["R"]])

dados["A_M"] = scaler2.fit_transform(dados[["A_M"]])

pca = PCA()

pca.fit(dados)
expl = pca.explained_variance_ratio_
svalues = pca.singular_values_
dados_reduzidos = pca.transform(dados)

explainability = pca.explained_variance_ratio_.cumsum()
factors = np.arange(1,dados.shape[1]+1,1)
plt.scatter(factors,explainability)
plt.hlines(0.9,0,20,'r')
plt.xlabel('Número de componentes')
plt.ylabel('Explicabilidade dos dados')
plt.show()
p = factors[explainability<0.9].max()+1

if p<2:
    p=2
    
print('90%% dos dados são explicados com as ' + str(p) + ' componentes.')
pca = PCA()

pca.fit(dados)
expl = pca.explained_variance_ratio_
svalues = pca.singular_values_
dados_reduzidos = pca.transform(dados)

explainability = pca.explained_variance_ratio_.cumsum()
factors = np.arange(1,dados.shape[1]+1,1)
'''
plt.scatter(factors,explainability)
plt.hlines(0.9,0,20,'r')
plt.xlabel('Número de componentes')
plt.ylabel('Explicabilidade dos dados')
plt.show()
p = factors[explainability<0.9].max()+1'''

if p<2:
    p=2
    
print('90%% dos dados são explicados com as ' + str(p) + ' componentes.')

pca = PCA(n_components=p)
pca.fit(dados)
dados_reduzidos = pca.transform(dados)

##
'''
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5).fit(dados_reduzidos)
categorias = kmeans.labels_

plt.scatter(dados_reduzidos[:,0], dados_reduzidos[:,1], c=categorias)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()'''

from sklearn.cluster import DBSCAN 
import ipywidgets as widgets 


@widgets.interact(epsilon=(1, 10, 0.1), minN=(1,400))

def dbscan(epsilon = 1.5, minN = 10):
    #X = np.array([dados['mean_ElectronAffinity'], dados['critical_temp']]).T
    dbscan = DBSCAN(eps=epsilon, min_samples=minN).fit(dados_reduzidos)
    categorias = dbscan.labels_
    plt.scatter(dados_reduzidos[:,0], dados_reduzidos[:,1], c=categorias)
    plt.xlabel('Afinidade eletrônica')
    plt.ylabel('Temperatura crítica')
    plt.text(-2,1.5,str((categorias == -1).sum())+' Outliers')
    plt.text(-2,1,str(categorias.max())+' Agrupamentos')
    plt.show()

