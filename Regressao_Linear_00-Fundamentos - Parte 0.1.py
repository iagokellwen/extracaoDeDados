#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Autor: Ricardo Roberto de Lima - Profº - Unipê - Centro Universitário de João Pessoa
# Ciência de Dados com Python + Pandas, Numpy, SkLearning.
# Machine Learning - Aula 01

#Importando a biblioteca pandas para carregar e visualizar a base de dados
import pandas as pd


# In[3]:


passageiros = pd.read_csv('Passageiros.csv')


# In[ ]:





# In[4]:


passageiros.head(30)


# In[5]:


#Exibindo as ultimas linhas do dataFrame.
passageiros.tail()


# In[6]:


#Exibindo os dados estatísticos do dataFrame
passageiros.describe()


# In[7]:


#Com as bibliotecas seaborn e matplotlib vamos conseguir gerar gráficos para visualizar a base de dados. 
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


plt.figure(figsize = (16,9))
sns.set(font_scale=2)
sns.lineplot(x="tempo", y="nPassageiros", data=passageiros)


# In[29]:


#Seleciona todas as colunas menos a última
Tempo = passageiros.iloc[:,:-1].values


# In[30]:


nPassageiros = passageiros.iloc[:,1].values


# In[31]:


#Podemos separar parte dos dados para treino e teste
from sklearn.model_selection import train_test_split


# In[32]:


Tempo_treino, Tempo_teste, nPassageiros_treino, nPassageiros_teste = train_test_split(Tempo, nPassageiros, 
                                                                                      test_size = 0.3)


# In[33]:


from sklearn import linear_model


# In[34]:


regressor = linear_model.LinearRegression()


# In[35]:


#Ajustando a reta aos dados de treino
regressor.fit(Tempo_treino,nPassageiros_treino)


# In[36]:


nPassageiros_predito = regressor.predict(Tempo_teste)


# In[37]:


import numpy as np


# In[38]:


#Colocando os dados em um data frame para posteriormente gerar uma figura com o 
# Seaborn
passageiros_predito = pd.DataFrame({'Tempo': np.ndarray.flatten(Tempo_teste),
                                    'nPassageiros': nPassageiros_predito,
             })
passageiros_teste = pd.DataFrame({'Tempo': np.ndarray.flatten(Tempo_teste),
                                    'nPassageiros': nPassageiros_teste,
             })


# In[39]:


plt.figure(figsize = (16,9))
sns.set(font_scale=2)
sns.lineplot(x="Tempo", y="nPassageiros", data = passageiros_teste,  marker='o', label = "Teste")
sns.lineplot(x='Tempo', y='nPassageiros', data = passageiros_predito, label = "Predito" )


# In[40]:


# Coeficientes
print('Coeficiente: \n', regressor.coef_)


# In[42]:


# MSE (mean square error)
print("MSE: %.2f" % np.mean((regressor.predict(Tempo) - nPassageiros) ** 2))

# Score de variação: 1 representa predição perfeita
print('Score de variação: %.2f' % regressor.score(Tempo, nPassageiros))


# In[43]:


# Scatter Plot representando a regressão linear
plt.scatter(Tempo, nPassageiros,  color = 'black')
plt.plot(Tempo, regressor.predict(Tempo), color = 'blue', linewidth = 3)
plt.xlabel('Tempo')
plt.ylabel('N. Passageiros')
plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:




