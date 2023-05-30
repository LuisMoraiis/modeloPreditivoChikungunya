import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

dataframe = pd.read_csv('chikungunya.csv')
                                                                                                            
dataframe = dataframe.dropna(how='all', axis=1)

sintomasParaInternacao = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE',
                                   'ARTRALGIA', 'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO']

df = dataframe.dropna(subset=sintomasParaInternacao )
df = df.dropna(subset=['HOSPITALIZ'])

df['HOSPITALIZ'] = pd.to_numeric(df['HOSPITALIZ'], errors='coerce')  
df = df.drop(df[df['HOSPITALIZ'] == 9.0].index)  


x = df[sintomasParaInternacao]
y = df['HOSPITALIZ']

x , y = RandomUnderSampler(random_state = 42).fit_resample(x, y)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size= 0.3, random_state= 42)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(treino_x, treino_y)

previsao = modelo.predict(teste_x)


print(classification_report(teste_y, previsao))
print(confusion_matrix(teste_y, previsao))
print(df['HOSPITALIZ'].value_counts())

