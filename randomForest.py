import pandas as pd
from imblearn.under_sampling import RandomUnderSampler #Sem balanceamento
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def chikungunyaBinaria(chykungunya):
    if chykungunya == 13.0:
        return 1
    else:
        return 0
    
dataframe = pd.read_csv('chikungunya.csv')
dataframe = dataframe.dropna(how= 'all', axis= 1)

sintomasParaInternacao = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE',
                                   'ARTRALGIA', 'PETEQUIA_N', 'DOR_RETRO']

df = dataframe.dropna(subset= sintomasParaInternacao)
df = df.dropna(subset= ['CLASSI_FIN'])

df['CLASSI_FIN'] = df['CLASSI_FIN'].apply(chikungunyaBinaria)

x = df[sintomasParaInternacao]
y = df['CLASSI_FIN']

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size= 0.3, random_state= 42)

modelo = RandomForestClassifier(n_estimators= 100, max_depth= None, random_state= 42)

modelo.fit(treino_x, treino_y)

previsao = modelo.predict(teste_x)

print('Modelo: RandomForest')
print('Previsão paciente com chikungunya:')
print(classification_report(teste_y, previsao))