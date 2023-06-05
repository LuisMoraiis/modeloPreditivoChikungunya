import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.tree import DecisionTreeClassifier

def chikungunyaBinaria(chikungunya):
    if chikungunya == 13.0:
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

x, y = RandomUnderSampler(random_state= 42).fit_resample(x, y)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size= 0.3, random_state= 42)

modelo = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 3, 4, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': [None, 'sqrt', 'log2']
}

grid = GridSearchCV(modelo, param_grid, cv= 5, scoring= 'accuracy', n_jobs=-1)
grid.fit(treino_x, treino_y)

bestScore = grid.best_score_
bestParams = grid.best_params_


print('\nModelo: DecisionTree')
print(f"Melhor score: {bestScore}")
print(f"Melhores parâmetros: {bestParams}")
modelo = DecisionTreeClassifier(**bestParams)

modelo.fit(treino_x, treino_y)

previsao = modelo.predict(teste_x)

print('Previsão paciente com chikungunya:')
print(classification_report(teste_y, previsao))
