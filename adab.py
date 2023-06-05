import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier

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

modelo = AdaBoostClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid = GridSearchCV(modelo, param_grid, cv= 3, scoring= 'accuracy')

grid.fit(treino_x, treino_y)

print(f"\nMelhor score: {grid.best_score_}")
print(f"Melhores parâmetros: {grid.best_params_}")

modelo = AdaBoostClassifier(**grid.best_params_)

modelo.fit(treino_x, treino_y)

previsao = modelo.predict(teste_x)

print('\nModelo: Adaboost')
print('Previsão paciente com chikungunya:')
print(classification_report(teste_y, previsao))