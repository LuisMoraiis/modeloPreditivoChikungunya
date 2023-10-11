import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

def chikungunyaBinaria(chikungunya):
    if chikungunya == 13.0:
        return 1
    else:
        return 0 
    
dataframe = pd.read_csv('/Users/luismorais/Documents/modeloPreditivoChikungunya/chikungunya.csv')
                                                                                                            
dataframe = dataframe.dropna(how='all', axis=1)


sintomasParaInternacao = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE',
                                   'ARTRALGIA', 'PETEQUIA_N', 'DOR_RETRO']

df = dataframe.dropna(subset=sintomasParaInternacao )
df = df.dropna(subset=['CLASSI_FIN'])

df['CLASSI_FIN'] = df['CLASSI_FIN'].apply(chikungunyaBinaria)


X = df[sintomasParaInternacao]
y = df['CLASSI_FIN']

X , y = RandomUnderSampler(random_state = 42).fit_resample(X, y)

treino_X, teste_X, treino_y, teste_y = train_test_split(X, y, test_size= 0.3, random_state= 42)

logistic_regression = LogisticRegression()

param_grid = {
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'C': [0.1, 1, 10],
    'fit_intercept': [True, False],
    'max_iter': [100, 200, 300],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(logistic_regression, param_grid, cv= 5, scoring= 'accuracy', n_jobs= -1)
grid.fit(treino_X, treino_y)

bestScore = grid.best_score_
bestParams = grid.best_params_

modelo = LogisticRegression(**bestParams)
modelo.fit(treino_X, treino_y)

