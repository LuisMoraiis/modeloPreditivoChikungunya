import input_user
import regressao_logistica
import pandas as pd
import numpy as np

input_user.cabecalho()

sintomas_apresentados = {}

input_user.questionario(sintomas_apresentados)

print(sintomas_apresentados)

sintomas_df = pd.DataFrame([sintomas_apresentados], columns=regressao_logistica.sintomasParaInternacao)


print(sintomas_df)

previsao = regressao_logistica.modelo.predict(sintomas_df)

print("previsao:",previsao[0])
