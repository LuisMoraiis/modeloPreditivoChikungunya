import pandas as pd

def cabecalho():
    print("********************")
    print("* modelo preditivo *")
    print("********************\n")


def questionario(sintomas_apresentados):
    chave = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE',
                                   'ARTRALGIA', 'PETEQUIA_N', 'DOR_RETRO']
    
    print("Responda ao seguinte questionario de acordo com os sintomas apresentados\n")
    print("Responda com 0 ou 1\n 1 - Sintoma presente\n 0 - Sintoma não presente")
    #Descricão de melhoria para a funcão no notion.
    for i in range(len(chave)):
        valor = input(f"{chave[i]}: ")
        while True:
            if(valor == "0" or valor == "1"):
                break
            else:
                print("Entrada invalida. Por favor digite 0 ou 1\n")
                valor = input(f"{chave[i]}: ")
        sintomas_apresentados[chave[i]] = int(valor)
