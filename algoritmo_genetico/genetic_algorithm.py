import numpy as np
import random as rd

tamCromossomo = 30
prob_crossover = 0.95
prob_mutations = 0.1
num_generations = 100
tam_population = 50

# Geração população inicial
p = np.zeros((tam_population, tamCromossomo))

for i in range(tam_population):
    for j in range(tamCromossomo):
        a = rd.uniform(0, 1)
        if (a > 0.5):
            p[i][j] = 1
        else:
            p[i][j] = 0

# Passo 2 - Criação de variaveis
ind = np.zeros((tamCromossomo))
individuo = np.zeros((tam_population))
Aptidao = np.zeros((tam_population))

novageracao = np.zeros((tam_population, tamCromossomo))
geracoes = 0

# Passo3 - Iniciando o algoritimo Genético
while (geracoes <= num_generations):
    novosindividuos = 0
    while (novosindividuos <= (tam_population-1)):
        # Transformando inviduos de binario para real
        for i in range(tam_population):
            ind[:] = p[i, :]
            conversao = 0
            for j in range(tamCromossomo):
                conversao = conversao+ind[j]*(2**(tamCromossomo-(j+1)))
            individuo[i] = (512/(2**tamCromossomo-1))*conversao

        # Calculo da aptidao dos individuos
        TotalAptidao = 0
        for i in range(tam_population):
            Aptidao[i] = abs(
                individuo[i]*np.sin(np.sin(np.sqrt(abs(individuo[i])))))+5
            TotalAptidao = TotalAptidao + Aptidao[i]
        # Seleção dos pais para o cruzamento - roleta
        # Indentificando a probabilidade de cada individuo
        pic = np.zeros(tam_population)
        pitotal = np.zeros(tam_population)

        pic = (1/TotalAptidao)*Aptidao

        # Criando a roleta
        for i in range(tam_population):
            if (i == 0):
                pitotal[i] = pic[i]
            else:
                pitotal[i] = pic[i]+pitotal[i-1]

        # Sorteando os pais de acordo com a probabilidade
        roleta1 = rd.uniform(0, 1)
        i = 0
        while (roleta1 > pitotal[i]):
            i += 1
        pai1 = i

        roleta2 = rd.uniform(0, 1)
        i = 0
        while (roleta2 > pitotal[i]):
            i += 1
        pai2 = i

        while (pai2 == pai1):
            roleta2 = rd.uniform(0, 1)
            i = 0
            while (roleta2 > pitotal[i]):
                i += 1
            pai2 = i

        # Operacao de Cruzamento
        if (prob_crossover > rd.uniform(0, 1)):
            c = round(1+(tamCromossomo-2)*rd.uniform(0, 1))
            gene11 = p[pai1][0:c]
            gene12 = p[pai1][c:tamCromossomo]
            gene21 = p[pai2][0:c]
            gene22 = p[pai2][c:tamCromossomo]
            filho1 = np.concatenate((gene11, gene22), axis=None)
            filho2 = np.concatenate((gene21, gene12), axis=None)

            novageracao[novosindividuos, :] = filho1
            novosindividuos += 1
            novageracao[novosindividuos, :] = filho2
            novosindividuos += 1

        # Operacao de mutacao
        if (prob_mutations > rd.uniform(0, 1)):
            d = round(1+(tamCromossomo-2)*rd.uniform(0, 1))
            if (novageracao[novosindividuos-2][d] == 0):
                novageracao[novosindividuos-2][d] = 1
            else:
                novageracao[novosindividuos-2][d] = 0
            if (novageracao[novosindividuos-1][d] == 0):
                novageracao[novosindividuos-1][d] = 1
            else:
                novageracao[novosindividuos-1][d] = 0

    indice = Aptidao.argmax()
    elem = individuo[indice]
    print(elem)
    p = novageracao
    geracoes += 1
