"""
Autores: Gabriel Fran�a, Guilherme Kenji e Pedro Minante 

Rede Multilayer Perceptron para a aproxima��o funcional.

Fun��o: cos(x) * cos(2x)
"""
# Importa��o de bibliotecas, ajustes de exibi��o e a defini��o de diret�rio
import PySimpleGUI as sg
import numpy as np
import random as rd
import matplotlib.pyplot as plt

print("\x1b[23x1b[1;1H")

# Declara��o de vari�veis
entradas = 1  # quantidade de entradas
neur = 200  # quantidade de neur�nios na camada intermedi�ria
limiar = 0.0  # valor para a limiariza��o
alfa = 0.005  # taxa de aprendizagem
# at� quando o treino deve ocorrer (erro aceito para um bom treinamento)
errotolerado = 0.25
listaciclo = []  # lista de ciclos do treinamento
listaerro = []  # lista dos erros do treinamento
xmin = -1
xmax = 1
npontos = 50

# -------------------------------TELA-------------------------------


class Tela:
    def __init__(self):
        sg.theme('Reddit')
        # Gerando o layout da tela
        layout = [
            [sg.Text('Limite inferior do domonio da funcao: '),
             sg.Input(size=(15, 0), key='xmin')],
            [sg.Text('Limite superior do  domonio da funcao: '),
             sg.Input(size=(15, 0), key='xmax')],
            [sg.Text('Numero de amostras para a funcao: '),
             sg.Input(size=(15, 0), key='npontos')],
            [sg.Text('Taxa de aprendizagem: '),
             sg.Input(size=(15, 0),  key='alfa')],
            [sg.Text('Quantidade de neuronios: '),
             sg.Input(size=(15, 0), key='neur')],
            [sg.Button('Executar aproximacao', key='Executar aproximacao')]
        ]
        # Gerando a janela da tela
        self.janela = sg.Window('MLP para domonio de funcões').layout(layout)

    # Ler layout da tela
    def Iniciar(self):
        while True:
            event, self.values = self.janela.Read()
            if event == sg.WINDOW_CLOSED:
                break
            if event == 'Executar aproximacao':
                xmin = (int((self.values['xmin'])))
                xmax = (int((self.values['xmax'])))
                npontos = (int((self.values['npontos'])))
                alfa = (float((self.values['alfa'])))
                npontos = (int((self.values['neur'])))
                # -------------------------------REDE MLP-------------------------------

                # Gerando o arquivo de entradas
                x1 = np.linspace(xmin, xmax, npontos)
                x = np.zeros((npontos, 1))
                for i in range(npontos):
                    x[i][0] = x1[i]
                (amostras, vsai) = np.shape(x)

                t1 = (np.cos(x))*(np.cos(2*x))
                t = np.zeros((1, amostras))
                for i in range(amostras):
                    t[0][i] = t1[i]
                (vsai, amostras) = np.shape(t)

                # Gerando os pesos sin�pticos aleatoriamente (entrada - intermedi�ria)
                vanterior = np.zeros((entradas, neur))
                aleatorio = 0.5
                for i in range(entradas):
                    for j in range(neur):
                        vanterior[i][j] = rd.uniform(-aleatorio, aleatorio)
                v0anterior = np.zeros((1, neur))
                for j in range(neur):
                    v0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

                # Gerando os pesos sin�pticos aleatoriamente (intermedi�ria - sa�da)
                wanterior = np.zeros((neur, vsai))
                aleatorio = 0.5
                for i in range(neur):
                    for j in range(vsai):
                        wanterior[i][j] = rd.uniform(-aleatorio, aleatorio)
                w0anterior = np.zeros((1, vsai))
                for j in range(vsai):
                    w0anterior[0][j] = rd.uniform(-aleatorio, aleatorio)

                # Gerando matrizes de atualiza��o de pesos e valores de sa�da da rede
                vnovo = np.zeros((entradas, neur))
                v0novo = np.zeros((1, neur))
                wnovo = np.zeros((neur, vsai))
                w0novo = np.zeros((1, vsai))
                zin = np.zeros((1, neur))
                z = np.zeros((1, neur))
                deltinhak = np.zeros((vsai, 1))
                deltaw0 = np.zeros((vsai, 1))
                deltinha = np.zeros((1, neur))
                xaux = np.zeros((1, entradas))
                h = np.zeros((vsai, 1))
                target = np.zeros((vsai, 1))
                deltinha2 = np.zeros((neur, 1))
                ciclo = 0
                errototal = 100000

                # Implementa��o da MLP

                while errotolerado < errototal:
                    errototal = 0
                    for padrao in range(amostras):
                        for j in range(neur):
                            # C�lculo dos valores (entrada - intermedi�ria)
                            zin[0][j] = np.dot(
                                x[padrao, :], vanterior[:, j])+v0anterior[0][j]
                        z = np.tanh(zin)
                        # C�lculo dos valores (intermedi�ria - sa�da)
                        yin = np.dot(z, wanterior)+w0anterior
                        y = np.tanh(yin)

                        # Obtendo a transposta de y
                        for m in range(vsai):
                            h[m][0] = y[0][m]
                        for m in range(vsai):
                            target[m][0] = t[0][padrao]

                        # C�lculo do erro total
                        errototal = errototal+np.sum(0.5*((target-h)**2))

                        # Obtendo matrizes para a atualiza��o dos pesos
                        deltinhak = (target-h)*(1+h)*(1-h)
                        deltaw = alfa*(np.dot(deltinhak, z))
                        deltaw0 = alfa*deltinhak
                        deltinhain = np.dot(np.transpose(
                            deltinhak), np.transpose(wanterior))
                        deltinha = deltinhain*(1+z)*(1-z)

                        # Obtendo a transposta de deltinha
                        for m in range(neur):
                            deltinha2[m][0] = deltinha[0][m]
                        for k in range(entradas):
                            xaux[0][k] = x[padrao][k]
                        deltav = alfa*np.dot(deltinha2, xaux)
                        deltav0 = alfa*deltinha

                        # Realizando as atualiza��es de peso
                        vnovo = vanterior+np.transpose(deltav)
                        v0novo = v0anterior+np.transpose(deltav0)
                        wnovo = wanterior+np.transpose(deltaw)
                        w0novo = w0anterior+np.transpose(deltaw0)
                        vanterior = vnovo
                        v0anterior = v0novo
                        wanterior = wnovo
                        w0anterior = w0novo
                    ciclo = ciclo + 1
                    listaciclo.append(ciclo)
                    listaerro.append(errototal)
                    print('Ciclo\t Erro')
                    print(ciclo, '\t', errototal)

                    # Verificando amostras para os pesos de cada ciclo e gerando sa�das aproximadas
                    zin2 = np.zeros((1, neur))
                    z2 = np.zeros((1, neur))
                    t2 = np.zeros((amostras, 1))
                    for i in range(amostras):
                        for j in range(neur):
                            zin2[0][j] = np.dot(
                                x[i, :], vanterior[:, j])+v0anterior[0][j]
                            z2 = np.tanh(zin2)
                        yin2 = np.dot(z2, wanterior)+w0anterior
                        y2 = np.tanh(yin2)
                        t2[i][0] = y2

                    plt.plot(x, t1, color='red')
                    plt.plot(x, t2, color='blue')

                plt.show()


tela = Tela()
tela.Iniciar()
