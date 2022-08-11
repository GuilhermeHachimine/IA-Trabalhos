import os
import random as rd
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt 

print("\x1b[2)\x1b[1;1H")
print("hello fuckin world!!")
os.chdir(r'C:\Users\Guilherme Hachimine\Desktop\IFTM\IA\Trabalhos\IA-Trabalhos\2')
x=np.loadtxt('x.txt')
(amostras,entradas)=np.shape(x)

t=np.loadtxt('target7.csv',delimiter=';',skiprows=0)
(numclasses,targets)=np.shape(t)
limiar=0.0
alfa=0.01
errotolerado=0.1
for i in range(entradas):
     for j in range(numclasses):
          v[i][j]=rd.uniform(-0.1,0.1)

for j in range(numclasses):
     v0[j]=rd.uniform(-0.1,0.1)


vetor1=[]
vetor2=[]

yin=np.zeros((numclasses,1))
y=np.zeros((numclasses,1))

erro=10
ciclo=0

while erro>errotolerado:
     ciclo+=1
     erro = 0
     for i in range(amostras):
          xaux = x[1,:]
          for m in range(numclasses):
               soma=0
               for n in range(entradas):
                    soma = soma+xaux[n]*v[n][m]
               yin[m]=soma+v0[m]
          for j in range(numclasses):
               if yin[j]>= limiar:
                    y[j]=1.0
               else:
                    y[j]=-1.0

          for j in range(numclasses):
               erro = erro+0.5*((t[j][i] - y[j])**2)
          vanterior=v

          for m in range(entradas):
               for n in range(numclasses):
                    v[m][n]=vanterior[m][n]+alfa*(t[n][i]-y[n])*xaux[m]
          v0anterior=v0
          for j in range(numclasses):
               v0=v0anterior[j]+alfa*(t[j][i]-y[j])
     vetor1.append(ciclo)
     vetor2.append(erro)

     plt.scatter(vetor1, vetor2,marker='*',color='#8cc63f')
     plt.xlabel('ciclo')
     plt.ylabel('erro')
     plt.show()


# class Tela:
#      def __init__(self):
          
#           layout = [
#                [sg.Text('Arquivo de entrada: ', size=(10,0)), 
#                  sg.Input(key='arq_entrada'), sg.FileBrowse('Abrir arquivo', key="entrada")],
#                [sg.Text('Arquivo de target: ', size=(10,0)), 
#                  sg.Input(key='arq_target'), sg.FileBrowse('Abrir arquivo')],
#                [sg.Text('Taxa de aprendizagem: ',size=(10,0)), sg.InputText(key='taxa')],
#                [sg.Radio('Ciclo', "criterio",size=(5,0), key='radio_ciclo', default=True),sg.InputText(key='ciclo')],
#                [sg.Radio('Erro', "criterio",size=(5,0), key='radio_erro'),sg.InputText(key='erro')],
         
#                [sg.Checkbox('', key='cb00'),
#                     sg.Checkbox('', key='cb01'),
#                     sg.Checkbox('', key='cb02'),
#                     sg.Checkbox('', key='cb03'),
#                     sg.Checkbox('', key='cb04')],
#                [sg.Checkbox('', key='cb10'),
#                     sg.Checkbox('', key='cb11'),
#                     sg.Checkbox('', key='cb12'),
#                     sg.Checkbox('', key='cb13'),
#                     sg.Checkbox('', key='cb14')],
#                [sg.Checkbox('', key='cb20'),
#                     sg.Checkbox('', key='cb21'),
#                     sg.Checkbox('', key='cb22'),
#                     sg.Checkbox('', key='cb23'),
#                     sg.Checkbox('', key='cb24')],
#                [sg.Checkbox('', key='cb30'),
#                     sg.Checkbox('', key='cb31'),
#                     sg.Checkbox('', key='cb32'),
#                     sg.Checkbox('', key='cb33'),
#                     sg.Checkbox('', key='cb34')],
#                [sg.Checkbox('', key='cb40'),
#                     sg.Checkbox('', key='cb41'),
#                     sg.Checkbox('', key='cb42'), 
#                     sg.Checkbox('', key='cb43'), 
#                     sg.Checkbox('', key='cb44')],
#                [sg.Checkbox('', key='cb50'),
#                     sg.Checkbox('', key='cb51'),
#                     sg.Checkbox('', key='cb52'),
#                     sg.Checkbox('', key='cb53'),
#                     sg.Checkbox('', key='cb54')],
#                [sg.Checkbox('', key='cb60'), 
#                     sg.Checkbox('', key='cb61'), 
#                     sg.Checkbox('', key='cb62'), 
#                     sg.Checkbox('', key='cb63'), 
#                     sg.Checkbox('', key='cb64')],
#                [sg.Button('Encontrar numero', key="resultado")],
#                [sg.Text('Resultado: '), sg.Text(key='valor_encontrado', size=(5,0))]
            
#         ]
        
#           self.window = sg.Window("Rede Adaline").layout(layout)
#           #dados
        
        
#      def Iniciar(self):
        
#         while True:
#           self.button, self.values= self.window.Read()  
          
#           if self.button == sg.WIN_CLOSED:
#                break
          
#           path_entrada = self.values['arq_entrada']
#           path_target = self.values['arq_target']
          
#           #criterio de parada funcionando
#           if(self.values['radio_ciclo'] == True):  
#                ciclos = self.values['ciclo']
#                ciclos = ciclos.replace(",", ".")
#                criterio = round(int(float(ciclos)))
#           else:
#                erros = self.values['erro']
#                erros = erros.replace(",", ".")
#                criterio = float(erros)
               
#           #recebimento da taxa de aprendizagem
#           if self.values['taxa']:
#                taxa = (self.values['taxa'])
#                taxa = taxa.replace(",", ".")
#                taxa_aprendizagem = (float(taxa))
          
          
#           graf = np.zeros((7,5))
#           vetor_graf = np.zeros([35,1])
#           k = 0
#           for i in range(7):
#                chave = 'cb'+ str(i)
#                for j in range(5):
#                     chave = chave + str(j)
                    
#                     if(self.values[chave] == True):
#                          graf[i][j] = 1
#                     else:
#                          graf[i][j] = -1
#                     vetor_graf[k] = graf[i][j]
#                     k += 1
                    
#                     chave = chave[:-1]
               
#                chave = chave[:-1]
          
          
          
#           if self.button == 'resultado':
#                x=np.loadtxt(path_entrada)
#                (amostras,entradas)=np.shape(x)

#                t=np.loadtxt(path_target,delimiter=';',skiprows=0)
#                (numclasses,targets)=np.shape(t)

#                limiar=0.0
#                alfa= taxa_aprendizagem
#                errotolerado= criterio
#                v = np.zeros((entradas,numclasses))
#                v0 = np.zeros((numclasses))

#                for i in range(entradas):
#                     for j in range(numclasses):
#                          v[i][j] = rd.uniform(-0.1,0.1)

#                for j in range(numclasses):
#                     v0[j] = rd.uniform(-0.1,0.1) 

#                vetor1=[]
#                vetor2=[]

#                yin=np.zeros((numclasses,1))
#                y=np.zeros((numclasses,1))

#                erro = 10
#                ciclo = 0
#                limite_ciclo = criterio
               
#                if(self.values['radio_ciclo'] == True):    
#                     while(ciclo < limite_ciclo):
#                          ciclo += 1
#                          erro = 0
#                          for i in range(amostras):
#                               xaux = x[i,:]
#                               for m in range(numclasses):
#                                    soma = 0
#                                    for n in range(entradas):
#                                         soma += xaux[n] * v[n][m]
#                                    yin[m] = soma + v0[m]

#                               for j in range(numclasses):
#                                    if yin[j] >= limiar:
#                                         y[j] = 1.0
#                                    else:
#                                         y[j] = -1.0
                                   
#                               for j in range(numclasses):
#                                    erro += 0.5*((t[j][i] - y[j])**2)

#                               vanterior = v

#                               for m in range(entradas):
#                                    for n in range(numclasses):
#                                         v[m][n] = vanterior[m][n] + alfa*(t[n][i]-y[n])*xaux[m]

#                               v0anterior = v0
                              
#                               for j in range(numclasses):
#                                    v0[j] = v0anterior[j]+alfa*(t[j][i]-y[j])
                                   
#                          vetor1.append(ciclo)
#                          vetor2.append(erro)
                         
#                     for m2 in range(numclasses):
#                          soma = 0
#                          for n2 in range(entradas):
#                               soma = soma + vetor_graf[n2]*v[n2][m2]
#                               yin[m2] = soma +v0[m2]
                         
#                     for j in range(numclasses):
#                          if yin[j] >= limiar:
#                               y[j] = 1.0
#                          else:
#                               y[j] = -1.0
                      
                      
                    
#                     aux = 0
#                     for i in range(len(y)):
#                          if y[i][0] != -1:
#                               aux = i
#                     self.window.FindElement("valor_encontrado").Update(value=aux+1)
#                     plt.scatter(vetor1, vetor2, marker='*', color= 'red')
#                     plt.xlabel('ciclo')
#                     plt.ylabel('erro')
#                     plt.show()
                       
#                else:
#                     while erro > errotolerado:
#                          ciclo += 1
#                          erro = 0
#                          for i in range(amostras):
#                               xaux = x[i,:]
#                               for m in range(numclasses):
#                                    soma = 0
#                                    for n in range(entradas):
#                                         soma += xaux[n] * v[n][m]
#                                    yin[m] = soma + v0[m]

#                               for j in range(numclasses):
#                                    if yin[j] >= limiar:
#                                         y[j] = 1.0
#                                    else:
#                                         y[j] = -1.0
                                   
#                               for j in range(numclasses):
#                                    erro += 0.5*((t[j][i] - y[j])**2)

#                               vanterior = v

#                               for m in range(entradas):
#                                    for n in range(numclasses):
#                                         v[m][n] = vanterior[m][n] + alfa*(t[n][i]-y[n])*xaux[m]

#                               v0anterior = v0

#                               for j in range(numclasses):
#                                    v0[j] = v0anterior[j]+alfa*(t[j][i]-y[j])
#                          vetor1.append(ciclo)
#                          vetor2.append(erro)

                    
               
#                     for m2 in range(numclasses):
#                          soma = 0
#                          for n2 in range(entradas):
#                               soma = soma + vetor_graf[n2]*v[n2][m2]
#                               yin[m2] = soma +v0[m2]
                         
#                     for j in range(numclasses):
#                          if yin[j] >= limiar:
#                               y[j] = 1.0
#                          else:
#                               y[j] = -1.0
                      
                      
                    
#                     aux = 0
#                     for i in range(len(y)):
#                          if y[i][0] != -1:
#                               aux = i
#                     self.window.FindElement("valor_encontrado").Update(value=aux+1)
#                     plt.scatter(vetor1, vetor2, marker='*', color= 'red')
#                     plt.xlabel('ciclo')
#                     plt.ylabel('erro')
#                     plt.show()
                    
                       
# tela = Tela()
# tela.Iniciar()