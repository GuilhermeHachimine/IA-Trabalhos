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

t=np.loadtxt('target7.csv',delimiter=',',skiprows=0)
(numclasses,targets)=np.shape(t)
limiar=0.0
alfa=0.01
errotolerado=0.1
v=np.zeros((entradas,numclasses))
v0=np.zeros((numclasses))
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
          xaux = x[i,:]
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

xteste=x[6,:]
for m2 in range(numclasses):
     soma=0
     for n2 in range(entradas):
          soma=soma+xteste[n2]*v[n2][m2]
          yin[m2]=soma+v0[m2]
print(yin)
for j in range(numclasses):
     if yin[j]>= limiar:
          y[j]=1.0
     else:
          y[j]=-1.0
print(y)

