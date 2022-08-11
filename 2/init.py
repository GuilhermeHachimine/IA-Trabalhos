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



