# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:44:09 2023

@author: jgall
"""

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
# Configuración matplotlib

arrays=np.load("min_max_kmeans.npz") #Cargo los minimos y maximos analizados en KMEANS_TRABAJO_1.py
ymax=arrays['arr_0']
ymin=arrays['arr_1']
umax=arrays['arr_2']
umin=arrays['arr_3']
emax=arrays['arr_4']
emin=arrays['arr_5']

# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

# Datos
# ==============================================================================

datosprueba= pd.ExcelFile('C:/Users/jgall/OneDrive/Documentos/Control Inteligente/Inteligente_2/parafallos.xlsx').parse(sheet_name='datosPrueba')

BD=datosprueba.values.T #se almacena la base de datos cargada en la variable BD y transpuesta

#print(len(BD[1])) codigo para conocer la longitud de la columna 1 de la BD cargada
#print (datosprueba)

t=BD[0]#Tiempo columna 1
y=BD[1]#Exitacion Columna 2
u=BD[2]#Salida columna 3
e=BD[3]#Error columna 4
p=BD[4]#perturbacion columna 5

'''
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, per, color='black')
plt.show()''' #Funcion para presentar graficos delas variables antes cargadas

t=BD[0]#Tiempo columna 1
y=BD[1]#Exitacion Columna 2
u=BD[2]#Salida columna 3
e=BD[3]#Error columna 4
p=BD[4]#perturbacion columna 5

'''
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, per, color='black')
plt.show()''' #Funcion para presentar graficos delas variables antes cargadas

tmin=min(t)
tmax=max(t)
ymax=max(y)
ymin=min(y)
umax=max(u)
umin=min(u)
emax=max(e)
emin=min(e) #Funciones para determinar los maximos y minimos de cada variable para luego normalizar

ny=[] #inicio un vector ny vacio
nu=[] #inicio un vector nu vacio
ne=[] #inicio un vector ne vacio
nt=[]

#normalizar informacion
for i in range(len(t)):

    normy= (y[i]-ymin)/(ymax-ymin)
    ny.append(normy)# funcion para que cada que calcule el dato de normy sea almacenado en ny en la misma posicion

    normu = (u[i] - umin) / (umax - umin)
    nu.append(normu)# funcion para que cada que calcule el dato de normu sea almacenado en ny en la misma posicion

    norme = (e[i] - emin) / (emax - emin)
    ne.append(norme)# funcion para que cada que calcule el dato de norme sea almacenado en ny en la misma posicion

    normt = (t[i] -tmin)/ (tmax - tmin)
    nt.append(normt)
    
infonorm=[ny, nu,nt] #almaceno las variables normalizadas en infonorm
#print(len(infonorm))
infonorm1=np.transpose(infonorm)#transponer los datos para volverlos columnas
#print (infonorm1)

###########################################################################
# Cargar el modelo desde el archivo
bosques_cargado = joblib.load('modelo_random_forest4.pkl')



# Clasificar los nuevos datos
prediccionesN= bosques_cargado.predict(infonorm1)

#plt. plot (nt,color='green')
#plt. plot (ny,color='blue')
#plt. plot (nu,color='pink')
#plt. plot(prediccionesN,color='red')

#kmeans2 = kmeans1.predict(infonorm1) #comando para evaluar nuevos datos
#print (kmeans2)
#plt.plot(t, kmeans2, color='purple')
#plt.plot(t, u, color='blue')
#plt.plot(t, e, color='green')
#plt.show()

# Cree una nueva subimagen, cuadrícula 2x1, número de serie 1, el primer número es el número de filas, el segundo número es el número de columnas, que indica la disposición de las subimágenes, y el tercer número es el número de serie de las subimágenes
plt.subplot(2, 1, 1)
plt.plot(t, y, color='red')
plt.plot(t, u, color='blue')
plt.plot(t, e, color='green')
plt.plot(t, p, color='orange')

plt.ylabel("DATOS DE PROCESO NORMALIZADOS")
# Establecer título de subimagen
plt.title("Proceso Real")
# Cree un nuevo subgrafo, cuadrícula 2x1, número de secuencia 2
plt.subplot(2, 1, 2)
plt.plot(t, prediccionesN, color='purple')
# Establecer título de subimagen
plt.title("Clasificador Entrenado")
plt.ylabel("CLASES ANALIZADAS")
plt.xlabel("TIEMPO")
# Establecer título
plt.suptitle("Proceso y Clasificador entrenado")

# Mostrar
plt.show()