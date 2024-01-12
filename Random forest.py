# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:15:31 2023

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
# Configuración matplotlib
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

datosprueba= pd.ExcelFile('C:/Users/jgall/OneDrive/Documentos/Control Inteligente/Inteligente_2/parafallos.xlsx').parse(sheet_name='SVM')

BD=datosprueba.values.T #se almacena la base de datos cargada en la variable BD y transpuesta

#print(len(BD[1])) codigo para conocer la longitud de la columna 1 de la BD cargada
#print (datosprueba)

t=BD[0]#Tiempo columna 1
y=BD[1]#Exitacion Columna 2
u=BD[2]#Salida columna 3
e=BD[3]#Error columna 4
#per=BD[4]#perturbacion columna 5

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
#per=BD[4]#perturbacion columna 5

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
bosques = RandomForestClassifier(n_estimators=50,
                                 criterion = "gini",
                                 max_features= "sqrt",
                                 bootstrap= True,
                                 max_samples=2/3,
                                 oob_score=True)
bosques.fit(infonorm1,e)

predicciones=bosques.predict(infonorm1)
#predicciones

print(bosques.score(infonorm1,e))
print(bosques.oob_score_)

accuracy = accuracy_score(
            y_true    = e,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy}%")

plt. plot (e,color='green')
plt. plot(predicciones,color='red')


# Arboles de decision 
from sklearn.tree import export_text

# Visualizar los primeros cinco árboles
#for i in range(5):
    #tree_i = bosques.estimators_[i]
    #tree_rules = export_text(tree_i, feature_names=['Exitacion', 'Salida', 'Tiempo'])
    #print(f"Árbol {i + 1}:\n{tree_rules}\n")

#################################################################################
import joblib
# Guardar el modelo en un archivo
joblib.dump(bosques, 'modelo_random_forest4.pkl')

# Cargar el modelo desde el archivo
#bosques_cargado = joblib.load('modelo_random_forest.pkl')



