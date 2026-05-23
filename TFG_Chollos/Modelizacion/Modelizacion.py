'''
preprocessing.py
====================
Dado el volumen del dataset, se optó por una partición holdout 70/15/15. Sobre el conjunto de entrenamiento se aplicó validación cruzada K-Fold (K=5) mediante GridSearchCV para el ajuste de hiperparámetros de cada modelo de forma independiente. Posteriormente, el conjunto de validación se empleó para la comparación y selección del modelo final entre los candidatos ya optimizados. El conjunto de test permaneció intacto hasta la evaluación final, garantizando una estimación del rendimiento libre de sesgos.
En resumen, hacemos una combinación entre partición train-validation-test y Cross-Validation con 5 folds.

Dado que todos los algoritmos de aprendizaje automático operan sobre espacios vectoriales numéricos, las variables categóricas presentes en el dataset fueron codificadas mediante get_dummies y LabelEncoder

Dependencias:
    - Python >= 3.10
    - pandas >= 3.0.1

Requisitos:
    uv
    uv add seaborn, scikit-learn --active --link-mode=copy

Uso:
    python preprocessing.py --input data_Booking/resultados/resultados_booking_{provincia}.csv  --output data_Booking/final/db_final_{provincia}.parquet
'''

# =============================================================================
# IMPORTS
# =============================================================================
#Librerías estándar (vienen incluidas con Python):
import ast
import json
import math

#Librerías de terceros (es necesario instalarlas):
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model   #Biblioteca Machine Learning
from sklearn.tree import DecisionTreeClassifier, plot_tree


#Módulos propios del proyecto
from TFG_Chollos.utils import configurar_logger, conseguir_ruta_general_TFG

# =============================================================================
# CONSTANTES
# =============================================================================
BASE = conseguir_ruta_general_TFG()

# =============================================================================
# CONFIGURACIÓN DEL LOGGER
# =============================================================================
logger = configurar_logger(__name__)

# =============================================================================
# FUNCIONES
# =============================================================================

def train_test_validation_particion(features:pd.DataFrame, target:pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:

    #Partición 70-30 (entrenamiento y temporal)
    X_train, X_temp, y_train, y_temp = train_test_split(
                                        features, 
                                        target, 
                                        train_size=0.7, 
                                        random_state=42)
    
    #Particion de los datos temporales en 50-50 (validación y test)
    X_val, X_test, y_val, y_test = train_test_split(
                                        X_temp, 
                                        y_temp, 
                                        train_size=0.5, 
                                        random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test      #Resultado final: 70% train / 15% validation / 15% test:



def crear_grafico_correlacion_lineal(db_codificada:pd.DataFrame):
    matriz_correlacion = db_codificada.corr(numeric_only=True).round(2)
    sns.heatmap(matriz_correlacion, annot=True, cmap="vlag", vmin=-1, vmax=1)
    plt.xticks(rotation=45)
    plt.yticks([i + 0.5 for i in range(len(matriz_correlacion.index))], matriz_correlacion.index)   #Se añade esto para que los yticks estén en medio de los recuadros y aparezcan todas las variables 
    plt.title('Matriz de correlación de variables')
    plt.show()



def crear_regresion_lineal(features_train:pd.DataFrame, target_train:pd.Series, variable_representar:pd.Series):
    '''
    Crea y entrena un modelo de regresión lineal Múltiple
    '''
    modelo = linear_model.LinearRegression()   #Creamos el modelo
    modelo.fit(features_train, target_train)   #Le pasamos al modelo nuestros datos (Entrenamos con nuestros datos)
    logger.info('✅ Creado y entrenado el modelo de "Regresión Lineal Múltiple" correctamente')

    sns.scatterplot(
        x=features_train[variable_representar],
        y=target_train)
    sns.lineplot(
        x=features_train[variable_representar],
        y=modelo.predict(features_train))   #Dibujamos la recta de regresión junto a nuestros datos
    
    plt.title(f'Regresión Lineal: {variable_representar} vs {target_train.name}')
    plt.show()

    return modelo



def crear_arbol_decision(features_train:pd.DataFrame, target_train:pd.Series):
    arbol = DecisionTreeClassifier()   #Creamos el modelo
    arbol.fit(features_train, target_train)   #Entrenamos el modelo con nuestros datos
    logger.info('✅ Creado y entrenado el modelo de "Árbol de Decisión" correctamente')

    # plot_tree(
    #     decision_tree = arbol,    #Representamos el árbol
    #      filled=True,   #Añadimos colores a la clase predicha en cada nodo
    #      class_names=['setosa', 'versicolor', 'virginica'],   #Añadimos etiquetas a las especies
    #      features_names=iris_df.columns.to_list())   #Añadimos los nombres de las variables predictoras
    
    return arbol

# def scatter_regresion():

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    db_Sevilla = pd.read_parquet(BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada_Sevilla.parquet')

    crear_grafico_correlacion_lineal(db_Sevilla)

    X = db_Sevilla.drop(columns=['precio'])
    y = db_Sevilla['precio']

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_particion(X, y)
    # regresion = crear_regresion_lineal(X_train, y_train, 'tamaño_habitacion')
    # arbol = crear_arbol_decision(X_train, y_train)

    # modelos = [
    #     ("Regresión Logística", regresion),
    #     ("Árbol de Decisión", arbol),
    #     ("Bosque Aleatorio", bosque_aleatorio)
    #     ]

    # for nombre, modelo in modelos:
    #     precisión = modelo.score(X_test, y_test)
    #     print(f'{nombre}: {precisión:.4f}')   #:.4f --> formato con 4 decimales

    # sns.histplot(data=db_Sevilla, x='precio')
    # plt.show()


if __name__ == '__main__':
    main()