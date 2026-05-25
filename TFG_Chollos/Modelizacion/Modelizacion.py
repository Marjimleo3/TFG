'''
preprocessing.py
====================
Dado el volumen del dataset, se optó por una partición holdout 70/15/15. Sobre el conjunto de entrenamiento se aplicó validación cruzada K-Fold (K=5) mediante GridSearchCV para el ajuste de hiperparámetros de cada modelo de forma independiente. Posteriormente, el conjunto de validación se empleó para la comparación y selección del modelo final entre los candidatos ya optimizados. El conjunto de test permaneció intacto hasta la evaluación final, garantizando una estimación del rendimiento libre de sesgos.
En resumen, hacemos una combinación entre partición train-validation-test y Cross-Validation con cv=5 folds.

 Lo que hace internamente con cv=5:

  X_train  →  fold1 | fold2 | fold3 | fold4 | fold5
                ↓
  Para cada combinación de hiperparámetros:
    entrena en 4 folds → evalúa en el fold restante → repite 5 veces → saca media
                ↓
  Se queda con la combinación que mejor media obtiene

Dado que la mayoría de los algoritmos de aprendizaje automático operan sobre espacios vectoriales numéricos, las variables categóricas presentes en el dataset fueron codificadas mediante get_dummies y LabelEncoder.

Este va a ser el flujo de trabajo completo:
  X_train  →  entrenar cada modelo
  X_val    →  ajustar hiperparámetros + elegir el mejor modelo   ← tomas decisiones
  X_test   →  evaluar el modelo elegido y reportar               ← no tomas decisiones

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor


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



def centrar_datos(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, conjunto_test:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series, target_test:pd.Series):
    '''
    Centrar transforma los datos para que cada columna tenga media 0
    Restamos la media en cada columna.
    Se suele utilizar en Análisis de Componentes Principales.
    Sensible a outliers
    '''
    media_X_train = np.mean(conjunto_ent, axis=0)
    media_y_train = np.mean(target_ent, axis=0)

    X_train = conjunto_ent - media_X_train
    y_train = target_ent - media_y_train

    X_val = conjunto_val - media_X_train     #Porque validación y test sólo comprueba, no actúa sobre la centralización de datos
    X_test = conjunto_test - media_X_train      #Porque validación y test sólo comprueba, no actúa sobre la centralización de datos
    y_val = target_val - media_y_train       #Porque validación y test sólo comprueba, no actúa sobre la centralización de datos
    y_test = target_test - media_y_train        #Porque validación y test sólo comprueba, no actúa sobre la centralización de datos

    return X_train, X_val, X_test, y_train, y_val, y_test, media_X_train, media_y_train
# Para hacer la transformación inversa: precio_real = pred_y + media_y_train


def estandarizar_datos(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, conjunto_test:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series, target_test:pd.Series):
    '''
    Estandarizar (Z-score scaling) transforma los datos para que cada columna tenga media 0, y desviación típica 1
    Fórmula: z = (x-\\mu)/\\sigma. --> Restamos la media y dividimos por la desviación típica de cada columna (z-score).
    Se suele utilizar para Regresión y SVM.
    Sensible a outliers
    '''
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # TRAIN: aprende y transforma
    X_train = scaler_X.fit_transform(conjunto_ent)
    y_train = scaler_y.fit_transform(target_ent.values.reshape(-1, 1)).ravel()    #y_train es una pd.Series (1D) y el StandardScaler necesita 2D. La solución es hacer .reshape(-1, 1) en la y antes de transformar. ravel devuelve el 2D generado a 1D

    # VALIDATION y TEST: transforma con los mismos parámetros (no aprende nada nuevo). Porque si llamas a estandarizar_datos(X_test) por separado, estás calculando la media y std del test, cuando deberías usar la media y std del train
    X_val = scaler_X.transform(conjunto_val)
    X_test = scaler_X.transform(conjunto_test)   # aplica los parámetros del train
    y_val = scaler_y.transform(target_val.values.reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(target_test.values.reshape(-1, 1)).ravel()

    #Pasamos los np.Arrays a pd.DataFrames o pd.Series:
    X_train, X_val, X_test = [pd.DataFrame(a, columns=conjunto_ent.columns) for a in [X_train, X_val, X_test]]    #a, b, c = [1, 2, 3] --> a=1, b=2, c=3
    y_train, y_val, y_test = [pd.Series(a, name=target_ent.name) for a in [y_train, y_val, y_test]]

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y
# Para hacer la transformación inversa: precio_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1))


def normalizar_datos(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, conjunto_test:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series, target_test:pd.Series):
    '''
    Normalizar (Min-Max Scaling, rescaling) transforma los datos para que todos ellos tengan valores entre 0 y 1.
    La fórmula utilizada es $$y = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$$. --> Restamos el mínimo, y dividimos entre la diferencia entre el máximo y el mínimo
    Se suele utilizar para Redes Neuronales.
    Muy sensible a outliers
    '''

    norm_X = MinMaxScaler(feature_range=(0,1))   #Construimos el modelo de normalización
    norm_y = MinMaxScaler(feature_range=(0,1))

    # TRAIN: aprende y transforma
    X_train = norm_X.fit_transform(conjunto_ent)   #Nos devuelve un array de np sin columnas
    y_train = norm_y.fit_transform(target_ent.values.reshape(-1, 1)).ravel()    #Nos devuelve un array de np sin columnas

    # VALIDATION y TEST: transforma con los mismos parámetros (no aprende nada nuevo). Porque si llamas a estandarizar_datos(X_test) por separado, estás calculando la media y std del test, cuando deberías usar la media y std del train
    X_val = norm_X.transform(conjunto_val)    #Nos devuelve un array de np sin columnas
    X_test = norm_X.transform(conjunto_test)    #Nos devuelve un array de np sin columnas
    y_val = norm_y.transform(target_val.values.reshape(-1, 1)).ravel()     #Nos devuelve un array de np sin columnas
    y_test = norm_y.transform(target_test.values.reshape(-1, 1)).ravel()    #Nos devuelve un array de np sin columnas

    X_train, X_val, X_test = [pd.DataFrame(a, columns=conjunto_ent.columns) for a in [X_train, X_val, X_test]]
    y_train, y_val, y_test = [pd.Series(a, name=target_ent.name) for a in [y_train, y_val, y_test]]

    return X_train, X_val, X_test, y_train, y_val, y_test, norm_X, norm_y
# Para hacer la transformación inversa: precio_real = norm_y.inverse_transform(y_pred.reshape(-1, 1))


def crear_grafico_correlacion_lineal(db_codificada:pd.DataFrame):
    if db_codificada['Piscina_infinita'].sum() == 0:
        db_codificada_modificada = db_codificada.drop(columns=['Piscina_infinita'])
    else:
        db_codificada_modificada = db_codificada
    matriz_correlacion = db_codificada_modificada.corr(numeric_only=True).round(2)
    sns.heatmap(matriz_correlacion, annot=True, cmap="vlag", vmin=-1, vmax=1)
    plt.xticks([i + 0.5 for i in range(len(matriz_correlacion.index))], matriz_correlacion.index, rotation=45, ha='right')    #Se añade esto para que los yticks estén en medio de los recuadros y aparezcan todas las variables 
    plt.yticks([i + 0.5 for i in range(len(matriz_correlacion.index))], matriz_correlacion.index)   #Se añade esto para que los yticks estén en medio de los recuadros y aparezcan todas las variables 
    plt.title('Matriz de correlación lineal de las variables')
    plt.show()



def crear_regresion_lineal(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, conjunto_test_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series, target_test_est:pd.Series, variable_representar:str):
    '''
    Crea y entrena un modelo de regresión lineal Múltiple.
    Estandarizamos los datos
    '''

    modelo = linear_model.LinearRegression()   #Creamos el modelo
    modelo.fit(conjunto_ent_est, target_ent_est)   #Le pasamos al modelo nuestros datos estandarizados(Entrenamos con nuestros datos)
    logger.info('✅ Creado y entrenado el modelo de "Regresión Lineal Múltiple" correctamente')

    sns.scatterplot(
        x=conjunto_ent_est[variable_representar],
        y=target_ent_est)
    sns.lineplot(
        x=conjunto_ent_est[variable_representar],
        y=modelo.predict(conjunto_ent_est))   #Dibujamos la recta de regresión junto a nuestros datos

    plt.title(f'Regresión Lineal: {variable_representar} vs {target_ent_est.name}')
    plt.show()

    return modelo



def crear_arbol_decision(conjunto_ent:pd.DataFrame, target_ent:pd.Series):
    '''
    No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Hay que decidir entre estos hiperparámetros:
        # max_depth : Profundidad máxima del árbol. Sin límite → overfitting. Valor bajo → underfitting
        min_samples_split : Mínimo de muestras que debe tener un nodo para poder dividirse. Se dejaron los valores por defecto por ser suficientemente robustos
        min_samples_leaf :  Mínimo de muestras que debe tener una hoja (nodo final). Evita hojas con 1 sola muestra. Se dejaron los valores por defecto por ser suficientemente robustos
    '''

    arbol = DecisionTreeRegressor(random_state=42)   #Creamos el modelo
    param_grid = {
      'max_depth': [3, 5, 10, 15, None],
    }

    grid_search = GridSearchCV(
      estimator=arbol,
      param_grid=param_grid,
      cv=5,                          # K-Fold con 5 folds, sobre X_train
      scoring='neg_mean_squared_error',  # métrica a optimizar
      n_jobs=-1                      # usa todos los núcleos del procesador
    )
    grid_search.fit(conjunto_ent, target_ent)   #Entrenamos el modelo con nuestros datos
    logger.info('✅ Creado y entrenado el modelo de "Árbol de Decisión" correctamente')

    mejor_arbol = grid_search.best_estimator_
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    plt.figure(figsize=(20, 10))
    plot_tree(
        decision_tree = mejor_arbol,    #Representamos el árbol
        filled=True,   #Añadimos colores a los valores predichos en cada nodo. filled no colorea cada valor individual colorea cada nodo del árbol. Coge la media de todas las muestras que caen en ese nodo, y cuanto más se aleje del promedio, más intenso el color
        max_depth=3,    #Con max_depth=3 vemos las primeras decisiones más importantes (las que más reducen el error)
        fontsize=9,    #Se ven muy pequeñas las letras
        feature_names=conjunto_ent.columns.to_list())   #Añadimos los nombres de las variables predictoras
    plt.show()
    logger.info('✅ Representado el "Árbol de Decisión" correctamente')

    #  Después, usas X_val para comparar modelos:
    # y_pred_val = mejor_arbol.predict(X_val)
    # print(r2_score(y_val, y_pred_val))
    # print(mean_squared_error(y_val, y_pred_val))
    
    return mejor_arbol

def crear_bosque_aleatorio(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, conjunto_test:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series, target_test:pd.Series):
    '''
    No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Hay que decidir entre estos hiperparámetros:
        # n_estimators : Número de árboles del bosque. Más árboles → más estable, más lento
        # max_depth :  Igual que en el árbol individual
        max_features : Cuántas features considera en cada división. Introduce aleatoriedad para que los árboles sean distintos entre sí. Se dejaron los valores por defecto por ser suficientemente robustos
    '''
    bosque = RandomForestRegressor(random_state=42)
    param_grid = {
      'n_estimators': [50, 100, 200],
      'max_depth':    [5, 10, 20, None],
    }
    grid_search = GridSearchCV(
      estimator=bosque,
      param_grid=param_grid,
      cv=5,                          # K-Fold con 5 folds, sobre X_train
      scoring='neg_mean_squared_error',  # métrica a optimizar
      n_jobs=-1                      # usa todos los núcleos del procesador
    )
    grid_search.fit(conjunto_ent, target_ent)

    mejor_bosque = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Bosque Aleatorio" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    #  Después, usas X_val para comparar modelos:
    # y_pred_val = mejor_arbol.predict(X_val)
    # print(r2_score(y_val, y_pred_val))
    # print(mean_squared_error(y_val, y_pred_val))

    r2_val = mejor_bosque.score(conjunto_val, target_val)    #Devuelve el porcentaje de predicciones correctas en clasificación, y la precisión o R^2 en regresión
    logger.info(f'R² en validación: {r2_val:.4f}')

    return mejor_bosque


def crear_maquinas_vectores_soporte(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, conjunto_test_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series, target_test_est:pd.Series):
    '''
    Estandarizamos los datos porque SVM maximiza márgenes → sensible a la magnitud.
    Hay que decidir entre estos hiperparámetros: 
        # C : Penalización por errores. C alto → intenta no fallar ningún punto (riesgo de overfitting). C bajo → acepta más errores a cambio de un margen más amplio
        # kernel : Función que transforma los datos (linear, rbf, poly). Permite separar datos no lineales 
        gamma : Controla el radio de influencia de cada punto. Solo aplica a kernels rbf y poly. Mencionar si solo si uso kernel rbf, si no, ni lo menciono
    '''
    param_grid = {
      'C':      [0.1, 1, 10, 100],
      'kernel': ['linear', 'rbf'],
    }


def crear_k_vecinos_cercanos(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, conjunto_test_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series, target_test_est:pd.Series):
    '''
    Estandarizamos los datos porque KNN mide distancia entre puntos → una variable en miles de euros dominaría sobre una en m².
    Hay que decidir entre estos hiperparámetros: 
        # k (número de vecinos) : Número de vecinos que consulta para predecir. k=1 → muy sensible al ruido. k grande → predicciones más suavizadas 
        weights : Si todos los vecinos pesan igual (uniform) o los más cercanos pesan más (distance) 
        metric : Cómo mide la distancia entre puntos (euclidean, manhattan...)
    '''
    param_grid = {
      'n_neighbors': [3, 5, 7, 10, 15],
    }


def crear_redes_neuronales(conjunto_ent_norm:pd.DataFrame, conjunto_val_norm:pd.DataFrame, conjunto_test_norm:pd.DataFrame, target_ent_norm:pd.Series, target_val_norm:pd.Series, target_test_norm:pd.Series):
    '''
    Normalizamos los datos porque las Redes Neuronales aprenden con gradientes → convergen mejor con datos en [0,1].
    Hay que decidir el 'early stopping' y entre estos hiperparámetros: 
        # hidden_layer_sizes : Número de capas ocultas y neuronas por capa  
        # learning_rate : Cuánto ajusta los pesos en cada paso. Alto → aprende rápido pero puede no converger. Bajo → aprende despacio pero más estable
        # epochs / max_iter : Número de veces que recorre todo el dataset entrenando. Se pone un valor alto y se delega en early stopping 
    '''
    # Redes Neuronales (MLPRegressor de sklearn)
    param_grid = {
      'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],   #(64,) → 1 capa oculta con 64 neuronas, (64, 64) → 2 capas ocultas con 64 neuronas cada una
      'learning_rate_init': [0.001, 0.01, 0.1],
    }



def crear_boosting():
    '''
    No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Hay que decidir el 'early stopping' y entre estos hiperparámetros: 
        # learning_rate : Número de árboles encadenados. Más árboles → más preciso, pero más riesgo de overfitting 
        # n_estimators : Cuánto contribuye cada árbol nuevo. Valores bajos necesitan más árboles   
        # max_depth : Profundidad de cada árbol individual. Suelen ser árboles poco profundos (3-5) 
    '''
    # Boosting (GradientBoostingRegressor de sklearn)
    param_grid = {
      'n_estimators':  [50, 100, 200],
      'learning_rate': [0.01, 0.05, 0.1],
      'max_depth':     [3, 5, 7],
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    db_Sevilla = pd.read_parquet(BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada_Almería.parquet')

    # crear_grafico_correlacion_lineal(db_Sevilla)

    X = db_Sevilla.drop(columns=['precio'])
    y = db_Sevilla['precio']

    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_particion(X, y)

    X_train_cen, X_val_cen, X_test_cen, y_train_cen, y_val_cen, y_test_cen, media_X_train, media_y_train = centrar_datos(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est, scaler_X, scaler_y = estandarizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_X, norm_y = normalizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)

    # regresion = crear_regresion_lineal(X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est, 'tamaño_habitacion')

    arbol = crear_arbol_decision(X_train, y_train)

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