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
    uv add xgboost, scikit-learn --active --link-mode=copy

Uso:
    python preprocessing.py --input data_Booking/resultados/resultados_booking_{provincia}.csv  --output data_Booking/final/db_final_{provincia}.parquet
'''

# =============================================================================
# IMPORTS
# =============================================================================
#Librerías estándar (vienen incluidas con Python):

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
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
from sklearn.metrics import f1_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC


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
    ruta = BASE / 'data' / 'graficos_modelizacion' / 'correlacion_lineal.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'✅ Gráfico guardado: {ruta}')



def crear_regresion_lineal(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series, variable_representar:str):
    '''
    Crea y entrena un modelo de regresión lineal Múltiple.
    Estandarizamos los datos
    '''
    logger.info('⏳ Creando "Regresión Lineal Múltiple"')

    regresion = linear_model.LinearRegression()   #Creamos el modelo
    regresion.fit(conjunto_ent_est, target_ent_est)   #Le pasamos al modelo nuestros datos estandarizados(Entrenamos con nuestros datos)
    logger.info('✅ Creado y entrenado el modelo de "Regresión Lineal Múltiple" correctamente')

    sns.scatterplot(
        x=conjunto_ent_est[variable_representar],
        y=target_ent_est)
    sns.lineplot(
        x=conjunto_ent_est[variable_representar],
        y=regresion.predict(conjunto_ent_est))   #Dibujamos la recta de regresión junto a nuestros datos

    plt.title(f'Regresión Lineal: {variable_representar} vs {target_ent_est.name}')
    ruta = BASE / 'data' / 'graficos_modelizacion' / f'regresion_lineal_{variable_representar}.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'✅ Gráfico guardado: {ruta}')

    r2_val = regresion.score(conjunto_val_est, target_val_est)
    logger.info(f'R² en validación: {r2_val:.4f}')

    ruta_modelo = BASE / 'data' / 'models' / 'regresion_lineal_reg.pkl'
    joblib.dump(regresion, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return regresion



def crear_arbol_decision(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series):
    '''
    Funcionamiento: Divide el espacio mediante reglas lógicas tipo if-else
    ¿Transformar datos?:No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Hay que decidir entre estos hiperparámetros:
        # max_depth : Profundidad máxima del árbol. Sin límite → overfitting. Valor bajo → underfitting
        min_samples_split : Mínimo de muestras que debe tener un nodo para poder dividirse. Se dejaron los valores por defecto por ser suficientemente robustos
        min_samples_leaf :  Mínimo de muestras que debe tener una hoja (nodo final). Evita hojas con 1 sola muestra. Se dejaron los valores por defecto por ser suficientemente robustos
    '''
    logger.info('⏳ Creando "Árbol de Decisión"')

    arbol = DecisionTreeRegressor(random_state=42)   #Creamos el modelo
    param_grid = {
      'max_depth': [3, 5, 10, 15],  #None provoca overfitting
    }

    grid_search = GridSearchCV(
        estimator=arbol,
        param_grid=param_grid,
        cv=3,                          # K-Fold con 3 folds, sobre X_train
        scoring='neg_mean_squared_error',  # métrica a optimizar
        n_jobs=-1,                      # usa todos los núcleos del procesador
        verbose=2)                     
    
    grid_search.fit(conjunto_ent, target_ent)   #Entrenamos el modelo con nuestros datos
    logger.info('✅ Creado y entrenado el modelo "Árbol de Decisión" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    mejor_arbol = grid_search.best_estimator_
    

    plt.figure(figsize=(20, 10))
    plot_tree(
        decision_tree = mejor_arbol,    #Representamos el árbol
        filled=True,   #Añadimos colores a los valores predichos en cada nodo. filled no colorea cada valor individual colorea cada nodo del árbol. Coge la media de todas las muestras que caen en ese nodo, y cuanto más se aleje del promedio, más intenso el color
        max_depth=3,    #Con max_depth=3 vemos las primeras decisiones más importantes (las que más reducen el error)
        fontsize=9,    #Se ven muy pequeñas las letras
        feature_names=conjunto_ent.columns.to_list())   #Añadimos los nombres de las variables predictoras
    ruta = BASE / 'data' / 'graficos_modelizacion' / 'arbol_decision.png'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta, bbox_inches='tight', dpi=150)
    plt.close()
    logger.info(f'✅ Gráfico guardado: {ruta}')
    logger.info('✅ Representado el "Árbol de Decisión" correctamente')

    r2_val = mejor_arbol.score(conjunto_val, target_val)
    logger.info(f'R² en validación: {r2_val:.4f}')

    ruta_modelo = BASE / 'data' / 'models' / 'arbol_decision_reg.pkl'

    joblib.dump(mejor_arbol, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_arbol



def crear_bosque_aleatorio(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series):
    '''
    Funcionamiento: "Sabiduría de las Masas", combina múltiples árboles
    Técnica: Bagging (Bootstrap Aggregating)
    ¿Transformar datos?: No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Hay que decidir entre estos hiperparámetros:
        # n_estimators : Número de árboles del bosque. Más árboles → más estable, más lento
        # max_depth :  Igual que en el árbol individual
        max_features : Cuántas features considera en cada división. Introduce aleatoriedad para que los árboles sean distintos entre sí. Se dejaron los valores por defecto por ser suficientemente robustos
    '''
    logger.info('⏳ Creando "Bosque Aleatorio"')

    bosque = RandomForestRegressor(random_state=42)
    param_grid = {
      'n_estimators': [50, 100],
      'max_depth':    [10, 20],
    }

    grid_search = GridSearchCV(
        estimator=bosque,
        param_grid=param_grid,
        cv=3,                          # K-Fold con 3 folds, sobre X_train
        scoring='neg_mean_squared_error',  # métrica a optimizar
        n_jobs=-1,                      # usa todos los núcleos del procesador
        verbose=2)                    
    
    grid_search.fit(conjunto_ent, target_ent)

    mejor_bosque = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo "Bosque Aleatorio" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    #  Después, usas X_val para comparar modelos:
    # y_pred_val = mejor_arbol.predict(X_val)
    # print(r2_score(y_val, y_pred_val))
    # print(mean_squared_error(y_val, y_pred_val))

    r2_val = mejor_bosque.score(conjunto_val, target_val)    #Devuelve el porcentaje de predicciones correctas en clasificación, y la precisión o R^2 en regresión
    logger.info(f'R² en validación: {r2_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl'

    joblib.dump(mejor_bosque, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_bosque



def crear_k_vecinos_cercanos(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series):
    '''
    Funcionamiento: Calcula la distancia entre el nuevo punto y todos los puntos de entrenamiento, selecciona los k puntos más cercanos y asigna la clase mayoritaria entre esos vecinos
    ¿Transformar datos?: Estandarizamos los datos porque KNN mide distancia entre puntos → una variable en miles de euros dominaría sobre una en m².
    Hay que decidir entre estos hiperparámetros: 
        # k (número de vecinos) : Número de vecinos que consulta para predecir. k=1 → muy sensible al ruido. k grande → predicciones más suavizadas 
        weights : Si todos los vecinos pesan igual (uniform) o los más cercanos pesan más (distance) 
        metric : Cómo mide la distancia entre puntos (euclidean, manhattan...)
    '''
    logger.info('⏳ Creando "KNN"')

    param_grid = {
      'n_neighbors': [3, 5, 7, 10, 15],
    }
    knn = KNeighborsRegressor()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=3,                          # K-Fold con 3 folds, sobre X_train
        scoring='neg_mean_squared_error',  # métrica a optimizar
        n_jobs=-1,                      # usa todos los núcleos del procesador
        verbose=2)                      
    grid_search.fit(conjunto_ent_est, target_ent_est)    

    mejor_knn = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo "KNN" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    r2_val = mejor_knn.score(conjunto_val_est, target_val_est)    #Devuelve el porcentaje de predicciones correctas en clasificación, y la precisión o R^2 en regresión
    logger.info(f'R² en validación: {r2_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'knn_reg.pkl'

    joblib.dump(mejor_knn, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_knn



def crear_maquinas_vectores_soporte(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, target_ent_est:pd.Series, target_val_est:pd.Series):
    '''
    Funcionamiento: Encuentra el hiperplano que mejor separa las clases, de forma que maximiza el margen (distancia entre el hiperplano y puntos cercanos)
    ¿Transformar datos?: Estandarizamos los datos porque SVM maximiza márgenes → sensible a la magnitud.
    Hay que decidir entre estos hiperparámetros: 
        # C : Penalización por errores. C alto → intenta no fallar ningún punto (riesgo de overfitting). C bajo → acepta más errores a cambio de un margen más amplio
        # kernel : Función que transforma los datos (linear, rbf, poly). Permite separar datos no lineales 
        gamma : Controla el radio de influencia de cada punto. Solo aplica a kernels rbf y poly. Mencionar si solo si uso kernel rbf, si no, ni lo menciono
    '''
    logger.info('⏳ Creando "SVM"')

    param_grid = [
        {'C': [0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale']},
    ]

    svm = SVR()
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=3,                          # K-Fold con 3 folds, sobre X_train
        scoring='neg_mean_squared_error',  # métrica a optimizar
        n_jobs=-1,                      # usa todos los núcleos del procesador
        verbose=2)                      
    grid_search.fit(conjunto_ent_est, target_ent_est)    

    mejor_svm = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo "SVM" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    r2_val = mejor_svm.score(conjunto_val_est, target_val_est)    #Devuelve la precisión o R^2 en regresión
    logger.info(f'R² en validación: {r2_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'svm_reg.pkl'

    joblib.dump(mejor_svm, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_svm


def crear_redes_neuronales(conjunto_ent_norm:pd.DataFrame, conjunto_val_norm:pd.DataFrame, target_ent_norm:pd.Series, target_val_norm:pd.Series):
    '''
    Funcionamiento: Divide el espacio mediante reglas lógicas tipo if-else
    ¿Transformar datos?: Normalizamos los datos porque las Redes Neuronales aprenden con gradientes → convergen mejor con datos en [0,1].
    Hay que decidir el 'early stopping' y entre estos hiperparámetros: 
        # hidden_layer_sizes : Número de capas ocultas y neuronas por capa  
        # learning_rate : Cuánto ajusta los pesos en cada paso. Alto → aprende rápido pero puede no converger. Bajo → aprende despacio pero más estable
        # epochs / max_iter : Número de veces que recorre todo el dataset entrenando. Se pone un valor alto y se delega en early stopping 
    '''
    logger.info('⏳ Creando "Redes Neuronales"')

    red = MLPRegressor(
        max_iter=1000,           # valor alto, se delega el paro en early_stopping
        early_stopping=True,     # para cuando la validación interna no mejora
        random_state=42)

    param_grid = {
      'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],   #(64,) → 1 capa oculta con 64 neuronas, (64, 64) → 2 capas ocultas con 64 neuronas cada una
      'learning_rate_init': [0.001, 0.01, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=red,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent_norm, target_ent_norm)

    mejor_red = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Redes Neuronales" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    r2_val = mejor_red.score(conjunto_val_norm, target_val_norm)
    logger.info(f'R² en validación: {r2_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'redes_neuronales_reg.pkl'

    joblib.dump(mejor_red, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_red



def crear_boosting(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, target_ent:pd.Series, target_val:pd.Series):
    '''
    Funcionamiento: Gradient Boosting optimizado con regularización L1/L2. Construye árboles secuencialmente corrigiendo los errores del anterior.
    ¿Transformar datos?: No es necesario escalar los datos porque solo pregunta "¿es X mayor que umbral?", la magnitud no importa.
    Implementación: XGBoost (más eficiente que sklearn GB al paralelizar con n_jobs).
    Hay que decidir entre estos hiperparámetros:
        # n_estimators : Número de árboles encadenados. Más árboles → más preciso, pero más riesgo de overfitting
        # learning_rate : Cuánto contribuye cada árbol nuevo. Valores bajos necesitan más árboles
        # max_depth : Profundidad de cada árbol individual. Suelen ser árboles poco profundos (3-5)
    '''
    logger.info('⏳ Creando "XGBoost"')

    boosting = XGBRegressor(random_state=42,
                            n_jobs=-1,     #Paraleliza el entrenamiento internamente, no solo los folds
                            verbosity=0)   #Suprime la salida propia de XGBoost para no mezclarla con el verbose del GridSearchCV

    param_grid = {
      'n_estimators':  [100, 300, 500],
      'learning_rate': [0.01, 0.05, 0.1],
      'max_depth':     [3, 5],
    }

    grid_search = GridSearchCV(
        estimator=boosting,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent, target_ent)

    mejor_boosting = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "XGBoost" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    r2_val = mejor_boosting.score(conjunto_val, target_val)
    logger.info(f'R² en validación: {r2_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'boosting_reg.pkl'

    joblib.dump(mejor_boosting, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_boosting


# =============================================================================
# MODELOS DE CLASIFICACIÓN (inflado / normal / chollo / super_chollo / hiper-chollo)
# =============================================================================

def crear_etiqueta_chollo(y_real:pd.Series, y_predicho:pd.Series) -> pd.Series:
    '''
    Clasifica cada alojamiento según la diferencia entre precio real y predicho:
        hiper_chollo : 4  -->  y_real/y_predicho <= 0.75  (>25% más barato)
        super_chollo : 3  -->  0.75 < y_real/y_predicho < 0.85  (15-25% más barato)
        chollo       : 2  -->  0.85 <= y_real/y_predicho < 0.99  (1-15% más barato)
        normal       : 1  -->  0.99 <= y_real/y_predicho < 1.01  (±1%)
        inflado      : 0  -->  y_real/y_predicho > 1.05   (>5% más caro)
    '''
    real_predicho = y_real / y_predicho

    condiciones = [
        real_predicho <= 0.75,
        (real_predicho > 0.75) & (real_predicho < 0.85),
        (real_predicho >= 0.85) & (real_predicho < 0.99),
        (real_predicho >= 0.99) & (real_predicho <= 1.01),
        real_predicho > 1.05
    ]
    etiquetas = [4, 3, 2, 1, 0]
    return pd.Series(np.select(condiciones, etiquetas), index=y_real.index, name='categoria')   #Condición: lista booleanos, etiquetas: lista de valores a asignar cuando condición sea True


def crear_regresion_logistica(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    Estandarizamos los datos. Versión clasificatoria de la regresión lineal.
    Hay que decidir entre estos hiperparámetros:
        # C : Inverso de la regularización. C alto → menos regularización (riesgo de overfitting)
        # solver : Algoritmo de optimización (lbfgs, liblinear...)
    '''
    logger.info('⏳ Creando "Regresión Logística"')

    regresion = linear_model.LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
    }
    grid_search = GridSearchCV(
        estimator=regresion,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent_est, objetivo_ent)

    mejor_regresion = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Regresión Logística" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_regresion.predict(conjunto_val_est), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'regresion_logistica_clf.pkl'

    joblib.dump(mejor_regresion, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_regresion


def crear_arbol_decision_clasificacion(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    No es necesario escalar los datos.
    Hay que decidir entre estos hiperparámetros:
        # max_depth : Profundidad máxima del árbol. Sin límite → overfitting. Valor bajo → underfitting
    '''
    logger.info('⏳ Creando "Árbol de Decisión Clasificación"')

    arbol = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 10, 15],
    }
    grid_search = GridSearchCV(
        estimator=arbol,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent, objetivo_ent)

    mejor_arbol = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Árbol de Decisión Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_arbol.predict(conjunto_val), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'arbol_decision_clf.pkl'

    joblib.dump(mejor_arbol, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_arbol


def crear_bosque_aleatorio_clasificacion(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    No es necesario escalar los datos.
    Hay que decidir entre estos hiperparámetros:
        # n_estimators : Número de árboles del bosque. Más árboles → más estable, más lento
        # max_depth : Igual que en el árbol individual
    '''
    logger.info('⏳ Creando "Bosque Aleatorio Clasificación"')

    bosque = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth':    [10, 20],
    }
    grid_search = GridSearchCV(
        estimator=bosque,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent, objetivo_ent)

    mejor_bosque = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Bosque Aleatorio Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_bosque.predict(conjunto_val), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'bosque_aleatorio_clf.pkl'

    joblib.dump(mejor_bosque, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_bosque


def crear_maquinas_vectores_soporte_clasificacion(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    Estandarizamos los datos porque SVM maximiza márgenes → sensible a la magnitud.
    Hay que decidir entre estos hiperparámetros:
        # C : Penalización por errores. C alto → intenta no fallar ningún punto (riesgo de overfitting)
        # kernel : Función que transforma los datos (linear, rbf, poly)
    '''
    logger.info('⏳ Creando "SVM Clasificación"')

    svm = SVC(random_state=42)
    param_grid = [
        {'C': [0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale']},
    ]
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent_est, objetivo_ent)

    mejor_svm = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "SVM Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_svm.predict(conjunto_val_est), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'svm_clf.pkl'

    joblib.dump(mejor_svm, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_svm


def crear_k_vecinos_cercanos_clasificacion(conjunto_ent_est:pd.DataFrame, conjunto_val_est:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    Estandarizamos los datos porque KNN mide distancia entre puntos.
    Hay que decidir entre estos hiperparámetros:
        # n_neighbors : Número de vecinos que consulta para predecir
    '''
    logger.info('⏳ Creando "KNN Clasificación"')

    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 10, 15],
    }
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent_est, objetivo_ent)

    mejor_knn = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "KNN Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_knn.predict(conjunto_val_est), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'knn_clf.pkl'

    joblib.dump(mejor_knn, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_knn



def crear_redes_neuronales_clasificacion(conjunto_ent_norm:pd.DataFrame, conjunto_val_norm:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    Normalizamos los datos porque las Redes Neuronales aprenden con gradientes → convergen mejor con datos en [0,1].
    Hay que decidir el 'early stopping' y entre estos hiperparámetros:
        # hidden_layer_sizes : Número de capas ocultas y neuronas por capa
        # learning_rate_init : Cuánto ajusta los pesos en cada paso
    '''
    logger.info('⏳ Creando "Redes Neuronales Clasificación"')

    red = MLPClassifier(
        max_iter=1000,
        early_stopping=True,
        random_state=42)
    param_grid = {
        'hidden_layer_sizes': [(64,), (128,), (64, 64), (128, 64)],
        'learning_rate_init': [0.001, 0.01, 0.1],
    }
    grid_search = GridSearchCV(
        estimator=red,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent_norm, objetivo_ent)

    mejor_red = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "Redes Neuronales Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_red.predict(conjunto_val_norm), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'redes_neuronales_clf.pkl'

    joblib.dump(mejor_red, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_red



def crear_boosting_clasificacion(conjunto_ent:pd.DataFrame, conjunto_val:pd.DataFrame, objetivo_ent:pd.Series, objetivo_val:pd.Series):
    '''
    No es necesario escalar los datos.
    Implementación: XGBoost (más eficiente que sklearn GB al paralelizar con n_jobs).
    Hay que decidir entre estos hiperparámetros:
        # n_estimators : Número de árboles encadenados
        # learning_rate : Cuánto contribuye cada árbol nuevo
        # max_depth : Profundidad de cada árbol individual
    '''
    logger.info('⏳ Creando "XGBoost Clasificación"')

    boosting = XGBClassifier(random_state=42, 
                            n_jobs=-1, 
                            verbosity=0, 
                            eval_metric='mlogloss')    #evita un warning de XGBoost en problemas multiclase
    param_grid = {
        'n_estimators':  [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth':     [3, 5],
    }
    grid_search = GridSearchCV(
        estimator=boosting,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2)
    grid_search.fit(conjunto_ent, objetivo_ent)

    mejor_boosting = grid_search.best_estimator_
    logger.info('✅ Creado y entrenado el modelo de "XGBoost Clasificación" correctamente')
    logger.info(f'Mejores hiperparámetros: {grid_search.best_params_}')

    f1_val = f1_score(objetivo_val, mejor_boosting.predict(conjunto_val), average='weighted')
    logger.info(f'F1 en validación: {f1_val:.4f}')
    ruta_modelo = BASE / 'data' / 'models' / 'boosting_clf.pkl'

    joblib.dump(mejor_boosting, ruta_modelo)
    logger.info(f'✅ Modelo guardado: {ruta_modelo}')

    return mejor_boosting


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    db = pd.read_parquet(BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet')

    # crear_grafico_correlacion_lineal(db)

    X = db.drop(columns=['precio'])
    y = db['precio']


    # 1. PARTICIÓN
    # -------------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_particion(X, y)


    # 2. TRANSFORMACIONES
    # -------------------------------------------------------------------------
    X_train_cen, X_val_cen, X_test_cen, y_train_cen, y_val_cen, y_test_cen, media_X_train, media_y_train = centrar_datos(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est, scaler_X, scaler_y = estandarizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)
    X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm, norm_X, norm_y = normalizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)


    # 3. MODELOS DE REGRESIÓN (predicción de precio)
    # -------------------------------------------------------------------------
    regresion   = crear_regresion_lineal(X_train_est, X_val_est, y_train_est, y_val_est, 'tamaño_habitacion')
    arbol       = crear_arbol_decision(X_train, X_val, y_train, y_val)
    # bosque      = crear_bosque_aleatorio(X_train, X_val, y_train, y_val)
    bosque      = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')
    # svm         = crear_maquinas_vectores_soporte(X_train_est, X_val_est, y_train_est, y_val_est)
    # knn         = crear_k_vecinos_cercanos(X_train_est, X_val_est, y_train_est, y_val_est)
    # red         = crear_redes_neuronales(X_train_norm, X_val_norm, y_train_norm, y_val_norm)
    boosting    = crear_boosting(X_train, X_val, y_train, y_val)


    # 4. SELECCIÓN DEL MEJOR MODELO DE REGRESIÓN (por R² en validación)
    # -------------------------------------------------------------------------
    modelos_regresion = [
        ('Regresión Lineal',  regresion,  X_train_est,  X_val_est,  X_test_est,  y_train_est,  y_val_est,  y_test_est),
        ('Árbol de Decisión', arbol,      X_train,      X_val,      X_test,      y_train,      y_val,      y_test),
        ('Bosque Aleatorio',  bosque,     X_train,      X_val,      X_test,      y_train,      y_val,      y_test),
        # ('SVM',               svm,        X_train_est,  X_val_est,  X_test_est,  y_train_est,  y_val_est,  y_test_est),
        # ('KNN',               knn,        X_train_est,  X_val_est,  X_test_est,  y_train_est,  y_val_est,  y_test_est),
        # ('Redes Neuronales',  red,        X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm),
        ('Boosting',          boosting,   X_train,      X_val,      X_test,      y_train,      y_val,      y_test),
    ]

    mejor_score = None
    mejor_nombre = mejor_modelo_reg = None
    mejor_X_train = mejor_X_val = mejor_X_test = None
    mejor_y_train = mejor_y_val = mejor_y_test = None
    for nombre, modelo, X_tr, X_va, X_te, y_tr, y_va, y_te in modelos_regresion:
        score = modelo.score(X_va, y_va)
        if mejor_score is None or score > mejor_score:
            mejor_score = score
            mejor_nombre, mejor_modelo_reg = nombre, modelo
            mejor_X_train, mejor_X_val, mejor_X_test, mejor_y_train, mejor_y_val, mejor_y_test = X_tr, X_va, X_te, y_tr, y_va, y_te

    logger.info(f'✅ Mejor modelo de regresión: {mejor_nombre} (R²={mejor_score:.4f})')


    # 5. CREAR ETIQUETA CHOLLO con el mejor modelo de regresión
    # -------------------------------------------------------------------------
    y_pred_train  = mejor_modelo_reg.predict(mejor_X_train)
    y_pred_val    = mejor_modelo_reg.predict(mejor_X_val)
    y_pred_test   = mejor_modelo_reg.predict(mejor_X_test)

    chollo_train  = crear_etiqueta_chollo(mejor_y_train, pd.Series(y_pred_train, index=mejor_y_train.index))
    chollo_val    = crear_etiqueta_chollo(mejor_y_val,   pd.Series(y_pred_val,   index=mejor_y_val.index))
    chollo_test   = crear_etiqueta_chollo(mejor_y_test,  pd.Series(y_pred_test,  index=mejor_y_test.index))
    logger.info(f'Distribución categorías train: {chollo_train.value_counts().to_dict()}')


    # 6. MODELOS DE CLASIFICACIÓN (chollo / no chollo)
    # -------------------------------------------------------------------------
    reg_log     = crear_regresion_logistica(X_train_est, X_val_est, chollo_train, chollo_val)
    arbol_clf   = crear_arbol_decision_clasificacion(X_train, X_val, chollo_train, chollo_val)
    bosque_clf  = crear_bosque_aleatorio_clasificacion(X_train, X_val, chollo_train, chollo_val)
    # svm_clf     = crear_maquinas_vectores_soporte_clasificacion(X_train_est, X_val_est, chollo_train, chollo_val)
    # knn_clf     = crear_k_vecinos_cercanos_clasificacion(X_train_est, X_val_est, chollo_train, chollo_val)
    # red_clf     = crear_redes_neuronales_clasificacion(X_train_norm, X_val_norm, chollo_train, chollo_val)
    boost_clf   = crear_boosting_clasificacion(X_train, X_val, chollo_train, chollo_val)


    # 7. SELECCIÓN DEL MEJOR MODELO DE CLASIFICACIÓN (por F1 en validación)
    # -------------------------------------------------------------------------
    modelos_clasificacion = [
        ('Regresión Logística',  reg_log,    X_train_est,  X_val_est,  X_test_est,  chollo_val),
        ('Árbol Clasificación',  arbol_clf,  X_train,      X_val,      X_test,      chollo_val),
        ('Bosque Clasificación', bosque_clf, X_train,      X_val,      X_test,      chollo_val),
        # ('SVM Clasificación',    svm_clf,    X_train_est,  X_val_est,  X_test_est,  chollo_val),
        # ('KNN Clasificación',    knn_clf,    X_train_est,  X_val_est,  X_test_est,  chollo_val),
        # ('Red Neuronal Clf',     red_clf,    X_train_norm, X_val_norm, X_test_norm, chollo_val),
        ('Boosting Clf',         boost_clf,  X_train,      X_val,      X_test,      chollo_val),
    ]

    mejor_f1 = None
    mejor_nombre_clf = mejor_modelo_clf = mejor_X_train_clf = mejor_X_val_clf = mejor_X_test_clf = None
    for nombre, modelo, X_tr, X_va, X_te, chollo_va in modelos_clasificacion:
        f1 = f1_score(chollo_va, modelo.predict(X_va), average='weighted')
        if mejor_f1 is None or f1 > mejor_f1:
            mejor_f1 = f1
            mejor_nombre_clf, mejor_modelo_clf = nombre, modelo
            mejor_X_train_clf, mejor_X_val_clf, mejor_X_test_clf = X_tr, X_va, X_te

    logger.info(f'✅ Mejor modelo de clasificación: {mejor_nombre_clf} (F1={mejor_f1:.4f})')


    # 8. EVALUACIÓN FINAL EN TEST
    # -------------------------------------------------------------------------
    # Regresión: R², MAE y RMSE sobre el conjunto de test
    y_pred_test_reg = mejor_modelo_reg.predict(mejor_X_test)
    r2_test   = r2_score(mejor_y_test, y_pred_test_reg)
    mae_test  = mean_absolute_error(mejor_y_test, y_pred_test_reg)
    rmse_test = np.sqrt(mean_squared_error(mejor_y_test, y_pred_test_reg))
    logger.info(f'[TEST Regresión - {mejor_nombre}] R²={r2_test:.4f} | MAE={mae_test:.2f} | RMSE={rmse_test:.2f}')

    # Clasificación: precision, recall y F1 por clase sobre el conjunto de test
    nombres_clases = ['inflado (0)', 'normal (1)', 'chollo (2)', 'super_chollo (3)', 'hiper_chollo (4)']
    y_pred_test_clf = mejor_modelo_clf.predict(mejor_X_test_clf)
    reporte = classification_report(chollo_test, y_pred_test_clf, labels=[0, 1, 2, 3, 4], target_names=nombres_clases, zero_division=0)
    logger.info(f'[TEST Clasificación - {mejor_nombre_clf}]\n{reporte}')

    f1_macro = f1_score(chollo_test, y_pred_test_clf, average='macro', zero_division=0)
    logger.info(f'[TEST Clasificación - {mejor_nombre_clf}] F1-macro={f1_macro:.4f}')


    # 9. GUARDAR SCALERS del modelo ganador (los modelos ya se guardan en cada función)
    # -------------------------------------------------------------------------
    MODELOS_EST      = {'Regresión Lineal', 'SVM', 'KNN'}
    MODELOS_NORM     = {'Redes Neuronales'}
    MODELOS_EST_CLF  = {'Regresión Logística', 'SVM Clasificación', 'KNN Clasificación'}
    MODELOS_NORM_CLF = {'Red Neuronal Clf'}

    modelos_dir = BASE / 'data' / 'models'

    if mejor_nombre in MODELOS_EST:
        joblib.dump(scaler_X, modelos_dir / 'scaler_X_regresion.pkl')
        joblib.dump(scaler_y, modelos_dir / 'scaler_y_regresion.pkl')
    elif mejor_nombre in MODELOS_NORM:
        joblib.dump(norm_X, modelos_dir / 'scaler_X_regresion.pkl')
        joblib.dump(norm_y, modelos_dir / 'scaler_y_regresion.pkl')

    if mejor_nombre_clf in MODELOS_EST_CLF:
        joblib.dump(scaler_X, modelos_dir / 'scaler_X_clasificacion.pkl')
    elif mejor_nombre_clf in MODELOS_NORM_CLF:
        joblib.dump(norm_X, modelos_dir / 'scaler_X_clasificacion.pkl')

    logger.info(f'✅ Scalers guardados en {modelos_dir}')


    # 10. GUARDAR RESULTADOS TEST
    # -------------------------------------------------------------------------
    resultados_test = pd.DataFrame({
        'precio_real':    mejor_y_test.values,
        'precio_predicho': y_pred_test_reg,
        'clase_real':     chollo_test.values,
        'clase_predicha': y_pred_test_clf,
    }, index=mejor_y_test.index)

    ruta_resultados = BASE / 'data' / 'resultados' / 'resultados_test.parquet'
    resultados_test.to_parquet(ruta_resultados)
    logger.info(f'✅ Resultados test guardados: {ruta_resultados}')


if __name__ == '__main__':
    main()