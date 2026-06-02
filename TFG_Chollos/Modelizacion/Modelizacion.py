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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    ruta = BASE / 'images' / f'regresion_lineal_{variable_representar}.png'
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
    ruta = BASE / 'images' / 'arbol_decision.png'
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
# ETIQUETADO DE CHOLLOS (inflado / normal / chollo / super_chollo / hiper-chollo)
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



# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
def main():

    db = pd.read_parquet(BASE / 'data' / 'processed' / 'modelizacion' / 'db_final_codificada.parquet')

    X = db.drop(columns=['precio'])
    y = db['precio']


    # 1. PARTICIÓN
    # -------------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_validation_particion(X, y)


    # 2. TRANSFORMACIONES
    # -------------------------------------------------------------------------
    X_train_est, X_val_est, X_test_est, y_train_est, y_val_est, y_test_est, scaler_X, scaler_y = estandarizar_datos(X_train, X_val, X_test, y_train, y_val, y_test)


    # 3. MODELOS DE REGRESIÓN (predicción de precio)
    # -------------------------------------------------------------------------
    regresion   = crear_regresion_lineal(X_train_est, X_val_est, y_train_est, y_val_est, 'tamaño_habitacion')
    arbol       = crear_arbol_decision(X_train, X_val, y_train, y_val)
    # bosque      = crear_bosque_aleatorio(X_train, X_val, y_train, y_val)
    bosque      = joblib.load(BASE / 'data' / 'models' / 'bosque_aleatorio_reg.pkl')
    boosting    = crear_boosting(X_train, X_val, y_train, y_val)


    # 4. SELECCIÓN DEL MEJOR MODELO DE REGRESIÓN (por R² en validación)
    # -------------------------------------------------------------------------
    modelos_regresion = [
        ('Regresión Lineal',  regresion,  X_train_est,  X_val_est,  X_test_est,  y_train_est,  y_val_est,  y_test_est),
        ('Árbol de Decisión', arbol,      X_train,      X_val,      X_test,      y_train,      y_val,      y_test),
        ('Bosque Aleatorio',  bosque,     X_train,      X_val,      X_test,      y_train,      y_val,      y_test),
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


    # 6. EVALUACIÓN FINAL EN TEST
    # -------------------------------------------------------------------------
    # Regresión: R², MAE y RMSE sobre el conjunto de test
    y_pred_test_reg = mejor_modelo_reg.predict(mejor_X_test)
    r2_test   = r2_score(mejor_y_test, y_pred_test_reg)
    mae_test  = mean_absolute_error(mejor_y_test, y_pred_test_reg)
    rmse_test = np.sqrt(mean_squared_error(mejor_y_test, y_pred_test_reg))
    logger.info(f'[TEST Regresión - {mejor_nombre}] R²={r2_test:.4f} | MAE={mae_test:.2f} | RMSE={rmse_test:.2f}')

    # 7. GUARDAR SCALERS del modelo ganador (los modelos ya se guardan en cada función)
    # -------------------------------------------------------------------------
    modelos_dir = BASE / 'data' / 'models'

    if mejor_nombre == 'Regresión Lineal':
        joblib.dump(scaler_X, modelos_dir / 'scaler_X_regresion.pkl')
        joblib.dump(scaler_y, modelos_dir / 'scaler_y_regresion.pkl')

    logger.info(f'✅ Scalers guardados en {modelos_dir}')


    # 10. GUARDAR RESULTADOS TEST
    # -------------------------------------------------------------------------
    resultados_test = pd.DataFrame({
        'precio_real':     mejor_y_test.values,
        'precio_predicho': y_pred_test_reg,
        'etiqueta':        chollo_test.values,
    }, index=mejor_y_test.index)

    ruta_resultados = BASE / 'data' / 'resultados' / 'resultados_test.parquet'
    resultados_test.to_parquet(ruta_resultados)
    logger.info(f'✅ Resultados test guardados: {ruta_resultados}')


if __name__ == '__main__':
    main()