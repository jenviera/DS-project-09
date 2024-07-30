#!/usr/bin/env python
# coding: utf-8

# # Hola Jenniffer!
# 
# Mi nombre es David Bautista, soy code reviewer de TripleTen y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión.
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres.**
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div
# 
# ¡Empecemos!

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# # Comentario General
# 
# Hola, Jennifer, te felicito por el desarrollo del proyecto. Completaste las diferentes secciones de muy buena manera. </div>

# # Introducción
# 
# En el entorno competitivo actual, la retención de clientes es crucial para el éxito y la sostenibilidad de las empresas, especialmente en el sector bancario. La capacidad de predecir qué clientes tienen una mayor probabilidad de abandonar permite a las instituciones financieras implementar estrategias proactivas para mejorar la retención y reducir la pérdida de ingresos.
# 
# Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje automático para predecir la salida de clientes de un banco. Utilizaremos un conjunto de datos que contiene diversas características demográficas y financieras de los clientes, como edad, género, saldo de cuenta, y comportamiento de uso de productos bancarios. El objetivo principal es identificar a los clientes que tienen una mayor probabilidad de dejar el banco, permitiendo así a la institución tomar medidas preventivas.
# 
# Para lograr esto, seguiremos los siguientes pasos:
# 
# 1. **Preparación de los Datos**: Descargaremos y prepararemos los datos para el análisis. Esto incluye la limpieza de datos, transformación de variables categóricas y normalización de características numéricas.
#    
# 2. **Análisis del Equilibrio de Clases**: Investigaremos el equilibrio de la variable objetivo (Exited) para entender la distribución de clientes que se quedan frente a los que se van.
# 
# 3. **Entrenamiento Inicial del Modelo**: Entrenaremos un modelo inicial sin tener en cuenta el desequilibrio de clases y evaluaremos su desempeño.
# 
# 4. **Mejora del Modelo**: Utilizaremos técnicas para manejar el desequilibrio de clases, como el sobremuestreo (SMOTE) y el submuestreo, y compararemos los resultados para seleccionar el mejor modelo.
# 
# 5. **Evaluación Final**: Realizaremos una prueba final del modelo seleccionado para medir su desempeño y su capacidad de generalización en datos no vistos.
# 
# Al final del proyecto, evaluaremos el modelo utilizando métricas clave como el F1 Score y el AUC-ROC, asegurándonos de que nuestro enfoque no solo sea preciso, sino también robusto y capaz de manejar el desequilibrio de clases presente en los datos. Este proyecto no solo proporciona una solución técnica para la predicción de la salida de clientes, sino que también ofrece una visión valiosa que puede ayudar al banco a mejorar su retención de clientes y su estrategia comercial a largo plazo.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, Jeniifer. Buen trabajo con el desarrollo de esta sección de introducción del proyecto.
# </div>
# 
# 

# ## Preparación de los Datos

# In[123]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Genial, buen trabajo importando las librerías necesarias para el desarrollo del proyecto.
# </div>
# 

# In[75]:


data = pd.read_csv('/datasets/Churn.csv')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, buen trabajo cargando los datos necesarios para el desarrollo del proyecto.
# </div>

# In[76]:


print(data.head())


# In[77]:


data.info()


# In[78]:


print(data.isnull().sum())


# In[79]:


# Rellenar valores faltantes con la mediana
median_tenure = data['Tenure'].median()
data['Tenure'].fillna(median_tenure, inplace=True)

print(data['Tenure'].isnull().sum())


# In[80]:


# Eliminar columnas irrelevantes
data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Convertir variables categóricas en variables dummy
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)

print(data.head())


# ### Conclusiones:
# - El conjunto de datos contiene 14 columnas y 10,000 filas, incluyendo la variable objetivo Exited. 
# - Las columnas RowNumber, CustomerId y Surname son irrelevantes para el análisis predictivo y pueden ser eliminadas. 
# - La columna Tenure tiene 909 valores NaN, se imputaron los valores faltantes con la mediana para darle una solución rápida y sencilla, eliminando todos los valores faltantes en Tenure.
# - Eliminamos las columnas irrelevantes y convertimos las variables categóricas (Geography y Gender) en variables dummy.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con el desarrollo de esta exploración y modificación inicial del contenido de los datos.⁣ </div>

# ## Análisis del Equilibrio de Clases

# In[81]:


# Distribución de la variable objetivo
data['Exited'].value_counts(normalize=True).plot(kind='bar')


# ### Conclusiones:
# - El conjunto de datos está desequilibrado, lo que puede afectar el rendimiento de los modelos predictivos si no se maneja adecuadamente. 
# - La distribución de la variable Exited tiene un desequilibrio significativo: aproximadamente el 80% de los clientes no abandonaron el banco (Exited=0) y solo el 20% lo hizo (Exited=1).

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, buen trabajo desarrollando este estudio del equilibrio de clases a predecir dentro del set de datos. ⁣ </div>

# ## Entrenamiento del Modelo sin Tener en Cuenta el Desequilibrio

# In[82]:


# División en conjuntos de entrenamiento (80%) y prueba (20%)
features = data.drop(columns=['Exited'])
target = data['Exited']

features_train, features_test, target_train, target_test = train_test_split(features,target, test_size=0.2, random_state=12345)

# Entrenamiento de un modelo inicial
model = RandomForestClassifier(random_state=12345)
model.fit(features_train, target_train)

target_pred = model.predict(features_test)

# Precision, Recall y F1-Score
precision = precision_score(target_test, target_pred)
recall = recall_score(target_test, target_pred)
f1 = f1_score(target_test, target_pred)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


# ### Conclusiones:
# El modelo inicial mostró un desempeño aceptable en términos de precisión general, pero la precisión y el recall para la clase minoritaria (Exited=1) fueron significativamente bajos, lo que confirma la necesidad de abordar el desequilibrio de clases.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo desarrollando ⁣el split correspondiente a los datos con el fin de poder desplegar de manera correcta los modelos, Poe otro lado, buen trabajo desplegando el Random Forest y generando métricas de desempeños que permiten entender las afecciones un set con desbalance de clases sobre la calidad de un modelo.  </div>

# ## Mejora del Modelo y Corrección del Desequilibrio de Clases

# Método 1: Submuestreo de la Clase Mayoritaria

# In[96]:


def downsample(features, target, fraction):
    features_zeros = features[ target == 0]
    features_ones = features[ target == 1]
    target_zeros = target[ target == 0]
    target_ones = target[ target == 1]
    
    features_downsampled = pd.concat([features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat([target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    
    features_downsampled, target_downsampled = shuffle(features_downsampled, target_downsampled, random_state=12345)
    return features_downsampled, target_downsampled

print(features_train.shape)

features_sample = features_train.sample(frac=0.1, random_state=12345)
print(features_sample.shape)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, Jennifer, buen trabajo aplicando el sub muestreo al conjunto de datos. </div>

# In[120]:


fraction = 0.2
features_downsampled, target_downsampled = downsample(features_train, target_train, fraction)

# Entrenamiento del modelo
model = RandomForestClassifier(random_state=12345)
model.fit(features_downsampled, target_downsampled)

# Predecir en el conjunto de prueba
prediction = model.predict(features_test)

# Evaluar el modelo
precision = precision_score(target_test, prediction)
recall = recall_score(target_test, prediction)
f1 = f1_score(target_test, prediction)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, Jennifer, buen trabajo desplegando y evaluando el Random Forest con el nuevo set que se aplicó el sub muestreo.  </div>

# Método 2: Modelo con ponderación de clases

# In[115]:


model_weighted = RandomForestClassifier(class_weight='balanced', random_state=12345)
model_weighted.fit(features_train, target_train)

# Predecir en el conjunto de prueba
y_pred_weighted = model_weighted.predict(features_test)

# Evaluar el modelo
precision = precision_score(target_test, y_pred_weighted)
recall = recall_score(target_test, y_pred_weighted)
f1 = f1_score(target_test, y_pred_weighted)
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Excelente, buen trabajo desplegando y evaluando el Random Forest con el argumento de balanceo.</div>

# ### Conclusiones:
# - El submuestreo equilibró las clases en el conjunto de entrenamiento, mejorando el rendimiento del modelo en términos de recall y F1-Score para la clase minoritaria. Sin embargo, puede introducir variabilidad debido a la reducción del tamaño del conjunto de entrenamiento.
# - La ponderación de clases permitió al modelo prestar más atención a la clase minoritaria sin reducir el tamaño del conjunto de datos. El rendimiento mejoró en términos de precisión, mostrando ser una técnica efectiva para manejar el desequilibrio de clases.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con el desarrollo de las conclusiones sobre lo desarrollado. </div>

# ## Prueba Final

# Comparamos los resultados de ambos métodos y seleccionamos el modelo entrenado con submuestreo como el mejor.

# In[134]:


# Entrenamiento del modelo
model = RandomForestClassifier(random_state=12345)
model.fit(features_downsampled, target_downsampled)

prediction = model.predict(features_test)

print("Reporte de clasificación del mejor modelo")
print("F1 Score:", f1_score(target_test, prediction))
print("AUC-ROC:", roc_auc_score(target_test, model.predict_proba(features_test)[:, 1]))


# ### Conclusiones: 
# - El modelo final entrenado mostró un buen equilibrio con un F1 Score y un AUC-ROC satisfactorios, demostrando ser efectivo para predecir la salida de clientes.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Perfecto, buen trabajo con el desarrollo y análisis de la prueba final. </div>

# # Conclusiones Finales
# En resumen, este proyecto desarrolló un modelo predictivo para la retención de clientes en un banco, abordando adecuadamente el problema del desequilibrio de clases. La imputación de valores faltantes y la aplicación de técnicas de submuestreo fueron cruciales para mejorar la precisión del modelo. El resultado final es un modelo que puede ayudar al banco a identificar a los clientes con mayor riesgo de abandono, permitiendo la implementación de estrategias de retención más efectivas.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con el desarrollo de la sección de conclusiones finales. </div>
