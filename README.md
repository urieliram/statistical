# Aprendizaje automático

Repositorio de actividades del curso de aprendizaje automático. La descripción del curso y las actividades se pueden encontrar en el [enlace del curso](https://github.com/satuelisa/StatisticalLearning). Los datos a usar del libro están disponibles en [datos](https://hastie.su.domains/ElemStatLearn/datasets/).
---

+ [Chapter 1: Introduction](#tarea-1-introduction)
+ [Chapter 2: Supervised learning]()
+ [Chapter 3: Linear Regresion]()

---

## Tarea 1: Introducción
>Instructions: Identify one or more learning problems in your thesis work and identify goals and elements.

En el trabajo de investigación del alumno se tienen resultados preliminares de 320 instancias resueltas de la programación de los generadores (Unit Commitment) para el día siguiente resuelto con un modelo de programación entera mixta.

Como entrada se tienen:
- Datos de disponibilidad de las unidades (variable categórica).
- La demanda eléctrica por regiones (variable escalar).
- La región a la que pertenecen los generadores (variable categórica).
- Requerimientos de reservas por tipo de reserva (variable escalar).

En las salidas de cada instancia se tiene el resultado de:
- Los generadores que fueron seleccionados para ser prendidos o apagados en cada hora (variable categórica).
- La producción de los generadores  en cada hora (variable escalar).
- Líneas de transmisión que fueron violadas (variable categórica).
- Las pérdidas en el sistema (variable escalar).
- Tiempos de ejecución de cada instancia (variable escalar).

Preguntas de investigación:
1. ¿Es posible determinar que conjunto de lineas de transmisión serán violadas conociendo los generadores que serán prendidos o la demanda por regiones?
2. ¿Es posible predecir las pérdidas en el sistema conociendo la salida de potencia de cada generador?
3. ¿Cómo puedo agrupar los generadores de acuerdo a aquellos que seguramente serán elegidos y a aquellos que es mas o menos probable que sean elegidos y aquellos que probablemente no?
4. ¿Es posible predecir el tiempo de ejecución de la programación conociendo la combinación de las lineas de transmisión violadas?
5. ¿Puede hacerse un modelo que explique los precios de las reservas en el sistema en función de los requerimientos de reservas requeridas?

Otras preguntas indirectamente relacionadas pero que se tienen datos para estudiar:

1. ¿Es posible predecir la aportación de agua de un embalse de un mes teniendo en cuenta las aportaciones mensuales de los demás embalses en una cuenca?. 
---
## Tarea 2: Aprendizaje supervisado

>Instructions: First carry out Exercise 2.8 of the textbook with their ZIP-code data and then replicate the process the best you manage to some data from you own problem that was established in the first homework.

El código completo de esta tarea se encuentra en [Tarea2.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea2.ipynb), aquí solo se presentan los resultados y partes importantes del código.

Los datos utilizados están disponibles en el [Repositorio](https://drive.google.com/drive/folders/159GnBJQDxTY9oYqPBZzdNghyb4Gd9pDS?usp=sharing).

### **Regresión líneal**
A continuación usaremos un modelo de regresión líneal para resolver el problema de ZIP-code [2,3] del libro [liga](https://link.springer.com/book/10.1007/978-0-387-84858-7). Usando la librería sklearn obtenemos un modelo de predicción de los datos de entrenamiento. Posteriomente, calculamos los errores entre la predicción 

y_pred y los datos de entrenamiento Y. Además, los errores de predicción son representados por un histograma.

```python
model = LinearRegression().fit(X, Y) #https://realpython.com/linear-regression-in-python/
y_pred = model.predict(X)
error = Y - y_pred
dfx = pd.DataFrame(error,Y)
plt = dfx.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf91', zorder=2, rwidth=0.9)
err_regress = mean_absolute_error(Y,y_pred)
```

![image](https://user-images.githubusercontent.com/54382451/150030914-b242c594-95f1-4124-b4a1-12f2a5f19f11.png)

Ahora, utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba. Además,  calculamos los errores entre la predicción y_pred2 y los datos de prueba Yt. Los errores de la predicción con datos de prueba son representados por un histograma.

```python
y_pred2 = model.predict(Xt)
error2 = Yt - y_pred2
df = pd.DataFrame(error2,Yt)
plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf40', zorder=2, rwidth=0.9)
err_regress_t = mean_absolute_error(Yt,y_pred2)
```

![image](https://user-images.githubusercontent.com/54382451/150031119-dc8de852-7b2d-4dbd-8d3b-8509bd57e46f.png)

Por último, calculamos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba.
>MAE del modelo de regresión con datos de entrenamiento: 7.02644
>
>MAE del modelo de regresión con datos de prueba: 6.88467


### **k-nearest neighbors**
A continuación usaremos un modelo de k-NN para resolver el problema de ZIP-code [2,3] del libro [liga](https://link.springer.com/book/10.1007/978-0-387-84858-7).

Para cada k se obtiene un modelo K-NN con los que se calculan el **error absoluto medio (MAE)** para los datos de entrenamiento X como de prueba Xt.

```python
k_list    = [1, 3, 5, 7, 15] ## Lista de parámetros k
mae_knn   = []               ## Guarda valores de error de diferentes k en datos de entrenamiento
mae_knn_t = []               ## Guarda valores de error de diferentes k en datos de prueba
```

Para cada k se obtiene un modelo K-NN con los que se calculan el **error absoluto medio (MAE)** para los datos de entrenamiento X como de prueba Xt.

```python
for k in k_list:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, Y)
    y_pred   = model.predict(X) 
    y_predt   = model.predict(Xt)
    mae_knn.append(mean_absolute_error(Y,y_pred))
    mae_knn_t.append(mean_absolute_error(Yt,y_predt))
    error = Y - y_pred
    errort = Yt - y_predt
    ## Líneas de código que imprimen los histogramas de los errores de cada modelo con un parámetro k.
    df = pd.DataFrame(error,Y)
    plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf91', zorder=2, rwidth=0.9)    
    dft = pd.DataFrame(errort,Yt)
    plt2 = dft.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf40', zorder=2, rwidth=0.9)
```

Por último mostramos el error absoluto medio (MAE) de los datos de entrenamiento así como de los datos de prueba del modelo K-NN para cada parámetro k=[1, 3, 5, 7, 15].
>MAE del modelo de KNN con datos de entrenamiento: [0.0, 0.00719, 0.010079, 0.01337, 0.02371]
>
>MAE del modelo de KNN con datos de prueba: [0.02472, 0.03021, 0.03296, 0.03767, 0.04761]

A continuación se muestran los histogramas de error del modelo de k-NN con parametro k=15 para las series de entrenamiento(derecha) y prueba(izquierda).

![image](https://user-images.githubusercontent.com/54382451/150032424-f01764a0-4645-4202-b76c-df40f1c37895.png)
![image](https://user-images.githubusercontent.com/54382451/150032720-8d57f11a-2fbc-4b99-a836-1036af9ba3d8.png)

Finalmente, comparamos graficamete los errores en la clasificación de ZIP-code entre modelo de regresión lineal y el K-NN con diferentes valores de k=[1, 3, 5, 7, 15].

![image](https://user-images.githubusercontent.com/54382451/150032555-9bb8e614-654e-433e-bf00-0471af83a8a5.png)

## **Estimación de pérdidas eléctricas en regiones con técnicas de  regresión líneal y k-NN**
Los datos que se usarán en este ejercicio son resultados de la planeación de la operación eléctrica del sistema eléctrico interconectado en México que consta de 320 instancias.
Las columnas de los datos son los resultados por región y por hora del día. Se dan resultados de generación térmica (GenTer), generación hidráulica (GenHid), generación renovable (GenRE), Gneración no programable (GenNP), Generación total (GenTot), demanda de la región (Demanda), Cortes de energía (Corte), Excedentes de energía(Excedente),Potencia (PotInt), precio Marginal y pérdidas de la región.

Una parte importante del proyecto de tesis es la estimación de pérdidas eléctricas en la red agregando restricciones al modelo MILP de programación de las unidades.

¿Es posible determinar un modelo lineal de regresión o k-NN que estime las pérdidas en función de la generación y demanda conocidas?

¿Puede agregarse este modelo de estimación de pérdidas al modelo MILP a manera de restricciónes para acelerar el proceso de convergencia?

### **Estimación de pérdidas eléctricas con regresión líneal**
A continuación, obtenemos un modelo de predicción de los datos de entrenamiento usando regresión lineal. Posteriomente, calculamos los errores entre la predicción y_pred y los datos de entrenamiento Y. Los errores de la predicción con datos de entrenamiento son representados por un histograma.

```python
model = LinearRegression().fit(X, Y) #https://realpython.com/linear-regression-in-python/
y_pred = model.predict(X)
error = Y - y_pred
dfx = pd.DataFrame(error,Y)
plt = dfx.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#777bd4', zorder=2, rwidth=0.9)
err_regress = mean_absolute_error(Y,y_pred)
```

![image](https://user-images.githubusercontent.com/54382451/150034666-8d10414c-cc85-47c2-996e-df00fd0835e3.png)

Ahora, utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba. Además, calculamos los errores entre la predicción y_pred2 y los datos de prueba Yt. Los errores de la predicción con datos de prueba son representados por un histograma.

```python
y_pred2 = model.predict(Xt)
error2 = Yt - y_pred2
df = pd.DataFrame(error2,Yt)
plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#76ced6', zorder=2, rwidth=0.9)
err_regress_t = mean_absolute_error(Yt,y_pred2)
```

![image](https://user-images.githubusercontent.com/54382451/150034749-5068f93a-1ed0-41f9-a6e3-91f9e9f23b56.png)

Por último, calculamos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba.

>MAE del modelo de regresión con datos de entrenamiento: 7.026442
>
>MAE del modelo de regresión con datos de prueba: 6.884672

### **Estimación de pérdidas eléctricas con k-NN**
Usaremos los arreglos mae_knn y mae_knn_y para guardar los resultados del error de predicción de cada modelo de K-NN con parámetro k=[1, 3, 5, 7, 15].

Para cada k se obtiene un modelo K-NN con los que se calculan el **error absoluto medio (MAE)** para los datos de entrenamiento X como de prueba Xt.

```python
for k in k_list:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X, Y)
    y_pred   = model.predict(X) 
    y_predt   = model.predict(Xt)
    mae_knn.append(mean_absolute_error(Y,y_pred))
    mae_knn_t.append(mean_absolute_error(Yt,y_predt))
    error = Y - y_pred
    errort = Yt - y_predt
    ## Líneas de código que imprimen los histogramas de los errores de cada modelo con un parámetro k.
    df = pd.DataFrame(error,Y)
    plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#777bd4', zorder=2, rwidth=0.9)    
    dft = pd.DataFrame(errort,Yt)
    plt2 = dft.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#76ced6', zorder=2, rwidth=0.9)
```

A continuación se muestran los histogramas de error del modelo de k-NN con parametro k=15 para las series de entrenamiento(derecha) y prueba(izquierda).

![image](https://user-images.githubusercontent.com/54382451/150038179-d0cdaf42-b93d-4e3d-8c93-ace876d511f3.png)
![image](https://user-images.githubusercontent.com/54382451/150038193-510363c3-97c5-45a6-88da-b0da768aa4b4.png)

Por último mostramos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba del modelo K-NN para cada parámetro k=[1, 3, 5, 7, 15].

>MAE del modelo de KNN con datos de entrenamiento: [1.3242, 1.2696, 1.2798, 1.2984, 1.4456, 1.6820, 1.3242, 1.2696, 1.2798, 1.2984, 1.4456, 1.6820]
>
>MAE del modelo de KNN con datos de prueba: [1.0412, 1.1468, 1.2613, 1.3769, 1.7017, 2.1300, 1.0412, 1.1468, 1.2613, 1.3769, 1.7017, 2.1300]

Finalmente comparamos los errores en la clasificación de ZIP-code del modelo de regresión lineal contra el de K-NN con diferentes valores de k.

![image](https://user-images.githubusercontent.com/54382451/150039065-a0d1c862-f82e-48fb-8ec5-025ac7691591.png)

## **Conclusiones de la tarea**
Las herramientas de **regresión lineal** y **k-NN** pueden ser útiles para predecir en base resultados de planeación de un sistema eléctrico las pérdidas eléctricas en una región usando como datos de entrada la demanda, y generación (térmica, hidráulica, renovable, etc) de las regiones. Se observó que el k-NN aplicadpo a los datos tiene un mejor desempeño que la regresión líneal, sin embargo el k-NN no genera un modelo matemático que podamos usar para obtener resultados de predicción sin consultar los datos de la instancia; el consultar los datos de la instancia cada vez que se hace una predicción implica más costo computacional que obtener un modelo de regresión líneal una sola vez. 
Una forma de utilizar el modelo líneal que se obtiene por la regresión para disminuir el tiempo de solución de la programación de las unidades es agregar al MILP el modelo de regresión por regiones como restricciones o cortes con le objetivo de acotar el espacio de solución. Siempre que se agrege estos cortes al mismo sistema eléctrico del que se obtuvo la información para hacer la regresión.



