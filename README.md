# Aprendizaje automático

Repositorio de actividades del curso de aprendizaje automático. La descripción del curso y las actividades se pueden encontrar en el [enlace del curso](https://github.com/satuelisa/StatisticalLearning). Los datos a usar del libro están disponibles aquí: [dataset del libro](https://hastie.su.domains/ElemStatLearn/datasets/).

---

+ [Tarea 1 Introducción](#tarea-1-introduction)
+ [Tarea 2 Aprendizaje supervisado](#tarea-2-aprendizaje-supervisado)
+ [Tarea 3 Regresión lineal](#tarea-3-regresión-lineal)
+ [Tarea 4 Clasificación](#tarea-4-clasificación)
+ [Tarea 5 Expansión de base](#tarea-5-expansión-de-base)
+ [Tarea 6 Suavizado](#tarea-6-suavizado)
+ [Tarea 7 Evaluación](#tarea-7-evaluación)
+ [Tarea 8 Inferencia](#tarea-8-inferencia)
+ [Tarea 9 Modelos Aditivos y Árboles](#tarea-9-modelos-aditivos-y-árboles) 
+ [Tarea 10 Impulso](#tarea-10-impulso)
+ [Tarea 11 Redes neuronales](#tarea-11-redes-neuronales)
+ [Tarea 12 Máquinas de Vectores de Soporte](#tarea-12-máquinas-de-vectores-de-soporte)
+ [Tarea 13 Prototipos y vecinos](#tarea-13-prototipos-y-vecinos)
+ [Tarea 14 Aprendizaje no supervisado](#tarea-14-aprendizaje-no-supervisado)
+ [Tarea 15 Bosque aleatorio](#tarea-15-bosque-aleatorio)

---

![image](https://github.com/urieliram/statistical/blob/main/figures/clasificaciondisco.PNG)


Figura de [Saul Dobilas](https://towardsdatascience.com/k-nearest-neighbors-knn-how-to-make-quality-predictions-with-supervised-learning-d5d2f326c3c2).

## Tarea 1 Introducción
>**Instructions:** Identify one or more learning problems in your thesis work and identify goals and elements.

En el trabajo de investigación del alumno se tienen resultados preliminares de 320 instancias resueltas de la programación de los generadores (Unit Commitment) para el día siguiente resuelto con un modelo de programación entera-mixta.

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

## Tarea 2 Aprendizaje supervisado

>Instructions: First carry out Exercise 2.8 of the textbook with their ZIP-code data and then replicate the process the best you manage to some data from you own problem that was established in the first homework.

El código completo de esta tarea se encuentra en [Tarea2.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea2.ipynb), aquí solo se presentan los resultados y partes importantes del código.

Los datos utilizados están disponibles en el [repositorio](https://drive.google.com/drive/folders/159GnBJQDxTY9oYqPBZzdNghyb4Gd9pDS?usp=sharing).

### Regresión líneal
A continuación usaremos un modelo de regresión líneal para resolver el problema de ZIP-code [2,3] del libro [liga](https://link.springer.com/book/10.1007/978-0-387-84858-7). Usando la librería sklearn obtenemos un modelo de predicción de los datos de entrenamiento. Posteriomente, calculamos los errores entre la predicción y_pred y los datos de entrenamiento Y. Además, los errores de predicción son representados por un histograma.

```python
model = LinearRegression().fit(X, Y) #https://realpython.com/linear-regression-in-python/
y_pred = model.predict(X)
error = Y - y_pred
dfx = pd.DataFrame(error,Y)
plt = dfx.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf91', zorder=2, rwidth=0.9)
err_regress = mean_absolute_error(Y,y_pred)
```
![image](https://github.com/urieliram/statistical/blob/main/figures/hist1.png)

Ahora, utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba. Además,  calculamos los errores entre la predicción y_pred2 y los datos de prueba Yt. Los errores de la predicción con datos de prueba son representados por un histograma.

```python
y_pred2 = model.predict(Xt)
error2 = Yt - y_pred2
df = pd.DataFrame(error2,Yt)
plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#86bf40', zorder=2, rwidth=0.9)
err_regress_t = mean_absolute_error(Yt,y_pred2)
```
![image](https://github.com/urieliram/statistical/blob/main/figures/hist2.png)

Por último, calculamos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba.
>MAE del modelo de regresión con datos de entrenamiento: 7.02644
>
>MAE del modelo de regresión con datos de prueba: 6.88467


### k-nearest neighbors

A continuación usaremos un modelo de k-NN para resolver el problema de ZIP-code [2,3] del libro [liga](https://link.springer.com/book/10.1007/978-0-387-84858-7).
Para cada k=[1, 3, 5, 7, 15] se obtiene un modelo k-NN con los que se calculan el **error absoluto medio (MAE)** para los datos de entrenamiento X como de prueba Xt.

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

Por último mostramos el error absoluto medio (MAE) de los datos de entrenamiento así como de los datos de prueba del modelo k-NN para cada parámetro k=[1, 3, 5, 7, 15].
>MAE del modelo de KNN con datos de entrenamiento: [0.0, 0.00719, 0.010079, 0.01337, 0.02371]
>
>MAE del modelo de KNN con datos de prueba: [0.02472, 0.03021, 0.03296, 0.03767, 0.04761]

A continuación se muestran los histogramas de error del modelo de k-NN con parametro k=15 para las series de entrenamiento (derecha) y prueba (izquierda).

![image](https://github.com/urieliram/statistical/blob/main/figures/hist3.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/hist4.png)

Finalmente, comparamos graficamete los errores en la clasificación de ZIP-code entre modelo de regresión lineal y el k-NN con diferentes valores de k=[1, 3, 5, 7, 15].

![image](https://github.com/urieliram/statistical/blob/main/figures/MAE1.png)

### Estimación de pérdidas eléctricas en regiones con técnicas de  regresión líneal y k-NN
Los datos que se usarán en este ejercicio son resultados de la planeación de la operación eléctrica del sistema eléctrico interconectado en México que consta de 320 instancias.
Las columnas de los datos son los resultados por región y por hora del día. Se dan resultados de generación térmica (GenTer), generación hidráulica (GenHid), generación renovable (GenRE), Gneración no programable (GenNP), Generación total (GenTot), demanda de la región (Demanda), Cortes de energía (Corte), Excedentes de energía(Excedente),Potencia (PotInt), precio Marginal y pérdidas de la región.

Una parte importante del proyecto de tesis es la estimación de pérdidas eléctricas en la red agregando restricciones al modelo MILP de programación de las unidades.

¿Es posible determinar un modelo lineal de regresión o k-NN que estime las pérdidas en función de la generación y demanda conocidas?

¿Puede agregarse este modelo de estimación de pérdidas al modelo MILP a manera de restricciónes para acelerar el proceso de convergencia?

#### Estimación de pérdidas eléctricas con regresión líneal
A continuación, obtenemos un modelo de predicción de los datos de entrenamiento usando regresión lineal. Posteriomente, calculamos los errores entre la predicción y_pred y los datos de entrenamiento Y. Los errores de la predicción con datos de entrenamiento son representados por un histograma.

```python
model = LinearRegression().fit(X, Y)
y_pred = model.predict(X)
error = Y - y_pred
dfx = pd.DataFrame(error,Y)
plt = dfx.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#777bd4', zorder=2, rwidth=0.9)
err_regress = mean_absolute_error(Y,y_pred)
```

![image](https://github.com/urieliram/statistical/blob/main/figures/hist5.png)

Ahora, utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba. Además, calculamos los errores entre la predicción y_pred2 y los datos de prueba Yt. Los errores de la predicción con datos de prueba son representados por un histograma.

```python
y_pred2 = model.predict(Xt)
error2 = Yt - y_pred2
df = pd.DataFrame(error2,Yt)
plt = df.hist(column=0, bins=25, grid=False, figsize=(6,3), color='#76ced6', zorder=2, rwidth=0.9)
err_regress_t = mean_absolute_error(Yt,y_pred2)
```

![image](https://github.com/urieliram/statistical/blob/main/figures/hist6.png)

Por último, calculamos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba.

>MAE del modelo de regresión con datos de entrenamiento: 7.026442
>
>MAE del modelo de regresión con datos de prueba: 6.884672

#### Estimación de pérdidas eléctricas con k-NN
Usaremos los arreglos mae_knn y mae_knn_y para guardar los resultados del error de predicción de cada modelo de k-NN con parámetro k=[1, 3, 5, 7, 15].

Para cada k se obtiene un modelo k-NN con los que se calculan el **error absoluto medio (MAE)** para los datos de entrenamiento X como de prueba Xt.

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

A continuación se muestran los histogramas de error del modelo de k-NN con parametro k=15 para las series de entrenamiento (derecha) y prueba (izquierda).

![image](https://github.com/urieliram/statistical/blob/main/figures/hist7.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/hist8.png)

Por último mostramos el **error absoluto medio (MAE)** de los datos de entrenamiento así como de los datos de prueba del modelo k-NN para cada parámetro k=[1, 3, 5, 7, 15].

>MAE del modelo de KNN con datos de entrenamiento: [1.3242, 1.2696, 1.2798, 1.2984, 1.4456, 1.6820, 1.3242, 1.2696, 1.2798, 1.2984, 1.4456, 1.6820]
>
>MAE del modelo de KNN con datos de prueba: [1.0412, 1.1468, 1.2613, 1.3769, 1.7017, 2.1300, 1.0412, 1.1468, 1.2613, 1.3769, 1.7017, 2.1300]

Finalmente comparamos los errores en la clasificación de ZIP-code del modelo de regresión lineal contra el de k-NN con diferentes valores de k.

![image](https://github.com/urieliram/statistical/blob/main/figures/MAE2.png)

### Conclusiones de la tarea 2
Las herramientas de **regresión lineal** y **k-NN** pueden ser útiles para predecir en base resultados de planeación de un sistema eléctrico las pérdidas eléctricas en una región usando como datos de entrada la demanda, y generación (térmica, hidráulica, renovable, etc) de las regiones. Se observó que el k-NN aplicado a los datos tiene un mejor desempeño que la regresión líneal, sin embargo el k-NN no genera un modelo matemático que podamos usar para obtener resultados de predicción sin consultar los datos de la instancia; el consultar los datos de la instancia cada vez que se hace una predicción implica más costo computacional que obtener un modelo de regresión líneal una sola vez. 
Una forma de utilizar el modelo líneal que se obtiene por la regresión para disminuir el tiempo de solución de la programación de las unidades es agregar al MILP el modelo de regresión por regiones como restricciones o cortes con le objetivo de acotar el espacio de solución. Siempre que se agregen estos cortes al mismo sistema eléctrico del que se obtuvo la información para hacer la regresión.

---

## **Tarea 3 Regresión Lineal**

>**Instructions:** Repeat the steps of the prostate cancer example in Section 3.2.1 using Python, first as a uni-variate problem using the book's data set and then as a multi-variate problem with data from your own project. Calculate also the p-values and the confidence intervals for the model's coefficients for the uni-variate version. Experiment, using libraries, also with subset selection.
>
El código completo de esta tarea se encuentra en [Tarea3.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea2.ipynb), aquí solo se presentan los resultados y partes importantes del código.

### Regresión líneal en cáncer de próstata
A continuación repetiremos el ejercicio 3.2.1 del [libro](https://link.springer.com/book/10.1007/978-0-387-84858-7) aplicando un modelo de regresión líneal para predecir cáncer de próstata.
Primero estandarizaremos los datos de los regresores `X_train` y `X_test` restando la media y dividiendo entre la varianza.
```python
df1.std(numeric_only = True) 
df1.mean(numeric_only = True)
df1 = df1 - df1.mean(numeric_only = True)
df1 = df1 / df1.std(numeric_only = True) 
X_train = df1.to_numpy()   ## Predictors
X_train = sm.add_constant(X_train)
y_train = df2.to_numpy()   ## Outcome
```

A continuación, obtenemos un modelo de predicción de los datos de entrenamiento usando regresión lineal usando la librería **statsmodels**.

```python
X_train = sm.add_constant(X_train)
model   = sm.OLS(y_train, X_train)
results = model.fit()
print(results.summary())
```

Los resultados de la regresión se muestran en la tabla siguiente. En las tres últimas columnas se muestran el valor **p** y los **intervalos de confianza** de los coeficientes.

```
OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.694
Model:                            OLS   Adj. R-squared:                  0.652
Method:                 Least Squares   F-statistic:                     16.47
Date:                Tue, 25 Jan 2022   Prob (F-statistic):           2.04e-12
Time:                        06:39:43   Log-Likelihood:                -67.505
No. Observations:                  67   AIC:                             153.0
Df Residuals:                      58   BIC:                             172.9
Df Model:                           8                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.4523      0.087     28.182      0.000       2.278       2.627
x1             0.7164      0.134      5.366      0.000       0.449       0.984
x2             0.2926      0.106      2.751      0.008       0.080       0.506
x3            -0.1425      0.102     -1.396      0.168      -0.347       0.062
x4             0.2120      0.103      2.056      0.044       0.006       0.418
x5             0.3096      0.125      2.469      0.017       0.059       0.561
x6            -0.2890      0.155     -1.867      0.067      -0.599       0.021
x7            -0.0209      0.143     -0.147      0.884      -0.306       0.264
x8             0.2773      0.160      1.738      0.088      -0.042       0.597
==============================================================================
Omnibus:                        0.825   Durbin-Watson:                   1.690
Prob(Omnibus):                  0.662   Jarque-Bera (JB):                0.389
Skew:                          -0.164   Prob(JB):                        0.823
Kurtosis:                       3.178   Cond. No.                         4.44
==============================================================================
```

Calculamos los errores entre la predicción `y_pred` y los datos de entrenamiento `y_train`. Los errores son representados en el siguiente histograma.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist9.png)

Utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba `y_test`. Además, calculamos los errores entre la predicción `y_pred2` y los datos de prueba `y_test`. Los errores de la predicción con datos de prueba son representados en el siguiente histograma.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist10.png)

Calculamos el error absoluto medio (MAE) de los datos de entrenamiento así como de los datos de prueba en la predicción de cancer de próstata.

>MAE del modelo de regresión con datos de entrenamiento: 0.4986
>MAE del modelo de regresión con datos de prueba: 0.5332


#### Regresión del mejor subconjunto aplicado a la predicción de cáncer de próstata 

A continuación aplicaremos la técnica de regresión del mejor subconjunto **(Best Subset Regression)** a los datos de entrenamiento de cáncer de próstata.

```python
## Loop over all possible numbers of features to be included
results = pd.DataFrame(columns=['num_features', 'features', 'MAE'])
for k in range(1, X_train.shape[1] + 1):
    # Loop over all possible subsets of size k
    for subset in itertools.combinations(range(X_train.shape[1]), k):
        subset = list(subset)        
        linreg_model = LinearRegression().fit(X_train[:, subset], y_train)
        linreg_prediction = linreg_model.predict(X_train[:, subset])
        linreg_mae = np.mean(np.abs(y_train - linreg_prediction))
        #print(subset," ",linreg_mae)
        results = results.append(pd.DataFrame([{'num_features': k,
                                                'features': subset,
                                                'MAE': linreg_mae}]))
print(results.sort_values('MAE'))
err_regress_subset = results.sort_values('MAE')['MAE'].head(1)
```
Los resultados del método sugiere utilizar las variables (features) [1, 2, 3, 4, 5, 6, 8]; es decir: ['lweight','age','lbph','svi','lcp','gleason','pgg45'] para lograr un mínimo error entre todas las combinaciones de las variables.

```
   num_features                     features       MAE
0             7        [1, 2, 3, 4, 5, 6, 8]  0.497997
0             8     [0, 1, 2, 3, 4, 5, 6, 8]  0.497997
0             9  [0, 1, 2, 3, 4, 5, 6, 7, 8]  0.498614
0             8     [1, 2, 3, 4, 5, 6, 7, 8]  0.498614
0             7        [1, 2, 3, 4, 5, 6, 7]  0.504766
..          ...                          ...       ...
0             2                       [0, 7]  0.904978
0             1                          [7]  0.904978
0             2                       [0, 3]  0.905767
0             1                          [3]  0.905767
0             1                          [0]  0.961085
```

### Regresión lineal en predicción de demanda eléctrica

A continuación aplicaremos **regresión líneal** a la predicción de demanda eléctrica. La variable independiente Y serán los datos de demanda de 24 horas antes en intervalos de 5 minutos (288 datos), y las variables independientes X serán los datos de otros días con una mayor correlación. Los datos se han dividido en datos de entrenamiento (`x_train`,`x_train`) y datos de prueba (`x_test`,`y_test`). El objetivo es encontrar el mejor modelo de pronóstico para los datos de demanda.

Los datos usados en esta sección están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing)

A continuación, obtenemos un modelo de predicción de los datos de entrenamiento usando **regresión lineal**. Ahora, calculamos los errores entre la predicción `y_pred` y los datos de entrenamiento `y_train`. Los errores son representados por un histograma.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist11.png)

Utilizamos el modelo obtenido con los datos de entrenamiento para predecir los datos de prueba `y_test`. Además, calculamos los errores entre la predicción `y_pred2` y los datos de prueba `y_test`. Los errores de la predicción con datos de prueba son representados por un histograma.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist12.png)

Comparamos el error absoluto medio (MAE) y bias de los datos de entrenamiento así como de los datos de prueba en la predicción de cancer de próstata.

>MAE y bias del modelo de regresión con datos de entrenamiento: 97.1445 , 0.0
>
>MAE y bias del modelo de regresión con datos de prueba: 176.1676 , 40.9495

A continuación aplicaremos Las técnicas de regresión lineal con reducción de dimensiones a nuestros datos de demanda. Estas técnicas pueden encontrarse en: [A Comparison of Shrinkage and Selection Methods for Linear Regression](https://towardsdatascience.com/a-comparison-of-shrinkage-and-selection-methods-for-linear-regression-ee4dd3a71f16).

#### Regresión del mejor subconjunto aplicado a la predicción de pronóstico de demanda eléctrica
Aplicamos la técnica de regresión del mejor subconjunto **(Best Subset Regression)** a los datos de entrenamiento de demanda electrica.

```python
results = pd.DataFrame(columns=['num_features', 'features', 'MAE'])
for k in range(1, X_train.shape[1] + 1):
    # Loop over all possible subsets of size k
    for subset in itertools.combinations(range(X_train.shape[1]), k):
        subset = list(subset)        
        linreg_model = LinearRegression().fit(X_train[:, subset], y_train)
        linreg_prediction = linreg_model.predict(X_train[:, subset])
        linreg_mae = np.mean(np.abs(y_train - linreg_prediction))
                                                'features': subset,
                                                'MAE': linreg_mae}]))
print(results.sort_values('MAE'))
subset_best = list(results.sort_values('MAE')['features'].head(1)[0]) ## Seleccionamos el mejor subconjunto con menor MAE
```
La salida del código nos muestra los subconjuntos (features) con el menor error absoluto medio (MAE). La aplicación de este método sugiere utilizar las variables (features) [1, 2, 3, 7, 8, 10]; es decir [X2, X3, X4, X8, X9, X11] para lograr un mínimo error entre todas las combinaciones de las variables.
```
   num_features                   features         MAE
0             6        [1, 2, 3, 7, 8, 10]   95.651466
0             7     [0, 1, 2, 3, 7, 8, 10]   95.651466
0             8  [0, 1, 2, 3, 6, 7, 8, 10]   95.724433
0             7     [1, 2, 3, 6, 7, 8, 10]   95.724433
0             8  [0, 1, 2, 3, 7, 8, 9, 10]   95.737479
..          ...                        ...         ...
0             1                       [10]  145.218521
0             2                    [0, 10]  145.218521
0             2                     [0, 5]  151.493611
0             1                        [5]  151.493611
0             1                        [0]  463.242794
```
Calculamos lo errores con los datos de prueba:

```python
modelsub = LinearRegression().fit(X_train[:, subset_best], y_train)
subset_prediction = modelsub.predict(X_test[:, subset_best])
err_test_subset  = np.mean(np.abs(y_test - subset_prediction))
bias_test_subset = bias.bias(y_test,subset_prediction,axis=0)
print("MAE y bias del modelo de regresión con datos de prueba (subset):" , err_test_subset, "," , bias_test_subset) 
```
Obtenemos los errores de los datos de prueba `y_test`.

>MAE y bias del modelo de regresión con datos de prueba (subset): 173.5274 , 40.9495

#### Regresión Ridge aplicada a la predicción de pronóstico de demanda eléctrica
Aplicaremos la técnica de regresión ridge **(Ridge Regression)** a los datos de entrenamiento de demanda eléctrica.
```python
ridge_cv = RidgeCV(normalize=True, alphas=np.logspace(-10, 1, 400))
ridge_model = ridge_cv.fit(X_train, y_train)
ridge_prediction = ridge_model.predict(X_test)
err_test_ridge = np.mean(np.abs(y_test - ridge_prediction))  ## MAE
bias_test_ridge = bias.bias(y_test,ridge_prediction,axis=0)
#print(ridge_model.intercept_)
#print(ridge_model.coef_)
print("MAE y bias del modelo de regresión con datos de prueba (ridge):" , err_test_ridge, "," , bias_test_ridge)
```
Obtenemos los errores de los datos de prueba `y_test`.

>MAE y bias del modelo de regresión con datos de prueba (ridge): 172.5657 , 40.9495

#### Regresión Lasso aplicada a la predicción de pronóstico de demanda eléctrica
Aplicaremos la técnica de regresión lasso **(lasso regression)** a los datos de entrenamiento de demanda eléctrica.

```python
lasso_cv         = LassoCV(normalize=True, alphas=np.logspace(-10, 1, 400))
lasso_model      = lasso_cv.fit(X_train, y_train)
lasso_prediction = lasso_model.predict(X_test)
err_test_lasso   = np.mean(np.abs(y_test - lasso_prediction)) ## MAE
bias_test_lasso  = bias.bias(y_test,lasso_prediction,axis=0)
#print(lasso_model.intercept_)
#print(lasso_model.coef_)
print("MAE y bias del modelo de regresión con datos de prueba (lasso):" , err_test_lasso, "," , bias_test_lasso)
```

Obtenemos los errores de los datos de prueba `y_test`.

>MAE y bias del modelo de regresión con datos de prueba (lasso): 169.1226 , 40.9495

#### Regresión de componentes principales aplicado a la predicción de pronóstico de demanda eléctrica
Aplicaremos la técnica de regresión de componentes principales **(Principal Components Regression)** a los datos de entrenamiento de demanda eléctrica.
```python
regression_model = LinearRegression(normalize=True)
pca_model = PCA()
pipe = Pipeline(steps=[('pca', pca_model), ('least_squares', regression_model)])
param_grid = {'pca__n_components': range(1, 9)}
search = GridSearchCV(pipe, param_grid)
pcr_model = search.fit(X_train, y_train)
pcr_prediction = pcr_model.predict(X_test)
err_test_pcr = np.mean(np.abs(y_test - pcr_prediction))  ## MAE
bias_test_pcr  = bias.bias(y_test,lasso_prediction,axis=0)
n_comp = list(pcr_model.best_params_.values())[0]
print("MAE y bias del modelo de regresión con datos de prueba (pcr):" , err_test_pcr, "," , bias_test_pcr)
```

Obtenemos los errores de los datos de prueba `y_test`.

>MAE y bias del modelo de regresión con datos de prueba (pcr): 164.4150 , 40.9495

#### Regresión por mínimos cuadrados parciales aplicado a la predicción de pronóstico de demanda eléctrica
Aplicaremos la técnica de regresión de componentes principales **(Partial Least Squares)** a los datos de entrenamiento de demanda eléctrica.

```python
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 9)}
search = GridSearchCV(pls_model_setup, param_grid)
pls_model = search.fit(X_train, y_train)
pls_prediction = pls_model.predict(X_test)
err_test_pls = np.mean(np.abs(y_test - pls_prediction))  ## MAE
bias_test_pls = bias.bias(y_test,lasso_prediction,axis=0)
print("MAE y bias del modelo de regresión con datos de prueba (pls):" , err_test_pls, "," , bias_test_pls)
```

Obtenemos los errores de los datos de prueba `y_test`.

MAE y bias del modelo de regresión con datos de prueba (pls): 545.5517 , 40.9495

Por último graficamos los resultados de predicción de las diferentes técnicas de regresión y los resultados de prueba Y.

![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda.png)

### Conclusiones tarea 3
En esta tarea se utilizó la **regresión lineal** para predecir demanda eléctrica en una región partir de datos de días semejantes (variable independientes) y datos de 24 horas antes (variable dependiente). Se utilizaron diversos métodos de reducción de dimensión de variables como: regresión de mejor subconjunto, ridge, lasso, componentes principales, regresión por mínimos cuadrados parciales. Estos métodos intentan reducir simultaneamente el sesgo o bias en la predicción y el número de variables. El método que tuvo un mejor desempeño en nuetros datos fue el de regresión de componentes principales. Por último, el uso de librerias estadísticas como sklear o statsmodels pueden ayudar mucho a obtener y probar diferentes modelos de regresión de manera rápida.

---

## **Tarea 4 Clasificación**
>**Instructions:** Pick one of the examples of the chapter that use the data of the book and replicate it in Python. Then, apply the steps in your own data.

El código completo de esta tarea se encuentra en [Tarea4.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea2.ipynb), aquí solo se presentan los resultados y partes importantes del código.

### Regresión logística en predicción de enfermedades cardiacas
A continuación repetiremos el ejemplo 4.4.2 de predicción de enfermedad cardiaca en Sudafrica **(South African Heart Disease)** del libro [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7).

A continuación, obtenemos un modelo de predicción de los datos de entrenamiento usando regresión logística de la librería **statsmodels**.
```python
model = sm.Logit(dfy, dfx)
results = model.fit()
print(results.summary())
```
 Como podemos en la tabla de resultados de la regresión algunas de las variables son no significativas al 95% con un valor P menor que 0.05. Tal es el caso de *alcohol, obesity, adiposity* y *sbp*. 
```
Optimization terminated successfully.
         Current function value: 0.522778
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    chd   No. Observations:                  462
Model:                          Logit   Df Residuals:                      453
Method:                           MLE   Df Model:                            8
Date:                Mon, 31 Jan 2022   Pseudo R-squ.:                  0.1897
Time:                        00:54:27   Log-Likelihood:                -241.52
converged:                       True   LL-Null:                       -298.05
Covariance Type:            nonrobust   LLR p-value:                 8.931e-21
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -3.9658      1.068     -3.715      0.000      -6.058      -1.873
sbp            0.0056      0.006      0.996      0.319      -0.005       0.017
tobacco        0.0795      0.026      3.033      0.002       0.028       0.131
ldl            0.1803      0.059      3.072      0.002       0.065       0.295
adiposity      0.0101      0.028      0.357      0.721      -0.046       0.066
famhist        0.9407      0.225      4.181      0.000       0.500       1.382
obesity       -0.0457      0.043     -1.067      0.286      -0.130       0.038
alcohol        0.0005      0.004      0.118      0.906      -0.008       0.009
age            0.0404      0.012      3.437      0.001       0.017       0.063
==============================================================================
```
Se aplicó una técnica de reducción de variables paso a paso **(Stepwise)** tal como el libro sugiere. Como resultado se encuentran un subconjunto de variables que son suficientes para explicar el efecto conjunto de los predictores sobre la variable *chd*. El procedimiento descarta una por una las variables con coeficiente P menos significativo `pmenor` y reajusta el modelo. Esto se hace repetidamente hasta que no se puedan eliminar más variables del modelo.

Los resultados obtenidos en la tabla coinciden con los del libro.
```python
## Se ordenan los valores p y se selecciona el más pequeño
p_values = results.pvalues.sort_values(ascending = False)
pmenor = p_values.head(1)
print("menorpi.item() ", pmenor.item())

## Proceso de stepwise
while pmenor.item() > 0.01:
    print(pmenor.index.tolist())
    dfx = dfx.drop(pmenor.index.tolist(), axis=1)
    model = sm.Logit(dfy, dfx)
    model = model.fit()
    # Se ordenan los valores p y se selecciona el más pequeño
    p_values = model.pvalues.sort_values(ascending = False)
    pmenor = p_values.head(1)    
print(model.summary())
```
El resultado de la stepwise se muestra a continuación:
```
menorpi.item()  0.9062
['alcohol']
Optimization terminated successfully.
         Current function value: 0.522793
         Iterations 6
['adiposity']
Optimization terminated successfully.
         Current function value: 0.522936
         Iterations 6
['sbp']
Optimization terminated successfully.
         Current function value: 0.524131
         Iterations 6
['obesity']
Optimization terminated successfully.
         Current function value: 0.525372
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                    chd   No. Observations:                  462
Model:                          Logit   Df Residuals:                      457
Method:                           MLE   Df Model:                            4
Date:                Mon, 31 Jan 2022   Pseudo R-squ.:                  0.1856
Time:                        00:54:29   Log-Likelihood:                -242.72
converged:                       True   LL-Null:                       -298.05
Covariance Type:            nonrobust   LLR p-value:                 5.251e-23
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -4.2043      0.498     -8.436      0.000      -5.181      -3.228
tobacco        0.0807      0.026      3.163      0.002       0.031       0.131
ldl            0.1676      0.054      3.093      0.002       0.061       0.274
famhist        0.9241      0.223      4.141      0.000       0.487       1.362
age            0.0440      0.010      4.520      0.000       0.025       0.063
==============================================================================
```
Finalmente, evaluamos el desempeño del modelo calculando la exactitud y la matriz de confusión.
```
Confusion Matrix : 
 [[254  48]
 [ 76  84]]
Test accuracy =  0.7316
```
### Análisis Discriminante Lineal y Cuadrático con datos de prueba en dos dimensiones
Haremos algunas pruebas con datos de dos dimensiones divididos en tres clases [1,2,3] usando el Análisis Discriminante Lineal **(LDA)** y Análisis Discriminante Cuadrático **(QDA)** usando la librería **sklearn**.

Iniciamos con el método **LDA** y obtenemos la exactitud **(score)**.
```python
LDA_model = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
LDA_model.fit(X_test, y_test)
y_pred_LDA = LDA_model.predict(X_test)
score = LDA_model.score(X_test, y_test)
parametros = LDA_model.get_params(deep=False)
print(score)
```
La exactitud  del método se muestra en una escala de 0 a 1.
```
Test accuracy LDA = 0.9906
```
Se dibuja un diagrama un dispersión con la separación por hiperplanos y los intervalos de confianza de cada clase al 95% (elipses) para el método **LDA**.

![image](https://github.com/urieliram/statistical/blob/main/figures/scatter1.png)

Graficamos la matriz de confusión del modelo **LDA**.

![image](https://github.com/urieliram/statistical/blob/main/figures/confusion1.png)
```
Matriz de confusión
[[228   3   0]
 [  1 269   1]
 [  0   2 246]]
```
Ahora, aplicamos un modelo **QDA** y obtenemos la exactitud **(score)**.
```python
QDA_model = QuadraticDiscriminantAnalysis(store_covariance=True)
QDA_model.fit(X_test, y_test)
y_pred_QDA = QDA_model.predict(X_test)
y_pred_QDA.tofile('predictionQDA.csv',sep=',')
score = QDA_model.score(X_test, y_test)
parametros = QDA_model.get_params(deep=False)
print(score)
```
La exactitud del método se muestra en una escala de 0 a 1.
```
Test accuracy QDA = 0.9933
```
Se dibuja un diagrama un dispersión con la separación por hiperplanos y los intervalos de confianza de cada clase al 95% (elipses) para el método **QDA**.

![image](https://github.com/urieliram/statistical/blob/main/figures/scatter2.png)

```
[[0.13 0.01]
 [0.01 0.06]]
[[ 0.07 -0.  ]
 [-0.    0.12]]
[[ 0.07 -0.  ]
 [-0.    0.31]]
```
Graficamos la matriz de confusión del modelo **QDA**

![image](https://github.com/urieliram/statistical/blob/main/figures/confusion2.png)
```
Matriz de confusión
[[230   1   0]
 [  2 268   1]
 [  0   1 247]]
``` 

### Análisis Discriminante Lineal y Cuadrático aplicada a clasificación de regiones de consumo de electricidad.
Los datos que se usarán en este ejercicio son resultados de la planeación de la operación eléctrica del sistema eléctrico interconectado en México que consta de 320 instancias. Las columnas de los datos son los resultados por región y por hora del día. Se dan resultados de generación térmica (GenTer), generación hidráulica (GenHid), generación renovable (GenRE), Generación no programable (GenNP), Generación total (GenTot), demanda de la región (Demanda), Cortes de energía (Corte), Excedentes de energía(Excedente),Potencia (PotInt), precio Marginal y pérdidas de la región.

La clases [1,2,3,4,..,67] son las regiones y los regresores son ['GenTer','GenHid','GenRE','GenNP','Demanda','Perdidas','PrecioMarginal'].

Preprocesamos los datos transformando algunas variables aplicando el logaritmo a algunas variables 
(el conjunto de variables a los que se les aplicó la transformación es la que mejor **score** de LDA y QDA nos dan)
```python
## Transformamos los datos aplicando el logaritmo a algunas variables.
dfx = dfx + 0.0001 ## Aplicamos un epsilon para evitar problemas con la transformación de log en valores cero.
dfx['Demanda' ] = np.log(dfx['Demanda']) 
dfx['Perdidas'] = np.log(dfx['Perdidas']) 

## Estandarizamos los datos
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
dfx = pd.DataFrame(scaler.fit_transform(dfx), columns = dfx.columns)
```
A continuación se muestran las proyecciones de los datos entre las variables de generación térmica por región ['GenTer'] y ['Demanda'] (derecha), así como ['GenTer'] y ['Pérdidas'] (izquierda).

![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_1.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_2.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_3.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_4.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection0_6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection1_2.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection1_3.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection1_4.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection1_5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection1_6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection2_3.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection2_4.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection2_5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection2_6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection3_4.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/projection3_5.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/projection3_6.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/projection4_5.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/projection4_6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/projection5_6.png)

Ahora, aplicamos un modelo **LDA** y obtenemos la exactitud **(score)** en una escala de 0 a 1.
```
Test accuracy LDA = 0.6549 
```
Ahora, aplicamos un modelo **QDA** y obtenemos la exactitud **(score)** en una escala de 0 a 1.
```
Test accuracy QDA = 0.4054 
```
Graficamos la matriz de confusión del modelo **QDA** aplicado a predicción de regiones eléctricas

![image](https://github.com/urieliram/statistical/blob/main/figures/confusion3.png) 

### Conclusiones Tarea 4
En esta tarea se utilizaron los métodos **Regresión Logística**, **LDA** y **QDA**. Se resolvió el ejemplo del libro sobre predicción de enfermedades cardiacas con multifactores usando regresión logística y **stepwise**. Además, se utilizó LDA y QDA para resolver un ejemplo sencillo de clasificación de dos variables con tres clases, se hizo el análisis exactitud de los métodos, resultando con muy buenos resultados (score LDA = 0.9906, score QDA = 0.9933), para ambos casos se realizaron figuras donde se observa la clasificación en áreas por medio de hiperplanos. Por último, se realizó el ejercicio de clasificación con datos de regiones eléctricas donde las clases son las regiones y los regresores los resultados de planeación de demanda, generación y pérdidas eléctricas en cada región. Los resultados de **score** sin tranformaciones logarítmicas fueron de LDA = 0.5760 y QDA = 0.3620. Aplicando la transformación logarítmica a las variables de pérdidas y demanda obtuvimos LDA = 0.6549 y QDA = 0.4054.

---

## **Tarea 5 Expansión de base**
>**Instructions:** Fit splines into single features in your project data. Explore options to obtain the best fit.

El código completo de esta tarea se encuentra en [Tarea5.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea5.ipynb), aquí solo se presentan los resultados y partes importantes del código.

### Interpolación spline en datos de demanda eléctrica
Aplicaremos el ajuste splines a datos de demanda eléctrica de una región en México. 

Queremos probar el uso de splines en la interpolación de datos perdidos. La serie original está completa y borraremos algunos datos aleatoriamente. Después los completaremos usando interpolación spline de primero hasta quinto orden y compararemos su desempeño con el error absoluto medio **Mean Absolute Error (MAE)**.

De la serie original borramos algunos datos aleatoriamente.
```python
r = round( len(df) * 0.2 )
np.random.seed(0)
random = []
for i in range(r):
    random.append(np.random.randint(0,len(df)))
notmiss = []
for i in random:
    if i not in notmiss:
        notmiss.append(i)
        
dfy_notmiss = df.drop( index = notmiss )

dfx_notmiss = dfx
dfx_notmiss = dfx_notmiss.drop( index = notmiss )
```
Una función calcula la interpolación de los datos y lo almacena en `xspline` y `yspline`. Además, calcula el error MAE y lo guarda en el arreglo  `err_or`
```python
def spline_by_order_k(x, y, dfx, dfy, k):
   for item in range(len(k)):
       tck  = interpolate.splrep(x, y, k=k[item])
       xor  = np.arange(0, n-1, 1/50)
       yor  = interpolate.splev(xor, tck)
       xspline.append( xor )
       yspline.append( yor )

       ## Comparamos contra los datos originales usando el MAE
       y_or   = interpolate.splev(dfx, tck) ## Esta variable se usará para comparar contra el original
       err_or = mean_absolute_error(dfy,y_or)
       print('!order:', k[item],'| mae',err_or,'|')
       
k = [1,2,3,4,5] ## orden del polinomio spline
xspline = []
yspline = []
spline_by_order_k(x, y, dfx, dfy, k)
```
Los resultados de la interpolación spline a los datos perdidos se muestra en la siguiente tabla:  
| Orden          | MAE          |
| :------------- |-------------:|
| 1              | 10.8581      |
| 2*              | 10.8100*      |
| 3              | 11.1600      |
| 4              | 13.0515      |
| 5              | 13.8680      |


\* *muestra el mejor ajuste según el MAE*

Las gráficas siguientes muestran el ajuste spline a los datos originales con puntos de color rojo, así como los datos perdidos en cruces de color rojo. Además se observan los distintos ajustes spline desde el orden uno hasta el cinco. Se puede ver que la mayoria de los datos perdidos si estan cerca del ajuste, todos con un desempeño muy parecido.
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda0.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda1.png)

## Interpolación spline en series de tiempo de aportaciones en embalses
A continuación, haremos ajustes spline a datos de aportaciones hidrológicas (lluvias) mensuales en el embalse (presa) Manuel Moreno Torres (MMT) una de las principales centrales hidroeléctricas en el país, ubicada en Chiapas, México. Solo tomaremos MMT pero podría ser cualquiera de los demás embalses de [Aportaciones_Embalses.csv](https://drive.google.com/drive/folders/159GnBJQDxTY9oYqPBZzdNghyb4Gd9pDS?usp=sharing).

Al igual que en el ejercicio anterior, probaremos el uso de splines en la interpolación de datos perdidos. La serie original está completa y borraremos algunos datos aleatoriamente. Después los completaremos usando interpolación spline de primero hasta quinto orden y compararemos su desempeño con el error absoluto medio **Mean Absolute Error (MAE)**.
Los resultados de la interpolación spline a los datos perdidos de aportaciones hidrológicas se muestra en la siguiente tabla:  
| Orden          | MAE          |
| :------------- |-------------:|
| 1* | 13.5144* |
| 2 | 16.4834 |
| 3 | 17.4133 |
| 4 | 20.9248 |
| 5 | 22.2923 |

\* *muestra el mejor ajuste según el MAE*

Las gráficas siguientes muestran algunos fragmentos de los datos con el ajuste spline. Los datos originales en color rojo, así como los datos perdidos en cruces de color rojo. En la siguiente figura se observan los distintos ajustes spline desde el orden uno hasta el cinco. 
![image](https://github.com/urieliram/statistical/blob/main/figures/splineEmbalse0.png)
En esta figura Se oberva algunos datos perdidos que no son cubiertos por el ajuste, además de que el ajuste incluye datos que en nuestro caso no pueden ser positivos. 
![image](https://github.com/urieliram/statistical/blob/main/figures/splineEmbalse1.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/splineEmbalse2.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/splineEmbalse3.png)


### Conclusiones Tarea 5
En esta tarea se utilizó el método **spline** para calcular datos perdidos en dos series de datos, una de demanda eléctrica y otra de aportaciones hidrológicas en presas. Se comparó la exactitud de la interpolación spline con grados polinómicos desde el uno al cinco. El mejor para el caso de demanda eléctrica es el de grado dos (cuadrático) con un error MAE=10.8100. Sin embargo, los resultados de los demás splines fueron muy cercanos. Para el caso de aportaciones hidrológicas, el mejor ajuste de interpolación fue el de orden uno (lineal) con un MAE = 13.5144. Además, se presentan figuras donde se observa el ajuste de los spline a los datos originales así como a los datos perdidos.

---

## **Tarea 6 Suavizado**
>**Instructions:** Build some local regression model for your data and adjust the parameters. Remember to read all of Chapter 6 first to get as many ideas as possible.

### Regresión local en la predicción de demanda eléctrica
En esta tarea se implementa el método de regresión (lineal) local con suavización con kernel en multiples dimensiones `k` en cada punto `xo ∈ Rp` de la variable `Y`. La implementación fue realizada con las consideraciones del libro [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7) en las secciones: *6.1 One-Dimensional Kernel Smoothers*; *6.1.1 Local Linear Regression*; y *6.3 Local Regression in Rp*.

Haremos la comparación de resultados de regresión para datos de demanda eléctrica. La variable independiente `X` serán los datos de demanda del día anterior, y la variable independiente `Y` serán los datos de días con una mayor correlación con `X`. En esta sección, aplicaremos técnicas de **regresión local con múltiples regresores `X`**.

Los datos usados en esta sección están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea6.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea6.ipynb), aquí solo se presentan los resultados y partes relevantes del código.

Adicionalmente, algunos ejemplos de uso de librerias de regresión local en dos dimensiones (`R2`) pueden  encontrarse aquí [Tarea6_b.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea6_b.ipynb).

Iniciamos calculando los pesos de los puntos `xi ∈ Rp` del vecindario alrededor del punto `xo`,  utilizando un kernel con distribución cuasi-normal que da mayor peso a las puntos `xi` mas cercanos al punto `xo` y menos peso a las observaciones más lejanas de acuerdo a un tamaño del vecindario `k`.
```python
# Calcula los pesos y regresa una matriz diagonal con los pesos
def get_weight_exp(xo, X, k): 
## k    : tamanio del vecindario (bandwidth)
## X    : Regresores
## xo   : punto donde se desea hacer la predicción.

    n = X.shape[0]             ## numero de datos
    weight = np.mat(np.eye(n)) ## Matriz de pesos identidad W.
    
  # Cálculo de pesos para todos los datos de entrenamiento xi.
    for i in range(n): 
        xi = X[i] 
        d = (-2 * k * k) 
        weight[i, i] = np.exp(np.dot((xi-xo), (xi-xo).T)/d) 
        
    return weight
```

A continuación, estimamos los coeficientes de regresión `β = (Xt W(xo) X)^{-1}) (Xt W Y)`. Note que el peso `W` obtenido por el Kernel se incluye en las operaciones matriciales.
```python
def local_regression(X,W,Xo):
    # W     --> Matriz diagonal de pesos
    # X     --> Regresores
    # xo    --> punto donde se desea hacer la predicción.
    Xt = X.T  # Calcula transpuesta de X
    A = np.matmul(Xt, np.matmul(W,X)) 
    A = np.linalg.inv(A)   # Calcula inversa de A
    B = np.matmul(Xt, np.matmul(W,Y)) 
    beta = np.matmul(A,B)
    prediccion = np.matmul(Xo,beta)
    return prediccion, beta
```

En el siguiente código se recorre uno a uno los puntos de `X` para calcular la predicción. Es decir, para cada uno de los datos, seleccionaremos una vecindad de `k` puntos muestreados y los usaremos como conjunto de entrenamiento para un modelo de regresión lineal con pesos. Aunque ajustamos un modelo lineal completo a los datos de la vecindad, solamente lo usamos para evaluar el ajuste en el único punto `xo`. 

En específico esta sección del código es para encontrar el valor mínimo del tamaño `k` sin caer en una singularidad.
```python
k = 0 # Tamanio del vecindario
kmin = 10
kmax = 19
Y_local_list= []
Y_local = []
aux = 0

for i in range(X.shape[0]):
    k = kmin
    flag = True
    while(flag==True):
        try:
            xo = X[[i]]
            W = get_weight_exp(xo, X, k)
            pred = local_regression(X, W, xo)
            Y_local.append(pred.item(0))
            aux = pred.item(0)
        except:
            print("Sorry! Singular matrix found in position i=",str(i),"k=", k)
            k = k + 1
            if(k>kmax):                
                flag = False
                Y_local.append(0)
        else:
            flag = False 
            
Y_local_list.append(Y_local)
```
Se han encontrado algunas singularidades que son manejadas incrementando el tamaño del vecindario `k`. Por ejemplo se encontró una singularidad en la posición 335 y se incrementó el tamaño del vecindario de `k`=10 hasta `k`=16, hasta encontrar un modelo para ese punto.
```
>Sorry! Singular matrix found in position i= 42 k= 10
>Sorry! Singular matrix found in position i= 93 k= 10
>Sorry! Singular matrix found in position i= 93 k= 11
>Sorry! Singular matrix found in position i= 245 k= 10
>Sorry! Singular matrix found in position i= 335 k= 10
>Sorry! Singular matrix found in position i= 335 k= 11
>Sorry! Singular matrix found in position i= 335 k= 12
>Sorry! Singular matrix found in position i= 335 k= 13
>Sorry! Singular matrix found in position i= 335 k= 14
>Sorry! Singular matrix found in position i= 335 k= 15
>Sorry! Singular matrix found in position i= 335 k= 16
```
En esta sección del código evaluamos el desempeño de la regresión local con otros `k` tamaños de vecindarios.
```python
k = 0 # Tamanio del vecindario
klist = [25,35,50,100]
aux=0
for item in klist:
    k = item
    Y_local = []
    for i in range(X.shape[0]):
        try:
            xo = X[[i]]
            W = get_weight_exp(xo, X, k)
            Ygorro = local_regression(X, W, xo)
            Y_local.append(Ygorro.item(0))
            aux = Ygorro.item(0)
        except:
            Y_local.append(aux)
            print("Sorry! Singular matrix found in position i=",str(i))
    Y_local_list.append(Y_local)
```

Con fines de comparar el desempeño de la regresión local calculamos el ajuste de `Y` usando unicamente la *regresión lineal múltiple*. Como se puede observar en este caso los pesos `W` son la matriz identidad.
```python
Y_pred = []
for i in range(X.shape[0]):
    xo = X[[i]]
    W = np.mat(np.eye(X.shape[0])) 
    Ygorro, beta = local_regression(X, W, xo)
    Y_pred.append(Ygorro.item(0))
```

En esta gráfica se observan los datos de demanda `Y` (puntos rojos), así como el ajuste usando regresión líneal (línea punteada roja) y regresión local con `k`= 10.
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda_t6_0.png)

Ahora mostramos los ajustes usando la regresión local con diferentes valores de `k`= [10,25,35,50,100] y regresión local (línea punteada roja) en diferentes intervalos de la serie de datos. En general podemos observar un mejor ajuste usando regresión local sobre la regresión lineal (línea punteada roja).
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda_t6_2.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda_t6_1.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda_t6_3.png)

Calculamos los errores de los métodos de regresión, para el caso de regresión local variamos los tamaños de las vecindades `k`.
| REGRESIÓN      | MAE            | MSD            | MAPE         |
| :------------- | -------------: | -------------: |-------------:|
| local, k=10 | 77.4973    | 323440.045    |    0.009 |
| local, k=25 | 83.5068    | 14118.722    |    0.0096 |
| local, k=35 | 98.6564    | 18483.2236    |    0.0113 |
| local, k=50 | 109.6632    | 21942.6356    |    0.0126 |
| local, k=100| 123.9285    | 26331.6078    |    0.0142 |
|    lineal      | 138.5861     | 32615.1951    |    0.0159 |

### **Conclusión tarea 6** 
En general la regresión local realizada punto por punto tuvo en general un mejor desempeño que el modelo de regresión lineal múltiple (en el caso en que el tamaño del vecindario `k` no conduce a singularidades en el cálculo de los coeficientes beta de la regresión). Además, podemos notar que mientras el tamaño del vecindario `k`=100,50,35,25,17 se hace más péqueño el error (MAE, MSD y MAPE) en el ajuste disminuye. El kernel usado para establecer los pesos fue una distribución radial cuasi-normal, sin embargo pueden hacerse pruebas cambiando el kernel a por ejemplo un tri-cúbico y analizar los resultados. Se ha observado que para algunos vecindarios de menor tamaño es posible encontrar singularidades que nos hace imposible encontrar un modelo de regresión, este problema posiblemente se presente por dependencias lineales en los regresores, para resolverlo se implementó  un procedimiento que incremeneta el tamaño de la ventana hasta encontrar un modelo. Otra opción alternativa para manejar la singularidad es identificar los regresores que pueden estar inflando la varianza y eliminarlos del cálculo de los coeficiente de la regresión, este procedimiento  alternativo puede verse aquí [Tarea6_c.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea6_c.ipynb).

---

## **Tarea 7 Evaluación**
>**Instructions:** Apply both cross-validation and bootstrap to your project data to study how variable your results are when you switch the test set around.

A continuación haremos la comparación de resultados de regresión para datos de demanda eléctrica y evaluaremos el error del modelo usando validación cruzada **(cross-validation)**  y **bootstrap**. La variable independiente `X` serán los datos de demanda del día anterior, y los datos independiente `Y` serán los datos de días con una mayor correlación con `X`. En esta sección, aplicaremos regresión lineal múltiple con multiples regresores `X`. Los datos usados en esta sección están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing) El código completo de esta tarea se encuentra en [Tarea7.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea7.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

### Muestreo **bootstrap** en estimación de error en la predicción de demanda eléctrica usando regresión líneal múltiple.
A continuación se calcula un modelo de regresión lineal múltiple para una de las muestras `X_train` elegidas aleatoriamente un 50% de datos del total del conjunto `X`. Los datos de error (MAE) de todas las réplicas del muestreo aleatorio se guardan en la lista `bootstrap_ols`.
```python
bootstrap_ols= []
replicas = 1000
for rep in range(replicas):
    a = np.arange(0,X.shape[0])
    b = np.sort(np.random.choice(a, replace=False, size = int(len(a)*0.9)))
    X_train = np.delete(X, b, axis = 0)
    Y_train = np.delete(Y, b, axis = 0)
    
    olsmod = sm.OLS(Y_train, X_train)
    olsres = olsmod.fit()
    Y_pred = olsres.predict(X_train)  
    bootstrap_ols.append(mean_absolute_error(Y_train,Y_pred))

dfb = pd.DataFrame((np.asarray(bootstrap_ols)).T)
bootstrap_mean = dfb.mean(numeric_only = True)
bootstrap_std = dfb.std(numeric_only = True)
dibuja_hist(dfb,colour='#76ced6',name='hist_t7_1.png',Xlabel="Error",Ylabel="Frecuencia",title="Error de predicción de demanda estimado con Boostrap")
print(bootstrap_mean)
print(bootstrap_std)
```
La media y la varianza del error del modelo calculada por medio de la distribución del error es: 111.18 y 12.54.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist_t7_1.png)

### Validación cruzada en estimación de error en la predicción de demanda eléctrica usando regresión líneal múltiple
En esta función se calcula un modelo de regresión lineal múltiple para cada una de las una de las muestras `X_test` extraidas del total del conjunto de entrenamiento `X_train`. Los datos de error del muestreo cross-validation se guardan en la lista cross_ols.
```python
cross_ols_fx = []
nblocks    = 100
nblocks    = X.shape[0] 
print(X.shape[0] )
size = int( X.shape[0] / nblocks)
intervals = np.arange(size, X.shape[0], size)

for i in intervals:  
    a = np.arange(0,X.shape[0])
    b = np.arange(i-size,i)
    c = np.sort(np.setdiff1d(a, b)) #El complemento del conjunto seleccionado
    X_test  = np.delete(X, b, axis = 0)
    Y_test  = np.delete(Y, b, axis = 0)    
    X_train = np.delete(X, c, axis = 0)
    Y_train = np.delete(Y, c, axis = 0)

    olsmod  = sm.OLS(Y_train, X_train)
    olsres  = olsmod.fit()
    Y_pred  = olsres.predict(X_test)  
    error = abs(Y_test - Y_pred)
    cross_ols_fx.append(mean_absolute_error(Y_test,Y_pred))

dfb = pd.DataFrame((np.asarray(cross_ols_fx)).T)
cross_mean_fx = dfb.mean(numeric_only = True)
cross_std_fx = dfb.std(numeric_only = True)
dibuja_hist(dfb,colour='#777bd4',name='hist_t7_2.png',Xlabel="Error",Ylabel="Frecuencia",title="Error de predicción de demanda estimado con Cross-validation")
print(cross_mean_fx)
print(cross_std_fx)
```

La media y la varianza del error del modelo calculada por medio de la distribución del error es: 184.10 y 76.38.

![image](https://github.com/urieliram/statistical/blob/main/figures/hist_t7_2.png)


Esta versión de validación cruzada elige aleatoriamente el inicio de las muestras de prueba X_test
```python
cross_ols = []
nblocks   = 10
size = int( X.shape[0] / nblocks)

replicas = 180
arr = np.sort(np.random.choice(a, replace=False, size = replicas))
print(arr)
for i in arr:

      a = np.arange(1,X.shape[0])
      b = np.arange(i-size,i)
      c = np.sort(np.setdiff1d(a, b)) #El complemento del conjunto seleccionado
      X_test  = np.delete(X, b, axis = 0)
      Y_test  = np.delete(Y, b, axis = 0)    
      X_train = np.delete(X, c, axis = 0)
      Y_train = np.delete(Y, c, axis = 0)

      olsmod  = sm.OLS(Y_train, X_train)
      olsres  = olsmod.fit()
      Y_pred  = olsres.predict(X_test)  
      error = abs(Y_test - Y_pred)
      cross_ols.append(mean_absolute_error(Y_test,Y_pred))

dfb = pd.DataFrame((np.asarray(cross_ols)).T)
cross_mean = dfb.mean(numeric_only = True)
cross_std = dfb.std(numeric_only = True)
dibuja_hist(dfb,colour='#17cb49',name='hist_t7_3.png',Xlabel="Error",Ylabel="Frecuencia",title="Error de predicción de demanda estimado con validación cruzada")
print(cross_mean)
print(cross_std)
```

La media y la varianza del error del modelo calculada por medio de la distribución del error es: 229.18 y 63.85

![image](https://github.com/urieliram/statistical/blob/main/figures/hist_t7_3.png)

### Evaluación del desempeño del bootstrap variando tamaño de las muestra y número de muestreos aleatorios (repeticiones).
Adicionalmente se ha hecho un análisis del error del **bootstrap**, variando el tamaño de la muestra en porciento del total de los datos `percent = [10,20,30,40,50,60,70,80,90]` y un número de simulaciones `replicas = [250,500,1000,1500,2000]`, el código utilizado y resultados obtenidos se muestran a continuación.

```python
from numpy.ma.core import mean
percent   = [10,20,30,40,50,60,70,80,90] # porcentaje de muestreo
replicas  = [250,500,1000,1500,2000] # porcentaje de muestreo 1500,2000,2500,5000
test      = []
ylist     = []

for rep in replicas:
    mean      = []
    desv_up   = []
    desv_down = []
    for per in percent:
        bootstrap_ols= []
        for rep in range(rep):
            a = np.arange(0,X.shape[0])
            b = np.sort(np.random.choice(a, replace=False, size = int(len(a) * (per/100))))
            X_train = np.delete(X, b, axis = 0)
            Y_train = np.delete(Y, b, axis = 0)            
            olsmod = sm.OLS(Y_train, X_train)
            olsres = olsmod.fit()
            Y_pred = olsres.predict(X_train)  
            bootstrap_ols.append(mean_absolute_error(Y_train, Y_pred))
        dfb = pd.DataFrame((np.asarray(bootstrap_ols)).T)
        bootstrap_mean = dfb.mean(numeric_only = True)
        bootstrap_std = dfb.std(numeric_only   = True)        
        mean.append(bootstrap_mean)
        desv_up.append(bootstrap_mean   + bootstrap_std*2)
        desv_down.append(bootstrap_mean - bootstrap_std*2)
    test.append(mean)

labels   = ['250 reps','500 reps','1000 reps','1500 reps','2000 reps']
Xlabel   = '% de muestreo'
Ylabel   = 'error'
title    = "Error en muestreo usando bootstrap con repeticiones"
namefile = 'fig_t7_4'
print(test)
dibuja_lineas(percent, test, labels, namefile, Xlabel, Ylabel, title)
```

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t7_4.png)

Se ha determinado que para obtener un resultado aceptable de error en estos datos el número mínimo de muestras aleatorias es de un 80% de los datos con 500 repeticiones, la gráfica de error vs número de réplicas se muestra a continuación. La línea roja representa la media del error, y las líneas azules los intervalos de confianza del 5% y 95% (asumiendo una distribución normal en el error). 

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t7_5.png)

**Con este análisis podemos cuantificar el efecto en la calidad del modelo en usar diferentes números de réplicas y/o muestras para conocer la cantidad de valores "suficientes" o "ideales" para los datos y determinar un punto en que agregar más datos o más réplicas ya no cambia nada en el modelo.**

### **Conclusión tarea 7** 
Se realizó un ejercicio de predicción de demanda eléctrica usando una regresión lineal múltiple, sin embargo debido a los pocos datos que se tienen para evaluar el modelo. Se aplicaron técnicas de validación cruzada y **bootstrap**, las cuales son una herramienta poderosa para evaluar la función del error. Ambas técnicas hacen un muestreo con los datos y evaluan el error en el modelo, resulta interesante observar las distribuciones que resultan parecidas a la distribución nornal para el **bootstrap** y para el caso de validación cruzada una distribución exponencial. El uso de estas técnicas tiene como ventaja obtener una distribución más realista del comportamiento del error e inclusive poder calcular intervalos de confianza.

---

## **Tarea 8 Inferencia**
>**Instrucciones:** Modelar la sobrecarga a base de observaciones que tienes para llegar a un modelo tipo "en estas condiciones, va a fallar con probabilidad tal".
Las variables independientes son el flujo neto máximo y mínimo en un área de control [CEN,GUA,NES,NOR,NTE,OCC,ORI,PEN] para un día y se calcula como la diferencia entre la demanda menos la generación en cada área de control. Los datos son obtenidos de 334 simulaciones de planeación de la operación de un día en adelanto de la red eléctrica en México. 
Los datos usados en esta sección están disponibles en [overload.csv](https://drive.google.com/file/d/1Q8Pk5apApNbcoqmKQp3RvQFvuk4DKylU/view?usp=sharing) [overload.csv](https://drive.google.com/file/d/1-ZCl-XLmmCpe_yNGryl7Eudg3Q_Xhyh8/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea8.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea8.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

### Inferencia Bayesiana
Queremos saber las distribuciones de probabilidad de los parámetros desconocidos de un modelo. Además, probar que tan buenos son estos parámetros. Cuanto mayor sea la probabilidad P(θ|x) de los valores de los parámetros dados los datos, más probable será que sean los parámetros "reales" de la distribución de la población ('θ' es la distribución a priori y 'x' la evidencia). Esto significa que podemos transformar nuestro problema de encontrar los parámetros de la distribución de la población a encontrar los valores de los parámetros que maximizan el valor P(θ|x).

### Un ejemplo básico de regresión logistica bayesiana
Se usará este ejemplo de un modelo de regresión logística básico, que simula fracturas óseas con variables independientes de edad y sexo. Fuente: [Lawrence Joseph](http://www.medicine.mcgill.ca/epidemiology/Joseph/courses/EPIB-621/main.html) [PDF](http://www.medicine.mcgill.ca/epidemiology/joseph/courses/EPIB-621/bayeslogit.pdf)

Al principio no sabemos nada sobre los parámetros `beta`, así que usaremos una distribución **uniforme**  con límites suficientemente grandes como los valores: `lower = -10**6; higher = 10**6 `
```python
lower = -10**6; higher = 10**6
with pm.Model() as first_model:
    ## Priors on parameters
    beta_0   = pm.Uniform('beta_0', lower=lower, upper= higher)
    beta_sex = pm.Uniform('beta_sex', lower, higher)
    beta_age = pm.Uniform('beta_age', lower, higher)
    
    #the probability of output equal to 1
    p = pm.Deterministic('p', pm.math.sigmoid(beta_0+beta_sex*df['sex']+ beta_age*df['age']))

with first_model:
    #fit the data 
    observed=pm.Bernoulli("frac", p, observed=df['frac'])
    start = pm.find_MAP()
    step  = pm.Metropolis()
    
    #samples from posterior distribution 
    trace=pm.sample(25000, step=step, start=start)
    first_burned_trace=trace[15000:]
```
Ahora, graficamos las distribuciones resultantes de nuestro primer modelo:

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t8_1.png)

Ahora calculamos la media de nuestra primera versión de las muestras generadas por la simulación:
```python
coeffs=['beta_0', 'beta_sex', 'beta_age']
d=dict()
for item in coeffs:
    d[item]=[first_burned_trace[item].mean()]
    
result_coeffs=pd.DataFrame.from_dict(d)    
result_coeffs
```
La media de las distribuciones de los parámetros de este primer modelo son:
beta_0	     beta_sex	beta_age
-24.843256	 1.619084	0.390901

Una ventaja del enfoque de inferencia bayesiana es que no solo nos da la media de los parámetros del modelo, también podemos obtener los intervalos de confianza al 95%. Es decir que los parámetros buscados se encontrarán entre los valores siguientes con una probabilidad del 95%. Lo hacemos de la siguiente manera:
```python
mean = first_burned_trace['beta_0'].mean()
hpd = az.hdi(first_burned_trace['beta_0'].flatten())

coeffs=['beta_0', 'beta_sex', 'beta_age']
interval=dict()
for item in coeffs:
    interval[item]=az.hdi(first_burned_trace[item].flatten()) #compute 95% high density interval
    
result_coeffs=pd.DataFrame.from_dict(interval).rename(index={0: 'lower', 1: 'upper'})
result_coeffs
```
Los intervalos al 95% del los parámetros del modelo son:
         beta_0	    beta_sex	beta_age
lower	-36.607193	0.694889	0.313219
upper	-19.923850	3.237995	0.575355

Los valores buscados se encuentran entre estos valores, ahora podemos refinar la búsqueda de los parámetros usando estos límites en un segundo modelo, por ejemplo cambiamos los límites superior e inferior de la distribución **uniforme** de la siguiente manera.
```python
with pm.Model() as second_model:
    ## Priors on parameters
    beta_0   = pm.Uniform('beta_0',   lower=-31.561240, upper= -20.376186)
    beta_sex = pm.Uniform('beta_sex', lower=0.555843,   upper=2.816436)
    beta_age = pm.Uniform('beta_age', lower=0.314423,   upper=0.487489)
```
Si graficamos las distribuciones resultantes de nuestro segundo modelo queda:
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t8_2.png)

Ahora, entrenemos el modelo asumiendo que los coeficientes de la regresión logistica siguen **distribuciones normales**. Es decir cambiaremos el conjunto de priors en un tercer modelo.
```python
with pm.Model() as third_model:  
    ## Priors on parameters
    beta_0   = pm.Normal('beta_0'  , mu=-23.764747, sd=10**4)
    beta_sex = pm.Normal('beta_sex', mu=1.572192,   sd=10**4)
    beta_age = pm.Normal('beta_age', mu=0.37384,    sd=10**4)
```
Si graficamos las distribuciones resultantes de nuestro segundo modelo queda:

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t8_3.png)

## Versión frecuentista de la regresión logística con la librería *statsmodel*.
Ahora, comparamos los resultados con un **análisis frecuentista** de regresión logística de la librería statsmodels.
```python
model = sm.Logit(dfy, dfx)
results = model.fit()
print(results.summary())
```
El resultado nos da la siguiente tabla, como vemos los parámetros son muy parecidos a los obtenidos por el **enfoque bayesiano**
```
Optimization terminated successfully.
         Current function value: 0.297593
         Iterations 8
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                   frac   No. Observations:                  100
Model:                          Logit   Df Residuals:                       97
Method:                           MLE   Df Model:                            2
Date:                Tue, 22 Feb 2022   Pseudo R-squ.:                  0.5484
Time:                        08:37:38   Log-Likelihood:                -29.759
converged:                       True   LL-Null:                       -65.896
Covariance Type:            nonrobust   LLR p-value:                 2.024e-16
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const        -21.8504      4.425     -4.938      0.000     -30.524     -13.177
sex            1.3611      0.733      1.856      0.063      -0.076       2.798
age            0.3447      0.069      5.027      0.000       0.210       0.479
==============================================================================
```

Hasta ahora se ha realizado un análisis inferencial bayesiano expresando suposiciones previas de distribución de cada una de las variables. Sin embargo, cuando el número de variables es muy grande se recomienda el uso de la librería **PyMC3** tiene un modelo lineal generalizado (GLM) que facilita el análisis. Se usará este modelo para ajustar los datos de sobrecarga en líneas de transmisión en la red eléctrica de México.

## **Predicción de sobrecarga en grupos de líneas de transmisión en la red eléctrica en México.**
En esta sección se usará inferencia bayesiana para ajustar un modelo de regresión logística a datos de sobrecarga en grupos de líneas de transmisión, que interconectan las regiones eléctricas en México. La variable dependientes es de naturaleza binaria con un valor de uno si la línea presenta sobrecarga y cero si no. Las variables independientes son el flujo neto máximo y mínimo en un área de control [CEN,GUA,NES,NOR,NTE,OCC,ORI,PEN] para un día y se calcula como la diferencia entre la demanda menos la generación en cada área de control. Los datos son obtenidos de 334 simulaciones de planeación de la operación de un día en adelanto de la red eléctrica en México. 

```python
with pm.Model() as fourth_model:
    pm.glm.GLM.from_formula('L3 ~ CEN + GUA + NES + NOR + NTE + OCC + ORI + PEE + PEN + CEN_min + GUA_min + NES_min + NOR_min + NTE_min + OCC_min + ORI_min + PEE_min + PEN_min',df, 
                            family=pm.glm.families.Binomial())
    fourth_trace = pm.sample(25000, tune=10000, init='adapt_diag')
pm.traceplot(fourth_trace)

plt.show()
```
Ahora, mostramos las distribuciones de los parámetros

Entrenamos el modelo:
```python
with fourth_model:
    map_solution=pm.find_MAP()
d=dict()
for item in map_solution.keys():
    d[item]=[float(map_solution[item])]
    
fourth_map_coeffs=pd.DataFrame.from_dict(d)    
fourth_map_coeffs
```

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t8_5.png)

Ahora calculamos la media de de las muestras generadas por la simulación.
```python
coeffs=['CEN','GUA','NES','NOR','NTE','OCC','ORI','PEE','PEN','CEN_min','GUA_min','NES_min','NOR_min','NTE_min','OCC_min','ORI_min','PEE_min','PEN_min','Intercept']
d=dict()
for item in coeffs:
    d[item]=[fourth_trace[item].mean()]
    
result_coeffs=pd.DataFrame.from_dict(d)    
print(result_coeffs)

        CEN       GUA       NES  ...   PEE_min   PEN_min  Intercept
0 -0.006587 -0.028223  0.003787  ... -6.568944  0.002085  20.682454
```

Ahora calculamos intervalos al 95%.
```python
mean = fourth_trace['Intercept'].mean()
hpd = az.hdi(fourth_trace['Intercept'].flatten())

coeffs=['CEN','GUA','NES','NOR','NTE','OCC','ORI','PEE','PEN','CEN_min','GUA_min','NES_min','NOR_min','NTE_min','OCC_min','ORI_min','PEE_min','PEN_min','Intercept']
interval=dict()
for item in coeffs:
    interval[item]=az.hdi(fourth_trace[item].flatten()) #compute 95% high density interval
    
result_coeffs=pd.DataFrame.from_dict(interval).rename(index={0: 'lower', 1: 'upper'})
print(result_coeffs)

            CEN       GUA      NES  ...      PEE_min   PEN_min  Intercept
lower -0.013823 -0.069692  0.00032  ... -1963.161634 -0.007922  -1.083443
upper  0.000798  0.014229  0.00739  ...  1812.848724  0.012059  42.543688
``` 
Por último, calculamos la matriz de confusión del ajuste al modelo de regresión logistica.

```python
with fourth_model:
    ppc = pm.sample_posterior_predictive(fourth_trace, samples=15000)
#compute y_score 
with fourth_model:
    fourth_y_score = np.mean(ppc['y'], axis=0)
#convert y_score into binary decisions    
fourth_model_prediction=[1 if x >0.5 else 0 for x in fourth_y_score]
#compute confussion matrix 
fourth_model_confusion_matrix = confusion_matrix(df['L3'], fourth_model_prediction)
fourth_model_confusion_matrix
array([[104,  42],
       [ 17, 172]])
```
### **Conclusión tarea 8** 
Hemos utilizado **PyMC3** para implementar la regresión logistica bayesiana para varias variables, además de la función **Logit** de la librería **statsmodel**, que implementa un enfoque frecuentista.
Los resultados de estimación de parámetros entre el enfoque frecuentista y el bayesiano son muy parecidos para el caso de estudio de fracturas de huesos, sin embargo, el enfoque bayesiano da algunas ventajas ya que da la posibilidad de actualizar el modelo con nueva información, mientras que los modelos de regresión lineal generan valores únicos de los parámetros de ajuste, mientras que los modelos de regresión lineal bayesianos pueden generar distribuciones de los parámetros, esto tiene la ventaja de que podemos cuantificar la incertidumbre de nuestra estimación.
Otra cosa que observamos es que a pesar de que los modelos modelos bayesianos que usamos usan distribuciones a priori diferentes, los rendimientos de predicción son similares. Esto quiere decir que a medida que crece el conjunto de datos los resultados deberían converger en la misma solución.
Para el caso de predicción de sobrecarga en líneas, se aplicó el modelo de regresión logistica ajustado con inferencia bayesiana con ayuda de la librería mencionada. Los resultados fueron predecidos correctamente en su mayoria, como lo evidencia la matriz de confisión. Las distribuciones de los parámetros se asemejan  en su mayoria a una distribución normal. Los resultados animan a seguir trabajando en mejorar la modelación del comportamiento de la sobrecarga eléctrica incluyendo más variables y tranformándolas así como variando las distribuciones a priori.

---

## **Tarea 9 Modelos Aditivos y Árboles**
>**Instrucciones:** Read through the spam example used throughout Chapter 9 and make an effort to replicate the steps for your own data. When something isn't quite applicable, discuss the reasons behind this. Be sure to read Sections 9.6 and 9.7 before getting started.

Los datos usados en esta sección están disponibles en [overloadlog.csv](https://drive.google.com/file/d/1-ZCl-XLmmCpe_yNGryl7Eudg3Q_Xhyh8/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea9.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea9.ipynb).

Aquí solo se presentan los resultados y las secciones relevantes del código.

### Modelos aditivos en predicción de sobrecarga en líneas de transmisión 
En esta sección se usará un modelo logístico aditivo para ajustar un modelo de regresión logística a datos de sobrecarga en grupos de líneas de transmisión, que interconectan las regiones eléctricas en México. La variable dependientes es de naturaleza binaria con un valor de uno si la línea presenta sobrecarga y cero si no. Las variables independientes son el flujo neto máximo y mínimo en un área de control [CEN,GUA,NES,NOR,NTE,OCC,ORI,PEN] para un día y se calcula como la diferencia entre la demanda menos la generación en cada área de control. Los datos son obtenidos de 334 simulaciones de planeación de la operación de un día en adelanto de la red eléctrica en México. 

Usaremos la función **LogisticGAM** de la librería **pygam**  que es la implemenetación logística de un modelo de Regresión Aditiva Generalizada **(GAM)**. Para resolver nuestro problema hemos seguido el procedimiento del libro en la sección `9.1.2 Example: Additive Logistic Regression`. Las diferencias entre el procedimiento original para predicción de spam y el de nuestro problema de predicción de sobrecarga en líneas de transmisión serán discutidas. El ejemplo de predicción de spam del libro se encuentra en el cuaderno [Tarea9_spam.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea9_spam.ipynb).

Los datos que utilizamos fueron transformados previamente a con `log(x + 0.1)` como lo sugiere el libro. Además se separan los datos de entrenamiento `X_train` y prueba `X_test` en un 70% entrenamiento y 30% de prueba.
```python
X = df[['CEN','NES','NOR','NTE','OCC','ORI','PEN','CEN_min','NES_min','NOR_min','NTE_min','OCC_min','ORI_min','PEN_min']] ## Predictors
y = df['L3']

## Crea conjuntos de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 5)
```
Al intentar entrenar un modelo con todas los regresores se presentaron problemas de convergencia, aún aumentando el número de iteraciones `max_iter` a 10000 y disminuyendo la tolerancia `tol` a 0.01. Por lo que hemos realizado un proceso para detectar aquellos regresores que pudieran estar causando problemas por multicolinealidad, debido a la alta correlación entre algunos de los regresores. El procedimiento implementado consiste en detectar cual de ellos es el que más factor de inflación de la varianza presenta y es retirado del modelo hasta encontrar un modelo válido. A diferencia del procedimiento del libro que usa un subconjunto de los regresores más significativos, usaremos todo el subconjunto de regresores que nos quedan despues del procedimiento descrito a continuación:

```python
while(Flag == True):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gam = LogisticGAM(max_iter=1000, tol=0.001, verbose=True).fit(X_train[column], y_train)
        
        if len(column) >1 :
            print(str(len(column))+'----------------------------------------------------')
            
            ## DETECTAMOS EL FACTOR DE INFLACIÓN DE LA VARIANZA
            # VIF dataframe 
            vif = pd.DataFrame()
            vif["feature"] = X_train.columns
            # calculating VIF for each feature
            vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
            #vif["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.values.shape[1])]

            column = vif["VIF"]
            max_value = column.max()
            indx = vif[vif['VIF'] == max_value].index

            X_train.drop(X_train.columns[indx.values[0]], axis=1, inplace=True)
            column = X_train.columns
            print(column)
            _ = plt.plot(gam.logs_['deviance'])
            
        else:
            Flag = False
```

El subconjunto de regresores que nos quedan despues del procedimiento de descarte de variables por inflación de varianza es:
```
X = df[['NOR', 'NTE', 'ORI', 'PEN', 'CEN_min', 'NTE_min', 'OCC_min', 'PEN_min']] ## Predictors
```

Se muestra un resumen de la regresión logística aditiva en la que vemos los regresores con una significancia al 0.001. 
```
LogisticGAM                                                                                               
=============================================== ==========================================================
Distribution:                      BinomialDist Effective DoF:                                     11.1091
Link Function:                        LogitLink Log Likelihood:                                   -35.2326
Number of Samples:                          100 AIC:                                               92.6832
                                                AICc:                                               96.337
                                                UBRE:                                               3.0157
                                                Scale:                                                 1.0
                                                Pseudo R-Squared:                                   0.4917
==========================================================================================================
Feature Function                  Lambda               Rank         EDoF         P > x        Sig. Code   
================================= ==================== ============ ============ ============ ============
s(0)                              [0.6]                20                        0.00e+00     ***         
s(1)                              [0.6]                20                        1.97e-04     ***         
s(2)                              [0.6]                20                        0.00e+00     ***         
s(3)                              [0.6]                20                        7.77e-16     ***         
s(4)                              [0.6]                20                        2.43e-03     **          
s(5)                              [0.6]                20                        4.15e-08     ***         
s(6)                              [0.6]                20                        0.00e+00     ***         
s(7)                              [0.6]                20                        1.37e-05     ***         
intercept                                              1                         1.66e-01                 
==========================================================================================================
Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```
La librería nos permite graficar los diagramas de dependencia parcial de los rgeresores que se muestran a continuación.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t9_1.png)

La exactitud del modelo y matriz de confusión obtenidas con el **modelo logístico aditivo** son: 
```
Test accuracy Logistic aditive =  0.7319
[[ 69  27]
 [ 36 103]]
 ```
En el procedimiento del libro los resultados se comparan con los obtenidos por una **Regresión logística**. Adicionalmente, en nuestro problema hemos comparado con una **Regresión Logística con Stepwise** con los resultados que se muestran a continuación.

Resultados con **Regresión logística**:
 ```
 Test accuracy RegLogit =  0.7531
[[79 17]
 [41 98]]
 ```
 
Resultados con **Regresión logística con stepwise**:
```
Test accuracy RegLogit + stepwise=  0.7702
[[ 65  31]
 [ 23 116]]
 ```
 El detalle de estos procedimientos se muestra en el cuaderno [Tarea9.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea9.ipynb)
 
#### Regresión logística aditiva con búsqueda de mejor subconjunto **(Best-Subset selection)**
Con el objetivo de encontrar el conjunto de regresores que mejor ajusten a nuestros datos, hemos realizado un procedimiento de búqueda del mejor subconjunto. Encontrando ['NTE_min','OCC_min'] como el conjunto con los mejores resultados con una exactitud de 0.8468. El código implemenetado se muestra a continuación. 

```python
## Loop over all possible numbers of features to be included
results = pd.DataFrame(columns=['num_features', 'features', 'accuracy'])
for k in range(1, X_train.shape[1] + 1):

    # Loop over all possible subsets of size k
    for subset in itertools.combinations(range(X_train.shape[1]), k):
        subset = list(subset)            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # Fit a logistic GAM    
            gam = LogisticGAM(max_iter=1000, tol=0.001, verbose=True).fit(X_train.iloc[:, subset], y_train)         
        
            accuracy = accuracy_score(y_test, gam.predict(X_test.iloc[:, subset]))
            results = results.append(pd.DataFrame([{'num_features': k,
                                                    'features': subset,
                                                    'accuracy': accuracy}]))
print(results.sort_values('accuracy'))
subset_best = list(results.sort_values('accuracy')['features'].head(1)[0]) ## Select the Best-Subset of variables
```

### Poda de árbol de decisión usando validación cruzadas **cross-validation**
Para obtener un árbol de decisión para predecir la sobrecarga en líneas de transmisión hemos utilizado la función **DecisionTreeClassifier** de la librería **sklearn**. Calcularemos un árbol para cada una de las muestras `X_test` extraidas del total del conjunto de entrenamiento `X_train`. De acuerdo al procedimeinto del libro se aplicó validación cruzada con `n_splits=10`. 

Para aplicar el procedimiento de validación cruzada hemos utilizado las funciones **KFold** y **cross_val_score** de la librería **sklearn**. A diferencia del libro que utiliza **Misclassification error** en el problema de spam, la medida de error que usaremos será **Gini index** que es más sensible a cambios en las probabilidades de cada nodo a diferencia del propuesto en el libro. 

Los datos de error de la validación cruzada se guardan en la lista `cross_ols`. 

```python
#Evaluación del desempeño del bootstrap variando tamaño de las muestra y número de muestreos aleatorios (repeticiones).
# evaluate a decision tree model using k-fold cross-validation
size =[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] 
test = []
accuracy = []

for i in size:
    cv = KFold(n_splits = 10, random_state = 1, shuffle = True)
    clf = tree.DecisionTreeClassifier(max_leaf_nodes = i,criterion = "entropy", random_state = 100,
                                  max_depth=100, min_samples_leaf=5)
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    accuracy.append(mean(scores))

test.append(accuracy)
```
La gráfica con los valores de el número de odos y la exactitud del modelo se muestran a continuación:
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t9_2.png)

El resultado de la validación cruzada para estimar la mejor exactitud nos da un óptimo de ocho nodos, el árbol resultante queda de la siguiente manera:

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t9_3.jpg)

```python
clf = tree.DecisionTreeClassifier(max_leaf_nodes=8, criterion = "gini", random_state = 100,
                               max_depth=10, min_samples_leaf=5)
clf = clf.fit(X_train, y_train)
print('Test accuracy árbol = ', accuracy_score(y_test, clf.predict(X_test)))

#compute confussion matrix 
confussion_matrix = confusion_matrix(y_test, clf.predict(X_test))
```
Además, con el objetivo de comparar el modelo del árbol con los demás procedimientos utilizados (regresión logística, regresión logística con stepwise, mejor subconjunto en regresión logística y el modelo logístico aditivo) calculamos la exactitud y su matriz de confusión.

```
Test accuracy árbol =  0.7489
[[77 19]
 [40 99]]
```

### **Conclusión tarea 9**
Hemos utilizado **pygam** para implementar regresión logistica aditiva y la librería **sklearn*** para calcular un árbol de decisión. Además, se utilizó la función **Logit** de la librería **statsmodel** con el objetivo de comparar los resultados con modelos logísticos con y sin stepwise, además de el mejor subconjunto. Aunque los resultados en exactitud fueron muy semejantes, se encontró que la exacttitud de la regresión logística aditiva fue de 0.7319 y para el árbol de ocho nodos la exactitud de 0.7489. De todos los modelos comparados el que mejor desempeño tuvo fue el de selección del mejor subconjunto con un 0.84680 de exactitud.

---

## **Tarea 10 Impulso**
>**Instrucciones:** Replicate the steps of the California housing example of Section 10.14.1 (with some library implementation) unless you really want to go all-in with this) to explore potential dependencies and interactions in the features of your data.

En esta tarea aplicaremos el procedimiento de predicción de precios de casas desarrollado en el libro [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/) de la `sección 10.14.1 California Housing` a nuestros datos para predecir demanda eléctrica. El código completo de esta tarea se encuentra en [Tarea10.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea10.ipynb). Aquí solo se presentan los resultados y las secciones mas relevantes del código. Los datos usados en esta sección están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing).

Usaremos un modelo de impulso de gradiente **(Gradient Boosting)** para producir un modelo predictivo a partir de un conjunto de modelos predictivos débiles **(weak)**, usando la función **ensemble.GradientBoostingRegressor** de la librería **sklearn**. El **Gradient Boosting** se puede utilizar para problemas de regresión y clasificación. En esta tarea, entrenaremos un modelo de regresión para predecir demanda eléctrica usando datos de demanda de días semejantes. La variable independiente `X` serán los datos de demanda del día anterior, y los datos independiente `Y` serán los datos de días con una mayor correlación con `X`.

Iniciamos definiendo los parámetros del modelo **GradientBoostingRegressor** con 1000 árboles de regresión `n_estimators=500`, con una profundidad de `max_depth=6` y una tasa de aprendizaje de `learning_rate": 0.1`, la función de pérdida utilizada será la desviación absoluta.

```python
params = { "n_estimators": 1000,
           "max_depth": 6,
           "min_samples_split": 5,
           "learning_rate": 0.1,
           "loss": "absolute_error",} 
```
Ahora, ajustaremos un modelo con nuestros datos de entrenamiento y calculamos algunas métricas de error en los datos de prueba.

```python
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mae = mean_absolute_error(y_test, reg.predict(X_test))
print("El error medio absoluto (MAE) en datos de prueba es: {:.4f}".format(mae))
```
```
El error medio absoluto (MAE) en datos de prueba es: **186.3801**
El error cuadrático medio (MSD) en datos de prueba es: **60367.8724**
El error medio absoluto porcentual (MAPE) en datos de prueba es: **0.0214**
```
En la figura siguiente visualizamos el proceso de ajuste con los datos de entrenamiento y prueba. Calculamos el error del conjunto de datos de entrenamiento y luego la compararemos con las iteraciones del conjunto de datos de prueba.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_1.png)

En la figura siguiente se representa la importancia relativa de los predictores. Podemos observar que los regresores `X11` Y `X1` tienen una ligera importancia sobre los demás, todas las demás variables tienen una relevancia ligeramente menor con un decremento monotónicamente decreciente.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_2.png)

La dependencia de cada una de las variables la podemos analizar por medio de las figuras siguientes. En el eje horizontal se observa el valor que ha tomado la variable con múltiples modelos débiles **(weak)**. Y en el eje vertical la relevancia relativa, todas las figuras tienen la misma escala así que la comparación es directa. Las discontinuidades que se observan en las figuras se deben al uso del modelo de árbol. Podemos observar algunas figuras con una curva cuasi-horizontal cercana al cero que indica baja relevancia, tal es el caso de `X6` o  `X7`, aunque se observan algunos valores extremos en los últimos deciles. Tambien se presentan algunos casos en que se observan en la misma figura simultamenamente relevancias altas positivas y negativas divididas por discontinuidades como por ejemplo en `X2`,`X3`, `X10` y `X11`. Otros predictores presentan relevancias relativamente más suaves en toda la distribución de los deciles como por ejemplo `X8`,`X9` y `X17`. Otras tienen una importancia más cercana al cero y son ruidosas como por ejemplo `X15` y `X18`.

Estas figuras nos pueden ayudar a tomar decisiones para hacer un modelo mas esbelto, eliminanado variables o poniendo especial atención en algunas de las variables mas relevantes y sus rangos de sensibilidad.  

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X1.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X2.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X3.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X4.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X7.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X8.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X9.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X10.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X11.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X12.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X13.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X14.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X15.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X16.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X17.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X18.png)

La figura que se muestra a continuación compara la relevancia entre las dos variables mas relevantes, las zonas de color muestran la dependencia parcial entre las dos variables con mayor o menor relevancia. Esta figura en dos dimensiones es semejante a la de tres dimensiones presentada en el libro en: *FIGURE 10.16. Partial dependence of house value on median age and average occupancy.*

En este caso se observa una relación fuerte entre las dos principales variables `X11` y `X1` principalmente en los cuadrantes inferior-izquierdo y superior-derecho. 

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t10_X_X_.png)

Por último, comparamos el error obtenido con los resultados de otros modelos de regresión líneal y local reportados en la [Tarea6.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea6.ipynb). Para el caso de regresión local, `k` son los tamaños de las vecindades.
| REGRESIÓN      | MAE            | MSD            | MAPE         |
| :------------- | -------------: | -------------: |-------------:|
| local, k=10    | 77.4973    | 323440.045    |    0.009 |
| local, k=25    | 83.5068    | 14118.722    |    0.0096 |
| local, k=35    | 98.6564    | 18483.2236    |    0.0113 |
| local, k=50    | 109.6632    | 21942.6356    |    0.0126 |
| local, k=100   | 123.9285    | 26331.6078    |    0.0142 |
| lineal         | 138.5861     | 32615.1951    |    0.0159 |
| **GradientBoostingRegressor** | **186.3801**     |  **60367.8724**   |  **0.0214**   |

Como puede verse el método **GradientBoostingRegressor** tiene un menor desempeño que otros métodos. Sin embargo, con esté método tenemos la posibilidad de hacer análisis del comportamiento de los predictores en la predicción de la variable dependiente, como fue demostrado.

### **Conclusión tarea 10**
Hemos utilizado la función **ensemble.GradientBoostingRegressor** de la librería **sklearn** para implementar el método de aumento de gradiente en árboles de decisión regresivos. Estos modelos pueden ser de gran utilidad ya que al crear muchas réplicas con modelos de árbol débiles (weak) podemos analizar la relevancia de los predictores en su capacidad de predicción de la variable dependiente e incluso hacer gráficas para analizar su importancia. Por ejemplo, podemos observar el comportamiento de gráficas de dependencia parcial. La densidad de la importancia de una variable se puede interpretar en deciles superpuestos en el eje horizontal. En nuestro caso se observaron una gran variedad de comportamientos de los predictores. Aunque otros métodos pueden dar mejores resultados en exactitud como los de regresión lineal o regresión local, la ventaja de estos métodos radica en poder hacer estudios de sensibilidad de los predictores.

---

## **Tarea 11 Redes Neuronales**
>**Instrucciones:** Go over the steps of the ZIP code examples in Chapter 11 and replicate as much as you can with your own project data. Don't forget to read the whole chapter before you start.


### Predicción de sobrecarga en grupos de líneas de transmisión usando Redes Neuronales Artificiales.
En esta sección se usaran redes neuronales para ajustar un modelo de predicción en datos de violación de flujo de potencia eléctrica en grupos de líneas de transmisión, que interconectan regiones eléctricas. La variable dependientes son un vector de naturaleza binaria dónde cada componente del vector representa una línea de transmisión. El valor cuando una línea presenta sobrecarga es uno y cero si no. Las variables independientes son el flujo neto máximo y mínimo en la región eléctrica [CEN,GUA,NES,NOR,NTE,OCC,ORI,PEN] en un día y se calcula como la diferencia entre la demanda menos la generación en cada región.

Los datos son obtenidos de 334 simulaciones de planeación de la operación de un día en adelanto de la red eléctrica en México y están disponibles en [overload.csv](https://drive.google.com/file/d/1Q8Pk5apApNbcoqmKQp3RvQFvuk4DKylU/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea11.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea11.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

A continuación se enlista las configuraciones de red usadas en el libro (para resolver el problema *11.7 Example: ZIP Code Data*), las que fueron aplicadas a nuestro problema de predicción de sobrecarga en líneas de transmisión. 

*   Net-1: Sin capa oculta, equivalente a regresión logística multinomial.
*   Net-2: Una capa oculta, 12 unidades ocultas totalmente conectadas.
*   Net-3: Dos capas ocultas conectadas localmente.
*   Net-4: Dos capas ocultas, conectadas localmente con peso compartido.
*   Net-5: dos capas ocultas, conectadas localmente, dos niveles de peso compartido.

Trataremos de replicar el ejercicio aplicando a nuestros datos. A diferencia del problema de 2 dimensiones del libro, usaremos redes para una sola dimensión. El objetivo es predecir un vector binarios (sobrecarga o no de algunas líneas de transmisión) a partir de un vector de datos reales (datos de demanda de las regiones eléctricas).

**Net-1**
```python
input_dim   = 14      ## {entradas}
num_classes = 26      ## {salidas}
model  = Sequential()
model.add(Dense(units = num_classes, input_dim = 14, activation='sigmoid' ))
```
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 26)                390       
                                                                 
=================================================================
Total params: 390
Trainable params: 390
Non-trainable params: 0
_________________________________________________________________
```
Exactitud obtenida  Net-1: 1.7021

**Net-2**
```python
model.add(Dense(units=12, input_dim = input_dim,  activation ='sigmoid' ))
model.add(Dense(units=num_classes,                activation ='sigmoid'))
```
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 12)                180       
                                                                 
 dense_1 (Dense)             (None, 26)                338       
                                                                 
=================================================================
Total params: 518
Trainable params: 518
Non-trainable params: 0
_________________________________________________________________
```
Exactitud obtenida Net-2: 14.4680


**Net-3**
```python
input_dim = (14,1)
input_ = Input(input_dim, name = 'the_input')
layer1 = LocallyConnected1D(1, 2, strides= 2, activation= 'sigmoid', name = 'layer1')(input_)
layer2 = LocallyConnected1D(1, 5, activation='sigmoid', name = 'layer2')(layer1)
layer3 = Flatten(name='layer3')(layer2) 
output = Dense(units=num_classes, activation='sigmoid', name = 'output')(layer3)
model = Model(inputs = input_, outputs = output)
input_dim = np.expand_dims(input_dim, axis=0)
```
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 the_input (InputLayer)      [(None, 14, 1)]           0         
                                                                 
 layer1 (LocallyConnected1D)  (None, 7, 1)             21        
                                                                 
 layer2 (LocallyConnected1D)  (None, 3, 1)             18        
                                                                 
 layer3 (Flatten)            (None, 3)                 0         
                                                                 
 output (Dense)              (None, 26)                104       
                                                                 
=================================================================
Total params: 143
Trainable params: 143
Non-trainable params: 0
_________________________________________________________________
```
Exactitud obtenida Net-3: 42.5531

**Net-4**
```python
input_dim = (14,1)
input_ = Input(input_dim, name = 'the_input')
layer1 = Conv1D(filters=2, kernel_size=2, strides=2, activation='sigmoid', name='layer1')(input_) 
layer2 = LocallyConnected1D(1, 5, activation='sigmoid', name='layer2')(layer1)
layer3 = Flatten(name='layer3')(layer2) 
output = Dense(units=num_classes, activation='sigmoid', name = 'output')(layer3)
model = Model(inputs = input_, outputs = output)
input_dim = np.expand_dims(input_dim, axis=0)
```
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 the_input (InputLayer)      [(None, 14, 1)]           0         
                                                                 
 layer1 (Conv1D)             (None, 7, 2)              6         
                                                                 
 layer2 (LocallyConnected1D)  (None, 3, 1)             33        
                                                                 
 layer3 (Flatten)            (None, 3)                 0         
                                                                 
 output (Dense)              (None, 26)                104       
                                                                 
=================================================================
Total params: 143
Trainable params: 143
Non-trainable params: 0
_________________________________________________________________
```
Exactitud obtenida Net-4: 0.0

**Net-5**
```python
input_dim = (14,1)
input_ = Input(input_dim, name = 'the_input')
layer1 = Conv1D(2, 2, strides= 2, activation= 'sigmoid', name = 'layer1')(input_)
layer2 = Conv1D(4, 5, activation='sigmoid', name = 'layer2')(layer1)
layer3 = Flatten(name='layer3')(layer2) 
output = Dense(units=num_classes, activation='sigmoid', name = 'output')(layer3)
model = Model(inputs = input_, outputs = output)
input_dim = np.expand_dims(input_dim, axis=0)
```
```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 the_input (InputLayer)      [(None, 14, 1)]           0         
                                                                 
 layer1 (Conv1D)             (None, 7, 2)              6         
                                                                 
 layer2 (Conv1D)             (None, 3, 4)              44        
                                                                 
 layer3 (Flatten)            (None, 12)                0         
                                                                 
 output (Dense)              (None, 26)                338       
                                                                 
=================================================================
Total params: 388
Trainable params: 388
Non-trainable params: 0
_________________________________________________________________
```
Exactitud obtenida Net-5 : 0.0

Los párámetros de ajuste de epoch (numero de macro-iteraciones) y de batch_size (número de muestras que se enviamos al modelo a la vez), Además se mueastran los procesos de compilación y ajuste del modelo con datos de entrenamiento y de validación en datos de prueba.
```python
epochs     = 100 
batch_size = 64
verbose    = 0
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose) 
history = model.fit(X_train, y_train, validation_data = (X_test,y_test), epochs=epochs, batch_size=batch_size,verbose=verbose)
```

En la siguiente tabla se resumen las exactitudes obtenidas por cada una de las configuraciones de red.
|                |Net-1           | Net-2          | Net-3          | Net-4          | Net-5          |  
| :------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
|Exactitud |1.7021|14.4680|42.5531|0.0|0.0|

***Como podemos ver la estategia de predicción de fallas simultanea de todas las líneas de transmisión no ha dado buenos resultados.*** Por lo que cambiaremos la estrategia a modelos de predicción separados por cada una de las líneas.

### Predicción de sobrecarga en líneas de transmisión por modelos independientes usando redes neuronales.
Repetiremos el ejercicio anterior con las cinco configuraciones de red: **Net-1**, **Net-2**, **Net-3**, **Net-4** y **Net-5** para predecir solo una línea a la vez como una variable bimodal, donde 1 = sobrecarga y 0 = no sobrecarga. Además, usaremos las series de datos más balanceadas entre el 30% y 70% de sobrecargas. Las líneas estudiadas fueron: **L3,L6,L7,L38,L39,L42,L50,L65,L67,L72,L75,L92,L93,L94,L95**. 

En las siguiente tabla podemos ver los resultados de exactitud de cada una de las líneas para cada una de las configuraciones de red.

**EXACTITUD**

|LÍNEA          |Net-1      | Net-2   | Net-3      | Net-4      | Net-5    |  
| :------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
|L3|73.1915|79.5745|40.8511|74.4681|58.7234|
|L6|81.2766|90.6383|64.6809|79.5745|76.5957|
|L7|78.2979|78.2979|54.8936|51.4894|53.1915|
|L38|74.4681|80|57.0213|56.1702|57.0213|
|L39|90.6383|97.4468|92.766|96.1702|95.3192|
|L42|90.2128|90.2128|85.9574|81.7021|90.6383|
|L50|99.1489|99.5745|76.5957|76.5957|76.5957|
|L65|90.6383|87.6596|74.8936|74.8936|57.8723|
|L67|99.1489|100|76.1702|76.1702|76.1702|
|L72|99.5745|99.5745|76.5957|76.5957|99.5745|
|L75|77.4468|94.0426|73.1915|73.1915|73.1915|
|L92|97.4468|99.1489|76.1702|76.1702|95.7447|
|L93|99.1489|99.1489|94.0426|99.5745|99.1489|
|L94|86.8085|96.5957|90.2128|81.2766|94.0426|
|L95|96.1702|98.7234|95.3192|92.766|95.3192|
|Promedio|88.9078|**92.7092**|75.2908|77.7872|79.9433|
|Desv.Estandar|9.7242|7.9445|15.7039|12.7479|17.0098|

En las siguiente tabla podemos ver los resultados de la función de pérdida de cada una de las líneas para cada una de las configuraciones de red.

**FUNCIÓN DE PÉRDIDA**
|LÍNEA          |Net-1      | Net-2   | Net-3      | Net-4      | Net-5      |
| :------------- | -------------: | -------------: | -------------: | -------------: | -------------: |
|L3|53.6114|46.4257|69.5333|67.3124|68.1058|
|L6|49.485|32.8382|58.6809|54.2588|55.9541|
|L7|60.7573|53.6467|75.1925|76.5696|75.5821|
|L38|62.0817|49.44|64.4072|65.1648|64.9379|
|L39|35.8939|14.7518|43.9997|36.0598|35.8088|
|L42|48.0594|27.3678|43.4613|47.2103|33.9506|
|L50|38.5398|7.939|71.9694|70.022|60.1689|
|L65|55.206|27.0214|55.7251|52.7564|60.57|
|L67|18.7448|9.894|62.4763|61.3409|52.1674|
|L72|7.815|4.1148|55.6773|54.6902|23.3024|
|L75|65.7197|28.2008|58.1139|60.2073|69.909|
|L92|25.182|7.1432|47.8633|42.7808|28.1238|
|L93|15.307|4.3076|29.9451|19.2315|11.3876|
|L94|43.1572|14.9036|32.9231|36.4453|19.3367|
|L95|35.0614|8.4832|21.32|22.504|12.8717|
|Promedio|40.9748|**22.4319**|52.7526|51.1036|44.8118|
|Desv.Estandar|17.9433|16.9910|15.9336|16.9863|22.2217|

Con el objetivo de analizar los resultados de exactitud y pérdida por cada modelo, hemos dibujado un diagrama de caja para exactitud y para pérdida. Cada serie nos representa los resultados de cada configuración de red en todas la líneas. En los diagramas podemos notar que los mejores resultados se han logrado con la red **Net-2** que tiene en promedio una mayor exactitud y una menor perdida. Otra red con resultados semejantes es la **Net-1**. Como podemos ver en los resultados, la calidad en la predicción mejora si hacemos un modelo por cada línea separadamente.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t11_6.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t11_7.png)

Algo interesante que se observó es que para algunos casos con niveles de exactitud muy altos cercanos al 100% la gráfica de exactitud vs epoch presenta una recta horizontal tanto en datos de entrenamiento como en datos de prueba. Este comportamiento se presenta en las líneas: **L50 L67 L72 L92 L93** con la red **Net-2** de mejor desempeño. A manera de ejemplo se muestra el comportamiento de la línea **L50**.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t11_2_a.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t11_2_b.png)


Paralelamente se utilizó un modelo de árbol de decisión para predecir la sobrecarga en todas las líneas, de este estudio se obtuvo la exactitud y la complejidad del árbol. La exactitud medida con el número de nodos en el árbol. Los resultados arrojaron niveles de exactitud semejantes al de la red neuronal y lo más interesante es que **las mismas líneas que presentaron el comportamiento de recta horizontal en redes neuronales coincidieron con los modelos de árbol menos complejos de tres nodos**. Estos resultados puede verse en la tabla siguiente:


**REDES NEURONALES VS ÁRBOL DE DECISIÓN**

|LINEA           |Net-2(EXACTITUD)|ÁRBOL(EXACTITUD)|NUM. DE NODOS   |
| :------------- | -------------: | -------------: | -------------: |
|L3|79.5745      |73.617  |19|
|L6|90.6383      |89.3617 |15|
|L7|78.2979      |76.1702 |13|
|L38|80          |78.7234 |15|
|L39|97.4468     |92.766  |9 |
|L42|90.2128     |88.0851 |11|
|L50|**99.5745** |98.7234 |**3** |
|L65|87.6596     |90.2128 |11|
|L67|**100**     |99.1489 |**3** |
|L72|**99.5745** |98.7234 |**3** |
|L75|94.0426     |95.7447 |5 |
|L92|**99.1489** |99.1489 |**3** |
|L93|**99.1489** |99.1489 |**3**|
|L94|96.5957     |92.3404 |5 |
|L95|98.7234     |92.3404 |5 |

De acuerdo a estos resultados, podemos deducir es que el modelo de predicción de sobrecarga de algunas líneas es muy sencillo tanto para una red neuronal de predicción como para un árbol de decisión tambien. Concluimos para nuestro problema que la complejidad del árbol está relacionada también con el tiempo de entrenamiento y complejidad de una red neuronal. 

Ejemplo de un árbol de decisión de tres nodos para la línea **L50** con perfil de entremiento horizontal en red neuronal **Net-2**.
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t11_tree.png)

### **Conclusión:** 
Hemos utilizado la API funcional de **Keras** para implementar diferentes estructuras de redes neuronales para predicción de sobrecarga en líneas de transmisión de acuerdo a la demanda en las regiones.

Sobre el primer ejercicio donde se pretende predecir de manera conjunta las sobrecargas podemos observar que mientras la complejidad de la estructura en las redes Net-1, Net-2 y Net-2 aumenta, la exactutud del modelo mejora. Sin embargo, en el caso de las redes Net-4 y Net-5 la exactitud del modelo fue de cero, la característica común de estas dos redes son que comparten pesos entre capas. Por lo que podemos suponer que para nuestros datos, esta estrategia puede no ser adecuada.
En general el desempeño de las cinco redes fue muy malo para nuestros datos, para poder obtener mejores resultados, podriamos cambiar las arquitecturas,  parámetros e incluso optimizadores (para esta tarea se usaron estrictamente las estructuras y parpametros del libro).

En el segundo ejercico donde se realizaron redes neuronales independientes uno por cada línea, se tuvieron mejores resultados con la red **Net-2** y la red  **Net-1**. La exactitud de los modelos de predicción por línea fue en general mayor que de manera conjunta. 

Concluimos para nuestro problema que la complejidad del árbol está relacionada también con el tiempo de entrenamiento y complejidad de una red neuronal. 

---

## **Tarea 12 Máquinas de Vectores de Soporte**
>**Instrucciones:** Pick either (a variant of) SVM or a generalization of LDA and apply it on your project data. Remember to analyze properly the effects of parameters and design choices in the prediction error.

### Predicción de demanda eléctrica usando máquinas de vectores de soporte (regresión)
A continuación utilizaremos **máquinas de vectores de soporte** en su extensión de regresión para predecir demanda eléctrica. La variable independiente `Y` serán los datos de demanda de 24 horas en intervalos de 5 minutos (288 datos), y las variables independientes X serán los datos de otros días con una mayor correlación. Los datos se han dividido en datos de entrenamiento (`X_train`,`y_train`) y datos de prueba (`X_test`,`y_test`). El objetivo es encontrar el mejor modelo de pronóstico para los datos de demanda.

Los datos usados en esta sección están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea12.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea12.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

Primero hacemos una lista con los parámetros a probar en los modelos de SVM, en este caso modificaremos el tipo de kernel y el valor epsilon, este valor de epsilon será el que defina el margen de error aceptable hacia arriba y hacia abajo del hiperplano que se está buscando.
```python
kernel_list = ['poly','rbf']
epsilon_list = [1,5,10,15,25]
```

Ahora, haremos un ciclo con estos parámetros modificando el valor del C="costo", este valor C es la penalización en la funcion objetivo (del problema de minimización del SVM) de la desviación de los datos a la banda de tolerancia de error.
```python
for k in kernel_list:
    for e in epsilon_list:
        mae_svm = []; mse_svm = []; mape_svm = []; Clist = []; perc_within_eps_list = []    
        for c in range(1, 55, 1):
            Clist.append(c)
            model = svm.SVR(kernel=k, C=c, epsilon=e)
            model.fit(X_train, y_train)            
            y_pred = model.predict(X_test)
            mae_svm.append(trunc(mean_absolute_error(y_test,y_pred),4))
            
            perc_within_eps = 100 * np.sum(abs(y_test-y_pred) <= e) / len(y_test)
            perc_within_eps_list.append(perc_within_eps)
```
Además guardamos `perc_within_eps_list` que es el porcentaje de datos que quedan fuera de la banda de tolerancia de error (que no se penaliza con ningún costo en la función objetivo). El objetivo es cuantificar la cantidad de datos que podrían estar equivocados en el ajuste.

Si graficamos los resultados de costo  `C` contra el porcentaje de datos `perc_within_eps_list` dentro de la tolerancia de error queda: 

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_poly_1.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_poly_5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_poly_10.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_poly_15.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_poly_25.png)

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_rbf_1.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_rbf_5.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_rbf_10.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_rbf_15.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t12_rbf_25.png)

Ahora seleccionaremos un modelo usando el método **GridSearchCV** de **sklearn** que nos da los parámetros `kernel`, `C` y `epsilon` del mejor ajuste de acuerdo a una métrica de error establecida (neg_mean_absolute_error). Repetiremos el procedimiento para cada tipo de kernel ('poly', 'rbf', 'linear')

```python
Clist = []
for c in range(1, 60, 1):
    Clist.append(c)

parameters = {'kernel': ('rbf'), 'C': Clist,'epsilon': [1,2,5,10,15,25]} 
model = svm.SVR()
clf   = GridSearchCV(model, parameters,scoring='neg_mean_absolute_error', cv=5)
clf.fit(X_train, y_train)
model = clf.best_estimator_
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("C:       {}".format(model.C))
print("Epsilon: {}".format(model.epsilon))
print("Kernel:  {}".format(model.kernel))    
mae = mean_absolute_error(y_test, model.predict(X_test))
print("MAE = {:,.2f}".format(1000*mae))    
perc_within_eps = 100*np.sum(y_test - model.predict(X_test) < model.epsilon) / len(y_test)
print("Percentage within Epsilon = {:,.2f}%".format(perc_within_eps))
```

El método **GridSearchCV** nos sugiere un conjunto de parámetros para cada kernel, por ejemplo para el kernel **radial** obtenemos los siguientes resultados:    
```  
coefficient of determination: 0.7466049842240431
C:       2
Epsilon: 10
Kernel:  poly
MAE = 170,731.45
Percentage within Epsilon = 37.15%
```
El mejor conjunto de parámetros del kernel **lineal**:
```
C:       0.0001
Epsilon: 0.0001
Kernel:  linear
coefficient of determination: 0.7663451754599173
mae_linear : 161.3644
mse_linear : 44209.1792
mape_linear : 0.0185
Percentage within Epsilon = 34.38%
```
El mejor conjunto de parámetros del kernel **polinómico**:
```
C:       1
Epsilon: 0
Kernel:  poly
coefficient of determination: 0.7524545558665545
mae_poly : 168.2798
mse_poly : 46837.3847
mape_poly : 0.0192
Percentage within Epsilon = 35.76%
```

A continuación, compararemos los resultados de predicción entre los modelos de **máquinas de vectores de soporte** y el método de regresión lineal múltiple. Ahora, calculamos los errores entre la predicción `y_pred` y los datos de entrenamiento `y_train`.

| REGRESIÓN      | MAE            | MSD            | MAPE         | 
| :------------- | -------------: | -------------: |-------------:|
|    lineal      | 167.1343       | 48064.1398     |     0.0192   |
|    SVM rbf     | 163.5971       | 46870.6049     |     0.0184   |
|    SVM poly    | 168.2798       | 46837.3847     |     0.0192   |
|    SVM linear  | 161.3644       | 44209.1792     |     0.0185   |

El ajuste de manera gráfica de las predicciones a los datos reales se ven en la siguiente gráfica:

![image](https://github.com/urieliram/statistical/blob/main/figures/pronodemanda_t12_1.png)

### Conclusiones tarea 12
En esta tarea se utilizó el método de **máquinas de vectores de soporte** (SVM) usado en su versión de regresión para predecir demanda eléctrica en una región partir de datos de días semejantes (variable independientes) y datos de 24 horas antes (variable dependiente). Para poder sintonizar los parámetros del modelo, se hicieron pruebas con diferentes kernels: líneal, polinómico, y radial. Tambien se modificaron los tamaños de una tolerancia epsilon que establece un rango de error aceptado de alejamiento del hiperplano. Tambien se modificó C, que es el "costo" de la distancia de los puntos al hiperplano que estan fuera de la banda de error permitida establecida en 2 unidades de epsilon. Con el objetivo de analizar el comportamiento de los parámetros se trazaron gráficas en las que podemos comparar el error de la predicción contra el porcentaje de datos qe caen dentro de la banda de tolerancia del error. Con esta información podemos elegir el mejor model. El método de SVM nos da esta flexibilidad de decidir el nivel de error aceptado en el modelo a traves del valor de epsilon. 
Usamos el método de **GridSearchCV** de **sklearn** que nos da los parámetros del mejor ajuste de acuerdo a una métrica de error establecida. Los resultados aunque minimizan el error, están condicionados a aceptar un margen de error alto que vemos con un alto porcentaje de datos en esta banda de tolerancia.
Además, se compararon los resultados con el método de regresión líneal múltiple. Se comprobó que el SVM tuvo un mejor desempeño en nuestros datos que el método de regresión linael. La idea principal de un SVM es identificar el hiperplano que cubre la mayoria de los datos teniendo en cuenta que se tolera parte del error.

---

## **Tarea 13 Prototipos y vecinos**
>**Instrucciones:** After a complete read-through of Chapter 13, make a comparison between (some variants) of k-means, LVQ, Gaussian mixtures, and KNN for your project data. Since these models are pretty simple, implement at least one of them fully from scratch without using a library that already does it for you.

Los datos de radiación solar se están disponibles en [rg1_horas.csv](https://drive.google.com/file/d/1jrrJgZhSsoQRWUUOZ0-I2ZCmMT1iDJT7/view?usp=sharing). Los datos de aportaciones hidráulicas en cuencas se encuentran disponibles en [Aportaciones_Embalses.csv](https://drive.google.com/file/d/1GJTg0J5W-061Dh8O4bNNCBiMpHwlCr9L/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea13.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea13.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

### Agrupamiento de perfiles de radiación solar horaria en una planta fotovoltaica.

A continuación utilizaremos las técnicas de **k-means, LVQ, Gaussian mixtures** y **KNN** usando las librerias de **sklearn** para clasificar los días en `n_clusters`. Además, implementaremos el método de **KNN** "a mano" para encontrar los `k` vecinos mas cercanos a prototipos (centroides) obtenidos previamente con **k-means**.

En la gráficas siguientes se muestra la radiación solar horaria. Cada serie representa un día y cada dato es el promedio de radiación de un día. Los datos fueron divididos en los meses de verano e invierno. Se han eliminado de la serie las horas de noche.
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_Verano.png)
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_Invierno.png)

#### **K-means**
El objetivo es agrupar las series para obtener prototipos, en el caso de **k-means** son llamados centroides.
```python
n_clusters = 10
k_means = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, n_init=10) # una variante  de KMeans
k_means = k_means.fit(X_train)
values  = k_means.cluster_centers_.squeeze()
labels  = k_means.labels_
kmeans_centers_= k_means.cluster_centers_
y_kmeans = k_means.predict(X_test)
y_kmeans_train = k_means.predict(X_train)
print_serie(kmeans_centers_,'Centroides','Radiación solar','horas',True,'fig_t13_centroids_kmeans')
```
Los centroides de **k-means** se muestran a continuación:
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_centroids_kmeans.png)

Ahora imprimimos los centroides con las series de datos que le corresponden. Cada subplot representa un conglomerado o cluster.
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_kmeans_train.png)


#### **LVQ**
```python
print('GLVQ')
glvq = GlvqModel(prototypes_per_class=1, initial_prototypes=None)
glvq.fit(X_train,labels)
glvq_pred = glvq.predict(X_test)
glvq_pred_train = glvq.predict(X_train)
print_patrones(list_series=X_train,list_categorias=glvq_pred_train,list_patrones=[],title_='Conglomerado ',namefile_='fig_t13_lvq_train')
```
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_lvq_train.png)

#### **GaussianMixture**
```python
gm = GaussianMixture(n_components=n_clusters, init_params='kmeans',covariance_type='full') #full, tied, diag, spherical
gm.fit(X_train)
gm_pred_train = gm.predict(X_train)
gm_pred       = gm.predict(X_test)
print('classification accuracy train:', gm.score(X_train, gm_pred_train))
print_patrones(list_series=X_train,list_categorias=gm_pred_train,list_patrones=[],title_='Conglomerado ',namefile_='fig_t13_gm_train')
```
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_gm_train.png)

#### **KNN**
```python
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, labels)
knn_pred_train = knn.predict(X_train)
knn_pred       = knn.predict(X_test)
print(knn.score(X_test,knn_pred))
print_patrones(list_series=X_train,list_categorias=knn_pred_train,list_patrones=[],title_='Conglomerado ',namefile_='fig_t13_knn_train')
```
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_knn_train.png)


#### **KNN form scratch** 
Implementaremos el método de KNN para dado un elemento, determinar los k vecinos más cercanos.
```python
## Calculamos la ditancia euclidiana de un elemento a todos los vecinos.
def euclidean(neig1, neig2):
	  distance = 0.0
	  for i in range(len(neig1)):
		    distance += (neig1[i] - neig2[i])**2
	  return sqrt(distance)

## Encuentra los vecinos más cercanos
def get_neighbors(train, test_row, num_neighbors):  
    distances = list()
    for train_row in train:
        dist = euclidean(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Predecimos un elemento con KNN
def predict_classification(train, test_row, num_neighbors):
	  neighbors = get_neighbors(train, test_row, num_neighbors)
	  output_values = [row[-1] for row in neighbors]
	  prediction = max(set(output_values), key=output_values.count)
	  return prediction
```

Probaremos el método para obtener los `k` vecinos mas cercanos a los centroides obtenidos por el método de **k-means**.
```python
serie = []
cat   = []
i     = 0 
for center in kmeans_centers_:
    neighbors = get_neighbors(X_train, center, num_neighbors=5)
    for neighbor in neighbors:
        serie.append(neighbor)
        cat.append(i)
    i = i  + 1
print_patrones(list_series=serie,list_categorias=cat,list_patrones=kmeans_centers_,title_='Centroide ',namefile_='fig_t13_knn_scratch')
```
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_knn_scratch.png)

### Agrupamiento de aportaciones KNN hidrológicas en presas.
Usaremos el método de **KNN** en un método que selecciona las `k` ventanas más parecidas en una serie de tiempo, usando como prototipo la última ventana de una serie, con los k vecinos seleccionados posteriormente se hace una predicción con los datos inmediatos para hacer un pronóstico. Para probar el método usaremos una serie de tiempo de aportaciones hidráulicas (lluvias) mensuales en la presa Peñitas en Tabasco.

Esta idea de usar **KNN** para extraer regresores de una serie de tiempo es obtenida de el artículo de [Grzegorz Dudek](https://doi.org/10.1016/j.epsr.2015.09.001) y es implementada y aplicada en esta tarea a aportaciones hidráulicas en presas.
```python
df = pd.read_csv('Aportaciones_Embalses.csv')
apor = df['PEA'].to_numpy()
v = 12                    ## tamanio de la ventana (un año)
k = 10                    ## número de vecinos a buscar k
vecindario    = []        ## vecindario completo
vecindario_b  = [] 
distances     = []
n             = len(apor) ## longitud total de la serie
tol           = 0.6       ## tolerancia de tamaño de ventanas para seleccion de vecinos

#print(apor[n-v:n],'**')  ## imprime el prototipo
## Se calcula la distancia euclidiana entre todos los vecinos.
for i in range(n-2*v+1):
    dist = euclidean(apor[n-v:n],apor[i:i+v])
    distances.append((i, dist))
    #print(apor[n-v:n],apor[i:i+v],i, dist)

## Se ordena el vecindario por distancia de menor a mayor y se guardan las posiciones.
distances.sort(key=lambda tup: tup[1])
neighbors  = []
neighbors2 = []
position   = []

## Se escogen los k vecinos mas cercanos y guardamos las posiciones.
i = 0
for pos, dis in distances:
    #print(apor[pos:pos+v],dis,pos)

    if i==0:      
        position.append(pos)   
        neighbors.append(apor[pos:pos+v])
        neighbors2.append(apor[pos+v:pos+2*v])
    else:
        bandera = True
        for p in position:
            if (abs(pos - p) < tol*v):
                bandera = False
                i = i - 1
                break
        if bandera == True:
            #print(pos,p)
            position.append(pos)   
            neighbors.append(apor[pos:pos+v])
            neighbors2.append(apor[pos+v:pos+2*v])
            bandera = False
    i = i + 1
    if i == k:
        break

## Convertimos a numpy.  
neighbors  = np.array(neighbors)
neighbors2 = np.array(neighbors2)
print('position',position)    ## posición de los k vecinos mas cercanos.
#print(neighbors)   ## k vecinos mas cercanos.
#print(neighbors2)  ## ventana de datos posterior a los k vecinos mas cercanos.

print_serie(neighbors, 'KNN - Aportaciones hidrológicas en Presa Peñitas','MMC','meses', False,'fig_t13_aportaciones')
```

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_aportaciones.png)

Aprovecharemos estos patrones para hacer un pronóstico usando regresión lineal múltiple con stepwise.

```python
X   = sm.add_constant(neighbors.T)
X_2 = sm.add_constant(neighbors2.T)
y   = apor[n-v:n]

model   = sm.OLS(y, X)
results = model.fit()
result_prediction = results.predict(X_2)
#print(result_prediction)
#print(results.summary())

## Se ordenan los valores p y se selecciona el más grande.
i = 0
pvalues = []
for pi in results.pvalues:
    pvalues.append((i,pi))
    i = i + 1
pvalues.sort(key=lambda tup: tup[1])
(i, pi) = pvalues[0]

## Proceso de stepwise
while pi > 0.05:
    print('Retiramos regresor X' + str(i))
    X   = np.delete(arr=X,   obj=i, axis=1)
    X_2 = np.delete(arr=X_2, obj=i, axis=1)
    model   = sm.OLS(y, X)
    results = model.fit()

    ## Se ordenan los valores p y se selecciona el más grande
    i = 0
    pvalues = []
    for pi in results.pvalues:
        pvalues.append((i,pi))
        i = i + 1
    pvalues.sort(key=lambda tup: tup[1])
    (i, pi) = pvalues[0]
    print(pi)

result_prediction = results.predict(X)
print(result_prediction)
print(results.summary())

# Exactitud del modelo
print('Test MAE OLS + stepwise= ', mean_absolute_error(y,y_real))
```
El resultado de la regresión stepwise y el pronóstico de aportaciones se muestran a continuación:
```
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.928
Model:                            OLS   Adj. R-squared (uncentered):              0.903
Method:                 Least Squares   F-statistic:                              38.44
Date:                Fri, 22 Apr 2022   Prob (F-statistic):                    1.85e-05
Time:                        22:56:32   Log-Likelihood:                         -71.320
No. Observations:                  12   AIC:                                      148.6
Df Residuals:                       9   BIC:                                      150.1
Df Model:                           3                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.3070      0.445      0.690      0.507      -0.699       1.313
x2             0.4654      0.188      2.479      0.035       0.041       0.890
x3             0.2166      0.479      0.452      0.662      -0.867       1.300
==============================================================================
Omnibus:                        0.830   Durbin-Watson:                   0.950
Prob(Omnibus):                  0.660   Jarque-Bera (JB):                0.586
Skew:                          -0.480   Prob(JB):                        0.746
Kurtosis:                       2.501   Cond. No.                         11.7
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
Test MAE OLS + stepwise=  149.97166666666666

```
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t13_aportaciones_prono3.png)

### Conclusiones tarea 13
En esta tarea se clasificaron diferentes dias de acuerdo a la radiación por hora. Diversas condiciones climáticas principalmente la nubosidad provocan que la producción de una planta fotovoltaica sea diferente a la de cielo despejado. Los métodos de agrupamiento utilizados fueron **k-means, LVQ, Gaussian mixtures** y **KNN** usando las librerias de **sklearn**. Además se implementó el **KNN** para encontrar los `k` dias mas parecidos a un día determinado. Este método se probó con los centroides previamente encontrados en el **k-means**. Adicionalmente, se implementó el método propuesto por [Grzegorz Dudek](https://doi.org/10.1016/j.epsr.2015.09.001) para extraer muestras de una serie de tiempo usando **KNN**  y haciendo una regresión con las muestras encontradas para obtener un pronóstico. Para probar el método se usaron datos de aportaciones hidrológicas de presa Peñitas en México.

---

## **Tarea 14 Aprendizaje no supervisado**
>**Instrucciones:** After reading the whole chapter, pick any three techniques introduced in it and apply them to your data. Make use of as many libraries as you please in this occasion. Discuss the drawbacks and advantages of each of the chosen techniques.

Los datos de demanda están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing). Los datos de generación eólica se encuentran disponibles en [Eolicas.csv](https://drive.google.com/file/d/1FNMdGkhjypcGTAtPeOfw12EuAolUJ4Fh/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea14.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea14.ipynb). Aquí solo se presentan los resultados y secciones relevantes del código.

### Análisis de componentes principales aplicado a reducir dimensiones en pronóstico de demanda eléctrica.

Análisis de componentes principales es un método de reducción de dimensiones que puede ser usado para representar con menos variables los datos originales. El método genera otras variables sintéticas llamadas **componentes** que pueden explicar partes importantes del fenómeno y demás ser ortogonales entre si, esto ayuda a prevenir [**multicolinealidad**](https://medium.com/@awabmohammedomer/principal-component-analysis-pca-in-python-6897664f97d6#:~:text=PCA%20aims%20to%20reduce%20dimensionality,original%20data%20with%20less%20noise.) en modelos de regresión. Estos **componentes** principales pueden utilizarse como regresores para ajustar un nuevo modelo.

Por ejemplo, tenemos un conjunto de datos de demanda de 18 dias y queremos obtener un modelo de regresión que explique los datos de hoy con los días pasados. Sin embargo, al graficar los días observamos una alta correlación entre estos dias, por supuesto este es un comportamiento esperado debido a que estos dias fueron seleccionados por una similitud con el actual. Por lo que necesitamos un método que además de reducir las dimensiones también sea capaz de reducir el factor de inflación de la varianza (**VIF**).

A continuación se muestra la matriz de correlación de los regresores, para una mayor claridad solo se han graficado las variables `X1` al `X12` de las 17.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_corr_dem.png)

Se calculan los datos del **VIF** de cada regresor, un **VIF** de arriba de 20 es muy alto y puede traer problemas para la solución del sistema lineal que encuentra los coeficientes de la regresión.

Ahora, calculamos la **regresión lineal múltiple** entre los datos de demanda del día de hoy `y_train` y los datos de los días pasados `X_train`. También calculamos el error MAE entre la regresión con los datos de entrenamiento y los de prueba `X_test`.
```python
model = LinearRegression().fit(X_train[:, :], y_train)
err1_mae = np.mean(np.abs(y_train - model.predict(X_train)))
err1_mae_test = np.mean(np.abs(y_test - model.predict(X_test)))
print("MAE del modelo de regresión con datos de entrenamiento con sklearn:", err1_mae)
print("MAE del modelo de regresión con datos de prueba con sklearn:", err1_mae_test)

MAE del modelo de regresión con datos de entrenamiento con sklearn: 102.03653229489723
MAE del modelo de regresión con datos de prueba con sklearn: 131.86027342669107
```

Además, se calculan los datos del **VIF** de cada regresor, un **VIF** de arriba de 20 es un factor muy alto, que puede traer problemas para la solución del sistema lineal que encuentra los coeficientes de la regresión. Como se puede ver los niveles de **VIF** son muy altos en nuestros datos originales.
```python
# calculating VIF for each feature
for i in range(X_train.shape[1]):
    print('X[',i,'] =',variance_inflation_factor(X_train,i))  
```
```    
VIF X[ 1 ] = 19.268815602047138
VIF X[ 2 ] = 25.900291272307413
VIF X[ 3 ] = 21.75005645732436
VIF X[ 4 ] = 31.3039369332227
VIF X[ 5 ] = 16.94503888144909
VIF X[ 6 ] = 23.182547018706778
VIF X[ 7 ] = 21.415405802141464
VIF X[ 8 ] = 32.21371857799564
VIF X[ 9 ] = 13.042998282179228
VIF X[ 10 ] = 19.990213915947887
VIF X[ 11 ] = 18.269618431784952
VIF X[ 12 ] = 9.088089141611725
VIF X[ 13 ] = 4.594616265979086
VIF X[ 14 ] = 33.45388030260464
VIF X[ 15 ] = 26.546509209585857
VIF X[ 16 ] = 31.938715377963362
VIF X[ 17 ] = 17.948772566166706
VIF X[ 18 ] = 21.219310770169713
```

Ahora, usaremos **PCA** para obtener los componentes que capturen la información de un porcentaje de la varianza. Los componenetes encontrados, serán usados como variables se utilizan como regresores al ajustar un nuevo modelo de regresión.
Obtenemos los **componentes principales** tratando de explicar la mayor cantidad de varianza posible al 99.99%.
```python
pca = PCA(0.9999) 
pca.fit(X_train)
X_pca = pca.transform(X_train)
X_pcat = pca.transform(X_test)
print(sum(pca.explained_variance_ratio_ * 100))
```
El diagrama siguiente muestra en el eje de las `x` los componentes y en el eje de las `y` el nivel de varianza explicado, la suma de toda la varianza explicada es aproximadamente de 99.99%. El número de componentes encontrado es de 17.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14__variance_pca_9999per.png)

Ahora, calculamos la **regresión lineal múltiple** entre los datos de demanda del día de hoy `y_train` y los componentes `X_pca`. También calculamos el error MAE entre la regresión con los datos de entrenamiento y los de prueba `X_pcat`.
```python
linreg_model = LinearRegression().fit(X_pca[:, :], y_train)
err2_mae = np.mean(np.abs(y_train - linreg_model.predict(X_pca)))
err2_mae_test = np.mean(np.abs(y_test - linreg_model.predict(X_pcat)))
print("MAE del modelo de regresión con datos de entrenamiento con sklearn:", err2_mae)
print("MAE del modelo de regresión con datos de prueba con sklearn:", err2_mae_test)
```
```
MAE del modelo de regresión con datos de entrenamiento con sklearn: 102.03653229489727
MAE del modelo de regresión con datos de prueba con sklearn: 131.86027342669115
```
Los resultados del error en el ajuste de los datos de entrenamiento y prueba entre los datos originales (`X_train`,`X_test`) y los componentes calculados con **PCA** (`X_train`,`X_test`) son muy parecidos.

Además, una ventaja de las nuevas variables es que el VIF se ha reducido tal como se muestra a continuación:
```python
# calculating VIF for each feature
for i in range(X_pcat.shape[1]):
    print('X_pca[',i,'] =',variance_inflation_factor(X_pcat,i))
```
```
VIF X_pca[ 0 ] = 1.0916455582847158
VIF X_pca[ 1 ] = 1.0601004320461906
VIF X_pca[ 2 ] = 1.1168117876529722
VIF X_pca[ 3 ] = 1.1457894613708417
VIF X_pca[ 4 ] = 1.1523412120435788
VIF X_pca[ 5 ] = 1.1536215578353162
VIF X_pca[ 6 ] = 1.1491477461449198
VIF X_pca[ 7 ] = 1.0731741487998225
VIF X_pca[ 8 ] = 1.1136973711933749
VIF X_pca[ 9 ] = 1.0868154380237036
VIF X_pca[ 10 ] = 1.0707778739647433
VIF X_pca[ 11 ] = 1.1465545728928228
VIF X_pca[ 12 ] = 1.1145628898435251
VIF X_pca[ 13 ] = 1.0872008821946384
VIF X_pca[ 14 ] = 1.1077344674401515
VIF X_pca[ 15 ] = 1.1459797454302063
VIF X_pca[ 16 ] = 1.08919860941622
VIF X_pca[ 17 ] = 1.0832319910231567
```
Finalmente, después de demostrar las ventajas y utilidad del **PCA** en la regresión, la utilizaremos para reducir las dimensiones en nuestros datos. Explicaremos el 98% de la varianza con nuestras nuevas variables. El número de **componentes** resultante es de ocho, en la gráfica siguiente se muestra cada **componente** y su varianza explicada.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_variance_pca2_98perc.png).

Además, la reducción del **VIF** es notable y el error MAE es muy semejante a los errores obtenidos con los datos originales.
```
VIF X_pca[ 0 ] = 1.0202126662701723
VIF X_pca[ 1 ] = 1.0309617986088888
VIF X_pca[ 2 ] = 1.0433254564831038
VIF X_pca[ 3 ] = 1.0802131687107597
VIF X_pca[ 4 ] = 1.0239019535969995
VIF X_pca[ 5 ] = 1.0637878942632697
VIF X_pca[ 6 ] = 1.0605053887139235
VIF X_pca[ 7 ] = 1.0088605587066386

MAE del modelo de regresión con datos de entrenamiento con sklearn: 111.79413703293265
MAE del modelo de regresión con datos de prueba con sklearn: 130.55280746630132
```
Adicionalmente, un diagrama de correlación entre los componentes puede confirmar que la baja correlación entre los componentes.
![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_corr_pca.png)


### Análisis factorial en la selección de parques eólicos representativos por regiones.

En México, se han instalado una gran cantidad de párques eólicos en los últimos años. Los parques están distribuidos en su mayoria en en el Itzmo de Tehuantepec, Oaxaca, pero tenemos algunos en Chiapas, Nuevo León, Tamaulipas, Baja California y otros estados. La generación eólica es tan intermitente como el viento, su fuente  de energía primaria. Por lo que es necesario pronosticar su producción de la mejor manera. Algunas técnicas de pronóstico son mas eficientes si se hace un pronóstico por región y no por parque, estas técnicas llamadas **Upscalling** fueron inicialmente propuestas por [Nils Siebert y Georges Kariniotakis](https://hal-mines-paristech.archives-ouvertes.fr/file/index/docid/526690/filename/EWEC_2006_SIEBERT_KARINIOTAKIS.pdf) y aplicadas en Dinamarca. 

En esta sección de la tarea usaremos la técnica de análisis factorial (**AF**) para hacer **upscalling** en 35 parques eólicos de diferentes regiones con datos de generación de aproximadamente un año. Si se desea conocer la ubicación de algunos de estos parques, ver el [mapa](https://consultaindigenajuchitan.files.wordpress.com/2015/01/01-parques-eolicos-istmo.jpg).

Iniciamos haciendo un análisis exploratorio para confirmar la alta correlación entre algunos de los parques debido a que la mayoría de ellos se encuentran en la región de Oaxaca y Chiapas.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_corr_eol.png)

Ahora, haremos el análisis factorial. Tal como lo sugiere el [libro](https://link.springer.com/book/10.1007/978-0-387-84858-7) en el apartado `14.7.1 Latent Variables and Factor Analysis`, compararemos los resultados de los componentes del **PCA**, los factores del **AF** con y sin la rotación **varimax**. El número de factores escogido es de cuatro.

```python
n_fact= 4
methods = [("PCA", PCA()),("FA sin rotación", FactorAnalysis()),("FA con varimax", FactorAnalysis(rotation="varimax")),]
fig, axes = plt.subplots(ncols=len(methods), figsize=(10,9))

for ax, (method, fa) in zip(axes, methods):
    fa.set_params(n_components=n_fact)
    fa.fit(X)

    components = fa.components_.T
    print("\n\n %s :\n" % method)
    print(components)

    vmax = np.abs(components).max()
    ax.imshow(components, cmap="bwr", vmax=vmax, vmin=-vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    if ax.is_first_col():
        ax.set_yticklabels(feature_names, color=LETRASNARA, fontsize='large')
        plt.tick_params(colors = LETRASNARA)
    else:
        ax.set_yticklabels([], color=LETRASNARA, fontsize='large')
        plt.tick_params(colors = LETRASNARA)

    ax.set_title(str(method), color=LETRASNARA, fontsize='x-large')
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["F1","F2","F3","F4"], color=LETRASNARA, fontsize='large')

    ax.spines['bottom'].set_color(LETRASNARA)
    ax.spines['top'   ].set_color(LETRASNARA) 
    ax.spines['right' ].set_color(LETRASNARA)
    ax.spines['left'  ].set_color(LETRASNARA)

plt.tight_layout()
plt.savefig('fig_t14_varimax', transparent=True) 
plt.show()
```
La técnica de **AF** a diferencia de **PCA**, puede ayudar a descubrir patrones latentes. Además, aunque la rotación **varimax** a los componentes no mejora la calidad de la predicción, puede ayudar a visualizar su estructura. Podemos decir que el FA con varimax puede dar más nitidéz de la pertenencia de las variables a los factores.

Aunque esta técnica es un poco subjetiva (tal como lo menciona el libro), puede ayudar a identificar los parques que tienen un comportamiento parecido. En el siguiente gráfico se observa que en *FA con rotación varimax* la pertenencia de las variables [EOL1,EOL2...EOL36] a los factores [F1,F2,F3,F4] es más evidente.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_varimax.png)

| FACTOR         | PARQUES              |
| :------------- | :-------------:      |
|    1           | EOL1-EOL22, EOL24, EOL27-EOL28, EOL30-EOL32       |
|    2           | EOL23, EOL26         |
|    3           | EOL33, EOL34, EOL25, EOL29 |
|    4           | EOL35, EOL36         |

Un tutorial para interpretar los resultados del **AF** puede verse en esta [liga al tutorial](https://support.minitab.com/es-mx/minitab/18/help-and-how-to/modeling-statistics/multivariate/how-to/factor-analysis/interpret-the-results/all-statistics-and-graphs/)

### Agrupamiento jerárquico

En la sección `14.3.12 Hierarchical Clustering` del [libro](https://link.springer.com/book/10.1007/978-0-387-84858-7) se discute la construcción de un **dendograma** como  herramienta de análisis posterior al agrupamiento de en este caso las variables de generación eólica de los 35 parques en México. Un tutorial para aprender a interpretar el dendograma es encontrado [Dendograma Minitab](https://support.minitab.com/es-mx/minitab/18/help-and-how-to/modeling-statistics/multivariate/how-to/cluster-observations/interpret-the-results/all-statistics-and-graphs/dendrogram/)

```python
import scipy.cluster.hierarchy as shc

fig, ax = plt.subplots(figsize=(10,10))
plt.tick_params(colors = LETRASNARA, which='both')
ax.spines['bottom'].set_color(LETRASNARA)
ax.spines['top'   ].set_color(LETRASNARA) 
ax.spines['right' ].set_color(LETRASNARA)
ax.spines['left'  ].set_color(LETRASNARA)    

plt.title("Dendograma de agrupamiento jerárquico",color=LETRASNARA,fontsize='x-large')

dend = shc.dendrogram(shc.linkage(X.T, method='ward'))

plt.savefig('fig_t14_dendogram2', transparent=True)  
plt.show()

```

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t14_dendogram.png)

El dendrograma presentado jerarquiza los parques por similitud, está dividido en dos conglomerados (uno en color verde y uno en azul), a un nivel de similitud de aproximadamente 160. El primer conglomerado (verde) se compone de las 26 variables, el segundo conglomerado (azul), se compone de las otras nueve variables. Los resultados de la agrupación se resumen en la siguiente tabla:

| CONGLOMERADO   | PARQUES                                        |
| :------------- | :-------------:                                |
|    1           | EOL1-EOL21, EOL23, EOL26, EOL27, EOL29, EOL31  |
|    2           | EOL22, EOL24, EOL25, EOL28, EOL30, EOL32-EOL35 |

Las agrupaciones encontradas por el **HC** corresponden en su mayoria con las propuestas por el método **AF**, sin embargo, a diferencia del **AF**, con el dendograma podemos hacer un análisis de similitud entre cada parque. Además, si cortamos el dendograma más abajo, podemos definir un mayor número de conglomerados de menor tamaño que hará que el nivel de similitud de los elementos sea mayor y crear nuevas agrupaciones.

### Conclusiones tarea 14
En esta tarea se demostró la utilidad del **PCA** como reductor de dimensiones y además como reductor del índice de inflación de la varianza **VIF**, se utilizaron los componentes principales para reducir el número de regresores en el pronóstico de demanda. Las pruebas demostraron la equivalencia en exactitud entre las variables originales y las variables calculadas por **PCA**. Muy importante que el método es eficaz para reducir la multicolinealidad.
Por otro lado, se agruparon parques eólicos usando la técnica de análisis factorial **AF**y el de **conglomerados jerárquicos**. En el método de **AF** se obtuvieron los pesos de cada variable sobre cada factor y se agruparon las que tenian un peso mayor. Para el caso del método de conglomerados jerárquicos, se dibujó un dendograma, en el que facilmente puede visualizarse la cercania o similitud entre los elementos o variables. Los resultados entre ambas técnicas fueron en su mayoria semejentes. Sin embargo el dendograma permite visualizar la relación entre las variables, es decir es posible ver en las últimas hojas del dendograma aquellos parques  que tienen un comportamiento más parecido.

---

## **Tarea 15 Bosque aleatorio**
>**Instrucciones:** After carefully reading all of Chapter 15 (regardless of how much of Section 15.4 results comprehensible), train (and evaluate) a random forest on your project data and compute also the variable importance and the proximity matrix corresponding to the forest..

Los datos de demanda están disponibles en [demanda.csv](https://drive.google.com/file/d/1KpY2p4bfVEwGRh5tJjMx9QpH6SEwrUwH/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea15.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea14.ipynb). Aquí solo se presentan los resultados y secciones relevantes del código.

En esta tarea usaremos la técnica de bosque aleatorio (**RF**) en su versión de regresión para predecir la demanda eléctrica de los próximos siete dias, a partir de datos de semanas con alta correlación con la semana actual, la selección de las semanas correlacionadas será utilizando vecinos más cercanos (**KNN**) y las medidas de correlación serán el coeficiente de correlación de pearson y la distancia euclidiana. Los resultados serán comparados con los obtenidos por **RF** y la regresión lineal múltiple (**OLS**) con su versión de **stepwise** con una significancia de 0.001.

A continuación, presentamos las serie de tiempo de demanda eléctrica que se desea pronosticar, se muestran solo nueve semanas, la última semana en azul, será usada como prueba, mientras que la serie color café será de entrenamiento.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_demanda.png)

En la gráfica siguiente mostramos las semanas seleccionadas con alta correlación usando el método de **KNN** usando el coeficiente de **correlación de pearson** como distancia. En color rojo se representan los datos de la semana actual.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_X_pearson_RF.png)

Ahora mostramos las semanas seleccionadas con alta correlación usando el método de **KNN** usando la **distancia euclidiana***. Nuevamente, en color rojo se representan los datos de la semana actual.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_X_euclidian_RF.png)

Para nuestras pruebas usaremos la implementación de **RandomForestRegressor** de la librería **sklearn**. Iniciaremos entrenando un **RF** utilizando los parámetros por defecto.

```python
    if typereg == 'RF':
        model         = RandomForestRegressor(random_state=42) 
        results       = model.fit(X, Y)
        prediction_Y2 = results.predict(X_2)
        print_importances(model_=model,labels_=positions,namefile_='fig_t15_importance_'+typedist+'_'+typereg)
```

En la gráfica siguiente mostramos la importancia de las variables en el **RF**, las variables en el eje de las 'y' representan las posiciones de inicio en la serie de demanda que fueron elegidas como regresores y en el eje 'x' la importancia relativa de cada una.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_importance_pearson_RF.png)

Otro modelo de **RT** es entrenado usando la librería **GridSearchCV** para sintonizar los parámetros del modelo. La selección es automática combinando los siguientes parámetros:

```python
        param_grid = { 
        'bootstrap': [True, False],
        'n_estimators': [10,60,110,160,210,260,310],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],}
```

Los parámetros que mejor ajustan a nuestros datos son: 
```
bootstrap:         True
n_estimators:      60
max_features:      sqrt
max_depth:         10
min_samples_leaf:  1
min_samples_split: 2
```

Ahora mostramos la importancia de las variables del **RF** entrenado con **GridSearchCV**:

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_importance_pearson_AutoRF.png)

Hemos repetido el ejercicio de seleccionar las semanas de mayor correlación usando **KNN** con la distancia euclidiana y aplicado la regresión con **RF** con y sin ajuste de parámetros por **GridSearchCV**. Además, hemos calculado la regresión lineal múltiple con el objetivo de comparar los resultados.

La comparación de los erores en los pronósticos se muestran en la tabla siguiente:

| SELECCIÓN      | REGRESIÓN      | MAE             | MAPE             | TIEMPO SELECCIÓN     | TIEMPO REGRESIÓN   |
| :------------- | :------------- | -------------:  | -------------:   |-------------:        |-------------:        |
| PEARSON        | OLS + STEP     | 0.4435          | 2.8163          | 3.2663               |    0.7149 |
| EUCLIDIAN      | OLS + STEP     | 0.8996           | 6.1071            | 21.7784              |    0.6169 |
| PEARSON        | RF             | 0.3975          | 3.1059           | 7.6678               |    3.9034 |
| EUCLIDIAN      | RF             | 0.2428           | 2.0313          | 21.7855               |    1.8006  |
| PEARSON        | AutoRF            | 0.3434          | 2.8154           | 8.1127               |    398.406 |
| EUCLIDIAN      | AutoRF            | 0.1982          | 1.2627           | 37.126               |    387.9376 |

Por último, se comparan los pronósticos obtenidos contra los datos reales (en color rojo). Los resultados son mucho mejor con la regresión con **RF** (series azul, naranja, gris) que con la regresión lineal (verde y cyan).

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t15_ajuste_prono1.png)

### Conclusiones tarea 15
En esta tarea se aplicó el **RF** en su versión de regresión para pronosticar demanda eléctrica a siete dias. Las pruebas demostraron un menor error sobre regresión lineal incluso con la versión de **RF** con los parámetros por defecto. Sin embargo, un mejor desempeño del modelo se puede lograr ajustando los parámetros, para esto usamos la librería **GridSearchCV** que lo hace automáticamente. Una desventaja de **RF** es que la selección de parámetros en estos modelos es mucho más lenta y costosa computacionalmente que los modelos lineales.
