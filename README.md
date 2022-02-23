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

---

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

### Evaluación del desempeño del muestreo bootstrap variando tamaño de las muestra y numero de muestreos akleatorios (repeticiones).
Adicionalmente se ha hecho un análisis del error del **bootstrap**, variando el tamaño de la muestra en porciento del total de los datos `percent = [10,20,30,40,50,60,70,80,90]` y un número de simulaciones `replicas = [250,500,1000,1500,2000]`, los resultados se muestran en la gráfica siguiente.

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t7_4.png)

Se ha determinado que para estos datos el número mínimo de muestras aleatorias es de 80% de los datos con 500 repeticiones para obtener un resultado aceptable, la gráfica de error contra número de réplicas se muestra a continuación. La línea roja representa la media del error, y las líneas pubnteadas los intervalos de confianza del 5% y 95% , Asumiendo una distribución normal en el error. 

![image](https://github.com/urieliram/statistical/blob/main/figures/fig_t7_5.png)

**Con este análisis podemos cuantificar el efecto en la calidad del modelo de usar diferente número de réplicas o muestras para conocer la cantidad de valores "suficientes" o "ideales" para los datos y determinar el punto en que agregar más datos o más réplicas ya no cambia nada en el modelo.**

### **Conclusión tarea 7** 
Se realizó un ejercicio de predicción de demanda eléctrica usando una regresión lineal múltiple, sin embargo debido a los pocos datos que se tienen para evaluar el modelo. Se aplicaron técnicas de validación cruzada y **bootstrap**, las cuales son una herramienta poderosa para evaluar la función del error. Ambas técnicas hacen un muestreo con los datos y evaluan el error en el modelo, resulta interesante observar las distribuciones que resultan parecidas a la distribución nornal para el **bootstrap** y para el caso de validación cruzada una distribución exponencial. El uso de estas técnicas tiene como ventaja obtener una distribución más realista del comportamiento del error e inclusive poder calcular intervalos de confianza.


## **Tarea 8 Inferencia**
>**Instrucciones:** Modelar la sobrecarga a base de observaciones que tienes para llegar a un modelo tipo "en estas condiciones, va a fallar con probabilidad tal".

Los datos usados en esta sección están disponibles en [bones.csv](https://drive.google.com/file/d/1Q8Pk5apApNbcoqmKQp3RvQFvuk4DKylU/view?usp=sharing) [overload.csv](https://drive.google.com/file/d/1-ZCl-XLmmCpe_yNGryl7Eudg3Q_Xhyh8/view?usp=sharing). El código completo de esta tarea se encuentra en [Tarea8.ipynb](https://github.com/urieliram/statistical/blob/main/Tarea8.ipynb), aquí solo se presentan los resultados y secciones relevantes del código.

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

