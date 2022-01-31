# Aprendizaje automático

Repositorio de actividades del curso de aprendizaje automático. La descripción del curso y las actividades se pueden encontrar en el [enlace del curso](https://github.com/satuelisa/StatisticalLearning). Los datos a usar del libro están disponibles aquí: [dataset del libro](https://hastie.su.domains/ElemStatLearn/datasets/).

---

+ [Tarea 1 Introducción](#tarea-1-introduction)
+ [Tarea 2 Aprendizaje supervisado](#tarea-2-aprendizaje-supervisado)
+ [Tarea 3 Regresión lineal](#tarea-3-regresión-lineal)
+ [Tarea 4 Clasificación](#tarea-4-clasificación)
+ [Tarea 5 Expansión de base](#tarea-5-expansión-de-base)

---

## Tarea 1 Introducción
>**Instructions:** Identify one or more learning problems in your thesis work and identify goals and elements.

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
        results = results.append(pd.DataFrame([{'num_features': k,
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
menorpi.item()  0.9062256410652616
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
## Análisis Discriminante Lineal y Cuadrático con datos de prueba en dos dimensiones
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

## Análisis Discriminante Lineal y Cuadrático aplicada a clasificación de regiones de consumo de electricidad.
Los datos que se usarán en este ejercicio son resultados de la planeación de la operación eléctrica del sistema eléctrico interconectado en México que consta de 320 instancias. Las columnas de los datos son los resultados por región y por hora del día. Se dan resultados de generación térmica (GenTer), generación hidráulica (GenHid), generación renovable (GenRE), Generación no programable (GenNP), Generación total (GenTot), demanda de la región (Demanda), Cortes de energía (Corte), Excedentes de energía(Excedente),Potencia (PotInt), precio Marginal y pérdidas de la región.

La clases [1,2,3,4,..,67] son las regiones y los regresores son ['GenTer','GenHid','GenRE','GenNP','Demanda','Perdidas','PrecioMarginal'].

Se muestra una proyección de los datos entre las variables de generación térmica por región ['GenTer'] y ['Demanda'] (derecha), así como ['GenTer'] y ['Demanda'] (izquierda).

![image](https://github.com/urieliram/statistical/blob/main/figures/scatter3.png) 
![image](https://github.com/urieliram/statistical/blob/main/figures/scatter4.png)

Ahora, aplicamos un modelo **LDA** y obtenemos la exactitud **(score)** en una escala de 0 a 1.
```
Test accuracy LDA = 0.5760
```
Ahora, aplicamos un modelo **QDA** y obtenemos la exactitud **(score)** en una escala de 0 a 1.
```
Test accuracy QDA = 0.3620
```
Graficamos la matriz de confusión del modelo **QDA** aplicado a predicción de regiones eléctricas

![image](https://github.com/urieliram/statistical/blob/main/figures/confusion3.png) 

### Conclusiones Tarea 4
En esta tarea se utilizaron los métodos **Regresión Logística**, **LDA** y **QDA**. Se resolvió el ejemplo del libro sobre predicción de enfermedades cardiacas con multifactores usando regresión logística y **stepwise**. Además, se utilizó LDA y QDA para resolver un ejemplo sencillo de clasificación de dos variables con tres clases, se hizo el análisis exactitud de los métodos, resultando con muy buenos resultados (score LDA = 0.9906, score QDA = 0.9933), para ambos casos se realizaron figuras donde se observa la clasificación en áreas por medio de hiperplanos. Por último, se realizó el ejercicio de clasificación con datos de regiones eléctricas donde las clases son las regiones y los regresores los resultados de planeación de demanda, generación y pérdidas eléctricas en cada región. Los resultados fueron mejores con el método LDA = 0.5760 que con el método QDA = 0.3620.

## Tarea 5 Expansión de base
>Fit splines into single features in your project data. Explore options to obtain the best fit.
